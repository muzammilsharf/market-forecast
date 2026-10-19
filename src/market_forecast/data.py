"""
Fetch OHLCV price data for tickers defined in tickers.py with batching, adaptive fallbacks,
and incremental updating with two weeks overlap.

Pipeline
--------
1. Ensure the raw data directory exists (RAW_DATA_DIR from paths.py).
2. Check existing parquet data to determine the last fetch date per ticker.
3. Fetch data with a two-week (14-day) overlap for existing tickers (START_DATE for new tickers).
4. Download in primary batches of 20 tickers.
   - If a batch fails as a whole, fall back to smaller batches (5 tickers).
   - If a smaller batch fails as a whole, fall back to single-ticker fetch.
5. Check and clean null rows for each ticker slice.
6. Retry any failed tickers with the same hierarchical fallback.
7. Standardise columns/dtypes, merge with existing data, and remove duplicates.
8. If permanently failed tickers exceed threshold, abort to prevent corrupted saves.
9. Export concatenated DataFrame to PRICES_PARQUET (paths.py).
10. Return the list of permanently failed tickers.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf  # pyright: ignore[reportMissingTypeStubs]

from market_forecast.config import (
    BATCH_SIZE,
    COLUMN_RENAME,
    MAX_ALLOWED_FAILED,
    OVERLAP_DAYS,
    SMALL_BATCH_SIZE,
    START_DATE,
)
from market_forecast.paths import PRICES_PARQUET, RAW_DATA_DIR
from market_forecast.tickers import TICKERS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Directory helper
# ---------------------------------------------------------------------------
def ensure_data_directory() -> None:
    """Create RAW_DATA_DIR (and parents) if they do not already exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Data directory ready: %s", RAW_DATA_DIR)


# ---------------------------------------------------------------------------
# Incremental fetch helpers: date inspection
# ---------------------------------------------------------------------------
def _get_existing_data(parquet_path: Path = PRICES_PARQUET) -> pd.DataFrame:
    """Load existing prices parquet if available, otherwise return empty DataFrame."""
    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            if not df.empty and "date" in df.columns and "ticker" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
                return df
        except Exception as exc:
            logger.warning("Could not read existing parquet file at %s: %s", parquet_path, exc)
    return pd.DataFrame()


def _determine_ticker_start_dates(
    tickers: list[dict[str, str]],
    existing_df: pd.DataFrame,
    default_start_date: date = START_DATE,
    overlap_days: int = OVERLAP_DAYS,
) -> dict[str, date]:
    """
    For each ticker, find its latest date in existing_df.
    If found, start_date = max(default_start_date, last_date - overlap_days).
    If not found, start_date = default_start_date.
    """
    start_dates: dict[str, date] = {}
    if existing_df.empty or "date" not in existing_df.columns or "ticker" not in existing_df.columns:
        for t in tickers:
            start_dates[t["symbol"]] = default_start_date
        return start_dates

    max_dates = existing_df.groupby("ticker")["date"].max()

    for t in tickers:
        symbol = t["symbol"]
        if symbol in max_dates:
            last_dt = max_dates[symbol]
            last_date = last_dt.date() if hasattr(last_dt, "date") else pd.to_datetime(last_dt).date()
            overlap_date = last_date - timedelta(days=overlap_days)
            start_dates[symbol] = max(default_start_date, overlap_date)
            logger.debug(
                "Ticker %s: last fetch date was %s. Overlap start date: %s",
                symbol,
                last_date,
                start_dates[symbol],
            )
        else:
            start_dates[symbol] = default_start_date

    return start_dates


# ---------------------------------------------------------------------------
# Basic checks for Null rows
# ---------------------------------------------------------------------------
def _clean_and_validate_ticker_slice(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Perform basic checks for null rows on a ticker's fetched slice:
    - Drop rows that are completely NaN across all columns.
    - Drop rows where all key price columns (open, high, low, close) are NaN.
    - If all rows are null or empty, return an empty DataFrame.
    """
    if df.empty:
        return pd.DataFrame()

    # Drop entirely empty rows
    cleaned = df.dropna(how="all").copy()
    if cleaned.empty:
        logger.warning("Ticker %s has only completely null rows.", symbol)
        return pd.DataFrame()

    # Check for rows where all price columns are NaN
    price_cols = [c for c in ["Open", "High", "Low", "Close", "open", "high", "low", "close"] if c in cleaned.columns]
    if price_cols:
        all_prices_nan = cleaned[price_cols].isna().all(axis=1)
        null_count = int(all_prices_nan.sum())
        if null_count > 0:
            logger.warning(
                "Ticker %s had %d row(s) with all-null price values; dropping these rows.",
                symbol,
                null_count,
            )
            cleaned = cleaned[~all_prices_nan]

    if cleaned.empty:
        logger.warning("Ticker %s has no valid non-null rows remaining after cleaning.", symbol)
        return pd.DataFrame()

    return cleaned


# ---------------------------------------------------------------------------
# Slice extraction
# ---------------------------------------------------------------------------
def _extract_ticker_slice(batch_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Extract a single ticker's OHLCV slice from a batch download DataFrame,
    and apply basic checks for null rows.
    """
    if batch_df.empty:
        return pd.DataFrame()

    if isinstance(batch_df.columns, pd.MultiIndex):
        # Case A: group_by='ticker' (ticker at level 0)
        if symbol in batch_df.columns.get_level_values(0):
            slice_df = batch_df[symbol]
            if isinstance(slice_df, pd.Series):
                slice_df = slice_df.to_frame()
            return _clean_and_validate_ticker_slice(slice_df, symbol)

        # Case B: group_by='column' (ticker at level 1)
        if symbol in batch_df.columns.get_level_values(1):
            slice_df = batch_df.xs(symbol, axis=1, level=1)
            if isinstance(slice_df, pd.Series):
                slice_df = slice_df.to_frame()
            return _clean_and_validate_ticker_slice(slice_df, symbol)

        return pd.DataFrame()

    # Flat column fallback (e.g. mock test data for a single ticker)
    return _clean_and_validate_ticker_slice(batch_df, symbol)


# ---------------------------------------------------------------------------
# Batch fetch with hierarchical fallback
# ---------------------------------------------------------------------------
def _download_batch(
    symbols: list[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Execute a yfinance download request for a list of *symbols*.
    Returns an empty DataFrame on failure or empty response.
    """
    if not symbols:
        return pd.DataFrame()
    try:
        result = yf.download(
            tickers=symbols,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
        )
        if result is None or result.empty:
            logger.warning("No data returned for batch of %d tickers: %s", len(symbols), symbols)
            return pd.DataFrame()
        return result
    except Exception as exc:
        logger.error("Error fetching batch %s: %s", symbols, exc)
        return pd.DataFrame()


def _fetch_symbols_with_fallback(
    symbols: list[str],
    start_date: date,
    end_date: date,
    small_batch_size: int = SMALL_BATCH_SIZE,
) -> dict[str, pd.DataFrame]:
    """
    Attempt to fetch data for `symbols`.
    - Level 1: Fetch full batch as a whole.
    - Level 2: If primary batch fails as a whole, fall back to smaller batches (default size 5).
    - Level 3: If a smaller batch fails as a whole, fall back to single-ticker fetch.

    Returns dict mapping symbol -> extracted, cleaned DataFrame.
    """
    results: dict[str, pd.DataFrame] = {}

    # Level 1: Primary batch fetch
    logger.info("Attempting primary batch fetch for %d tickers: %s", len(symbols), symbols)
    batch_df = _download_batch(symbols, start_date=start_date, end_date=end_date)
    batch_succeeded = not batch_df.empty and not batch_df.dropna(how="all").empty

    if batch_succeeded:
        for symbol in symbols:
            results[symbol] = _extract_ticker_slice(batch_df, symbol)
        return results

    # Level 2: Primary batch failed as a whole -> Fall back to smaller batches
    logger.warning(
        "Batch of %d tickers failed as a whole. Falling back to smaller batches of size %d ...",
        len(symbols),
        small_batch_size,
    )
    small_batches = [symbols[i : i + small_batch_size] for i in range(0, len(symbols), small_batch_size)]

    for s_idx, small_batch in enumerate(small_batches, start=1):
        logger.info(
            "Attempting small batch %d/%d (%d tickers: %s) ...",
            s_idx,
            len(small_batches),
            len(small_batch),
            small_batch,
        )
        small_df = _download_batch(small_batch, start_date=start_date, end_date=end_date)
        small_succeeded = not small_df.empty and not small_df.dropna(how="all").empty

        if small_succeeded:
            for symbol in small_batch:
                results[symbol] = _extract_ticker_slice(small_df, symbol)
        else:
            # Level 3: Smaller batch failed as a whole -> Fall back to single-ticker fetch
            logger.warning(
                "Smaller batch of %d tickers failed as a whole. Falling back to single-ticker fetch ...",
                len(small_batch),
            )
            for symbol in small_batch:
                logger.info("Fetching single ticker: %s ...", symbol)
                single_df = _download_batch([symbol], start_date=start_date, end_date=end_date)
                results[symbol] = _extract_ticker_slice(single_df, symbol)

    return results


def _download_ticker(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Download OHLCV data for a single symbol between start_date and end_date.

    Convenience wrapper around _download_batch. Returns an empty DataFrame on failure.
    """
    batch_df = _download_batch([symbol], start_date=start_date, end_date=end_date)
    return _extract_ticker_slice(batch_df, symbol)


# ---------------------------------------------------------------------------
# Column standardisation
# ---------------------------------------------------------------------------
def _standardise(df: pd.DataFrame, symbol: str, sector: str) -> pd.DataFrame:
    """
    Rename columns, cast dtypes, add *ticker* / *sector* metadata,
    and save date as a column (not an index).

    Volume is cast to nullable Int64 so that missing volume values
    do not raise an exception here, allowing quality checks to detect them.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    keep = [c for c in COLUMN_RENAME if c in df.columns]
    df = df[keep].copy()

    df.rename(columns=COLUMN_RENAME, inplace=True)

    # Cast price columns to float64
    for col in ["open", "high", "low", "close", "adj_close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.dtype("float64"))

    # Cast volume to nullable Int64
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype(pd.Int64Dtype())

    # Convert index to a timezone-naive 'date' column
    date_col = pd.to_datetime(df.index).tz_localize(None)
    df.reset_index(drop=True, inplace=True)
    df.insert(0, "date", date_col)
    df.insert(1, "ticker", symbol)
    df.insert(2, "sector", sector)

    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_datafetch(
    start_date: date = START_DATE,
    end_date: date | None = None,
    max_allowed_failed: int = MAX_ALLOWED_FAILED,
    batch_size: int = BATCH_SIZE,
    small_batch_size: int = SMALL_BATCH_SIZE,
    overlap_days: int = OVERLAP_DAYS,
) -> list[str]:
    """
    Full data-fetch pipeline:
      - compute end_date (defaults to date.today() at runtime)
      - ensure directory exists
      - inspect existing prices parquet to check last fetch date per ticker
      - fetch data with two weeks overlap for existing tickers (start_date for new tickers)
      - download in batches of `batch_size` (default 20 tickers, no request delay)
      - if a batch fails as a whole, fall back to smaller batches of `small_batch_size` (5),
        and if that fails, fall back to single-ticker fetch
      - apply basic checks for null rows on each ticker slice
      - retry failed tickers using the same adaptive fallback
      - merge with existing data, sort, and deduplicate overlapping records
      - abort if permanently failed tickers exceed max_allowed_failed
      - export concatenated DataFrame to Parquet
      - return list of permanently failed tickers

    Parameters
    ----------
    start_date : date
        Default start date for new tickers (default: START_DATE = 2020-01-01).
    end_date : date | None
        End date for data fetch. Computed at runtime as date.today() if None.
    max_allowed_failed : int
        Maximum number of tickers allowed to fail permanently. If more than this
        number fail, the pipeline aborts without saving a partial file.
    batch_size : int
        Primary batch size for yfinance download requests (default: 20).
    small_batch_size : int
        Fallback batch size when a primary batch fails as a whole (default: 5).
    overlap_days : int
        Days of overlap to fetch against each ticker's last fetch date (default: 14).

    Returns
    -------
    list[str]
        List of ticker symbols that permanently failed to fetch.

    Raises
    ------
    RuntimeError
        If no data was fetched or if permanently failed tickers exceed max_allowed_failed.
    """
    if end_date is None:
        end_date = date.today()

    ensure_data_directory()

    # Inspect existing data for incremental update
    existing_df = _get_existing_data(PRICES_PARQUET)
    ticker_start_dates = _determine_ticker_start_dates(
        TICKERS,
        existing_df,
        default_start_date=start_date,
        overlap_days=overlap_days,
    )

    # Group tickers by their required fetch start_date
    tickers_by_start: dict[date, list[dict[str, str]]] = defaultdict(list)
    for t in TICKERS:
        t_start = ticker_start_dates[t["symbol"]]
        tickers_by_start[t_start].append(t)

    successful: list[pd.DataFrame] = []
    failed_tickers: list[dict[str, str]] = []

    logger.info(
        "Starting download for %d tickers across %d start-date group(s) to %s ...",
        len(TICKERS),
        len(tickers_by_start),
        end_date,
    )

    # ------------------------------------------------------------------ #
    # First pass: batch downloads with fallback (no request delay)
    # ------------------------------------------------------------------ #
    for fetch_start, group in tickers_by_start.items():
        batches = [group[i : i + batch_size] for i in range(0, len(group), batch_size)]
        for batch_idx, batch in enumerate(batches, start=1):
            symbols = [t["symbol"] for t in batch]
            logger.info(
                "[1st pass] Group (%s): batch %d/%d (%d tickers: %s) ...",
                fetch_start,
                batch_idx,
                len(batches),
                len(symbols),
                symbols,
            )
            batch_results = _fetch_symbols_with_fallback(
                symbols,
                start_date=fetch_start,
                end_date=end_date,
                small_batch_size=small_batch_size,
            )

            for ticker_info in batch:
                symbol = ticker_info["symbol"]
                sector = ticker_info["sector"]
                ticker_df = batch_results.get(symbol, pd.DataFrame())

                if ticker_df.empty:
                    logger.warning("[1st pass] No valid data returned for %s.", symbol)
                    failed_tickers.append(ticker_info)
                else:
                    try:
                        successful.append(_standardise(ticker_df, symbol, sector))
                    except Exception as exc:
                        logger.error("Standardisation failed for %s: %s", symbol, exc)
                        failed_tickers.append(ticker_info)

    # ------------------------------------------------------------------ #
    # Retry pass: failed tickers with fallback (no request delay)
    # ------------------------------------------------------------------ #
    permanently_failed: list[str] = []

    if failed_tickers:
        logger.warning(
            "%d ticker(s) failed in first pass; retrying: %s",
            len(failed_tickers),
            [t["symbol"] for t in failed_tickers],
        )

        # Group failed tickers by their start dates
        retry_by_start: dict[date, list[dict[str, str]]] = defaultdict(list)
        for t in failed_tickers:
            t_start = ticker_start_dates.get(t["symbol"], start_date)
            retry_by_start[t_start].append(t)

        for fetch_start, retry_group in retry_by_start.items():
            retry_batches = [retry_group[i : i + batch_size] for i in range(0, len(retry_group), batch_size)]
            for r_idx, retry_batch in enumerate(retry_batches, start=1):
                symbols = [t["symbol"] for t in retry_batch]
                logger.info(
                    "[Retry] Group (%s): batch %d/%d (%d tickers: %s) ...",
                    fetch_start,
                    r_idx,
                    len(retry_batches),
                    len(symbols),
                    symbols,
                )
                batch_results = _fetch_symbols_with_fallback(
                    symbols,
                    start_date=fetch_start,
                    end_date=end_date,
                    small_batch_size=small_batch_size,
                )

                for ticker_info in retry_batch:
                    symbol = ticker_info["symbol"]
                    sector = ticker_info["sector"]
                    ticker_df = batch_results.get(symbol, pd.DataFrame())

                    if ticker_df.empty:
                        permanently_failed.append(symbol)
                        logger.error("Retry failed for %s -- permanently failed.", symbol)
                    else:
                        try:
                            successful.append(_standardise(ticker_df, symbol, sector))
                        except Exception as exc:
                            logger.error("Standardisation retry failed for %s: %s", symbol, exc)
                            permanently_failed.append(symbol)

        if permanently_failed:
            logger.error(
                "Permanently failed tickers (%d): %s",
                len(permanently_failed),
                permanently_failed,
            )
        else:
            logger.info("All previously-failed tickers recovered on retry.")

    # ------------------------------------------------------------------ #
    # Failure threshold gatekeep
    # ------------------------------------------------------------------ #
    if len(permanently_failed) > max_allowed_failed:
        error_msg = (
            f"Data fetch aborted: {len(permanently_failed)} tickers failed permanently, "
            f"exceeding the maximum allowed threshold of {max_allowed_failed}. "
            f"Failed tickers: {permanently_failed}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # ------------------------------------------------------------------ #
    # Export & deduplication against existing data
    # ------------------------------------------------------------------ #
    if not successful:
        error_msg = "No data fetched for any ticker -- aborting without saving."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    new_combined = pd.concat(successful, axis=0, ignore_index=True)

    if not existing_df.empty:
        logger.info(
            "Merging %d newly fetched rows with %d existing rows ...",
            len(new_combined),
            len(existing_df),
        )
        combined = pd.concat([existing_df, new_combined], axis=0, ignore_index=True)
    else:
        combined = new_combined

    combined["date"] = pd.to_datetime(combined["date"]).dt.tz_localize(None)
    combined.sort_values(["ticker", "date"], inplace=True)

    # Deduplicate overlapping rows resulting from the two weeks overlap (keep latest)
    initial_count = len(combined)
    combined.drop_duplicates(subset=["ticker", "date"], keep="last", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    deduped_count = initial_count - len(combined)

    if deduped_count > 0:
        logger.info("Deduplicated %d overlapping row(s) from incremental update.", deduped_count)

    logger.info(
        "Saving %d rows x %d columns -> %s",
        len(combined),
        combined.shape[1],
        PRICES_PARQUET,
    )
    combined.to_parquet(PRICES_PARQUET, index=False, engine="auto")
    logger.info("Export complete.")

    return permanently_failed