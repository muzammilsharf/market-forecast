"""
Fetch OHLCV price data for all tickers defined in tickers.py.

Pipeline
--------
1. Ensure the raw data directory exists (RAW_DATA_DIR from paths.py).
2. Attempt to download each ticker with a 2-second inter-request delay.
3. Retry every ticker that failed in the first pass.
4. If permanently failed tickers exceed threshold, abort to prevent corrupted/partial saves.
5. Standardise column names and dtypes, storing date as a column and keeping volume nullable.
6. Concatenate all DataFrames and export to PRICES_PARQUET (paths.py).
7. Return the list of permanently failed tickers.
"""

from __future__ import annotations

import logging
import time
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf  # pyright: ignore[reportMissingTypeStubs]

from market_forecast.config import (
    COLUMN_RENAME,
    MAX_ALLOWED_FAILED,
    REQUEST_DELAY,
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
# Single-ticker fetch
# ---------------------------------------------------------------------------
def _download_ticker(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Download OHLCV data for *symbol* between start_date and end_date.

    Returns an empty DataFrame on failure.
    """
    try:
        result = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
        )
        if result is None or result.empty:
            logger.warning("No data returned for %s.", symbol)
            return pd.DataFrame()
        return result
    except Exception as exc:
        logger.error("Error fetching %s: %s", symbol, exc)
        return pd.DataFrame()


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
    # yfinance >= 0.2 returns a MultiIndex when group_by='column' (default).
    # Flatten if needed.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only the columns we care about (present columns only)
    keep = [c for c in COLUMN_RENAME if c in df.columns]
    df = df[keep].copy()

    df.rename(columns=COLUMN_RENAME, inplace=True)

    # Cast price columns to float64
    for col in ["open", "high", "low", "close", "adj_close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.dtype("float64"))

    # Cast volume to nullable Int64 so NaNs don't raise and can be caught by quality checks
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
) -> list[str]:
    """
    Full data-fetch pipeline:
      - compute end_date (defaults to date.today() at runtime)
      - ensure directory exists
      - first-pass download (all tickers, 2-second delay)
      - retry failed tickers
      - abort if permanently failed tickers exceed max_allowed_failed
      - standardise columns / dtypes (date as a column, volume nullable)
      - export concatenated DataFrame to Parquet
      - return list of permanently failed tickers

    Parameters
    ----------
    start_date : date
        Start date for data fetch (default: START_DATE = 2020-01-01).
    end_date : date | None
        End date for data fetch. Computed at runtime as date.today() if None.
    max_allowed_failed : int
        Maximum number of tickers allowed to fail permanently. If more than this
        number fail, the pipeline aborts without saving a partial file.

    Returns
    -------
    list[str]
        List of ticker symbols that permanently failed to fetch.

    Raises
    ------
    RuntimeError
        If no data was fetched or if permanently failed tickers exceed max_allowed_failed.
    """
    # Compute end_date dynamically inside function
    if end_date is None:
        end_date = date.today()

    ensure_data_directory()

    successful: list[pd.DataFrame] = []
    failed_tickers: list[dict[str, str]] = []

    # ------------------------------------------------------------------ #
    # First pass
    # ------------------------------------------------------------------ #
    logger.info(
        "Starting first-pass download for %d tickers from %s to %s ...",
        len(TICKERS),
        start_date,
        end_date,
    )

    for ticker_info in TICKERS:
        symbol = ticker_info["symbol"]
        sector = ticker_info["sector"]

        logger.info("[1st pass] Fetching %s ...", symbol)
        df = _download_ticker(symbol, start_date=start_date, end_date=end_date)

        if df.empty:
            failed_tickers.append(ticker_info)
        else:
            try:
                successful.append(_standardise(df, symbol, sector))
            except Exception as exc:
                logger.error("Standardisation failed for %s: %s", symbol, exc)
                failed_tickers.append(ticker_info)

        time.sleep(REQUEST_DELAY)

    # ------------------------------------------------------------------ #
    # Retry pass
    # ------------------------------------------------------------------ #
    permanently_failed: list[str] = []

    if failed_tickers:
        logger.warning(
            "%d ticker(s) failed in first pass; retrying: %s",
            len(failed_tickers),
            [t["symbol"] for t in failed_tickers],
        )

        for ticker_info in failed_tickers:
            symbol = ticker_info["symbol"]
            sector = ticker_info["sector"]

            logger.info("[Retry] Fetching %s ...", symbol)
            df = _download_ticker(symbol, start_date=start_date, end_date=end_date)

            if df.empty:
                permanently_failed.append(symbol)
                logger.error("Retry failed for %s -- permanently failed.", symbol)
            else:
                try:
                    successful.append(_standardise(df, symbol, sector))
                except Exception as exc:
                    logger.error("Standardisation retry failed for %s: %s", symbol, exc)
                    permanently_failed.append(symbol)

            time.sleep(REQUEST_DELAY)

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
    # Export
    # ------------------------------------------------------------------ #
    if not successful:
        error_msg = "No data fetched for any ticker -- aborting without saving."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    combined = pd.concat(successful, axis=0, ignore_index=True)
    combined.sort_values(["ticker", "date"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    logger.info(
        "Saving %d rows x %d columns -> %s",
        len(combined),
        combined.shape[1],
        PRICES_PARQUET,
    )
    combined.to_parquet(PRICES_PARQUET, index=False, engine="auto")
    logger.info("Export complete.")

    return permanently_failed