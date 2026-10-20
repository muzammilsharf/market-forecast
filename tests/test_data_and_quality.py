from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from market_forecast.data import (
    _clean_and_validate_ticker_slice,
    _determine_ticker_start_dates,
    _download_batch,
    _download_ticker,
    _extract_ticker_slice,
    _fetch_symbols_with_fallback,
    _standardise,
    check_overlap_adjustment,
    OverlapCheckResult,
    run_datafetch,
)
from market_forecast.quality import check_ticker, run_quality_checks
from market_forecast.tickers import TICKERS


def _sample_raw_df() -> pd.DataFrame:
    """Generate mock yfinance download output."""
    dates = pd.date_range("2020-01-02", periods=150, freq="B")
    return pd.DataFrame(
        {
            "Open": np.linspace(100.0, 150.0, len(dates)),
            "High": np.linspace(105.0, 155.0, len(dates)),
            "Low": np.linspace(95.0, 145.0, len(dates)),
            "Close": np.linspace(102.0, 152.0, len(dates)),
            "Adj Close": np.linspace(102.0, 152.0, len(dates)),
            "Volume": np.random.randint(1000, 50000, len(dates)),
        },
        index=dates,
    )


def test_standardise_date_is_column():
    raw_df = _sample_raw_df()
    std_df = _standardise(raw_df, "AAPL", "Technology")

    # date must be a column, not the index
    assert "date" in std_df.columns
    assert isinstance(std_df.index, pd.RangeIndex)
    assert std_df.columns[0] == "date"
    assert std_df.columns[1] == "ticker"
    assert std_df.columns[2] == "sector"
    assert std_df["open"].dtype == np.dtype("float64")
    assert isinstance(std_df["volume"].dtype, pd.Int64Dtype)
    assert std_df["ticker"].iloc[0] == "AAPL"
    assert std_df["sector"].iloc[0] == "Technology"


def test_standardise_with_nan_volume_is_nullable():
    raw_df = _sample_raw_df()
    # Introduce NaN into volume
    raw_df.loc[raw_df.index[0], "Volume"] = np.nan
    std_df = _standardise(raw_df, "AAPL", "Technology")

    # Volume must not crash and should preserve the missing value as pd.NA
    assert pd.isna(std_df["volume"].iloc[0])

    # Quality check should flag the missing volume
    report = check_ticker(
        std_df,
        "AAPL",
        start_date=date(2020, 1, 2),
        end_date=date(2020, 8, 1),
    )
    assert not report.passed
    assert any("Null values found" in issue for issue in report.issues)


def test_clean_and_validate_ticker_slice_null_rows():
    """Verify that completely null rows and all-null price rows are removed."""
    raw_df = _sample_raw_df()
    # Insert entirely null row
    raw_df.loc[raw_df.index[1]] = np.nan
    # Insert row with null prices but non-null volume
    raw_df.loc[raw_df.index[2], ["Open", "High", "Low", "Close"]] = np.nan
    raw_df.loc[raw_df.index[2], "Volume"] = 1000

    cleaned = _clean_and_validate_ticker_slice(raw_df, "AAPL")
    # Must drop both corrupt rows
    assert len(cleaned) == len(raw_df) - 2

    # Entirely null DataFrame should return empty
    all_null_df = pd.DataFrame(np.nan, index=raw_df.index, columns=raw_df.columns)
    empty_result = _clean_and_validate_ticker_slice(all_null_df, "AAPL")
    assert empty_result.empty


def test_download_ticker_error_returns_empty_frame():
    """Assert _download_ticker returns an empty frame if yf.download raises an exception."""
    with patch("yfinance.download", side_effect=RuntimeError("API Network Error")):
        result = _download_ticker("AAPL", start_date=date(2020, 1, 1), end_date=date(2020, 1, 10))
        assert isinstance(result, pd.DataFrame)
        assert result.empty


def test_extract_ticker_slice_multiindex_and_missing():
    raw_df = _sample_raw_df()
    frames = {}
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        frames[("AAPL", col)] = raw_df[col]
    batch_df = pd.DataFrame(frames, index=raw_df.index)
    batch_df.columns = pd.MultiIndex.from_tuples(list(frames.keys()), names=["Ticker", "Price"])

    # Extract existing symbol
    aapl_slice = _extract_ticker_slice(batch_df, "AAPL")
    assert not aapl_slice.empty
    assert list(aapl_slice.columns) == ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    # Extract missing symbol
    msft_slice = _extract_ticker_slice(batch_df, "MSFT")
    assert msft_slice.empty


def test_batch_fallback_hierarchy():
    """
    Test hierarchical fallback:
    1. Primary batch (20 tickers) fails as a whole.
    2. Falls back to small batches (5 tickers).
    3. One small batch fails as a whole -> falls back to single-ticker fetch.
    """
    symbols = [f"T{i}" for i in range(10)]
    mock_df = _sample_raw_df()

    def mock_download(*args, **kwargs):
        tickers = kwargs.get("tickers") or (args[0] if args else [])
        # Fail the 10-ticker batch as a whole
        if len(tickers) == 10:
            return pd.DataFrame()
        # For small batch of 5: fail T0..T4 as a whole, succeed for T5..T9
        if set(tickers) == {f"T{i}" for i in range(5)}:
            return pd.DataFrame()
        # When called as single ticker for T0..T4, succeed
        frames = {}
        for sym in tickers:
            for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
                frames[(sym, col)] = mock_df[col]
        res = pd.DataFrame(frames, index=mock_df.index)
        res.columns = pd.MultiIndex.from_tuples(list(frames.keys()), names=["Ticker", "Price"])
        return res

    with patch("market_forecast.data.yf.download", side_effect=mock_download):
        results = _fetch_symbols_with_fallback(
            symbols,
            start_date=date(2020, 1, 1),
            end_date=date(2020, 8, 1),
            small_batch_size=5,
        )

    # All 10 symbols should have succeeded via the fallbacks
    assert len(results) == 10
    for sym in symbols:
        assert not results[sym].empty


def test_determine_ticker_start_dates_with_two_weeks_overlap():
    existing_records = [
        {"ticker": "AAPL", "date": pd.Timestamp("2024-05-20"), "close": 150.0},
        {"ticker": "MSFT", "date": pd.Timestamp("2024-06-01"), "close": 400.0},
    ]
    existing_df = pd.DataFrame(existing_records)

    tickers = [
        {"symbol": "AAPL", "sector": "Technology"},
        {"symbol": "MSFT", "sector": "Technology"},
        {"symbol": "NVDA", "sector": "Technology"},  # new ticker
    ]

    start_dates = _determine_ticker_start_dates(
        tickers,
        existing_df,
        default_start_date=date(2020, 1, 1),
        overlap_days=14,
    )

    # AAPL last date was 2024-05-20 -> start date should be 2024-05-06 (14 days earlier)
    assert start_dates["AAPL"] == date(2024, 5, 6)
    # MSFT last date was 2024-06-01 -> start date should be 2024-05-18 (14 days earlier)
    assert start_dates["MSFT"] == date(2024, 5, 18)
    # NVDA is new -> start date should be default 2020-01-01
    assert start_dates["NVDA"] == date(2020, 1, 1)


def test_quality_checks_removes_duplicate_data():
    raw_df = _sample_raw_df()
    std_df = _standardise(raw_df, "AAPL", "Technology")

    # Introduce duplicate date rows (e.g. from 2-week overlap)
    dupe_row = std_df.iloc[[0]].copy()
    dupe_row["close"] = 103.0  # new corrected value within OHLC
    dupe_row["high"] = 106.0
    dupe_row["low"] = 94.0
    df_with_dupes = pd.concat([std_df, dupe_row], ignore_index=True)

    clean_df, reports = run_quality_checks(
        df_with_dupes,
        start_date=date(2020, 1, 2),
        end_date=date(2020, 8, 1),
    )

    # Clean df must have duplicates removed
    assert len(clean_df) == len(std_df)
    assert clean_df.duplicated(subset=["ticker", "date"]).sum() == 0
    # Keep="last" preserves the latest value
    assert clean_df["close"].iloc[0] == 103.0

    # Report should record duplicate data warning but pass
    assert reports[0].passed
    assert any("Duplicate data detected" in issue for issue in reports[0].issues)


def test_run_datafetch_too_many_failures_raises(tmp_path):
    raw_dir = tmp_path / "raw"
    parquet_path = raw_dir / "prices.parquet"

    # Mock _download_batch to always return empty DataFrame (simulate all failures)
    with patch("market_forecast.data._download_batch", return_value=pd.DataFrame()):
        with patch("market_forecast.data.RAW_DATA_DIR", raw_dir):
            with patch("market_forecast.data.PRICES_PARQUET", parquet_path):
                with pytest.raises(RuntimeError, match="exceeding the maximum allowed threshold"):
                    run_datafetch(max_allowed_failed=2)

    # Parquet file must NOT be written when threshold is exceeded
    assert not parquet_path.exists()


def test_run_datafetch_incremental_overlap_and_deduplication(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = raw_dir / "prices.parquet"

    # 1. Create existing parquet with data up to 2024-01-10
    dates_existing = pd.date_range("2024-01-02", "2024-01-10", freq="B")
    existing_df = pd.DataFrame(
        {
            "date": dates_existing,
            "ticker": "AAPL",
            "sector": "Technology",
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 100.0,
            "adj_close": 100.0,
            "volume": 10000,
        }
    )
    existing_df.to_parquet(parquet_path, index=False)

    # 2. Simulate incremental fetch returning dates from 2023-12-27 (2 weeks overlap) to 2024-01-15
    dates_new = pd.date_range("2023-12-27", "2024-01-15", freq="B")
    mock_new_raw = pd.DataFrame(
        {
            "Open": np.full(len(dates_new), 110.0),
            "High": np.full(len(dates_new), 115.0),
            "Low": np.full(len(dates_new), 105.0),
            "Close": np.full(len(dates_new), 112.0),
            "Adj Close": np.full(len(dates_new), 112.0),
            "Volume": np.full(len(dates_new), 20000),
        },
        index=dates_new,
    )

    def mock_batch_download(symbols, start_date, end_date):
        frames = {}
        for sym in symbols:
            for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
                frames[(sym, col)] = mock_new_raw[col]
        result = pd.DataFrame(frames, index=mock_new_raw.index)
        result.columns = pd.MultiIndex.from_tuples(list(frames.keys()), names=["Ticker", "Price"])
        return result

    with patch("market_forecast.data._download_batch", side_effect=mock_batch_download):
        with patch("market_forecast.data.RAW_DATA_DIR", raw_dir):
            with patch("market_forecast.data.PRICES_PARQUET", parquet_path):
                failed = run_datafetch()

    assert failed == []
    assert parquet_path.exists()

    saved = pd.read_parquet(parquet_path)
    aapl_saved = saved[saved["ticker"] == "AAPL"]
    # Check no duplicate dates exist
    assert aapl_saved["date"].duplicated().sum() == 0
    # Overlapping dates should have the newer close value (112.0 instead of old 100.0)
    overlap_date_row = aapl_saved[aapl_saved["date"] == pd.Timestamp("2024-01-10")]
    assert not overlap_date_row.empty
    assert overlap_date_row["close"].iloc[0] == 112.0


def test_check_overlap_adjustment_unchanged():
    dates = pd.date_range("2024-01-02", periods=10, freq="B")
    old_df = pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAPL",
            "close": np.full(10, 150.0),
            "adj_close": np.full(10, 150.0),
        }
    )
    new_df = old_df.copy()

    res = check_overlap_adjustment(old_df, new_df, "AAPL")
    assert res.adjustment_type is None
    assert not res.is_glitch
    assert res.ratio is None


def test_check_overlap_adjustment_dividend():
    dates = pd.date_range("2024-01-02", periods=10, freq="B")
    old_df = pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAPL",
            "close": np.full(10, 100.0),
            "adj_close": np.full(10, 100.0),
        }
    )
    new_df = old_df.copy()
    # Dividend: only adj_close changed, ratio 0.95 across all overlapping dates
    new_df["adj_close"] = 95.0

    res = check_overlap_adjustment(old_df, new_df, "AAPL")
    assert res.adjustment_type == "dividend"
    assert not res.is_glitch
    assert res.ratio is not None
    assert np.isclose(res.ratio, 0.95, atol=1e-3)


def test_check_overlap_adjustment_split():
    dates = pd.date_range("2024-01-02", periods=10, freq="B")
    old_df = pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAPL",
            "close": np.full(10, 200.0),
            "adj_close": np.full(10, 200.0),
        }
    )
    new_df = old_df.copy()
    # 2:1 Split: close and adj_close halved (ratio 0.5) across all overlapping dates
    new_df["close"] = 100.0
    new_df["adj_close"] = 100.0

    res = check_overlap_adjustment(old_df, new_df, "AAPL")
    assert res.adjustment_type == "split"
    assert not res.is_glitch
    assert res.ratio is not None
    assert np.isclose(res.ratio, 0.5, atol=1e-3)


def test_check_overlap_adjustment_irregular_data_glitch(caplog):
    dates = pd.date_range("2024-01-02", periods=10, freq="B")
    old_df = pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAPL",
            "close": np.full(10, 100.0),
            "adj_close": np.full(10, 100.0),
        }
    )
    new_df = old_df.copy()
    # Irregular change: day 0 has 105.0, day 1 has 95.0, rest 100.0
    new_df.loc[0, "close"] = 105.0
    new_df.loc[1, "close"] = 95.0

    res = check_overlap_adjustment(old_df, new_df, "AAPL")
    assert res.adjustment_type == "split"
    assert res.is_glitch is True
    assert res.ratio is None
    assert any("Data glitch detected" in msg for msg in caplog.messages)


def test_run_datafetch_overlap_unchanged_no_refetch(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = raw_dir / "prices.parquet"

    # Existing parquet with data up to 2024-01-10 for AAPL and MSFT
    dates_hist = pd.date_range("2020-01-02", "2024-01-10", freq="B")
    records = []
    for sym in ["AAPL", "MSFT"]:
        for d in dates_hist:
            records.append(
                {
                    "date": d,
                    "ticker": sym,
                    "sector": "Technology",
                    "open": 100.0,
                    "high": 105.0,
                    "low": 95.0,
                    "close": 100.0,
                    "adj_close": 100.0,
                    "volume": 10000,
                }
            )
    pd.DataFrame(records).to_parquet(parquet_path, index=False)

    # New incremental data: overlap dates (from 2023-12-27 to 2024-01-10) are UNCHANGED (close=100, adj_close=100)
    # Plus new dates up to 2024-01-15
    dates_new = pd.date_range("2023-12-27", "2024-01-15", freq="B")
    mock_raw = pd.DataFrame(
        {
            "Open": np.full(len(dates_new), 100.0),
            "High": np.full(len(dates_new), 105.0),
            "Low": np.full(len(dates_new), 95.0),
            "Close": np.full(len(dates_new), 100.0),
            "Adj Close": np.full(len(dates_new), 100.0),
            "Volume": np.full(len(dates_new), 10000),
        },
        index=dates_new,
    )

    download_calls = []

    def mock_download(symbols, start_date, end_date):
        download_calls.append({"symbols": list(symbols), "start_date": start_date, "end_date": end_date})
        frames = {}
        for sym in symbols:
            for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
                frames[(sym, col)] = mock_raw[col]
        res = pd.DataFrame(frames, index=mock_raw.index)
        res.columns = pd.MultiIndex.from_tuples(list(frames.keys()), names=["Ticker", "Price"])
        return res

    mock_tickers = [
        {"symbol": "AAPL", "sector": "Technology"},
        {"symbol": "MSFT", "sector": "Technology"},
    ]

    with patch("market_forecast.data.TICKERS", mock_tickers):
        with patch("market_forecast.data._download_batch", side_effect=mock_download):
            with patch("market_forecast.data.RAW_DATA_DIR", raw_dir):
                with patch("market_forecast.data.PRICES_PARQUET", parquet_path):
                    failed = run_datafetch()

    assert failed == []
    # Assert NO full refetch with start_date=START_DATE (2020-01-01) was executed!
    for call in download_calls:
        assert call["start_date"] != date(2020, 1, 1)


def test_run_datafetch_overlap_adj_close_differs_only_that_ticker_refetched(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = raw_dir / "prices.parquet"

    # Existing parquet with AAPL and MSFT from 2020-01-02 to 2024-01-10
    dates_hist = pd.date_range("2020-01-02", "2024-01-10", freq="B")
    records = []
    for sym in ["AAPL", "MSFT"]:
        for d in dates_hist:
            records.append(
                {
                    "date": d,
                    "ticker": sym,
                    "sector": "Technology",
                    "open": 100.0,
                    "high": 105.0,
                    "low": 95.0,
                    "close": 100.0,
                    "adj_close": 100.0,
                    "volume": 10000,
                }
            )
    pd.DataFrame(records).to_parquet(parquet_path, index=False)

    dates_new = pd.date_range("2023-12-27", "2024-01-15", freq="B")
    download_history = []

    def mock_download(symbols, start_date, end_date):
        download_history.append({"symbols": list(symbols), "start_date": start_date, "end_date": end_date})
        # If full refetch for AAPL from 2020-01-01
        if start_date == date(2020, 1, 1):
            dates = pd.date_range("2020-01-02", "2024-01-15", freq="B")
            # Refetched full history has adj_close adjusted to 90.0 for dividend
            frames = {}
            for sym in symbols:
                frames[(sym, "Open")] = np.full(len(dates), 100.0)
                frames[(sym, "High")] = np.full(len(dates), 105.0)
                frames[(sym, "Low")] = np.full(len(dates), 95.0)
                frames[(sym, "Close")] = np.full(len(dates), 100.0)
                frames[(sym, "Adj Close")] = np.full(len(dates), 90.0)
                frames[(sym, "Volume")] = np.full(len(dates), 10000)
            res = pd.DataFrame(frames, index=dates)
            res.columns = pd.MultiIndex.from_tuples(list(frames.keys()), names=["Ticker", "Price"])
            return res

        # Normal incremental fetch
        frames = {}
        for sym in symbols:
            frames[(sym, "Open")] = np.full(len(dates_new), 100.0)
            frames[(sym, "High")] = np.full(len(dates_new), 105.0)
            frames[(sym, "Low")] = np.full(len(dates_new), 95.0)
            frames[(sym, "Close")] = np.full(len(dates_new), 100.0)
            # MSFT adj_close is unchanged (100.0)
            # AAPL adj_close changed to 90.0 (dividend across overlap)
            adj_val = 90.0 if sym == "AAPL" else 100.0
            frames[(sym, "Adj Close")] = np.full(len(dates_new), adj_val)
            frames[(sym, "Volume")] = np.full(len(dates_new), 10000)
        res = pd.DataFrame(frames, index=dates_new)
        res.columns = pd.MultiIndex.from_tuples(list(frames.keys()), names=["Ticker", "Price"])
        return res

    mock_tickers = [
        {"symbol": "AAPL", "sector": "Technology"},
        {"symbol": "MSFT", "sector": "Technology"},
    ]

    with patch("market_forecast.data.TICKERS", mock_tickers):
        with patch("market_forecast.data._download_batch", side_effect=mock_download):
            with patch("market_forecast.data.RAW_DATA_DIR", raw_dir):
                with patch("market_forecast.data.PRICES_PARQUET", parquet_path):
                    failed = run_datafetch()

    assert failed == []
    # Verify: AAPL was refetched from 2020-01-01, but MSFT was NOT refetched!
    full_refetches = [c for c in download_history if c["start_date"] == date(2020, 1, 1)]
    assert len(full_refetches) == 1
    assert "AAPL" in full_refetches[0]["symbols"]
    assert "MSFT" not in full_refetches[0]["symbols"]

    # In saved parquet, ALL AAPL rows have replaced adj_close = 90.0
    saved = pd.read_parquet(parquet_path)
    aapl_saved = saved[saved["ticker"] == "AAPL"]
    msft_saved = saved[saved["ticker"] == "MSFT"]

    assert len(aapl_saved) > 100
    assert (aapl_saved["adj_close"] == 90.0).all()
    # MSFT should still have old 100.0 rows preserved from 2020
    assert (msft_saved["adj_close"] == 100.0).all()


def test_run_datafetch_safety_valve_stops_refetch_when_too_many_triggers(tmp_path, caplog):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = raw_dir / "prices.parquet"

    # Create 22 tickers
    symbols = [f"T{i}" for i in range(22)]
    mock_tickers = [{"symbol": sym, "sector": "Tech"} for sym in symbols]

    dates_hist = pd.date_range("2024-01-02", "2024-01-10", freq="B")
    records = []
    for sym in symbols:
        for d in dates_hist:
            records.append(
                {
                    "date": d,
                    "ticker": sym,
                    "sector": "Tech",
                    "open": 100.0,
                    "high": 105.0,
                    "low": 95.0,
                    "close": 100.0,
                    "adj_close": 100.0,
                    "volume": 10000,
                }
            )
    pd.DataFrame(records).to_parquet(parquet_path, index=False)

    dates_new = pd.date_range("2023-12-27", "2024-01-15", freq="B")
    download_calls = []

    def mock_download(syms, start_date, end_date):
        download_calls.append({"symbols": list(syms), "start_date": start_date})
        frames = {}
        for sym in syms:
            for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
                # Change adj_close to 90.0 for ALL 22 tickers -> all 22 trigger
                val = 90.0 if col == "Adj Close" else 100.0
                frames[(sym, col)] = np.full(len(dates_new), val)
        res = pd.DataFrame(frames, index=dates_new)
        res.columns = pd.MultiIndex.from_tuples(list(frames.keys()), names=["Ticker", "Price"])
        return res

    with patch("market_forecast.data.TICKERS", mock_tickers):
        with patch("market_forecast.data._download_batch", side_effect=mock_download):
            with patch("market_forecast.data.RAW_DATA_DIR", raw_dir):
                with patch("market_forecast.data.PRICES_PARQUET", parquet_path):
                    with patch("market_forecast.data.BATCH_SIZE", 25):
                        run_datafetch(max_adjustment_threshold=20)

    # Safety valve must trigger: warning logged
    assert any("Safety valve triggered" in msg for msg in caplog.messages)
    # NO full refetches (start_date=2020-01-01) should have been executed!
    assert not any(c["start_date"] == date(2020, 1, 1) for c in download_calls)

