from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from market_forecast.data import _download_ticker, _standardise, run_datafetch
from market_forecast.quality import check_ticker, run_quality_checks


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


def test_download_ticker_error_returns_empty_frame():
    """Assert _download_ticker returns an empty frame if yf.download raises an exception."""
    with patch("yfinance.download", side_effect=RuntimeError("API Network Error")):
        result = _download_ticker("AAPL", start_date=date(2020, 1, 1), end_date=date(2020, 1, 10))
        assert isinstance(result, pd.DataFrame)
        assert result.empty


def test_quality_checks_with_date_column():
    raw_df = _sample_raw_df()
    std_df = _standardise(raw_df, "AAPL", "Technology")

    # Pass date bounds matching the sample
    report = check_ticker(
        std_df,
        "AAPL",
        start_date=date(2020, 1, 2),
        end_date=date(2020, 8, 1),
    )
    assert report.passed, f"Report failed with issues: {report.issues}"


def test_run_datafetch_too_many_failures_raises(tmp_path):
    raw_dir = tmp_path / "raw"
    parquet_path = raw_dir / "prices.parquet"

    # Mock _download_ticker to always return empty DataFrame (simulate all failures)
    with patch("market_forecast.data._download_ticker", return_value=pd.DataFrame()):
        with patch("market_forecast.data.REQUEST_DELAY", 0):
            with patch("market_forecast.data.RAW_DATA_DIR", raw_dir):
                with patch("market_forecast.data.PRICES_PARQUET", parquet_path):
                    with pytest.raises(RuntimeError, match="exceeding the maximum allowed threshold"):
                        run_datafetch(max_allowed_failed=2)

    # Parquet file must NOT be written when threshold is exceeded
    assert not parquet_path.exists()


def test_run_datafetch_success_and_returns_failed_list(tmp_path):
    raw_dir = tmp_path / "raw"
    parquet_path = raw_dir / "prices.parquet"
    mock_df = _sample_raw_df()

    def mock_fetch(symbol, start_date, end_date):
        if symbol == "AAPL":
            return pd.DataFrame()  # simulate AAPL failing
        return mock_df

    with patch("market_forecast.data._download_ticker", side_effect=mock_fetch):
        with patch("market_forecast.data.REQUEST_DELAY", 0):
            with patch("market_forecast.data.RAW_DATA_DIR", raw_dir):
                with patch("market_forecast.data.PRICES_PARQUET", parquet_path):
                    # 1 failure is under the threshold of 5
                    failed = run_datafetch(max_allowed_failed=5)

    assert failed == ["AAPL"]
    assert parquet_path.exists()

    # Read parquet and verify 'date' is a column
    saved = pd.read_parquet(parquet_path)
    assert "date" in saved.columns
    assert "ticker" in saved.columns
    assert "AAPL" not in saved["ticker"].values
