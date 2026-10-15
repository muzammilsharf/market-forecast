from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from scripts.ingest import main


def _sample_clean_df() -> pd.DataFrame:
    """Generate a clean dataset covering START_DATE to today."""
    today = pd.Timestamp.now().normalize()
    dates = pd.date_range(start="2020-01-02", end=today, freq="B")
    n = len(dates)
    base = np.linspace(100.0, 200.0, n)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAPL",
            "sector": "Technology",
            "open": base,
            "high": base + 5.0,
            "low": base - 5.0,
            "close": base + 1.0,
            "adj_close": base + 1.0,
            "volume": np.random.randint(1000, 50000, n),
        }
    )


def test_ingest_main_success(tmp_path):
    parquet_path = tmp_path / "prices.parquet"
    df = _sample_clean_df()
    df.to_parquet(parquet_path, index=False)

    with patch("scripts.ingest.run_datafetch", return_value=[]):
        with patch("scripts.ingest.PRICES_PARQUET", parquet_path):
            exit_code = main([])

    assert exit_code == 0


def test_ingest_main_exits_1_on_download_failures_exceeding_threshold(tmp_path):
    parquet_path = tmp_path / "prices.parquet"
    df = _sample_clean_df()
    df.to_parquet(parquet_path, index=False)

    # 3 failures, threshold is 1
    with patch("scripts.ingest.run_datafetch", return_value=["AAPL", "MSFT", "NVDA"]):
        with patch("scripts.ingest.PRICES_PARQUET", parquet_path):
            exit_code = main(["--max-download-failures", "1"])

    assert exit_code == 1


def test_ingest_main_exits_1_on_hard_qc_failure(tmp_path):
    parquet_path = tmp_path / "prices.parquet"
    df = _sample_clean_df()
    # Introduce negative price (hard QC failure)
    df.loc[0, "close"] = -10.0
    df.to_parquet(parquet_path, index=False)

    with patch("scripts.ingest.run_datafetch", return_value=[]):
        with patch("scripts.ingest.PRICES_PARQUET", parquet_path):
            exit_code = main([])

    assert exit_code == 1


def test_ingest_main_warning_detected_with_issue_count(tmp_path):
    parquet_path = tmp_path / "prices.parquet"
    df = _sample_clean_df()

    # Introduce a massive jump in close price, keeping high/low consistent (warning only: passed=True, issue_count=1)
    jump_close = 300.0
    df.loc[50, "close"] = jump_close
    df.loc[50, "high"] = jump_close + 10.0
    df.loc[50, "low"] = 50.0
    df.loc[50, "adj_close"] = jump_close
    df.to_parquet(parquet_path, index=False)

    with patch("scripts.ingest.run_datafetch", return_value=[]):
        with patch("scripts.ingest.PRICES_PARQUET", parquet_path):
            # Default run should pass (warning is logged but does not fail exit code)
            exit_code_default = main([])
            assert exit_code_default == 0

            # With --fail-on-warnings, it should fail with exit code 1
            exit_code_strict = main(["--fail-on-warnings"])
            assert exit_code_strict == 1
