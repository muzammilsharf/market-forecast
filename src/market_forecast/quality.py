"""
Quality-gate module for stock time-series data.

Checks performed on a per-ticker basis
---------------------------------------
1.  Minimum row count          — at least MIN_ROWS trading rows must exist.
2.  Date range coverage        — first and last dates must be within tolerance
                                 of START_DATE / end_date.
3.  Null / NaN values          — no null values allowed in OHLCV columns.
4.  Zero or negative prices    — open, high, low, close, adj_close > 0.
5.  Negative volume            — volume must be >= 0.
6.  OHLC consistency           — low <= open, close <= high for every row.
7.  Duplicate dates            — no duplicate date entries per ticker.
8.  Monotonic date order       — dates must be strictly increasing.
9.  Extreme daily returns      — flags rows where |pct_change(close)| > 50 %
                                 (configurable via MAX_DAILY_RETURN_PCT).
10. Forward-fill gap detection — flags tickers with suspiciously long runs of
                                 identical close prices (> MAX_FLATLINE_DAYS).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from market_forecast.config import (
    DATE_TOLERANCE_DAYS,
    MAX_DAILY_RETURN_PCT,
    MAX_FLATLINE_DAYS,
    MIN_ROWS,
    OHLCV_COLS,
    PRICE_COLS,
    START_DATE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class QualityReport:
    ticker: str
    passed: bool = True
    issues: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        """Record a failing check and mark the report as not passed."""
        self.passed = False
        self.issues.append(message)

    def warn(self, message: str) -> None:
        """Record a non-fatal warning (does not flip *passed*)."""
        self.issues.append(f"[WARN] {message}")

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"{self.ticker}: {status}"]
        for issue in self.issues:
            lines.append(f"  • {issue}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual check helpers
# ---------------------------------------------------------------------------
def _check_minimum_rows(df: pd.DataFrame, report: QualityReport) -> None:
    n = len(df)
    if n < MIN_ROWS:
        report.fail(f"Too few rows: {n} (minimum {MIN_ROWS}).")


def _check_date_coverage(
    df: pd.DataFrame,
    report: QualityReport,
    start_date: date = START_DATE,
    end_date: date | None = None,
) -> None:
    if df.empty:
        return
    if end_date is None:
        end_date = date.today()

    if "date" in df.columns:
        date_series = pd.to_datetime(df["date"])
    else:
        date_series = pd.to_datetime(df.index)

    first = date_series.min().date()
    last = date_series.max().date()
    tol = timedelta(days=DATE_TOLERANCE_DAYS)

    if first > start_date + tol:
        report.fail(
            f"Data starts too late: {first} (expected ~{start_date})."
        )
    if last < end_date - tol:
        report.fail(
            f"Data ends too early: {last} (expected ~{end_date})."
        )


def _check_nulls(df: pd.DataFrame, report: QualityReport) -> None:
    present_cols = [c for c in OHLCV_COLS if c in df.columns]
    null_counts = df[present_cols].isnull().sum()
    nulls = null_counts[null_counts > 0]
    if not nulls.empty:
        report.fail(
            f"Null values found — {nulls.to_dict()}."
        )


def _check_non_positive_prices(df: pd.DataFrame, report: QualityReport) -> None:
    present_cols = [c for c in PRICE_COLS if c in df.columns]
    for col in present_cols:
        bad = (df[col] <= 0).sum()
        if bad > 0:
            report.fail(f"Column '{col}' has {bad} zero/negative value(s).")


def _check_volume(df: pd.DataFrame, report: QualityReport) -> None:
    if "volume" not in df.columns:
        return
    bad = (df["volume"] < 0).sum()
    if bad > 0:
        report.fail(f"'volume' has {bad} negative value(s).")


def _check_ohlc_consistency(df: pd.DataFrame, report: QualityReport) -> None:
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        return
    violations = (
        (df["low"] > df["open"])
        | (df["low"] > df["close"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["high"])
    ).sum()
    if violations > 0:
        report.fail(
            f"OHLC consistency violated on {violations} row(s) "
            "(e.g. low > open/close, or high < open/close)."
        )


def _check_duplicate_dates(df: pd.DataFrame, report: QualityReport) -> None:
    if "date" in df.columns:
        dupes = df["date"].duplicated().sum()
    else:
        dupes = df.index.duplicated().sum()
    if dupes > 0:
        report.fail(f"Duplicate date entries: {dupes}.")


def _check_monotonic_index(df: pd.DataFrame, report: QualityReport) -> None:
    if "date" in df.columns:
        if not pd.to_datetime(df["date"]).is_monotonic_increasing:
            report.fail("Date column is not strictly monotonically increasing.")
    else:
        if not df.index.is_monotonic_increasing:
            report.fail("Date index is not strictly monotonically increasing.")


def _check_extreme_returns(df: pd.DataFrame, report: QualityReport) -> None:
    if "close" not in df.columns or len(df) < 2:
        return
    pct = df["close"].pct_change().abs() * 100
    extreme = (pct > MAX_DAILY_RETURN_PCT).sum()
    if extreme > 0:
        report.warn(
            f"{extreme} row(s) with daily |return| > {MAX_DAILY_RETURN_PCT}%."
        )


def _check_flatline(df: pd.DataFrame, report: QualityReport) -> None:
    if "close" not in df.columns or len(df) < 2:
        return
    # Length of longest run of equal close prices
    same = (df["close"] == df["close"].shift()).astype(int)
    run_len = same.groupby((same != same.shift()).cumsum()).transform("sum").max()
    if run_len > MAX_FLATLINE_DAYS:
        report.warn(
            f"Longest run of identical close prices: {run_len} days "
            f"(threshold {MAX_FLATLINE_DAYS})."
        )


# ---------------------------------------------------------------------------
# Per-ticker gate
# ---------------------------------------------------------------------------
def check_ticker(
    df: pd.DataFrame,
    ticker: str,
    start_date: date = START_DATE,
    end_date: date | None = None,
) -> QualityReport:
    """
    Run all quality checks for a single ticker's DataFrame slice.

    Parameters
    ----------
    df : pd.DataFrame
        Slice for one ticker (with a 'date' column or DatetimeIndex).
    ticker : str
        Ticker symbol (used for reporting only).
    start_date : date
        Expected start date.
    end_date : date | None
        Expected end date (defaults to date.today() if None).

    Returns
    -------
    QualityReport
        Contains pass/fail status and a list of issues.
    """
    report = QualityReport(ticker=ticker)

    _check_minimum_rows(df, report)
    _check_date_coverage(df, report, start_date=start_date, end_date=end_date)
    _check_nulls(df, report)
    _check_non_positive_prices(df, report)
    _check_volume(df, report)
    _check_ohlc_consistency(df, report)
    _check_duplicate_dates(df, report)
    _check_monotonic_index(df, report)
    _check_extreme_returns(df, report)
    _check_flatline(df, report)

    return report


# ---------------------------------------------------------------------------
# Dataset-level gate
# ---------------------------------------------------------------------------
def run_quality_checks(
    df: pd.DataFrame,
    drop_failed: bool = False,
    start_date: date = START_DATE,
    end_date: date | None = None,
) -> tuple[pd.DataFrame, list[QualityReport]]:
    """
    Run quality checks on the full combined prices DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Combined prices DataFrame (multi-ticker, with a 'ticker' column).
    drop_failed : bool
        If True, rows belonging to tickers that fail checks are removed
        from the returned DataFrame.
    start_date : date
        Expected start date for coverage checks.
    end_date : date | None
        Expected end date for coverage checks.

    Returns
    -------
    clean_df : pd.DataFrame
        The (optionally filtered) DataFrame.
    reports : list[QualityReport]
        One QualityReport per ticker.
    """
    if "ticker" not in df.columns:
        raise ValueError("DataFrame must have a 'ticker' column.")

    reports: list[QualityReport] = []
    failed_tickers: list[str] = []

    for ticker, group in df.groupby("ticker"):
        report = check_ticker(
            group,
            str(ticker),
            start_date=start_date,
            end_date=end_date,
        )
        reports.append(report)

        if report.passed:
            logger.info("QC PASS  %s", ticker)
        else:
            logger.warning("QC FAIL  %s\n%s", ticker, report)
            failed_tickers.append(str(ticker))

    total = len(reports)
    passed = sum(1 for r in reports if r.passed)
    logger.info("Quality check summary: %d/%d tickers passed.", passed, total)

    clean_df = df.copy()
    if drop_failed and failed_tickers:
        clean_df = clean_df[~clean_df["ticker"].isin(failed_tickers)]
        logger.info(
            "Dropped %d failed ticker(s) from DataFrame.", len(failed_tickers)
        )

    return clean_df, reports


# ---------------------------------------------------------------------------
# Convenience: summarise results as a DataFrame
# ---------------------------------------------------------------------------
def reports_to_dataframe(reports: list[QualityReport]) -> pd.DataFrame:
    """Convert a list of QualityReports into a tidy summary DataFrame."""
    rows = []
    for r in reports:
        rows.append(
            {
                "ticker": r.ticker,
                "passed": r.passed,
                "issue_count": len(r.issues),
                "issues": "; ".join(r.issues) if r.issues else "",
            }
        )
    return pd.DataFrame(rows).set_index("ticker")
