"""
Shared configuration constants for market_forecast data pipelines and quality checks.
"""

from __future__ import annotations

from datetime import date

# ---------------------------------------------------------------------------
# Pipeline Constants
# ---------------------------------------------------------------------------
START_DATE: date = date(2020, 1, 1)
BATCH_SIZE: int = 20            # number of tickers per primary yfinance download batch
SMALL_BATCH_SIZE: int = 5       # fallback batch size when primary batch fails as a whole
OVERLAP_DAYS: int = 14          # two weeks (14 days) overlap for incremental ticker fetches
MAX_ALLOWED_FAILED: int = 10    # maximum permanently failed tickers before aborting
MAX_ADJUSTMENT_REFETCH_THRESHOLD: int = 20  # safety valve: max corporate action adjustments before aborting full refetches

# Canonical column name mapping (yfinance -> project standard)
COLUMN_RENAME: dict[str, str] = {
    "Open":         "open",
    "High":         "high",
    "Low":          "low",
    "Close":        "close",
    "Adj Close":    "adj_close",
    "Volume":       "volume",
}

# Column categories
PRICE_COLS: list[str] = ["open", "high", "low", "close", "adj_close"]
OHLCV_COLS: list[str] = ["open", "high", "low", "close", "adj_close", "volume"]

# ---------------------------------------------------------------------------
# Quality Check Thresholds
# ---------------------------------------------------------------------------
MIN_ROWS: int = 100                  # minimum trading rows per ticker
DATE_TOLERANCE_DAYS: int = 10        # leeway for first/last date checks
MAX_DAILY_RETURN_PCT: float = 50.0   # flag extreme single-day moves
MAX_FLATLINE_DAYS: int = 5           # max consecutive identical close prices

# ---------------------------------------------------------------------------
# Ingestion Script & CI Exit Policy
# ---------------------------------------------------------------------------
MAX_ALLOWED_DOWNLOAD_FAILURES: int = 5  # max allowed failed ticker downloads before CLI exit 1
MAX_ALLOWED_QC_FAILURES: int = 0        # max allowed hard QC failures (0 = any hard failure exits 1)
