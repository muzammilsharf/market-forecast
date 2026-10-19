"""
Entry-point script to ingest stock data and execute quality checks.

Exit Code Policy (CI / GitHub Actions):
---------------------------------------
- Exits 0 if all operations succeed within configured failure tolerances.
- Exits 1 if:
    1. An unhandled exception occurs during ingestion.
    2. The generated prices parquet file is missing.
    3. The number of permanently failed downloads exceeds --max-download-failures (default: 5).
    4. The number of hard QC failures exceeds --max-qc-failures (default: 0 = zero tolerance).
    5. --fail-on-warnings (or --strict) is enabled and any ticker has quality check warnings.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from market_forecast.config import (
    MAX_ALLOWED_DOWNLOAD_FAILURES,
    MAX_ALLOWED_QC_FAILURES,
)
from market_forecast.data import run_datafetch
from market_forecast.paths import PRICES_PARQUET
from market_forecast.quality import reports_to_dataframe, run_quality_checks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest stock data from yfinance and run time-series quality checks."
    )
    parser.add_argument(
        "--max-download-failures",
        type=int,
        default=MAX_ALLOWED_DOWNLOAD_FAILURES,
        help=f"Maximum allowed failed downloads before exiting with code 1 (default: {MAX_ALLOWED_DOWNLOAD_FAILURES}).",
    )
    parser.add_argument(
        "--max-qc-failures",
        type=int,
        default=MAX_ALLOWED_QC_FAILURES,
        help=f"Maximum allowed hard quality-check failures before exiting with code 1 (default: {MAX_ALLOWED_QC_FAILURES}).",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Treat quality-check warnings (e.g. extreme returns, flatlines) as failures (exit 1).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: zero tolerance for download failures, hard QC failures, or warnings.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Apply strict preset if requested
    max_download_failures = 0 if args.strict else args.max_download_failures
    max_qc_failures = 0 if args.strict else args.max_qc_failures
    fail_on_warnings = True if args.strict else args.fail_on_warnings

    # Configure logging for the ingest CLI execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("ingest")

    logger.info("=== Starting Market Forecast Data Ingestion ===")
    logger.info(
        "Thresholds: max_download_failures=%d, max_qc_failures=%d, fail_on_warnings=%s",
        max_download_failures,
        max_qc_failures,
        fail_on_warnings,
    )

    try:
        failed_downloads = run_datafetch()
    except Exception as exc:
        logger.error("Ingestion pipeline raised an unhandled exception: %s", exc)
        return 1

    download_failed_count = len(failed_downloads)
    if download_failed_count > 0:
        logger.warning(
            "Download completed with %d failed ticker(s): %s",
            download_failed_count,
            failed_downloads,
        )
    else:
        logger.info("Download completed: all tickers successfully fetched.")

    if not PRICES_PARQUET.exists():
        logger.error("Expected parquet file does not exist at %s", PRICES_PARQUET)
        return 1

    logger.info("=== Running Quality Checks on %s ===", PRICES_PARQUET)
    df = pd.read_parquet(PRICES_PARQUET)
    clean_df, reports = run_quality_checks(df)
    if len(clean_df) < len(df):
        logger.info(
            "Quality checks deduplicated dataset from %d to %d rows. Updating %s ...",
            len(df),
            len(clean_df),
            PRICES_PARQUET,
        )
        clean_df.to_parquet(PRICES_PARQUET, index=False, engine="auto")
    summary_df = reports_to_dataframe(reports)

    # Filter for all tickers with any issues (both hard failures and warnings)
    tickers_with_issues = summary_df[summary_df["issue_count"] > 0]
    hard_failures = summary_df[~summary_df["passed"]]
    warnings_only = summary_df[summary_df["passed"] & (summary_df["issue_count"] > 0)]

    if not tickers_with_issues.empty:
        logger.warning(
            "Quality checks detected issues on %d ticker(s) total (%d hard failure(s), %d warning(s)):\n%s",
            len(tickers_with_issues),
            len(hard_failures),
            len(warnings_only),
            tickers_with_issues[["passed", "issue_count", "issues"]],
        )
    else:
        logger.info("All quality checks passed clean with zero issues or warnings.")

    # ------------------------------------------------------------------ #
    # Exit Code Evaluation (for CI / GitHub Actions)
    # ------------------------------------------------------------------ #
    has_error = False

    if download_failed_count > max_download_failures:
        logger.error(
            "CI Failure: %d download failure(s) exceeded the allowed threshold of %d.",
            download_failed_count,
            max_download_failures,
        )
        has_error = True

    if len(hard_failures) > max_qc_failures:
        logger.error(
            "CI Failure: %d hard QC failure(s) exceeded the allowed threshold of %d.",
            len(hard_failures),
            max_qc_failures,
        )
        has_error = True

    if fail_on_warnings and len(warnings_only) > 0:
        logger.error(
            "CI Failure: %d warning(s) detected with --fail-on-warnings/--strict enabled.",
            len(warnings_only),
        )
        has_error = True

    if has_error:
        logger.error("=== Ingestion pipeline completed with ERRORS (exit code 1) ===")
        return 1

    logger.info("=== Ingestion pipeline completed SUCCESSFULLY (exit code 0) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
