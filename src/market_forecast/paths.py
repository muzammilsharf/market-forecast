'''This file contains all the necessary paths used in the project.'''
from pathlib import Path

# Base directory of the project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# Parquet and DuckDB paths
PRICES_PARQUET = RAW_DATA_DIR / "prices.parquet"
DUCKDB_PATH = DATA_DIR / "market.duckdb"