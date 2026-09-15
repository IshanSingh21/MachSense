"""Reusable data loading and header normalization module."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import pandas as pd

from machsense.config.settings import get_settings
from machsense.data.download import download_ai4i_dataset
from machsense.data.schema import COLUMN_MAPPING, EXPECTED_FEATURE_COLUMNS
from machsense.utils.logger import get_logger

logger = get_logger("machsense.data.loader")


def load_raw_data(
    file_path: Optional[Union[Path, str]] = None,
    auto_download: bool = True,
) -> pd.DataFrame:
    """Load raw dataset and standardize column headers to snake_case.

    Args:
        file_path: Optional path to raw CSV file. Defaults to 'data/raw/ai4i2020.csv'.
        auto_download: If True and file is missing, automatically downloads it.

    Returns:
        pd.DataFrame with standardized snake_case column names.
    """
    settings = get_settings()

    if file_path is None:
        target_path = settings.resolve_path("raw_data_dir") / "ai4i2020.csv"
    else:
        target_path = Path(file_path)

    if not target_path.exists():
        if auto_download:
            logger.info("Raw data file not found at %s. Triggering automatic download...", target_path)
            target_path = download_ai4i_dataset(target_path=target_path)
        else:
            raise FileNotFoundError(f"Raw data file does not exist at: {target_path}")

    logger.info("Reading raw dataset from: %s", target_path)
    df = pd.read_csv(target_path)
    logger.info("Raw dataset loaded successfully with %d rows and %d columns.", len(df), len(df.columns))

    # Standardize column headers
    standardized_df = standardize_columns(df)
    return standardized_df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw dataset headers to canonical snake_case identifiers.

    Args:
        df: Input DataFrame with original headers.

    Returns:
        DataFrame with standardized column names.
    """
    rename_dict = {}
    for col in df.columns:
        clean_col = col.strip()
        if clean_col in COLUMN_MAPPING:
            rename_dict[col] = COLUMN_MAPPING[clean_col]
        else:
            # Fallback normalization: lowercase and replace special chars with underscores
            normalized = clean_col.lower().replace(" ", "_").replace("[", "").replace("]", "").replace("/", "_")
            rename_dict[col] = normalized

    renamed_df = df.rename(columns=rename_dict)

    # Verify required feature columns are present
    missing_features = [col for col in EXPECTED_FEATURE_COLUMNS if col not in renamed_df.columns]
    if missing_features:
        logger.warning("Loaded dataset is missing expected features: %s", missing_features)

    return renamed_df
