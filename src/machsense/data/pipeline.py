"""End-to-end reusable data pipeline orchestrator for MachSense."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd

from machsense.config.settings import get_settings
from machsense.data.cleaner import clean_dataset, validate_dataset
from machsense.data.download import download_ai4i_dataset
from machsense.data.loader import load_raw_data
from machsense.data.schema import ValidationReport
from machsense.data.splitter import DataSplit, split_data
from machsense.utils.logger import get_logger, setup_logging

logger = get_logger("machsense.data.pipeline")


def run_data_pipeline(
    raw_file_path: Optional[Path | str] = None,
    save_artifacts: bool = True,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> DataSplit:
    """Execute complete end-to-end data pipeline.

    Workflow:
    1. Ingestion: Download / load raw sensor telemetry.
    2. Header Normalization: Standardize column names.
    3. Validation: Verify physical ranges and thermodynamic laws.
    4. Cleaning: Deduplicate, handle missing values, filter physical outliers.
    5. Storage: Save interim cleaned dataset.
    6. Stratification & Splitting: Produce isolated train/val/test splits without leakage.
    7. Storage: Save processed train/val/test feature matrices and targets.

    Args:
        raw_file_path: Optional path to raw CSV file.
        save_artifacts: Whether to persist interim and processed files to disk.
        val_size: Fraction of samples for validation.
        test_size: Fraction of samples for test.
        random_state: Random seed for deterministic reproducibility.

    Returns:
        DataSplit object containing X_train, X_val, X_test, y_train, y_val, y_test.
    """
    setup_logging()
    settings = get_settings()
    logger.info("=========================================================")
    logger.info("Starting MachSense Data Pipeline Execution")
    logger.info("=========================================================")

    # 1. Load Raw Data
    raw_df = load_raw_data(file_path=raw_file_path, auto_download=True)

    # 2. Initial Validation
    initial_report = validate_dataset(raw_df)
    logger.info("Initial Data Validation:\n%s", initial_report.summary())

    # 3. Clean Dataset
    clean_df, clean_report = clean_dataset(raw_df, drop_inconsistent=True)

    # 4. Save Interim Data
    if save_artifacts:
        interim_dir = settings.resolve_path("interim_data_dir")
        interim_dir.mkdir(parents=True, exist_ok=True)
        interim_csv_path = interim_dir / "clean_machsense.csv"
        clean_df.to_csv(interim_csv_path, index=False)
        logger.info("Saved interim cleaned dataset to: %s", interim_csv_path)

    # 5. Split Dataset
    split = split_data(
        clean_df,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        stratify=True,
        target_col=settings.data.target_column,
    )

    # 6. Save Processed Splits
    if save_artifacts:
        processed_dir = settings.resolve_path("processed_data_dir")
        processed_dir.mkdir(parents=True, exist_ok=True)

        split.X_train.to_csv(processed_dir / "X_train.csv", index=False)
        split.X_val.to_csv(processed_dir / "X_val.csv", index=False)
        split.X_test.to_csv(processed_dir / "X_test.csv", index=False)
        split.y_train.to_csv(processed_dir / "y_train.csv", index=False)
        split.y_val.to_csv(processed_dir / "y_val.csv", index=False)
        split.y_test.to_csv(processed_dir / "y_test.csv", index=False)

        if split.failure_modes_train is not None:
            split.failure_modes_train.to_csv(processed_dir / "failure_modes_train.csv", index=False)
            split.failure_modes_val.to_csv(processed_dir / "failure_modes_val.csv", index=False)
            split.failure_modes_test.to_csv(processed_dir / "failure_modes_test.csv", index=False)

        logger.info("Saved processed train/val/test splits to: %s", processed_dir)

    logger.info("MachSense Data Pipeline completed successfully.")
    return split


if __name__ == "__main__":
    run_data_pipeline()
