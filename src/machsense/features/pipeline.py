"""Feature engineering and preprocessing pipeline execution orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from machsense.config.settings import get_settings
from machsense.data.pipeline import run_data_pipeline
from machsense.features.preprocessor import MachSensePreprocessor
from machsense.utils.logger import get_logger, setup_logging

logger = get_logger("machsense.features.pipeline")


def run_feature_pipeline(
    save_artifacts: bool = True,
    scaling_method: Optional[str] = None,
) -> Tuple[MachSensePreprocessor, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute complete preprocessing and feature engineering pipeline.

    Workflow:
    1. Ingestion: Retrieve train/val/test splits from data/processed/.
    2. Initialization: Instantiate MachSensePreprocessor with configured scaling.
    3. Fit on Train: Fit imputers, encoders, and scalers strictly on X_train.
    4. Transform: Transform X_train, X_val, and X_test without leakage.
    5. Artifact Persistence: Save fitted preprocessor in models/ and transformed datasets.

    Args:
        save_artifacts: Whether to save preprocessor.joblib and transformed CSVs to disk.
        scaling_method: Optional scaling method override ('standard', 'robust', 'none').

    Returns:
        Tuple of (Fitted Preprocessor, X_train_trans, X_val_trans, X_test_trans).
    """
    setup_logging()
    settings = get_settings()
    actual_scaling = scaling_method or settings.features.scaling_method

    logger.info("=========================================================")
    logger.info("Starting MachSense Feature Pipeline Execution")
    logger.info("Scaling Method: %s", actual_scaling)
    logger.info("=========================================================")

    processed_dir = settings.resolve_path("processed_data_dir")
    train_file = processed_dir / "X_train.csv"

    if not train_file.exists():
        logger.info("Processed data splits not found. Triggering data pipeline first...")
        run_data_pipeline(save_artifacts=True)

    # 1. Load split partitions
    logger.info("Loading processed split partitions from %s...", processed_dir)
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_val = pd.read_csv(processed_dir / "X_val.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")

    # 2. Fit preprocessor strictly on X_train
    preprocessor = MachSensePreprocessor(scaling_method=actual_scaling)
    X_train_trans = preprocessor.fit_transform(X_train)

    # 3. Transform X_val and X_test using frozen training statistics
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

    # 4. Save artifacts
    if save_artifacts:
        # Save preprocessor artifact to models/
        preprocessor.save()

        # Save transformed feature matrices
        X_train_trans.to_csv(processed_dir / "X_train_transformed.csv", index=False)
        X_val_trans.to_csv(processed_dir / "X_val_transformed.csv", index=False)
        X_test_trans.to_csv(processed_dir / "X_test_transformed.csv", index=False)

        logger.info("Saved transformed datasets to: %s", processed_dir)

    logger.info(
        "Feature pipeline completed successfully. Shapes: Train %s, Val %s, Test %s",
        X_train_trans.shape,
        X_val_trans.shape,
        X_test_trans.shape,
    )
    return preprocessor, X_train_trans, X_val_trans, X_test_trans


if __name__ == "__main__":
    run_feature_pipeline()
