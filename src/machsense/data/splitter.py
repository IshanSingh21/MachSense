"""Leakage-proof stratified dataset splitting for predictive maintenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from machsense.config.settings import get_settings
from machsense.data.schema import FAILURE_MODE_COLUMNS
from machsense.utils.logger import get_logger

logger = get_logger("machsense.data.splitter")


@dataclass
class DataSplit:
    """Strongly-typed container holding train, validation, and test partitions."""
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    failure_modes_train: Optional[pd.DataFrame] = None
    failure_modes_val: Optional[pd.DataFrame] = None
    failure_modes_test: Optional[pd.DataFrame] = None

    def verify_no_leakage(self) -> bool:
        """Verify that train, validation, and test sets have disjoint index sets."""
        train_idx = set(self.X_train.index)
        val_idx = set(self.X_val.index)
        test_idx = set(self.X_test.index)

        train_val_overlap = train_idx.intersection(val_idx)
        train_test_overlap = train_idx.intersection(test_idx)
        val_test_overlap = val_idx.intersection(test_idx)

        if train_val_overlap or train_test_overlap or val_test_overlap:
            raise ValueError(
                f"Data leakage detected! Overlaps: Train-Val={len(train_val_overlap)}, "
                f"Train-Test={len(train_test_overlap)}, Val-Test={len(val_test_overlap)}"
            )
        return True

    def summary(self) -> str:
        """Generate human-readable summary of splits and class distribution."""
        total = len(self.X_train) + len(self.X_val) + len(self.X_test)
        train_pos = int(self.y_train.sum())
        val_pos = int(self.y_val.sum())
        test_pos = int(self.y_test.sum())

        return (
            f"DataSplit Summary (Total: {total} records):\n"
            f"  - Train: {len(self.X_train):>5} samples ({len(self.X_train)/total:>5.1%}) | "
            f"Failures: {train_pos:>3} ({train_pos/max(1, len(self.y_train)):>5.2%})\n"
            f"  - Val:   {len(self.X_val):>5} samples ({len(self.X_val)/total:>5.1%}) | "
            f"Failures: {val_pos:>3} ({val_pos/max(1, len(self.y_val)):>5.2%})\n"
            f"  - Test:  {len(self.X_test):>5} samples ({len(self.X_test)/total:>5.1%}) | "
            f"Failures: {test_pos:>3} ({test_pos/max(1, len(self.y_test)):>5.2%})\n"
            f"  - Features: {list(self.X_train.columns)}"
        )


def separate_features_and_target(
    df: pd.DataFrame,
    target_col: str = "machine_failure",
    drop_metadata_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Separate input features, target variable, and failure mode flags.

    Args:
        df: Cleaned DataFrame.
        target_col: Name of binary failure target column.
        drop_metadata_cols: List of metadata identifiers to exclude from features.

    Returns:
        Tuple of (X features DataFrame, y target Series, failure modes DataFrame).
    """
    if drop_metadata_cols is None:
        drop_metadata_cols = ["udi", "product_id"]

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    # Target
    y = df[target_col].copy()

    # Failure mode multi-label indicators
    available_failure_modes = [col for col in FAILURE_MODE_COLUMNS if col in df.columns]
    failure_modes_df = df[available_failure_modes].copy() if available_failure_modes else pd.DataFrame(index=df.index)

    # Exclude target, failure modes, and metadata identifiers from training features X
    cols_to_drop = set(drop_metadata_cols + [target_col] + available_failure_modes)
    feature_cols = [c for c in df.columns if c not in cols_to_drop]

    X = df[feature_cols].copy()
    return X, y, failure_modes_df


def split_data(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True,
    target_col: str = "machine_failure",
) -> DataSplit:
    """Split cleaned dataset into train, validation, and test partitions with stratification.

    Prevents data leakage by ensuring complete isolation of indices and sets.

    Args:
        df: Cleaned input DataFrame.
        val_size: Fraction for validation set.
        test_size: Fraction for test set.
        random_state: Random seed for reproducibility.
        stratify: Whether to use stratified sampling on the target column.
        target_col: Target column name for stratification.

    Returns:
        DataSplit object containing X_train, X_val, X_test, y_train, y_val, y_test, etc.
    """
    settings = get_settings()
    actual_random_state = random_state if random_state is not None else settings.project.random_seed

    X, y, failure_modes = separate_features_and_target(df, target_col=target_col)

    # Calculate first split: (Train + Val) vs Test
    stratify_target = y if stratify else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=actual_random_state,
        stratify=stratify_target,
    )

    # Calculate second split: Train vs Val from Temp
    val_relative_ratio = val_size / (1.0 - test_size)
    stratify_temp = y_temp if stratify else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_relative_ratio,
        random_state=actual_random_state,
        stratify=stratify_temp,
    )

    # Align failure modes with the respective splits
    fm_train = failure_modes.loc[X_train.index] if not failure_modes.empty else None
    fm_val = failure_modes.loc[X_val.index] if not failure_modes.empty else None
    fm_test = failure_modes.loc[X_test.index] if not failure_modes.empty else None

    split = DataSplit(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        failure_modes_train=fm_train,
        failure_modes_val=fm_val,
        failure_modes_test=fm_test,
    )

    split.verify_no_leakage()
    logger.info("Stratified data split completed successfully:\n%s", split.summary())
    return split
