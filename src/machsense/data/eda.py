"""Reusable Exploratory Data Analysis (EDA) and statistical computation engine for MachSense."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from machsense.config.settings import get_settings
from machsense.data.loader import load_raw_data
from machsense.data.schema import EXPECTED_FEATURE_COLUMNS, FAILURE_MODE_COLUMNS
from machsense.utils.logger import get_logger, setup_logging

logger = get_logger("machsense.data.eda")


@dataclass
class DatasetSummary:
    """Statistical summary of dataset distributions and imbalance metrics."""
    total_samples: int
    total_features: int
    target_counts: Dict[int, int]
    failure_rate_pct: float
    imbalance_ratio: float
    failure_mode_counts: Dict[str, int]
    numerical_stats: pd.DataFrame
    outlier_counts: Dict[str, int]
    skewness: Dict[str, float]

    def summary(self) -> str:
        """Render readable text summary of statistical metrics."""
        return (
            f"=== MachSense Dataset Statistical Overview ===\n"
            f"Total Samples:          {self.total_samples:,}\n"
            f"Target Non-Failures:    {self.target_counts.get(0, 0):,} ({(100 - self.failure_rate_pct):.2f}%)\n"
            f"Target Failures:        {self.target_counts.get(1, 0):,} ({self.failure_rate_pct:.2f}%)\n"
            f"Imbalance Ratio (0:1):  {self.imbalance_ratio:.1f} : 1\n"
            f"\nFailure Modes Breakdown:\n"
            + "\n".join([f"  - {k.upper():<5}: {v:>4} occurrences" for k, v in self.failure_mode_counts.items()])
            + f"\n\nOutliers (1.5*IQR bounds):\n"
            + "\n".join([f"  - {k:<25}: {v:>4} outliers" for k, v in self.outlier_counts.items()])
        )


def compute_numerical_outliers(df: pd.DataFrame, columns: List[str]) -> Dict[str, int]:
    """Calculate the number of outliers using the 1.5 * IQR rule for each column."""
    outlier_counts = {}
    for col in columns:
        if col in df.columns:
            q25 = df[col].quantile(0.25)
            q75 = df[col].quantile(0.75)
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            count = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
            outlier_counts[col] = count
    return outlier_counts


def compute_dataset_summary(df: pd.DataFrame) -> DatasetSummary:
    """Compute comprehensive statistical profile of the predictive maintenance dataset.

    Args:
        df: Input standardized DataFrame.

    Returns:
        DatasetSummary dataclass instance.
    """
    total_samples = len(df)
    target_counts = df["machine_failure"].value_counts().to_dict() if "machine_failure" in df.columns else {0: total_samples}
    pos_count = target_counts.get(1, 0)
    neg_count = target_counts.get(0, 0)

    failure_rate = (pos_count / max(1, total_samples)) * 100.0
    imbalance_ratio = neg_count / max(1, pos_count)

    # Failure modes
    fm_counts = {}
    for fm in FAILURE_MODE_COLUMNS:
        if fm in df.columns:
            fm_counts[fm] = int(df[fm].sum())

    # Numerical statistics
    num_cols = [c for c in EXPECTED_FEATURE_COLUMNS if c != "type" and c in df.columns]
    num_stats = df[num_cols].describe().T
    num_stats["median"] = df[num_cols].median()
    num_stats["skewness"] = df[num_cols].skew()
    num_stats["kurtosis"] = df[num_cols].kurtosis()

    outlier_counts = compute_numerical_outliers(df, num_cols)
    skewness_dict = df[num_cols].skew().to_dict()

    return DatasetSummary(
        total_samples=total_samples,
        total_features=len(df.columns),
        target_counts=target_counts,
        failure_rate_pct=failure_rate,
        imbalance_ratio=imbalance_ratio,
        failure_mode_counts=fm_counts,
        numerical_stats=num_stats,
        outlier_counts=outlier_counts,
        skewness=skewness_dict,
    )


def compute_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute domain-informed engineering features derived from physical sensor telemetry.

    Features derived:
    1. power_w: Mechanical rotational power = Torque (Nm) * Angular velocity (rad/s)
       Angular velocity = speed_rpm * 2 * pi / 60
    2. temp_difference_k: Thermal gradient = Process Temp (K) - Air Temp (K)
    3. overstrain_index: Mechanical tool strain = Torque (Nm) * Tool Wear (min)
    4. temp_ratio: Relative thermal ratio = Process Temp / Air Temp
    """
    derived_df = df.copy()

    if "torque_nm" in df.columns and "rotational_speed_rpm" in df.columns:
        # Power in Watts: Power = Torque * (RPM * 2 * pi / 60)
        derived_df["power_w"] = derived_df["torque_nm"] * (derived_df["rotational_speed_rpm"] * (2 * np.pi / 60.0))

    if "process_temperature_k" in df.columns and "air_temperature_k" in df.columns:
        # Temperature difference in Kelvin
        derived_df["temp_difference_k"] = derived_df["process_temperature_k"] - derived_df["air_temperature_k"]
        derived_df["temp_ratio"] = derived_df["process_temperature_k"] / derived_df["air_temperature_k"]

    if "torque_nm" in df.columns and "tool_wear_min" in df.columns:
        # Strain index (Torque * Tool Wear)
        derived_df["overstrain_index"] = derived_df["torque_nm"] * derived_df["tool_wear_min"]

    return derived_df


def compute_correlations(df: pd.DataFrame, target_col: str = "machine_failure") -> pd.Series:
    """Compute Pearson correlation of all numerical features against the target variable."""
    num_df = df.select_dtypes(include=[np.number])
    if target_col in num_df.columns:
        corrs = num_df.corr()[target_col].drop(target_col, errors="ignore")
        return corrs.sort_values(ascending=False)
    return pd.Series(dtype=float)


def run_eda_cli() -> None:
    """CLI execution for exploratory data analysis statistics."""
    setup_logging()
    logger.info("Running MachSense Exploratory Data Analysis computation...")
    df = load_raw_data()
    summary = compute_dataset_summary(df)
    print(summary.summary())

    print("\n=== Numerical Features Descriptive Statistics ===")
    print(summary.numerical_stats[["mean", "std", "min", "50%", "max", "skewness"]])

    derived = compute_domain_features(df)
    corrs = compute_correlations(derived)
    print("\n=== Feature Correlations with Target (Machine Failure) ===")
    print(corrs.to_string())


if __name__ == "__main__":
    run_eda_cli()
