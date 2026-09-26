"""Data cleaning, validation engine, and physical inconsistency filter."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from machsense.data.schema import (
    EXPECTED_FEATURE_COLUMNS,
    MachineType,
    SensorBoundaries,
    ValidationReport,
)
from machsense.utils.logger import get_logger

logger = get_logger("machsense.data.cleaner")


def validate_dataset(df: pd.DataFrame) -> ValidationReport:
    """Perform rigorous domain and physical constraint checks on the dataset.

    Args:
        df: Input DataFrame with standardized column names.

    Returns:
        ValidationReport containing all diagnostics and validation status.
    """
    total_records = len(df)
    missing_vals = df.isnull().sum().to_dict()
    errors: List[str] = []
    warnings: List[str] = []

    # Check for duplicate machine identifiers (UDI)
    duplicate_udi = 0
    if "udi" in df.columns:
        duplicate_udi = int(df.duplicated(subset=["udi"]).sum())
        if duplicate_udi > 0:
            errors.append(f"Found {duplicate_udi} duplicate UDI entries.")

    # Check required columns
    for col in EXPECTED_FEATURE_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing required feature column: '{col}'")

    # Boundary checks
    out_of_bounds = 0
    if "air_temperature_k" in df.columns:
        oob = (
            (df["air_temperature_k"] < SensorBoundaries.AIR_TEMP_MIN_K)
            | (df["air_temperature_k"] > SensorBoundaries.AIR_TEMP_MAX_K)
        ).sum()
        out_of_bounds += int(oob)

    if "process_temperature_k" in df.columns:
        oob = (
            (df["process_temperature_k"] < SensorBoundaries.PROCESS_TEMP_MIN_K)
            | (df["process_temperature_k"] > SensorBoundaries.PROCESS_TEMP_MAX_K)
        ).sum()
        out_of_bounds += int(oob)

    if "rotational_speed_rpm" in df.columns:
        oob = (
            (df["rotational_speed_rpm"] < SensorBoundaries.ROTATIONAL_SPEED_MIN_RPM)
            | (df["rotational_speed_rpm"] > SensorBoundaries.ROTATIONAL_SPEED_MAX_RPM)
        ).sum()
        out_of_bounds += int(oob)

    if "torque_nm" in df.columns:
        oob = (
            (df["torque_nm"] < SensorBoundaries.TORQUE_MIN_NM)
            | (df["torque_nm"] > SensorBoundaries.TORQUE_MAX_NM)
        ).sum()
        out_of_bounds += int(oob)

    if "tool_wear_min" in df.columns:
        oob = (
            (df["tool_wear_min"] < SensorBoundaries.TOOL_WEAR_MIN_MIN)
            | (df["tool_wear_min"] > SensorBoundaries.TOOL_WEAR_MAX_MIN)
        ).sum()
        out_of_bounds += int(oob)

    # Physical inconsistency: process temp must not be significantly colder than ambient air
    thermodynamic_violations = 0
    if "air_temperature_k" in df.columns and "process_temperature_k" in df.columns:
        # Allowing 0.5 K measurement noise tolerance
        viol_mask = df["process_temperature_k"] < (df["air_temperature_k"] - 0.5)
        thermodynamic_violations = int(viol_mask.sum())
        if thermodynamic_violations > 0:
            warnings.append(
                f"Found {thermodynamic_violations} records where process temp is colder than air temp."
            )

    # Valid machine type checks
    if "type" in df.columns:
        valid_types = {e.value for e in MachineType}
        invalid_types = int((~df["type"].isin(valid_types)).sum())
        if invalid_types > 0:
            errors.append(f"Found {invalid_types} records with invalid machine type.")

    is_valid = len(errors) == 0 and out_of_bounds == 0
    valid_records_count = total_records - out_of_bounds - duplicate_udi - thermodynamic_violations

    return ValidationReport(
        is_valid=is_valid,
        total_records=total_records,
        valid_records_count=max(0, valid_records_count),
        missing_values_per_col=missing_vals,
        out_of_bound_records=out_of_bounds,
        duplicate_udi_count=duplicate_udi,
        thermodynamic_inconsistencies=thermodynamic_violations,
        errors=errors,
        warnings=warnings,
    )


def clean_dataset(
    df: pd.DataFrame,
    drop_inconsistent: bool = True,
    impute_numerical_strategy: str = "median",
) -> Tuple[pd.DataFrame, ValidationReport]:
    """Clean dataset by handling missing values, duplicates, and physical anomalies.

    Args:
        df: Standardized input DataFrame.
        drop_inconsistent: If True, removes physical boundary violations and duplicates.
        impute_numerical_strategy: Strategy for handling missing numerical data ('median' or 'drop').

    Returns:
        Tuple of (Cleaned DataFrame, ValidationReport).
    """
    initial_count = len(df)
    logger.info("Starting data cleaning pipeline on %d records...", initial_count)
    clean_df = df.copy()

    # 1. Deduplication on UDI if present
    if "udi" in clean_df.columns:
        dups = clean_df.duplicated(subset=["udi"])
        if dups.any():
            dup_count = dups.sum()
            logger.warning("Dropping %d duplicate UDI records.", dup_count)
            clean_df = clean_df.drop_duplicates(subset=["udi"], keep="first")

    # 2. Missing Value Treatment
    missing_counts = clean_df.isnull().sum()
    if missing_counts.sum() > 0:
        logger.info("Handling missing values across columns: %s", missing_counts[missing_counts > 0].to_dict())
        if impute_numerical_strategy == "drop":
            clean_df = clean_df.dropna()
        elif impute_numerical_strategy == "median":
            numerical_cols = clean_df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if clean_df[col].isnull().any():
                    median_val = clean_df[col].median()
                    clean_df[col] = clean_df[col].fillna(median_val)
                    logger.info("Imputed missing values in '%s' with median (%s)", col, median_val)
            # Categorical columns imputation with mode
            categorical_cols = clean_df.select_dtypes(include=["object", "category"]).columns
            for col in categorical_cols:
                if clean_df[col].isnull().any():
                    mode_val = clean_df[col].mode()[0]
                    clean_df[col] = clean_df[col].fillna(mode_val)
                    logger.info("Imputed missing values in '%s' with mode (%s)", col, mode_val)

    # 3. Physical Boundary & Inconsistency Filtering
    if drop_inconsistent:
        mask = pd.Series(True, index=clean_df.index)

        if "air_temperature_k" in clean_df.columns:
            mask &= (clean_df["air_temperature_k"] >= SensorBoundaries.AIR_TEMP_MIN_K) & (
                clean_df["air_temperature_k"] <= SensorBoundaries.AIR_TEMP_MAX_K
            )

        if "process_temperature_k" in clean_df.columns:
            mask &= (clean_df["process_temperature_k"] >= SensorBoundaries.PROCESS_TEMP_MIN_K) & (
                clean_df["process_temperature_k"] <= SensorBoundaries.PROCESS_TEMP_MAX_K
            )

        if "rotational_speed_rpm" in clean_df.columns:
            mask &= (clean_df["rotational_speed_rpm"] >= SensorBoundaries.ROTATIONAL_SPEED_MIN_RPM) & (
                clean_df["rotational_speed_rpm"] <= SensorBoundaries.ROTATIONAL_SPEED_MAX_RPM
            )

        if "torque_nm" in clean_df.columns:
            mask &= (clean_df["torque_nm"] >= SensorBoundaries.TORQUE_MIN_NM) & (
                clean_df["torque_nm"] <= SensorBoundaries.TORQUE_MAX_NM
            )

        if "tool_wear_min" in clean_df.columns:
            mask &= (clean_df["tool_wear_min"] >= SensorBoundaries.TOOL_WEAR_MIN_MIN) & (
                clean_df["tool_wear_min"] <= SensorBoundaries.TOOL_WEAR_MAX_MIN
            )

        if "air_temperature_k" in clean_df.columns and "process_temperature_k" in clean_df.columns:
            mask &= clean_df["process_temperature_k"] >= (clean_df["air_temperature_k"] - 0.5)

        if "type" in clean_df.columns:
            valid_types = {e.value for e in MachineType}
            mask &= clean_df["type"].isin(valid_types)

        filtered_out = int((~mask).sum())
        if filtered_out > 0:
            logger.warning("Filtered out %d physically inconsistent/out-of-bounds records.", filtered_out)
            clean_df = clean_df[mask].reset_index(drop=True)

    report = validate_dataset(clean_df)
    logger.info("Cleaning completed: %d -> %d records retained.", initial_count, len(clean_df))
    return clean_df, report
