"""Unit and integration tests for MachSense data ingestion, validation, and splitting pipeline."""

import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from machsense.data.cleaner import clean_dataset, validate_dataset
from machsense.data.download import download_ai4i_dataset, generate_synthetic_ai4i_benchmark
from machsense.data.loader import load_raw_data, standardize_columns
from machsense.data.pipeline import run_data_pipeline
from machsense.data.schema import (
    COLUMN_MAPPING,
    EXPECTED_FEATURE_COLUMNS,
    MachineType,
    SensorRecord,
    ValidationReport,
)
from machsense.data.splitter import (
    DataSplit,
    separate_features_and_target,
    split_data,
)


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Fixture providing a small synthetic raw DataFrame matching UCI format."""
    return generate_synthetic_ai4i_benchmark(n_samples=200, seed=123)


def test_generate_synthetic_benchmark(sample_raw_df: pd.DataFrame):
    """Verify synthetic dataset generator produces correct shape and columns."""
    assert len(sample_raw_df) == 200
    assert "UDI" in sample_raw_df.columns
    assert "Machine failure" in sample_raw_df.columns
    assert "Air temperature [K]" in sample_raw_df.columns
    assert sample_raw_df["Machine failure"].sum() > 0


def test_standardize_columns(sample_raw_df: pd.DataFrame):
    """Verify header normalization converts raw headers to standard snake_case."""
    std_df = standardize_columns(sample_raw_df)
    assert "udi" in std_df.columns
    assert "air_temperature_k" in std_df.columns
    assert "process_temperature_k" in std_df.columns
    assert "rotational_speed_rpm" in std_df.columns
    assert "torque_nm" in std_df.columns
    assert "tool_wear_min" in std_df.columns
    assert "machine_failure" in std_df.columns


def test_sensor_record_validation_success():
    """Verify valid sensor record passes schema validation."""
    record = SensorRecord(
        udi=1,
        product_id="M14860",
        type=MachineType.MEDIUM,
        air_temperature_k=298.1,
        process_temperature_k=308.6,
        rotational_speed_rpm=1551.0,
        torque_nm=42.8,
        tool_wear_min=0.0,
        machine_failure=0,
    )
    assert record.udi == 1
    assert record.air_temperature_k == 298.1


def test_sensor_record_validation_failure_on_thermodynamic_violation():
    """Verify schema rejects process temp colder than air temp."""
    with pytest.raises(ValidationError):
        SensorRecord(
            udi=2,
            product_id="L47181",
            type=MachineType.LOW,
            air_temperature_k=310.0,
            process_temperature_k=290.0,  # Physically impossible under active cutting
            rotational_speed_rpm=1500.0,
            torque_nm=40.0,
            tool_wear_min=10.0,
            machine_failure=0,
        )


def test_validate_dataset_diagnostics(sample_raw_df: pd.DataFrame):
    """Verify validate_dataset accurately reports valid dataset status."""
    std_df = standardize_columns(sample_raw_df)
    report = validate_dataset(std_df)
    assert isinstance(report, ValidationReport)
    assert report.is_valid is True
    assert report.total_records == 200
    assert report.duplicate_udi_count == 0


def test_validate_dataset_detects_anomalies(sample_raw_df: pd.DataFrame):
    """Verify validate_dataset catches out-of-bounds anomalies and duplicates."""
    std_df = standardize_columns(sample_raw_df)
    # Inject corrupt records
    corrupt_df = std_df.copy()
    corrupt_df.loc[0, "air_temperature_k"] = 999.0  # Extreme out-of-bound
    corrupt_df.loc[1, "udi"] = corrupt_df.loc[2, "udi"]  # Duplicate UDI

    report = validate_dataset(corrupt_df)
    assert report.is_valid is False
    assert report.out_of_bound_records >= 1
    assert report.duplicate_udi_count >= 1


def test_clean_dataset_handles_missing_and_outliers(sample_raw_df: pd.DataFrame):
    """Verify clean_dataset removes duplicates, imputes missing values, and filters violations."""
    std_df = standardize_columns(sample_raw_df)
    corrupt_df = std_df.copy()

    # Introduce missing values and duplicate
    corrupt_df.loc[5, "torque_nm"] = np.nan
    corrupt_df = pd.concat([corrupt_df, corrupt_df.iloc[[0]]], ignore_index=True)

    clean_df, report = clean_dataset(corrupt_df, drop_inconsistent=True, impute_numerical_strategy="median")

    assert not clean_df["torque_nm"].isnull().any()
    assert clean_df.duplicated(subset=["udi"]).sum() == 0
    assert len(clean_df) <= len(sample_raw_df)


def test_separate_features_and_target(sample_raw_df: pd.DataFrame):
    """Verify features and target separation strips metadata and target columns."""
    std_df = standardize_columns(sample_raw_df)
    X, y, failure_modes = separate_features_and_target(std_df)

    assert "machine_failure" not in X.columns
    assert "udi" not in X.columns
    assert "product_id" not in X.columns
    assert "twf" not in X.columns
    assert len(X) == len(y)
    assert set(EXPECTED_FEATURE_COLUMNS).issubset(set(X.columns))


def test_stratified_split_and_no_leakage(sample_raw_df: pd.DataFrame):
    """Verify train/val/test splitting maintains stratification and strict index disjointness."""
    std_df = standardize_columns(sample_raw_df)
    clean_df, _ = clean_dataset(std_df)

    split = split_data(clean_df, val_size=0.15, test_size=0.15, random_state=42, stratify=True)

    # 1. Verification of no leakage
    assert split.verify_no_leakage() is True

    # 2. Proportion checks (approx 70 / 15 / 15)
    total = len(clean_df)
    assert len(split.X_train) == pytest.approx(int(total * 0.70), abs=3)
    assert len(split.X_val) == pytest.approx(int(total * 0.15), abs=2)
    assert len(split.X_test) == pytest.approx(int(total * 0.15), abs=2)

    # 3. Stratification checks: positive failure rates should be closely aligned
    train_rate = split.y_train.mean()
    val_rate = split.y_val.mean()
    test_rate = split.y_test.mean()

    assert abs(train_rate - val_rate) < 0.08
    assert abs(train_rate - test_rate) < 0.08


def test_run_data_pipeline_end_to_end(tmp_path: Path):
    """Verify full end-to-end data pipeline runs and creates saved artifacts."""
    raw_file = tmp_path / "raw_test_data.csv"
    df = generate_synthetic_ai4i_benchmark(n_samples=300, seed=789)
    df.to_csv(raw_file, index=False)

    split = run_data_pipeline(
        raw_file_path=raw_file,
        save_artifacts=True,
        val_size=0.2,
        test_size=0.1,
        random_state=42,
    )

    assert isinstance(split, DataSplit)
    assert len(split.X_train) > 0
    assert len(split.X_val) > 0
    assert len(split.X_test) > 0
    assert split.verify_no_leakage() is True
