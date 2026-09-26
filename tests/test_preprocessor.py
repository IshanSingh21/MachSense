"""Unit and integration tests for MachSense preprocessing and feature engineering."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from machsense.data.download import generate_synthetic_ai4i_benchmark
from machsense.data.loader import standardize_columns
from machsense.data.splitter import separate_features_and_target
from machsense.features.domain_features import DomainFeatureExtractor
from machsense.features.pipeline import run_feature_pipeline
from machsense.features.preprocessor import MachSensePreprocessor


@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    """Provide clean X feature DataFrame for preprocessing tests."""
    raw = generate_synthetic_ai4i_benchmark(n_samples=150, seed=42)
    std = standardize_columns(raw)
    X, y, _ = separate_features_and_target(std)
    return X


def test_domain_feature_extractor(sample_feature_df: pd.DataFrame):
    """Verify DomainFeatureExtractor creates all 5 domain physical features."""
    extractor = DomainFeatureExtractor(include_wear_rate=True)
    transformed = extractor.fit_transform(sample_feature_df)

    expected_new = ["power_w", "temp_difference_k", "temp_ratio", "overstrain_index", "tool_wear_rate"]
    for feat in expected_new:
        assert feat in transformed.columns
        assert not transformed[feat].isnull().any()

    # Physical properties
    assert (transformed["power_w"] > 0).all()
    assert (transformed["temp_difference_k"] >= 0).all()


def test_preprocessor_fit_transform(sample_feature_df: pd.DataFrame):
    """Verify MachSensePreprocessor fits and transforms with correct output dimensions."""
    preprocessor = MachSensePreprocessor(scaling_method="standard")
    X_trans = preprocessor.fit_transform(sample_feature_df)

    assert isinstance(X_trans, pd.DataFrame)
    assert len(X_trans) == len(sample_feature_df)
    assert "type_L" in X_trans.columns
    assert "type_M" in X_trans.columns
    assert "type_H" in X_trans.columns
    assert "power_w" in X_trans.columns
    assert "overstrain_index" in X_trans.columns


def test_preprocessor_handles_missing_values(sample_feature_df: pd.DataFrame):
    """Verify preprocessor imputes missing numerical and categorical values."""
    corrupt_df = sample_feature_df.copy()
    corrupt_df.loc[0, "torque_nm"] = np.nan
    corrupt_df.loc[1, "type"] = np.nan

    preprocessor = MachSensePreprocessor()
    preprocessor.fit(sample_feature_df)
    X_trans = preprocessor.transform(corrupt_df)

    assert not X_trans.isnull().any().any()


def test_preprocessor_handles_unseen_categories(sample_feature_df: pd.DataFrame):
    """Verify preprocessor handles unknown categorical levels without crashing."""
    corrupt_df = sample_feature_df.copy()
    corrupt_df.loc[0, "type"] = "UNKNOWN_TYPE"

    preprocessor = MachSensePreprocessor()
    preprocessor.fit(sample_feature_df)
    X_trans = preprocessor.transform(corrupt_df)

    # One-hot encoded columns for unknown will all be 0
    assert X_trans.loc[0, "type_L"] == 0.0
    assert X_trans.loc[0, "type_M"] == 0.0
    assert X_trans.loc[0, "type_H"] == 0.0


def test_preprocessor_transform_instance(sample_feature_df: pd.DataFrame):
    """Verify transform_instance successfully processes a single telemetry dictionary payload."""
    preprocessor = MachSensePreprocessor()
    preprocessor.fit(sample_feature_df)

    payload = {
        "type": "M",
        "air_temperature_k": 300.2,
        "process_temperature_k": 310.5,
        "rotational_speed_rpm": 1500,
        "torque_nm": 42.0,
        "tool_wear_min": 120.0,
    }

    result = preprocessor.transform_instance(payload)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result.loc[0, "type_M"] == 1.0
    assert result.loc[0, "type_L"] == 0.0


def test_preprocessor_save_and_load(sample_feature_df: pd.DataFrame, tmp_path: Path):
    """Verify serialization and deserialization roundtrip preserves identical transforms."""
    save_file = tmp_path / "preprocessor.joblib"

    preprocessor = MachSensePreprocessor(scaling_method="robust")
    X_trans_orig = preprocessor.fit_transform(sample_feature_df)

    # Save
    preprocessor.save(save_file)
    assert save_file.exists()

    # Load
    loaded = MachSensePreprocessor.load(save_file)
    X_trans_loaded = loaded.transform(sample_feature_df)

    pd.testing.assert_frame_equal(X_trans_orig, X_trans_loaded)


def test_run_feature_pipeline_execution():
    """Verify complete feature pipeline execution and artifact creation."""
    preprocessor, X_train, X_val, X_test = run_feature_pipeline(save_artifacts=True)

    assert preprocessor.is_fitted_ is True
    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
