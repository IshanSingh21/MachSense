"""Unit tests for EDA and statistical computation routines."""

import numpy as np
import pandas as pd
import pytest

from machsense.data.download import generate_synthetic_ai4i_benchmark
from machsense.data.eda import (
    DatasetSummary,
    compute_correlations,
    compute_dataset_summary,
    compute_domain_features,
    compute_numerical_outliers,
)
from machsense.data.loader import standardize_columns


@pytest.fixture
def clean_test_df() -> pd.DataFrame:
    """Provide standardized DataFrame for testing EDA routines."""
    raw = generate_synthetic_ai4i_benchmark(n_samples=250, seed=42)
    return standardize_columns(raw)


def test_compute_dataset_summary(clean_test_df: pd.DataFrame):
    """Verify compute_dataset_summary correctly calculates imbalance metrics and summary stats."""
    summary = compute_dataset_summary(clean_test_df)
    assert isinstance(summary, DatasetSummary)
    assert summary.total_samples == 250
    assert summary.failure_rate_pct > 0
    assert summary.imbalance_ratio > 1.0
    assert "air_temperature_k" in summary.numerical_stats.index
    assert "torque_nm" in summary.outlier_counts
    assert isinstance(summary.summary(), str)


def test_compute_numerical_outliers(clean_test_df: pd.DataFrame):
    """Verify outlier computation using IQR."""
    cols = ["air_temperature_k", "rotational_speed_rpm"]
    outliers = compute_numerical_outliers(clean_test_df, cols)
    assert "air_temperature_k" in outliers
    assert "rotational_speed_rpm" in outliers
    assert all(isinstance(v, int) for v in outliers.values())


def test_compute_domain_features(clean_test_df: pd.DataFrame):
    """Verify domain-engineered features are correctly computed from telemetry."""
    derived = compute_domain_features(clean_test_df)
    assert "power_w" in derived.columns
    assert "temp_difference_k" in derived.columns
    assert "overstrain_index" in derived.columns
    assert "temp_ratio" in derived.columns

    # Verify physical calculations
    assert (derived["temp_difference_k"] >= 0).all()
    assert (derived["power_w"] > 0).all()
    assert (derived["overstrain_index"] >= 0).all()


def test_compute_correlations(clean_test_df: pd.DataFrame):
    """Verify target correlation calculation."""
    derived = compute_domain_features(clean_test_df)
    corrs = compute_correlations(derived, target_col="machine_failure")
    assert isinstance(corrs, pd.Series)
    assert "machine_failure" not in corrs.index
    assert "torque_nm" in corrs.index
