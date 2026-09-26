"""Unit and integration tests for MachSense hyperparameter tuning and model registry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from machsense.features.preprocessor import MachSensePreprocessor
from machsense.models.registry import ModelMetadata, ModelRegistry
from machsense.models.tuner import (
    get_tuning_search_spaces,
    run_model_optimization,
    tune_model,
)


@pytest.fixture
def dummy_train_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provide synthetic training data for fast CV tuning tests."""
    rng = np.random.default_rng(42)
    n = 100
    X = pd.DataFrame(
        {
            "feature1": rng.normal(0, 1, n),
            "feature2": rng.normal(5, 2, n),
        }
    )
    y = pd.Series((rng.uniform(0, 1, n) < 0.15).astype(int), name="machine_failure")
    return X, y


def test_get_tuning_search_spaces():
    """Verify search spaces are defined for RandomForest and HistGradientBoosting."""
    spaces = get_tuning_search_spaces()
    assert "tuned_random_forest" in spaces
    assert "tuned_hist_gradient_boosting" in spaces

    rf_est, rf_grid = spaces["tuned_random_forest"]
    assert "n_estimators" in rf_grid
    assert "max_depth" in rf_grid


def test_tune_model_cv(dummy_train_data: tuple[pd.DataFrame, pd.Series]):
    """Verify tune_model executes Stratified K-Fold CV search and returns best estimator."""
    X, y = dummy_train_data
    param_grid = {"n_estimators": [5, 10], "max_depth": [2, 4]}
    base_est = RandomForestClassifier(random_state=42)

    best_est, best_params, best_score = tune_model(
        model_name="test_tune_rf",
        base_estimator=base_est,
        param_grid=param_grid,
        X_train=X,
        y_train=y,
        cv_splits=2,
    )

    assert hasattr(best_est, "predict")
    assert "n_estimators" in best_params
    assert 0.0 <= best_score <= 1.0


def test_model_registry_save_and_load(tmp_path: Path, dummy_train_data: tuple[pd.DataFrame, pd.Series]):
    """Verify ModelRegistry saves and loads versioned artifacts with complete metadata."""
    X, y = dummy_train_data
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)

    preprocessor = MachSensePreprocessor()
    # Create minimal raw-like frame to fit preprocessor
    raw_df = pd.DataFrame(
        {
            "type": ["L", "M", "H"] * 10,
            "air_temperature_k": [300.0] * 30,
            "process_temperature_k": [310.0] * 30,
            "rotational_speed_rpm": [1500] * 30,
            "torque_nm": [40.0] * 30,
            "tool_wear_min": [10.0] * 30,
        }
    )
    preprocessor.fit(raw_df)

    metadata = ModelMetadata(
        model_version="v1.0.0",
        model_name="test_rf",
        model_type="RandomForestClassifier",
        created_at="2026-09-19T00:00:00Z",
        hyperparameters={"n_estimators": 5},
        feature_names=["f1", "f2"],
        optimal_threshold=0.55,
        cv_mean_pr_auc=0.88,
        validation_metrics={"pr_auc": 0.88},
        test_metrics={"pr_auc": 0.87},
    )

    registry = ModelRegistry(models_dir=tmp_path / "models")
    saved = registry.save_versioned_artifacts(
        model=model,
        preprocessor=preprocessor,
        metadata=metadata,
        version="v1.0.0",
        set_as_active=True,
    )

    assert saved["model_path"].exists()
    assert saved["preprocessor_path"].exists()
    assert saved["metadata_path"].exists()

    # Load versioned
    loaded_model, loaded_prep, loaded_meta = registry.load_versioned_artifacts(version="v1.0.0")
    assert hasattr(loaded_model, "predict")
    assert loaded_prep.is_fitted_ is True
    assert loaded_meta["model_version"] == "v1.0.0"
    assert loaded_meta["optimal_threshold"] == 0.55


def test_registry_verify_artifacts_integrity():
    """Verify registry diagnostic test loads active artifacts and executes valid inference."""
    registry = ModelRegistry()
    diag = registry.verify_artifacts_integrity()

    assert diag["status"] == "PASSED"
    assert "failure_probability" in diag
    assert 0.0 <= diag["failure_probability"] <= 1.0
    assert diag["predicted_class"] in [0, 1]


def test_run_model_optimization_pipeline():
    """Verify full Day 6 model optimization pipeline execution and leaderboard creation."""
    champion_model, metadata, comparison_df = run_model_optimization(
        version="v1.0.0",
        cv_splits=2,
        save_artifacts=False,
    )

    assert hasattr(champion_model, "predict")
    assert isinstance(metadata, ModelMetadata)
    assert len(comparison_df) >= 2
    assert "val_pr_auc" in comparison_df.columns
