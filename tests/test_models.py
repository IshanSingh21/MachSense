"""Unit and integration tests for MachSense model training and evaluation engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from machsense.models.evaluator import (
    ModelEvaluationResult,
    evaluate_classifier,
    find_optimal_threshold,
)
from machsense.models.trainer import (
    get_baseline_models,
    train_and_evaluate_models,
)


@pytest.fixture
def sample_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Provide synthetic feature matrix and binary target for model tests."""
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame(
        {
            "feature1": rng.normal(0, 1, n),
            "feature2": rng.normal(5, 2, n),
            "feature3": rng.uniform(0, 10, n),
        }
    )
    # Synthetic imbalanced target (10% positive)
    y = pd.Series((rng.uniform(0, 1, n) < 0.10).astype(int), name="machine_failure")
    return X, y


def test_get_baseline_models():
    """Verify get_baseline_models returns all required model families."""
    models = get_baseline_models()
    assert "dummy_most_frequent" in models
    assert "logistic_regression" in models
    assert "decision_tree" in models
    assert "random_forest" in models
    assert "hist_gradient_boosting" in models


def test_evaluate_classifier(sample_dataset: tuple[pd.DataFrame, pd.Series]):
    """Verify evaluate_classifier computes all required classification metrics."""
    X, y = sample_dataset
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    res = evaluate_classifier(model, X, y, model_name="TestRF", split_name="test")

    assert isinstance(res, ModelEvaluationResult)
    assert res.model_name == "TestRF"
    assert 0.0 <= res.accuracy <= 1.0
    assert 0.0 <= res.precision <= 1.0
    assert 0.0 <= res.recall <= 1.0
    assert 0.0 <= res.f1_minority <= 1.0
    assert 0.0 <= res.pr_auc <= 1.0
    assert 0.0 <= res.roc_auc <= 1.0
    assert res.true_positives + res.false_negatives == int(y.sum())
    assert 0.0 <= res.false_negative_rate <= 1.0
    assert isinstance(res.summary(), str)
    assert isinstance(res.to_dict(), dict)


def test_find_optimal_threshold():
    """Verify optimal threshold sweep correctly identifies valid threshold."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    thresh, f1, rec = find_optimal_threshold(y_true, y_prob)
    assert 0.0 <= thresh <= 1.0
    assert 0.0 <= f1 <= 1.0
    assert 0.0 <= rec <= 1.0


def test_train_and_evaluate_models_pipeline():
    """Verify end-to-end model benchmark execution, leaderboard generation, and serialization."""
    val_df, experiments, best_model = train_and_evaluate_models(save_artifacts=False)

    assert isinstance(val_df, pd.DataFrame)
    assert len(val_df) == 5  # 5 model families
    assert "pr_auc" in val_df.columns
    assert "false_negatives" in val_df.columns
    assert "dummy_most_frequent" in experiments
    assert hasattr(best_model, "predict")
