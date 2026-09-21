"""Unit tests for MachSense Explainable AI (SHAP) module."""

from pathlib import Path
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from machsense.models.explainability import (
    FEATURE_DISPLAY_NAMES,
    NON_CAUSAL_DISCLAIMER,
    FeatureAttribution,
    LocalExplanationResult,
    MachSenseExplainer,
)


@pytest.fixture
def synthetic_tree_data():
    """Create a minimal synthetic dataset and trained RandomForestClassifier for fast deterministic tests."""
    np.random.seed(42)
    n_samples = 100
    feature_names = [
        "type_L",
        "type_M",
        "type_H",
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        "power_w",
        "temp_difference_k",
        "temp_ratio",
        "overstrain_index",
        "tool_wear_rate",
    ]
    X_mat = np.random.randn(n_samples, len(feature_names))
    # Synthetic target driven primarily by power_w (col 8) and overstrain_index (col 11)
    y_vec = (X_mat[:, 8] + X_mat[:, 11] > 0.5).astype(int)

    X_df = pd.DataFrame(X_mat, columns=feature_names)
    y_series = pd.Series(y_vec)

    model = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
    model.fit(X_df, y_series)

    return model, X_df, y_series, feature_names


def test_explainer_initialization(synthetic_tree_data):
    """Test initializing MachSenseExplainer with model and feature names."""
    model, X_df, _, feature_names = synthetic_tree_data
    explainer = MachSenseExplainer(model=model, feature_names=feature_names)

    assert explainer.model is model
    assert explainer.feature_names == feature_names
    assert isinstance(explainer.expected_value, float)
    assert 0.0 <= explainer.expected_value <= 1.0


def test_compute_global_importance(synthetic_tree_data):
    """Test calculating global feature importance table."""
    model, X_df, _, feature_names = synthetic_tree_data
    explainer = MachSenseExplainer(model=model, feature_names=feature_names)

    importance_df = explainer.compute_global_importance(X_df)

    assert isinstance(importance_df, pd.DataFrame)
    assert len(importance_df) == len(feature_names)
    assert list(importance_df.columns) == [
        "feature",
        "display_name",
        "mean_abs_shap",
        "relative_importance",
        "rank",
    ]
    # Check descending order
    assert (importance_df["mean_abs_shap"].diff().dropna() <= 1e-6).all()
    # Check relative importances sum to 1.0
    assert np.isclose(importance_df["relative_importance"].sum(), 1.0, atol=1e-4)
    # Check rank order is 1 to M
    assert list(importance_df["rank"]) == list(range(1, len(feature_names) + 1))


def test_explain_instance_structure(synthetic_tree_data):
    """Test generating structured local explanation for a single instance."""
    model, X_df, _, feature_names = synthetic_tree_data
    explainer = MachSenseExplainer(model=model, feature_names=feature_names)

    instance = X_df.iloc[0]
    result = explainer.explain_instance(instance, threshold=0.5, top_k=3)

    assert isinstance(result, LocalExplanationResult)
    assert 0.0 <= result.predicted_probability <= 1.0
    assert result.predicted_class in (0, 1)
    assert result.risk_level in ("NOMINAL", "MODERATE_WARNING", "ELEVATED", "CRITICAL")
    assert len(result.all_attributions) == len(feature_names)
    assert len(result.top_risk_escalators) <= 3
    assert len(result.top_stabilizers) <= 3
    assert NON_CAUSAL_DISCLAIMER in result.non_causal_disclaimer
    assert len(result.operator_summary) > 0


def test_explain_instance_efficiency_additivity(synthetic_tree_data):
    """Verify Shapley efficiency property (sum of SHAP values + base_val ~ predicted score)."""
    model, X_df, _, feature_names = synthetic_tree_data
    explainer = MachSenseExplainer(model=model, feature_names=feature_names)

    instance = X_df.iloc[5]
    result = explainer.explain_instance(instance)

    sum_shap = sum(attr.shap_value for attr in result.all_attributions)
    reconstructed_pred = result.base_value + sum_shap

    # TreeSHAP values for probability output in scikit-learn sum to model.predict_proba
    assert np.isclose(reconstructed_pred, result.predicted_probability, atol=1e-3)


def test_operator_summary_formatting(synthetic_tree_data):
    """Test plain-English operator summary formatting and domain terms."""
    model, X_df, _, feature_names = synthetic_tree_data
    explainer = MachSenseExplainer(model=model, feature_names=feature_names)

    instance = X_df.iloc[1]
    raw_vals = {"power_w": 2400.0, "torque_nm": 65.0}
    result = explainer.explain_instance(instance, original_unscaled_values=raw_vals)

    summary = result.operator_summary
    assert "Failure Probability:" in summary
    assert "Safety Note:" in summary
    assert "DISCLAIMER (NON-CAUSAL ATTRIBUTION)" in summary

    # Verify dict conversion
    summary_dict = result.to_dict()
    assert isinstance(summary_dict, dict)
    assert "predicted_probability" in summary_dict
    assert "all_attributions" in summary_dict


def test_plot_generation(synthetic_tree_data):
    """Test global and waterfall plot generator methods."""
    model, X_df, _, feature_names = synthetic_tree_data
    explainer = MachSenseExplainer(model=model, feature_names=feature_names)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        bar_path = tmp_path / "bar.png"
        waterfall_path = tmp_path / "waterfall.png"

        fig1 = explainer.plot_global_importance(X_df, save_path=bar_path)
        assert isinstance(fig1, plt.Figure)
        assert bar_path.exists()
        plt.close(fig1)

        fig2 = explainer.plot_waterfall(X_df.iloc[0], save_path=waterfall_path)
        assert isinstance(fig2, plt.Figure)
        assert waterfall_path.exists()
        plt.close(fig2)


def test_numpy_array_input(synthetic_tree_data):
    """Test explainer with raw 1D and 2D numpy arrays."""
    model, X_df, _, feature_names = synthetic_tree_data
    explainer = MachSenseExplainer(model=model, feature_names=feature_names)

    arr_1d = X_df.iloc[0].to_numpy()
    result_1d = explainer.explain_instance(arr_1d)
    assert isinstance(result_1d, LocalExplanationResult)

    arr_2d = X_df.to_numpy()
    shap_vals = explainer.compute_shap_values(arr_2d)
    assert shap_vals.shape == arr_2d.shape
