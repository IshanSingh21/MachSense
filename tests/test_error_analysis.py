"""Unit tests for MachSense error analysis, prediction slicing, and failure mode attribution."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from machsense.models.error_analysis import (
    ErrorAnalysisReport,
    FailureModeSliceResult,
    analyze_model_errors,
    categorize_predictions,
    compare_error_features,
    compute_failure_mode_slices,
    evaluate_threshold_grid,
)


@pytest.fixture
def sample_error_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, np.ndarray]:
    """Provide synthetic dataset with known failure modes and prediction probabilities."""
    n = 100
    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        {
            "speed": rng.normal(1500, 100, n),
            "torque": rng.normal(40, 10, n),
            "temp_diff": rng.normal(10, 2, n),
        }
    )
    # Synthetic failure modes
    hdf = np.zeros(n, dtype=int)
    hdf[:10] = 1  # 10 HDF events
    pwf = np.zeros(n, dtype=int)
    pwf[10:15] = 1  # 5 PWF events

    fm_df = pd.DataFrame({"hdf": hdf, "pwf": pwf, "twf": np.zeros(n, dtype=int), "osf": np.zeros(n, dtype=int), "rnf": np.zeros(n, dtype=int)})
    y_true = pd.Series((hdf | pwf), name="machine_failure")

    # Probabilities: High for HDF (caught), Low for PWF (missed)
    y_prob = np.zeros(n)
    y_prob[:8] = 0.90   # 8 HDF caught
    y_prob[8:10] = 0.20 # 2 HDF missed
    y_prob[10:15] = 0.10 # 5 PWF missed
    y_prob[50] = 0.85   # 1 False Alarm (FP)

    return X, y_true, fm_df, y_prob


def test_categorize_predictions(sample_error_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, np.ndarray]):
    """Verify categorize_predictions accurately separates TP, FP, TN, and FN masks."""
    _, y_true, _, y_prob = sample_error_data
    tp_mask, fp_mask, tn_mask, fn_mask = categorize_predictions(y_true, y_prob, threshold=0.5)

    assert tp_mask.sum() == 8   # 8 True Positives
    assert fp_mask.sum() == 1   # 1 False Positive
    assert fn_mask.sum() == 7   # 7 False Negatives (2 HDF + 5 PWF)
    assert tn_mask.sum() == 84  # 84 True Negatives
    assert tp_mask.sum() + fp_mask.sum() + tn_mask.sum() + fn_mask.sum() == len(y_true)


def test_compute_failure_mode_slices(sample_error_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, np.ndarray]):
    """Verify slice evaluation computes exact catch rates per failure mode."""
    _, y_true, fm_df, y_prob = sample_error_data
    slices = compute_failure_mode_slices(fm_df, y_true, y_prob, threshold=0.5)

    assert "hdf" in slices
    assert "pwf" in slices

    hdf_slice = slices["hdf"]
    assert isinstance(hdf_slice, FailureModeSliceResult)
    assert hdf_slice.total_occurrences == 10
    assert hdf_slice.detected_count == 8
    assert hdf_slice.missed_count == 2
    assert hdf_slice.recall_rate == 0.80

    pwf_slice = slices["pwf"]
    assert pwf_slice.total_occurrences == 5
    assert pwf_slice.detected_count == 0
    assert pwf_slice.missed_count == 5
    assert pwf_slice.recall_rate == 0.0


def test_compare_error_features(sample_error_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, np.ndarray]):
    """Verify feature comparison between TP and FN sets."""
    X, y_true, _, y_prob = sample_error_data
    tp_mask, _, _, fn_mask = categorize_predictions(y_true, y_prob, threshold=0.5)

    comp = compare_error_features(X, tp_mask, fn_mask)
    assert isinstance(comp, pd.DataFrame)
    assert "TP_Mean" in comp.columns
    assert "FN_Mean" in comp.columns
    assert "Difference" in comp.columns
    assert len(comp) == X.shape[1]


def test_evaluate_threshold_grid(sample_error_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, np.ndarray]):
    """Verify threshold grid evaluation computes valid metrics across range."""
    _, y_true, _, y_prob = sample_error_data
    grid = evaluate_threshold_grid(y_true, y_prob, thresholds=[0.2, 0.5, 0.8])

    assert isinstance(grid, pd.DataFrame)
    assert len(grid) == 3
    assert "precision" in grid.columns
    assert "recall" in grid.columns
    assert "f1_score" in grid.columns
    assert "false_negatives" in grid.columns


def test_analyze_model_errors_end_to_end(sample_error_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, np.ndarray]):
    """Verify full end-to-end diagnostic report generation."""
    X, y_true, fm_df, _ = sample_error_data
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y_true)

    report = analyze_model_errors(
        model=model,
        X_trans=X,
        y_true=y_true,
        failure_modes_df=fm_df,
        threshold=0.5,
        split_name="test",
    )

    assert isinstance(report, ErrorAnalysisReport)
    assert report.total_samples == len(y_true)
    assert len(report.untrusted_operating_conditions) > 0
    assert isinstance(report.summary(), str)
