"""Predictive modeling, evaluation, tuning, registry, and error analysis package for MachSense."""

from machsense.models.error_analysis import (
    ErrorAnalysisReport,
    FailureModeSliceResult,
    analyze_model_errors,
    categorize_predictions,
    compute_failure_mode_slices,
    evaluate_threshold_grid,
)
from machsense.models.evaluator import (
    ModelEvaluationResult,
    evaluate_classifier,
    find_optimal_threshold,
)
from machsense.models.explainability import (
    FEATURE_DISPLAY_NAMES,
    NON_CAUSAL_DISCLAIMER,
    FeatureAttribution,
    LocalExplanationResult,
    MachSenseExplainer,
)
from machsense.models.registry import (
    ModelMetadata,
    ModelRegistry,
)
from machsense.models.trainer import (
    get_baseline_models,
    train_and_evaluate_models,
)
from machsense.models.tuner import (
    get_tuning_search_spaces,
    run_model_optimization,
    tune_model,
)

__all__ = [
    "ModelEvaluationResult",
    "evaluate_classifier",
    "find_optimal_threshold",
    "get_baseline_models",
    "train_and_evaluate_models",
    "ModelMetadata",
    "ModelRegistry",
    "get_tuning_search_spaces",
    "tune_model",
    "run_model_optimization",
    "ErrorAnalysisReport",
    "FailureModeSliceResult",
    "analyze_model_errors",
    "categorize_predictions",
    "compute_failure_mode_slices",
    "evaluate_threshold_grid",
    "MachSenseExplainer",
    "FeatureAttribution",
    "LocalExplanationResult",
    "FEATURE_DISPLAY_NAMES",
    "NON_CAUSAL_DISCLAIMER",
]
