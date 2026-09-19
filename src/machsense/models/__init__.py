"""Predictive modeling, evaluation, tuning, and registry package for MachSense."""

from machsense.models.evaluator import (
    ModelEvaluationResult,
    evaluate_classifier,
    find_optimal_threshold,
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
]
