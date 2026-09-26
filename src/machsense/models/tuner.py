"""Hyperparameter tuning and model optimization engine for MachSense."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from machsense.config.settings import get_settings
from machsense.features.pipeline import run_feature_pipeline
from machsense.features.preprocessor import MachSensePreprocessor
from machsense.models.evaluator import (
    ModelEvaluationResult,
    evaluate_classifier,
)
from machsense.models.registry import ModelMetadata, ModelRegistry
from machsense.utils.logger import get_logger, setup_logging

logger = get_logger("machsense.models.tuner")


def get_tuning_search_spaces() -> Dict[str, Tuple[Any, Dict[str, List[Any]]]]:
    """Define candidate model architectures and parameter grids for tuning."""
    return {
        "tuned_random_forest": (
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [100, 150],
                "max_depth": [10, 16],
                "min_samples_split": [2, 5],
                "class_weight": ["balanced", "balanced_subsample"],
            },
        ),
        "tuned_hist_gradient_boosting": (
            HistGradientBoostingClassifier(random_state=42),
            {
                "max_iter": [100, 150],
                "learning_rate": [0.05, 0.1],
                "max_leaf_nodes": [31, 63],
                "class_weight": ["balanced", None],
            },
        ),
    }


def tune_model(
    model_name: str,
    base_estimator: Any,
    param_grid: Dict[str, List[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
    random_state: int = 42,
) -> Tuple[Any, Dict[str, Any], float]:
    """Execute Stratified K-Fold Cross-Validation hyperparameter search strictly on training data."""
    logger.info("Starting hyperparameter tuning for %s (CV: %d folds)...", model_name, cv_splits)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=base_estimator,
        param_grid=param_grid,
        cv=cv,
        scoring="average_precision",  # Primary imbalanced metric: PR-AUC
        n_jobs=1,
        refit=True,
        verbose=0,
    )

    grid_search.fit(X_train, y_train)

    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = float(grid_search.best_score_)

    logger.info(
        "Tuning complete for %s | Best 5-Fold CV PR-AUC: %.4f | Best Params: %s",
        model_name,
        best_score,
        best_params,
    )
    return best_estimator, best_params, best_score


def run_model_optimization(
    version: str = "v1.0.0",
    cv_splits: int = 5,
    save_artifacts: bool = True,
) -> Tuple[Any, ModelMetadata, pd.DataFrame]:
    """Execute complete hyperparameter tuning, validation, threshold calibration, and registration.

    Workflow:
    1. Load preprocessed feature matrices (X_train, X_val, X_test).
    2. Perform 5-fold Stratified CV hyperparameter tuning strictly on X_train.
    3. Evaluate tuned candidates on Validation set for threshold calibration.
    4. Select champion model and perform final unbiased test evaluation.
    5. Save versioned artifacts and experiment records via ModelRegistry.

    Args:
        version: Version tag for artifact registration (e.g., 'v1.0.0').
        cv_splits: Number of stratified cross-validation folds.
        save_artifacts: Whether to persist artifacts to disk.

    Returns:
        Tuple of (Champion Model, ModelMetadata, Comparison DataFrame).
    """
    setup_logging()
    settings = get_settings()
    logger.info("=========================================================")
    logger.info("Starting MachSense Model Optimization Pipeline (Day 6)")
    logger.info("Target Version: %s | CV Strategy: Stratified %d-Fold", version, cv_splits)
    logger.info("=========================================================")

    processed_dir = settings.resolve_path("processed_data_dir")
    train_trans_file = processed_dir / "X_train_transformed.csv"

    if not train_trans_file.exists():
        logger.info("Transformed features missing. Running feature pipeline...")
        run_feature_pipeline(save_artifacts=True)

    # 1. Load partitions
    X_train = pd.read_csv(processed_dir / "X_train_transformed.csv")
    X_val = pd.read_csv(processed_dir / "X_val_transformed.csv")
    X_test = pd.read_csv(processed_dir / "X_test_transformed.csv")

    y_train = pd.read_csv(processed_dir / "y_train.csv")["machine_failure"]
    y_val = pd.read_csv(processed_dir / "y_val.csv")["machine_failure"]
    y_test = pd.read_csv(processed_dir / "y_test.csv")["machine_failure"]

    search_spaces = get_tuning_search_spaces()
    tuning_records: Dict[str, Any] = {}
    comparison_rows: List[Dict[str, Any]] = []

    best_val_pr_auc = -1.0
    champion_model: Optional[Any] = None
    champion_name = ""
    champion_params: Dict[str, Any] = {}
    champion_cv_score = 0.0
    champion_val_res: Optional[ModelEvaluationResult] = None
    champion_test_res: Optional[ModelEvaluationResult] = None

    # 2. Tune each candidate architecture
    for name, (estimator, param_grid) in search_spaces.items():
        best_est, best_params, cv_score = tune_model(
            model_name=name,
            base_estimator=estimator,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
            cv_splits=cv_splits,
            random_state=42,
        )

        # Evaluate on validation split
        val_res = evaluate_classifier(best_est, X_val, y_val, model_name=name, split_name="val")
        # Evaluate on test split
        test_res = evaluate_classifier(best_est, X_test, y_test, model_name=name, split_name="test")

        tuning_records[name] = {
            "best_params": best_params,
            "cv_mean_pr_auc": cv_score,
            "validation": val_res.to_dict(),
            "test": test_res.to_dict(),
        }

        row = {
            "model_name": name,
            "cv_pr_auc": cv_score,
            "val_pr_auc": val_res.pr_auc,
            "val_recall": val_res.recall,
            "val_precision": val_res.precision,
            "val_f1": val_res.f1_minority,
            "val_fn": val_res.false_negatives,
            "test_pr_auc": test_res.pr_auc,
            "test_recall": test_res.recall,
            "test_f1": test_res.f1_minority,
            "test_fn": test_res.false_negatives,
            "optimal_threshold": val_res.optimal_threshold,
        }
        comparison_rows.append(row)

        # Select candidate with highest validation PR-AUC
        if val_res.pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_res.pr_auc
            champion_model = best_est
            champion_name = name
            champion_params = best_params
            champion_cv_score = cv_score
            champion_val_res = val_res
            champion_test_res = test_res

    comparison_df = pd.DataFrame(comparison_rows).sort_values(by="val_pr_auc", ascending=False).reset_index(drop=True)
    logger.info("=========================================================")
    logger.info("Selected Champion Model: %s (Val PR-AUC: %.4f | Test PR-AUC: %.4f)", champion_name, champion_val_res.pr_auc, champion_test_res.pr_auc)
    logger.info("=========================================================")

    # 3. Create Model Metadata
    metadata = ModelMetadata(
        model_version=version,
        model_name=champion_name,
        model_type=type(champion_model).__name__,
        created_at=datetime.now(timezone.utc).isoformat(),
        hyperparameters=champion_params,
        feature_names=list(X_train.columns),
        optimal_threshold=champion_val_res.optimal_threshold,
        cv_mean_pr_auc=champion_cv_score,
        validation_metrics=champion_val_res.to_dict(),
        test_metrics=champion_test_res.to_dict(),
        description=f"Tuned {champion_name} for predictive maintenance failure classification.",
    )

    # 4. Save Versioned Artifacts
    if save_artifacts:
        models_dir = settings.resolve_path("models_dir")
        # Save tuning experiments log
        exp_file = models_dir / "tuning_experiments.json"
        with open(exp_file, "w", encoding="utf-8") as f:
            json.dump(tuning_records, f, indent=2)
        logger.info("Saved tuning experiments to: %s", exp_file)

        # Register versioned artifacts
        preprocessor = MachSensePreprocessor.load()
        registry = ModelRegistry(models_dir=models_dir)
        registry.save_versioned_artifacts(
            model=champion_model,
            preprocessor=preprocessor,
            metadata=metadata,
            version=version,
            set_as_active=True,
        )

        # Verify integrity of newly saved artifacts
        registry.verify_artifacts_integrity(version=version)

    return champion_model, metadata, comparison_df


if __name__ == "__main__":
    run_model_optimization()
