"""Multi-model training orchestrator and experiment tracking engine for MachSense."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from machsense.config.settings import get_settings
from machsense.features.pipeline import run_feature_pipeline
from machsense.models.evaluator import ModelEvaluationResult, evaluate_classifier
from machsense.utils.logger import get_logger, setup_logging

logger = get_logger("machsense.models.trainer")


def get_baseline_models(random_state: int = 42) -> Dict[str, Any]:
    """Instantiate diverse classification model families with class balancing."""
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced",
            max_iter=100,
            learning_rate=0.1,
            random_state=random_state,
        ),
    }


def train_and_evaluate_models(
    random_state: int = 42,
    save_artifacts: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any], Any]:
    """Train all candidate models, evaluate on val and test splits, and track experiments.

    Args:
        random_state: Seed for deterministic model initialization.
        save_artifacts: Whether to save experiment results and best model to disk.

    Returns:
        Tuple of (Comparison DataFrame, Results Dict, Best Model instance).
    """
    setup_logging()
    settings = get_settings()
    logger.info("=========================================================")
    logger.info("Starting MachSense Multi-Model Baseline Benchmark")
    logger.info("=========================================================")

    processed_dir = settings.resolve_path("processed_data_dir")
    train_trans_file = processed_dir / "X_train_transformed.csv"

    # Ensure preprocessed matrices exist
    if not train_trans_file.exists():
        logger.info("Transformed feature matrices missing. Running feature pipeline...")
        run_feature_pipeline(save_artifacts=True)

    # 1. Load transformed feature matrices and targets
    logger.info("Loading training, validation, and test partitions...")
    X_train = pd.read_csv(processed_dir / "X_train_transformed.csv")
    X_val = pd.read_csv(processed_dir / "X_val_transformed.csv")
    X_test = pd.read_csv(processed_dir / "X_test_transformed.csv")

    y_train = pd.read_csv(processed_dir / "y_train.csv")["machine_failure"]
    y_val = pd.read_csv(processed_dir / "y_val.csv")["machine_failure"]
    y_test = pd.read_csv(processed_dir / "y_test.csv")["machine_failure"]

    models = get_baseline_models(random_state=random_state)
    trained_models: Dict[str, Any] = {}
    val_results: List[ModelEvaluationResult] = []
    test_results: List[ModelEvaluationResult] = []
    all_experiments_data: Dict[str, Any] = {}

    # 2. Train and Evaluate each model
    for model_name, model in models.items():
        logger.info("---------------------------------------------------------")
        logger.info("Training candidate model: %s ...", model_name)
        model.fit(X_train, y_train)
        trained_models[model_name] = model

        # Evaluate on validation split
        val_res = evaluate_classifier(model, X_val, y_val, model_name=model_name, split_name="val")
        val_results.append(val_res)

        # Evaluate on test split
        test_res = evaluate_classifier(model, X_test, y_test, model_name=model_name, split_name="test")
        test_results.append(test_res)

        all_experiments_data[model_name] = {
            "validation": val_res.to_dict(),
            "test": test_res.to_dict(),
        }

    # 3. Create Leaderboard Comparison Table
    val_df = pd.DataFrame([r.to_dict() for r in val_results])
    val_df = val_df.sort_values(by="pr_auc", ascending=False).reset_index(drop=True)

    # Select Champion Model based on highest PR-AUC on validation split
    best_model_name = val_df.iloc[0]["model_name"]
    best_model = trained_models[best_model_name]
    logger.info("=========================================================")
    logger.info("Champion Model Selected: %s (Val PR-AUC: %.4f)", best_model_name, val_df.iloc[0]["pr_auc"])
    logger.info("=========================================================")

    # 4. Save Experiment Results and Champion Artifacts
    if save_artifacts:
        models_dir = settings.resolve_path("models_dir")
        models_dir.mkdir(parents=True, exist_ok=True)

        # Save experiment JSON
        json_path = models_dir / "experiment_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_experiments_data, f, indent=2)
        logger.info("Saved experiment results to: %s", json_path)

        # Save comparison CSV
        csv_path = models_dir / "model_comparison.csv"
        val_df.to_csv(csv_path, index=False)
        logger.info("Saved model comparison table to: %s", csv_path)

        # Save best model joblib
        best_model_path = models_dir / "best_model.joblib"
        joblib.dump(best_model, best_model_path)
        logger.info("Saved champion model to: %s", best_model_path)

    return val_df, all_experiments_data, best_model


if __name__ == "__main__":
    train_and_evaluate_models()
