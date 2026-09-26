"""Evaluation metrics computation, confusion matrix analysis, and threshold optimization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from machsense.utils.logger import get_logger

logger = get_logger("machsense.models.evaluator")


@dataclass
class ModelEvaluationResult:
    """Structured container holding comprehensive classification metrics."""
    model_name: str
    split_name: str
    accuracy: float
    precision: float
    recall: float
    f1_minority: float
    f1_macro: float
    pr_auc: float
    roc_auc: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    false_negative_rate: float
    optimal_threshold: float
    f1_at_optimal_threshold: float
    recall_at_optimal_threshold: float
    additional_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation result to dictionary for JSON/DataFrame persistence."""
        return asdict(self)

    def summary(self) -> str:
        """Render clean readable performance summary."""
        return (
            f"[{self.model_name.upper()} on {self.split_name.upper()}]\n"
            f"  - PR-AUC (Average Precision):  {self.pr_auc:.4f}  <-- Primary Metric\n"
            f"  - Recall (Failure Detection):   {self.recall:.4f} ({self.true_positives}/{self.true_positives + self.false_negatives} caught)\n"
            f"  - Precision (Alert Accuracy):   {self.precision:.4f}\n"
            f"  - F1-Score (Minority Class):    {self.f1_minority:.4f}\n"
            f"  - ROC-AUC:                      {self.roc_auc:.4f}\n"
            f"  - Accuracy:                     {self.accuracy:.4f} (Caution: Deceptive under 28.5:1 Imbalance)\n"
            f"  - Confusion Matrix:             TP={self.true_positives}, FP={self.false_positives}, "
            f"TN={self.true_negatives}, FN={self.false_negatives} (FN Rate: {self.false_negative_rate:.2%})\n"
            f"  - Optimal Threshold:            {self.optimal_threshold:.2f} (F1: {self.f1_at_optimal_threshold:.4f}, Recall: {self.recall_at_optimal_threshold:.4f})"
        )


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, float]:
    """Find classification threshold that maximizes F1-Score on the failure class.

    Returns:
        Tuple of (optimal_threshold, max_f1, recall_at_optimal_threshold).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # Calculate F1 score for all threshold points
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)

    if best_idx < len(thresholds):
        optimal_thresh = float(thresholds[best_idx])
    else:
        optimal_thresh = 0.5

    best_f1 = float(f1_scores[best_idx])
    best_recall = float(recalls[best_idx])

    return optimal_thresh, best_f1, best_recall


def evaluate_classifier(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y_true: pd.Series | np.ndarray,
    model_name: str = "Classifier",
    split_name: str = "val",
    threshold: float = 0.5,
) -> ModelEvaluationResult:
    """Perform rigorous multi-metric evaluation of a classifier.

    Args:
        model: Trained scikit-learn compatible classifier.
        X: Feature matrix.
        y_true: Ground truth binary targets.
        model_name: Name identifier for the model.
        split_name: Name of data split ('train', 'val', 'test').
        threshold: Standard decision threshold for binary prediction.

    Returns:
        ModelEvaluationResult dataclass with all metrics.
    """
    y_arr = np.asarray(y_true)

    # 1. Predictions and Probabilities
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        # Convert decision function to pseudo-probabilities via sigmoid
        df_vals = model.decision_function(X)
        y_prob = 1.0 / (1.0 + np.exp(-df_vals))
    else:
        # Dummy or hard prediction model
        y_pred_direct = model.predict(X)
        y_prob = np.asarray(y_pred_direct, dtype=float)

    y_pred = (y_prob >= threshold).astype(int)

    # 2. Confusion Matrix Calculation
    cm = confusion_matrix(y_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # 3. Core Metrics
    acc = float(accuracy_score(y_arr, y_pred))
    prec = float(precision_score(y_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_arr, y_pred, zero_division=0))
    f1_min = float(f1_score(y_arr, y_pred, pos_label=1, zero_division=0))
    f1_mac = float(f1_score(y_arr, y_pred, average="macro", zero_division=0))

    # 4. Probabilistic Metrics (PR-AUC & ROC-AUC)
    try:
        pr_auc = float(average_precision_score(y_arr, y_prob))
    except Exception:
        pr_auc = 0.0

    try:
        roc_auc = float(roc_auc_score(y_arr, y_prob))
    except Exception:
        roc_auc = 0.5

    # 5. False Negative Metrics
    total_positives = tp + fn
    fn_rate = float(fn / max(1, total_positives))

    # 6. Optimal Threshold Sweep
    opt_thresh, opt_f1, opt_rec = find_optimal_threshold(y_arr, y_prob)

    result = ModelEvaluationResult(
        model_name=model_name,
        split_name=split_name,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1_minority=f1_min,
        f1_macro=f1_mac,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        false_negative_rate=fn_rate,
        optimal_threshold=opt_thresh,
        f1_at_optimal_threshold=opt_f1,
        recall_at_optimal_threshold=opt_rec,
    )

    logger.info("Evaluation complete for %s on %s:\n%s", model_name, split_name, result.summary())
    return result
