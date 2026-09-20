"""Diagnostic error analysis, prediction slicing, and failure mode attribution engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from machsense.config.settings import get_settings
from machsense.data.schema import FAILURE_MODE_COLUMNS
from machsense.models.evaluator import evaluate_classifier
from machsense.models.registry import ModelRegistry
from machsense.utils.logger import get_logger, setup_logging

logger = get_logger("machsense.models.error_analysis")


@dataclass
class FailureModeSliceResult:
    """Detection metrics for a specific failure mechanism (e.g. HDF, PWF, OSF, TWF, RNF)."""
    failure_mode: str
    total_occurrences: int
    detected_count: int
    missed_count: int
    recall_rate: float
    mean_predicted_prob: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ErrorAnalysisReport:
    """Comprehensive diagnostic breakdown of model error cases and limitations."""
    split_name: str
    total_samples: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    failure_mode_slices: Dict[str, FailureModeSliceResult]
    fn_feature_comparison: pd.DataFrame
    threshold_grid: pd.DataFrame
    untrusted_operating_conditions: List[str]

    def summary(self) -> str:
        """Render human-readable diagnostic error summary."""
        lines = [
            f"=== Error Analysis Diagnostic Report [{self.split_name.upper()}] ===",
            f"Total Samples: {self.total_samples:,} | Confusion: TP={self.true_positives}, FP={self.false_positives}, TN={self.true_negatives}, FN={self.false_negatives}",
            "\n--- Performance Slice by Specific Failure Mode ---",
        ]
        for fm, slice_res in self.failure_mode_slices.items():
            lines.append(
                f"  - {fm.upper():<5}: {slice_res.detected_count:>2}/{slice_res.total_occurrences:>2} caught "
                f"({slice_res.recall_rate:>6.1%}) | Missed (FN): {slice_res.missed_count:>2} | Avg Prob: {slice_res.mean_predicted_prob:.3f}"
            )

        lines.append("\n--- Untrusted Machine Operating Conditions ---")
        for cond in self.untrusted_operating_conditions:
            lines.append(f"  [!] {cond}")

        return "\n".join(lines)


def categorize_predictions(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Categorize prediction indices into TP, FP, TN, and FN boolean masks."""
    y_arr = np.asarray(y_true)
    y_pred = (y_prob >= threshold).astype(int)

    tp_mask = (y_arr == 1) & (y_pred == 1)
    fp_mask = (y_arr == 0) & (y_pred == 1)
    tn_mask = (y_arr == 0) & (y_pred == 0)
    fn_mask = (y_arr == 1) & (y_pred == 0)

    return tp_mask, fp_mask, tn_mask, fn_mask


def compute_failure_mode_slices(
    failure_modes_df: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, FailureModeSliceResult]:
    """Compute detection rates for each individual physical failure mode."""
    slices = {}
    y_pred = (y_prob >= threshold).astype(int)

    for col in FAILURE_MODE_COLUMNS:
        if col in failure_modes_df.columns:
            fm_mask = failure_modes_df[col].values == 1
            total_mode_events = int(fm_mask.sum())

            if total_mode_events > 0:
                detected = int(((y_pred == 1) & fm_mask).sum())
                missed = total_mode_events - detected
                recall = float(detected / total_mode_events)
                avg_prob = float(y_prob[fm_mask].mean()) if total_mode_events > 0 else 0.0

                slices[col] = FailureModeSliceResult(
                    failure_mode=col,
                    total_occurrences=total_mode_events,
                    detected_count=detected,
                    missed_count=missed,
                    recall_rate=recall,
                    mean_predicted_prob=avg_prob,
                )
    return slices


def compare_error_features(
    X_df: pd.DataFrame,
    tp_mask: np.ndarray,
    fn_mask: np.ndarray,
) -> pd.DataFrame:
    """Compare mean and standard deviation of features between detected (TP) and missed (FN) failures."""
    comparison = pd.DataFrame(index=X_df.columns)

    if tp_mask.sum() > 0:
        comparison["TP_Mean"] = X_df[tp_mask].mean()
        comparison["TP_Std"] = X_df[tp_mask].std()
    else:
        comparison["TP_Mean"] = np.nan
        comparison["TP_Std"] = np.nan

    if fn_mask.sum() > 0:
        comparison["FN_Mean"] = X_df[fn_mask].mean()
        comparison["FN_Std"] = X_df[fn_mask].std()
    else:
        comparison["FN_Mean"] = np.nan
        comparison["FN_Std"] = np.nan

    comparison["Difference"] = comparison["TP_Mean"] - comparison["FN_Mean"]
    return comparison


def evaluate_threshold_grid(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Evaluate precision, recall, F1, and false negatives across an operational threshold grid."""
    if thresholds is None:
        thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.57, 0.70, 0.80, 0.90]

    y_arr = np.asarray(y_true)
    records = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_arr, preds, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * (prec * rec) / (prec + rec + 1e-10))

        records.append(
            {
                "threshold": t,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "fn_rate": round(fn / max(1, tp + fn), 4),
            }
        )
    return pd.DataFrame(records)


def analyze_model_errors(
    model: Any,
    X_trans: pd.DataFrame,
    y_true: pd.Series,
    failure_modes_df: pd.DataFrame,
    threshold: float = 0.5,
    split_name: str = "test",
) -> ErrorAnalysisReport:
    """Execute end-to-end diagnostic error analysis on a given split."""
    y_prob = model.predict_proba(X_trans)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_trans)

    # 1. Categorize predictions
    tp_mask, fp_mask, tn_mask, fn_mask = categorize_predictions(y_true, y_prob, threshold=threshold)

    # 2. Slice by failure mode
    slices = compute_failure_mode_slices(failure_modes_df, y_true, y_prob, threshold=threshold)

    # 3. Feature comparison
    feat_comp = compare_error_features(X_trans, tp_mask, fn_mask)

    # 4. Threshold grid
    thresh_grid = evaluate_threshold_grid(y_true, y_prob)

    # 5. Define documented untrusted operating conditions
    untrusted_conditions = [
        "Random Failures (RNF): Hardware faults occurring from stochastic background noise (mean catch rate ~0-15%). Telemetry shows zero thermal or mechanical degradation prior to failure.",
        "Borderline Tool Wear (190 - 205 min): Machines near the 200 min threshold exhibit high ambiguity between normal wear and impending tool breakage.",
        "Out-of-Range Ambient Temperatures: Predictions with Air Temp < 270 K or > 350 K violate training distribution and should be flagged for sensor inspection.",
        "Missing/Corrupted Telemetry Packets: If rotational speed drops to 0 RPM while torque > 0 Nm without an active stop signal, predictions are invalid.",
    ]

    report = ErrorAnalysisReport(
        split_name=split_name,
        total_samples=len(y_true),
        true_positives=int(tp_mask.sum()),
        false_positives=int(fp_mask.sum()),
        true_negatives=int(tn_mask.sum()),
        false_negatives=int(fn_mask.sum()),
        failure_mode_slices=slices,
        fn_feature_comparison=feat_comp,
        threshold_grid=thresh_grid,
        untrusted_operating_conditions=untrusted_conditions,
    )

    logger.info("Error analysis complete:\n%s", report.summary())
    return report


def run_error_analysis_cli() -> None:
    """CLI execution for model error analysis."""
    setup_logging()
    settings = get_settings()
    processed_dir = settings.resolve_path("processed_data_dir")

    registry = ModelRegistry()
    model, _, meta = registry.load_versioned_artifacts()

    X_test = pd.read_csv(processed_dir / "X_test_transformed.csv")
    y_test = pd.read_csv(processed_dir / "y_test.csv")["machine_failure"]
    fm_test = pd.read_csv(processed_dir / "failure_modes_test.csv")

    threshold = meta.get("optimal_threshold", 0.5)
    report = analyze_model_errors(
        model=model,
        X_trans=X_test,
        y_true=y_test,
        failure_modes_df=fm_test,
        threshold=threshold,
        split_name="test",
    )
    print(report.summary())
    print("\n=== Threshold Sensitivity Matrix ===")
    print(report.threshold_grid.to_string(index=False))


if __name__ == "__main__":
    run_error_analysis_cli()
