"""Explainable AI (XAI) module for MachSense.

Provides SHAP (SHapley Additive exPlanations) attribution analysis for tree-based
predictive maintenance models, including:
1. Global feature importance calculation (mean absolute SHAP).
2. Local instance attribution decomposing failure probability into risk escalators and stabilizers.
3. Plain-English operator summaries for shop-floor maintenance engineers.
4. Non-causal safety notices distinguishing statistical attribution from physical causation.
5. Plot generation utilities for waterfall and summary bar/beeswarm charts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from machsense.utils.logger import get_logger

logger = get_logger(__name__)

# Human-readable domain descriptions for sensor and engineered features
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "type_L": "Product Variant: Low Quality (L)",
    "type_M": "Product Variant: Medium Quality (M)",
    "type_H": "Product Variant: High Quality (H)",
    "air_temperature_k": "Air Temperature [K]",
    "process_temperature_k": "Process Temperature [K]",
    "rotational_speed_rpm": "Spindle Rotational Speed [rpm]",
    "torque_nm": "Mechanical Torque [Nm]",
    "tool_wear_min": "Tool Wear Accumulated [min]",
    "power_w": "Electric/Mechanical Power [W]",
    "temp_difference_k": "Process-Air Temp Difference [K]",
    "temp_ratio": "Process/Air Temp Ratio",
    "overstrain_index": "Mechanical Overstrain Index [Nm * min]",
    "tool_wear_rate": "Tool Wear Rate [min / rpm]",
}

NON_CAUSAL_DISCLAIMER: str = (
    "DISCLAIMER (NON-CAUSAL ATTRIBUTION): SHAP values reflect the mathematical "
    "contribution of each feature to the model's output probability under the learned decision trees. "
    "Attribution indicates correlation and predictive association within the historical training distribution, "
    "not necessarily a direct physical root cause. Always verify physical machinery before taking destructive action."
)


@dataclass
class FeatureAttribution:
    """Individual feature contribution to a prediction."""

    feature: str
    display_name: str
    feature_value: float
    shap_value: float
    contribution_type: str  # "escalator", "stabilizer", "neutral"
    percent_contribution: float = 0.0


@dataclass
class LocalExplanationResult:
    """Structured local explanation for a single telemetry instance."""

    predicted_probability: float
    predicted_class: int
    base_value: float
    risk_level: str  # "NOMINAL", "ELEVATED", "CRITICAL"
    top_risk_escalators: list[FeatureAttribution] = field(default_factory=list)
    top_stabilizers: list[FeatureAttribution] = field(default_factory=list)
    all_attributions: list[FeatureAttribution] = field(default_factory=list)
    operator_summary: str = ""
    non_causal_disclaimer: str = NON_CAUSAL_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        """Serialize explanation to dictionary."""
        return asdict(self)


class MachSenseExplainer:
    """SHAP-based explainability engine for MachSense machine failure classification models.

    Computes mathematically rigorous Shapley additive attributions using tree path traversal.
    """

    def __init__(
        self,
        model: Any,
        feature_names: Optional[list[str]] = None,
        background_data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ) -> None:
        """Initialize MachSenseExplainer.

        Args:
            model: Trained tree-based classification model (e.g. RandomForestClassifier).
            feature_names: Optional list of feature names.
            background_data: Optional background dataset for TreeExplainer interventional conditioning.
        """
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data

        logger.info("Initializing SHAP TreeExplainer for MachSense model...")
        if background_data is not None:
            self.explainer = shap.TreeExplainer(model, data=background_data)
        else:
            self.explainer = shap.TreeExplainer(model)

        # Determine expected value (base value) for failure class (class 1)
        if isinstance(self.explainer.expected_value, (list, np.ndarray)):
            if len(self.explainer.expected_value) == 2:
                self.expected_value: float = float(self.explainer.expected_value[1])
            else:
                self.expected_value = float(self.explainer.expected_value[0])
        else:
            self.expected_value = float(self.explainer.expected_value)

        logger.info(f"TreeExplainer initialized with base expected value: {self.expected_value:.4f}")

    def _prepare_data(
        self, X: Union[pd.DataFrame, np.ndarray, pd.Series, list[float]]
    ) -> Tuple[pd.DataFrame, list[str]]:
        """Normalize input into DataFrame and retrieve feature names."""
        if isinstance(X, pd.Series):
            df = X.to_frame().T
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                df = pd.DataFrame(X.reshape(1, -1))
            else:
                df = pd.DataFrame(X)
            if self.feature_names is not None and len(self.feature_names) == df.shape[1]:
                df.columns = self.feature_names
        elif isinstance(X, list):
            arr = np.array(X)
            if arr.ndim == 1:
                df = pd.DataFrame(arr.reshape(1, -1))
            else:
                df = pd.DataFrame(arr)
            if self.feature_names is not None and len(self.feature_names) == df.shape[1]:
                df.columns = self.feature_names
        else:
            raise TypeError(f"Unsupported data type for explainability: {type(X)}")

        feature_names = list(df.columns)
        if self.feature_names is None:
            self.feature_names = feature_names

        return df, feature_names

    def get_shap_explanation(self, X: Union[pd.DataFrame, np.ndarray]) -> shap.Explanation:
        """Compute SHAP Explanation object for positive failure class (class 1).

        Args:
            X: Input dataset.

        Returns:
            shap.Explanation object corresponding to class 1.
        """
        df, feature_names = self._prepare_data(X)
        exp = self.explainer(df)

        # If multi-class output (samples, features, classes), slice class 1
        if len(exp.shape) == 3 and exp.shape[2] == 2:
            return exp[:, :, 1]
        return exp

    def compute_shap_values(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Compute raw SHAP values matrix for positive failure class.

        Args:
            X: Input dataset of shape (N, n_features).

        Returns:
            np.ndarray of shape (N, n_features).
        """
        exp = self.get_shap_explanation(X)
        return np.asarray(exp.values)

    def compute_global_importance(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> pd.DataFrame:
        """Compute global feature importance using mean absolute SHAP values.

        Formula: I_j = (1/N) * sum_i |phi_{i,j}|

        Args:
            X: Feature matrix.

        Returns:
            DataFrame with columns ['feature', 'display_name', 'mean_abs_shap', 'relative_importance', 'rank']
        """
        df, feature_names = self._prepare_data(X)
        shap_values = self.compute_shap_values(df)

        mean_abs = np.mean(np.abs(shap_values), axis=0)
        total_abs = np.sum(mean_abs)
        rel_importance = (mean_abs / total_abs) if total_abs > 0 else np.zeros_like(mean_abs)

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "display_name": [FEATURE_DISPLAY_NAMES.get(f, f) for f in feature_names],
            "mean_abs_shap": mean_abs,
            "relative_importance": rel_importance,
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

        importance_df["rank"] = np.arange(1, len(importance_df) + 1)
        return importance_df

    def explain_instance(
        self,
        instance: Union[pd.Series, pd.DataFrame, np.ndarray, list[float]],
        threshold: float = 0.5,
        top_k: int = 5,
        original_unscaled_values: Optional[dict[str, float]] = None,
    ) -> LocalExplanationResult:
        """Generate local attribution and plain-English breakdown for a single instance.

        Args:
            instance: 1D or 1-row feature vector.
            threshold: Decision threshold for failure classification.
            top_k: Number of top risk escalators and stabilizers to highlight.
            original_unscaled_values: Optional dict of raw physical sensor values for human-readable report.

        Returns:
            LocalExplanationResult object with complete diagnostics.
        """
        df, feature_names = self._prepare_data(instance)
        if len(df) != 1:
            raise ValueError(f"explain_instance requires a single instance, got {len(df)} rows.")

        # Compute predicted probability and class
        if hasattr(self.model, "predict_proba"):
            prob_arr = self.model.predict_proba(df)
            prob = float(prob_arr[0, 1]) if prob_arr.shape[1] > 1 else float(prob_arr[0, 0])
        elif hasattr(self.model, "predict"):
            prob = float(self.model.predict(df)[0])
        else:
            prob = self.expected_value

        pred_class = 1 if prob >= threshold else 0

        # Compute SHAP values for this instance
        exp = self.get_shap_explanation(df)
        shap_vals = np.asarray(exp.values[0])
        base_val = float(exp.base_values[0]) if hasattr(exp, "base_values") else self.expected_value

        # Build feature attribution records
        total_abs_shap = np.sum(np.abs(shap_vals))
        attributions: list[FeatureAttribution] = []

        for i, f_name in enumerate(feature_names):
            f_val = float(df.iloc[0, i])
            s_val = float(shap_vals[i])
            pct = (abs(s_val) / total_abs_shap * 100.0) if total_abs_shap > 0 else 0.0

            if s_val > 0.001:
                ctype = "escalator"
            elif s_val < -0.001:
                ctype = "stabilizer"
            else:
                ctype = "neutral"

            attributions.append(
                FeatureAttribution(
                    feature=f_name,
                    display_name=FEATURE_DISPLAY_NAMES.get(f_name, f_name),
                    feature_value=f_val,
                    shap_value=s_val,
                    contribution_type=ctype,
                    percent_contribution=pct,
                )
            )

        # Separate risk escalators and stabilizers
        escalators = sorted(
            [a for a in attributions if a.contribution_type == "escalator"],
            key=lambda x: x.shap_value,
            reverse=True,
        )
        stabilizers = sorted(
            [a for a in attributions if a.contribution_type == "stabilizer"],
            key=lambda x: x.shap_value,
        )

        # Determine qualitative risk level
        if prob >= 0.70:
            risk_level = "CRITICAL"
        elif prob >= threshold:
            risk_level = "ELEVATED"
        elif prob >= 0.25:
            risk_level = "MODERATE_WARNING"
        else:
            risk_level = "NOMINAL"

        # Construct result object
        result = LocalExplanationResult(
            predicted_probability=prob,
            predicted_class=pred_class,
            base_value=base_val,
            risk_level=risk_level,
            top_risk_escalators=escalators[:top_k],
            top_stabilizers=stabilizers[:top_k],
            all_attributions=attributions,
            operator_summary="",
            non_causal_disclaimer=NON_CAUSAL_DISCLAIMER,
        )

        # Generate plain-English operator summary
        result.operator_summary = self.generate_operator_summary(
            result, original_unscaled_values=original_unscaled_values
        )

        return result

    def generate_operator_summary(
        self,
        explanation: LocalExplanationResult,
        original_unscaled_values: Optional[dict[str, float]] = None,
    ) -> str:
        """Create plain-English shop-floor operator narrative.

        Args:
            explanation: Computed LocalExplanationResult.
            original_unscaled_values: Optional raw engineering units.

        Returns:
            Human-readable diagnostic string.
        """
        prob_pct = explanation.predicted_probability * 100.0
        base_pct = explanation.base_value * 100.0
        lines = []

        # Header with Risk Level
        if explanation.risk_level == "CRITICAL":
            lines.append(f"[CRITICAL ALERT] Failure Probability: {prob_pct:.1f}% (Baseline: {base_pct:.1f}%)")
            lines.append("Immediate inspection recommended. Machine is exhibiting severe failure precursor signatures.")
        elif explanation.risk_level == "ELEVATED":
            lines.append(f"[ELEVATED RISK] Failure Probability: {prob_pct:.1f}% (Baseline: {base_pct:.1f}%)")
            lines.append("Telemetry exceeds normal operating safety margin. Schedule preventive maintenance check.")
        elif explanation.risk_level == "MODERATE_WARNING":
            lines.append(f"[MODERATE WARNING] Failure Probability: {prob_pct:.1f}% (Baseline: {base_pct:.1f}%)")
            lines.append("Minor telemetry drift observed. Monitor machine parameters during upcoming production runs.")
        else:
            lines.append(f"[NOMINAL STATE] Failure Probability: {prob_pct:.1f}% (Baseline: {base_pct:.1f}%)")
            lines.append("Machine telemetry is within standard healthy operating envelope.")

        lines.append("")

        # Key Drivers (Escalators)
        if explanation.top_risk_escalators:
            lines.append("Key Risk Escalators (Factors Increasing Failure Likelihood):")
            for esc in explanation.top_risk_escalators:
                val_str = ""
                if original_unscaled_values and esc.feature in original_unscaled_values:
                    val_str = f" [Raw: {original_unscaled_values[esc.feature]:.2f}]"
                lines.append(
                    f"  - {esc.display_name}{val_str}: +{esc.shap_value:.4f} SHAP impact ({esc.percent_contribution:.1f}% attribution)"
                )
        else:
            lines.append("No significant risk-escalating factors detected.")

        lines.append("")

        # Stabilizing Factors
        if explanation.top_stabilizers:
            lines.append("Key Stabilizing Factors (Factors Maintaining Safety Margin):")
            for stb in explanation.top_stabilizers:
                val_str = ""
                if original_unscaled_values and stb.feature in original_unscaled_values:
                    val_str = f" [Raw: {original_unscaled_values[stb.feature]:.2f}]"
                lines.append(
                    f"  - {stb.display_name}{val_str}: {stb.shap_value:.4f} SHAP impact ({stb.percent_contribution:.1f}% attribution)"
                )

        lines.append("")
        lines.append(f"Safety Note: {explanation.non_causal_disclaimer}")

        return "\n".join(lines)

    def plot_global_importance(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        max_display: int = 10,
        save_path: Optional[Union[str, Path]] = None,
        title: str = "MachSense Global Feature Importance (SHAP)",
    ) -> plt.Figure:
        """Generate and optionally save global feature importance bar plot.

        Args:
            X: Dataset to evaluate.
            max_display: Number of top features to show.
            save_path: Optional path to save image file.
            title: Chart title.

        Returns:
            matplotlib Figure object.
        """
        importance_df = self.compute_global_importance(X).head(max_display)

        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        y_pos = np.arange(len(importance_df))
        ax.barh(y_pos, importance_df["mean_abs_shap"][::-1], color="#1f77b4", edgecolor="black", alpha=0.85)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance_df["display_name"][::-1], fontsize=10)
        ax.set_xlabel("Mean Absolute SHAP Value (Impact on Failure Probability)", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        plt.tight_layout()

        if save_path is not None:
            save_p = Path(save_path)
            save_p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_p, bbox_inches="tight", dpi=150)
            logger.info(f"Saved global importance plot to {save_p}")

        return fig

    def plot_waterfall(
        self,
        instance: Union[pd.Series, pd.DataFrame, np.ndarray],
        max_display: int = 10,
        save_path: Optional[Union[str, Path]] = None,
        title: str = "MachSense Local Instance Attribution (SHAP Waterfall)",
    ) -> plt.Figure:
        """Generate and save local waterfall plot for a single instance.

        Args:
            instance: Single input row.
            max_display: Maximum features to display.
            save_path: Optional path to save figure.
            title: Chart title.

        Returns:
            matplotlib Figure.
        """
        exp = self.get_shap_explanation(instance)
        # Map feature names to display names in the Explanation object
        if hasattr(exp, "feature_names") and exp.feature_names is not None:
            exp.feature_names = [FEATURE_DISPLAY_NAMES.get(f, f) for f in exp.feature_names]

        fig = plt.figure(figsize=(10, 6), dpi=100)
        shap.plots.waterfall(exp[0], max_display=max_display, show=False)
        plt.title(title, fontsize=12, fontweight="bold", pad=12)
        plt.tight_layout()

        if save_path is not None:
            save_p = Path(save_path)
            save_p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_p, bbox_inches="tight", dpi=150)
            logger.info(f"Saved waterfall plot to {save_p}")

        return fig


def main() -> None:
    """Demonstrate SHAP explainability on saved MachSense production model."""
    import joblib

    print("=" * 70)
    print("MachSense Explainable AI (SHAP) Demonstration")
    print("=" * 70)

    model_path = Path("models/best_model.joblib")
    data_path = Path("data/processed/X_test_transformed.csv")
    y_test_path = Path("data/processed/y_test.csv")

    if not model_path.exists() or not data_path.exists():
        print(f"Artifacts missing. Please ensure {model_path} and {data_path} exist.")
        return

    model = joblib.load(model_path)
    X_test = pd.read_csv(data_path)
    y_test = pd.read_csv(y_test_path).squeeze() if y_test_path.exists() else None

    explainer = MachSenseExplainer(model=model, feature_names=list(X_test.columns))

    print("\n--- 1. Computing Global Feature Importance ---")
    importance_df = explainer.compute_global_importance(X_test)
    print(importance_df[["rank", "feature", "mean_abs_shap", "relative_importance"]].to_string(index=False))

    # Identify a high-risk failure case and nominal case
    if y_test is not None:
        failure_indices = np.where(y_test == 1)[0]
        nominal_indices = np.where(y_test == 0)[0]

        if len(failure_indices) > 0:
            fail_idx = failure_indices[0]
            print(f"\n--- 2. Local Instance Explanation (True Failure Sample #{fail_idx}) ---")
            fail_exp = explainer.explain_instance(X_test.iloc[fail_idx])
            print(fail_exp.operator_summary)

        if len(nominal_indices) > 0:
            nom_idx = nominal_indices[0]
            print(f"\n--- 3. Local Instance Explanation (Nominal Sample #{nom_idx}) ---")
            nom_exp = explainer.explain_instance(X_test.iloc[nom_idx])
            print(nom_exp.operator_summary)


if __name__ == "__main__":
    main()
