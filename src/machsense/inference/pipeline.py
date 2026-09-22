from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from machsense.config.settings import get_settings
from machsense.features.preprocessor import MachSensePreprocessor
from machsense.inference.schema import (
    BatchPredictionResult,
    InferenceStatus,
    PredictionResult,
    RiskLevel,
    SensorPayload,
)
from machsense.inference.validator import InferenceValidator
from machsense.models.explainability import (
    NON_CAUSAL_DISCLAIMER,
    LocalExplanationResult,
    MachSenseExplainer,
)
from machsense.models.registry import ModelRegistry
from machsense.utils.logger import get_logger

logger = get_logger(__name__)


class MachSensePredictionService:
    """Production-grade inference service for real-time and batch machine health predictions.

    Guarantees:
    1. Zero Retraining: Only pre-fitted, serialized transformers are used.
    2. Strict Validation: Input payloads are verified against domain physics.
    3. Integrated Explainability: Attributions and operator diagnostics generated via TreeSHAP.
    4. Graceful Error Handling: Structured errors are returned without server crashes.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        preprocessor: Optional[MachSensePreprocessor] = None,
        metadata: Optional[dict[str, Any]] = None,
        explainer: Optional[MachSenseExplainer] = None,
        models_dir: Optional[Union[Path, str]] = None,
        version: Optional[str] = None,
        eager_load_explainer: bool = True,
    ) -> None:
        """Initialize the Prediction Service.

        If components are not explicitly provided, they are loaded from the ModelRegistry.
        """
        self.models_dir = Path(models_dir) if models_dir else get_settings().resolve_path("models_dir")
        self.version = version
        self.registry = ModelRegistry(models_dir=self.models_dir)

        # 1. Load Model & Preprocessor
        if model is not None and preprocessor is not None:
            self.model = model
            self.preprocessor = preprocessor
            self.metadata = metadata or {}
        else:
            logger.info("Loading model artifacts from registry (version: %s)...", version or "active champion")
            self.model, self.preprocessor, self.metadata = self.registry.load_versioned_artifacts(version=version)

        # Verify preprocessor invariant
        if not getattr(self.preprocessor, "is_fitted_", False):
            raise RuntimeError("CRITICAL: Loaded MachSensePreprocessor is not fitted. Cannot serve inference.")

        # 2. Extract operational parameters
        self.model_version: str = self.metadata.get("model_version", version or "v1.0.0")
        self.model_name: str = self.metadata.get("model_name", type(self.model).__name__)
        self.optimal_threshold: float = float(self.metadata.get("optimal_threshold", 0.5))

        # 3. Initialize SHAP Explainer
        if explainer is not None:
            self.explainer: Optional[MachSenseExplainer] = explainer
        elif eager_load_explainer:
            try:
                feature_names = getattr(self.preprocessor, "feature_names_out_", None)
                self.explainer = MachSenseExplainer(model=self.model, feature_names=feature_names)
            except Exception as e:
                logger.warning("Could not initialize SHAP explainer eagerly: %s", str(e))
                self.explainer = None
        else:
            self.explainer = None

        logger.info(
            "MachSensePredictionService ready: Model=%s (%s), Optimal Threshold=%.4f",
            self.model_name,
            self.model_version,
            self.optimal_threshold,
        )

    def get_service_info(self) -> dict[str, Any]:
        """Return diagnostic metadata about the running prediction service."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "optimal_threshold": self.optimal_threshold,
            "expected_input_features": ["type", "air_temperature_k", "process_temperature_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min"],
            "transformed_feature_count": len(getattr(self.preprocessor, "feature_names_out_", [])),
            "explainer_ready": self.explainer is not None,
        }

    def predict(
        self,
        payload: Union[dict[str, Any], SensorPayload],
        explain: bool = True,
        threshold: Optional[float] = None,
    ) -> PredictionResult:
        """Perform end-to-end inference on a single machine telemetry record.

        Args:
            payload: Raw dictionary of sensor telemetry or validated SensorPayload.
            explain: If True, computes local SHAP attribution and operator summary.
            threshold: Optional custom decision threshold (defaults to calibrated optimal threshold).

        Returns:
            PredictionResult containing probability, class, risk level, and diagnostics.
        """
        start_time = time.perf_counter()
        active_threshold = threshold if threshold is not None else self.optimal_threshold

        # Step 1: Input Validation
        val_result = InferenceValidator.validate_payload(payload)
        if not val_result.is_valid:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return PredictionResult(
                status=InferenceStatus.VALIDATION_ERROR,
                threshold_used=active_threshold,
                model_version=self.model_version,
                errors=val_result.errors,
                warnings=val_result.warnings,
                latency_ms=round(latency_ms, 2),
            )

        try:
            cleaned_payload = val_result.cleaned_data or {}

            # Step 2: Preprocessing (Strictly no fitting / no leakage)
            X_trans = self.preprocessor.transform_instance(cleaned_payload)

            # Step 3: Model Scoring
            if hasattr(self.model, "predict_proba"):
                prob_arr = self.model.predict_proba(X_trans)
                failure_prob = float(prob_arr[0, 1]) if prob_arr.shape[1] > 1 else float(prob_arr[0, 0])
            elif hasattr(self.model, "predict"):
                failure_prob = float(self.model.predict(X_trans)[0])
            else:
                raise AttributeError(f"Model {type(self.model)} has neither predict_proba nor predict method.")

            # Step 4: Threshold Classification & Risk Mapping
            pred_class = 1 if failure_prob >= active_threshold else 0
            pred_label = "FAILURE_IMMINENT" if pred_class == 1 else "HEALTHY"

            if failure_prob >= 0.70:
                risk_level = RiskLevel.CRITICAL
            elif failure_prob >= active_threshold:
                risk_level = RiskLevel.ELEVATED
            elif failure_prob >= 0.25:
                risk_level = RiskLevel.MODERATE_WARNING
            else:
                risk_level = RiskLevel.NOMINAL

            # Step 5: SHAP Local Explainability
            top_escalators: list[dict[str, Any]] = []
            top_stabilizers: list[dict[str, Any]] = []
            operator_summary: Optional[str] = None

            if explain and self.explainer is not None:
                try:
                    local_exp: LocalExplanationResult = self.explainer.explain_instance(
                        X_trans,
                        threshold=active_threshold,
                        top_k=4,
                        original_unscaled_values=cleaned_payload,
                    )
                    top_escalators = [asdict(e) for e in local_exp.top_risk_escalators]
                    top_stabilizers = [asdict(s) for s in local_exp.top_stabilizers]
                    operator_summary = local_exp.operator_summary
                except Exception as expl_err:
                    logger.warning("Failed to generate local SHAP explanation: %s", str(expl_err))
                    operator_summary = f"[DIAGNOSTIC NOTICE] Automated explanation unavailable: {str(expl_err)}"

            latency_ms = (time.perf_counter() - start_time) * 1000.0

            return PredictionResult(
                status=InferenceStatus.SUCCESS,
                failure_probability=round(failure_prob, 4),
                predicted_class=pred_class,
                predicted_label=pred_label,
                risk_level=risk_level,
                threshold_used=float(active_threshold),
                model_version=self.model_version,
                top_risk_escalators=top_escalators,
                top_stabilizers=top_stabilizers,
                operator_summary=operator_summary,
                non_causal_disclaimer=NON_CAUSAL_DISCLAIMER,
                warnings=val_result.warnings,
                latency_ms=round(latency_ms, 2),
            )

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error("Unexpected error during inference: %s", str(exc), exc_info=True)
            return PredictionResult(
                status=InferenceStatus.PROCESSING_ERROR,
                threshold_used=active_threshold,
                model_version=self.model_version,
                errors=[f"Inference pipeline execution error: {str(exc)}"],
                latency_ms=round(latency_ms, 2),
            )

    def predict_batch(
        self,
        df: pd.DataFrame,
        explain: bool = False,
        threshold: Optional[float] = None,
    ) -> BatchPredictionResult:
        """Perform vectorized batch inference on a DataFrame of telemetry records.

        Args:
            df: Input DataFrame containing required sensor features.
            explain: If True, computes local SHAP explanations for individual records.
            threshold: Optional custom decision threshold.

        Returns:
            BatchPredictionResult containing aggregated metrics and individual predictions.
        """
        start_time = time.perf_counter()
        active_threshold = threshold if threshold is not None else self.optimal_threshold

        is_valid, val_errors, cleaned_df = InferenceValidator.validate_dataframe(df)
        if not is_valid:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return BatchPredictionResult(
                status=InferenceStatus.VALIDATION_ERROR,
                total_records=len(df) if isinstance(df, pd.DataFrame) else 0,
                successful_predictions_count=0,
                errors=val_errors,
                total_latency_ms=round(latency_ms, 2),
            )

        try:
            # Transform batch
            X_trans = self.preprocessor.transform(cleaned_df, return_dataframe=True)

            # Score batch
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X_trans)[:, 1]
            else:
                probs = self.model.predict(X_trans)

            predictions: list[PredictionResult] = []
            failure_count = 0

            for i in range(len(cleaned_df)):
                prob = float(probs[i])
                pred_class = 1 if prob >= active_threshold else 0
                if pred_class == 1:
                    failure_count += 1

                pred_label = "FAILURE_IMMINENT" if pred_class == 1 else "HEALTHY"

                if prob >= 0.70:
                    risk_level = RiskLevel.CRITICAL
                elif prob >= active_threshold:
                    risk_level = RiskLevel.ELEVATED
                elif prob >= 0.25:
                    risk_level = RiskLevel.MODERATE_WARNING
                else:
                    risk_level = RiskLevel.NOMINAL

                # Local explanation only if requested (to keep batch speed high)
                top_escalators: list[dict[str, Any]] = []
                top_stabilizers: list[dict[str, Any]] = []
                operator_summary: Optional[str] = None

                if explain and self.explainer is not None:
                    try:
                        row_dict = cleaned_df.iloc[i].to_dict()
                        local_exp = self.explainer.explain_instance(
                            X_trans.iloc[[i]],
                            threshold=active_threshold,
                            original_unscaled_values=row_dict,
                        )
                        top_escalators = [asdict(e) for e in local_exp.top_risk_escalators]
                        top_stabilizers = [asdict(s) for s in local_exp.top_stabilizers]
                        operator_summary = local_exp.operator_summary
                    except Exception:
                        pass

                predictions.append(
                    PredictionResult(
                        status=InferenceStatus.SUCCESS,
                        failure_probability=round(prob, 4),
                        predicted_class=pred_class,
                        predicted_label=pred_label,
                        risk_level=risk_level,
                        threshold_used=float(active_threshold),
                        model_version=self.model_version,
                        top_risk_escalators=top_escalators,
                        top_stabilizers=top_stabilizers,
                        operator_summary=operator_summary,
                    )
                )

            total_latency = (time.perf_counter() - start_time) * 1000.0

            return BatchPredictionResult(
                status=InferenceStatus.SUCCESS,
                total_records=len(df),
                successful_predictions_count=len(predictions),
                predictions=predictions,
                failure_count=failure_count,
                mean_failure_probability=float(np.mean(probs)) if len(probs) > 0 else 0.0,
                total_latency_ms=round(total_latency, 2),
            )

        except Exception as exc:
            total_latency = (time.perf_counter() - start_time) * 1000.0
            logger.error("Error during batch inference: %s", str(exc), exc_info=True)
            return BatchPredictionResult(
                status=InferenceStatus.PROCESSING_ERROR,
                total_records=len(df),
                successful_predictions_count=0,
                errors=[f"Batch execution error: {str(exc)}"],
                total_latency_ms=round(total_latency, 2),
            )
