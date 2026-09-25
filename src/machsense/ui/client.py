"""UI Client wrapper for MachSense prediction service and physical presets."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd

from machsense.inference.pipeline import MachSensePredictionService
from machsense.inference.schema import (
    BatchPredictionResult,
    InferenceStatus,
    PredictionResult,
    RiskLevel,
    SensorPayload,
)
from machsense.utils.logger import get_logger

logger = get_logger(__name__)

# Standard domain failure presets for rapid shop-floor testing
TELEMETRY_PRESETS: dict[str, dict[str, Any]] = {
    "Nominal Healthy Machine": {
        "type": "M",
        "air_temperature_k": 298.1,
        "process_temperature_k": 308.6,
        "rotational_speed_rpm": 1550.0,
        "torque_nm": 40.0,
        "tool_wear_min": 30.0,
        "description": "Standard nominal cutting cycle under balanced speed and torque with minimal tool wear.",
    },
    "Heat Dissipation Failure (HDF)": {
        "type": "L",
        "air_temperature_k": 302.8,
        "process_temperature_k": 311.2,
        "rotational_speed_rpm": 1360.0,
        "torque_nm": 52.0,
        "tool_wear_min": 105.0,
        "description": "Cooling dissipation collapse: Low speed with ambient-process temp delta under 8.5 K.",
    },
    "Power Failure (PWF)": {
        "type": "L",
        "air_temperature_k": 300.5,
        "process_temperature_k": 310.8,
        "rotational_speed_rpm": 2840.0,
        "torque_nm": 65.0,
        "tool_wear_min": 35.0,
        "description": "Power overload breakdown: Spindle wattage exceeds 19,000 W (far exceeding 9,000 W limit).",
    },
    "Overstrain Failure (OSF)": {
        "type": "L",
        "air_temperature_k": 302.5,
        "process_temperature_k": 311.8,
        "rotational_speed_rpm": 1380.0,
        "torque_nm": 62.5,
        "tool_wear_min": 215.0,
        "description": "Severe mechanical overload: High torque combined with tool wear exceeding 200 min.",
    },
    "Tool Wear Failure (TWF)": {
        "type": "M",
        "air_temperature_k": 299.0,
        "process_temperature_k": 309.2,
        "rotational_speed_rpm": 1420.0,
        "torque_nm": 48.0,
        "tool_wear_min": 240.0,
        "description": "Exhausted cutting insert life: Tool wear duration exceeding 220 min threshold.",
    },
}


class MachSenseUIClient:
    """Client providing a clean interface for the Streamlit UI without exposing ML internals.

    Supports dual operating modes:
    1. HTTP API Mode: Connects to a running FastAPI backend (e.g. http://127.0.0.1:8000).
    2. In-Process Mode: Directly queries MachSensePredictionService in-memory.
    """

    def __init__(
        self,
        service: Optional[MachSensePredictionService] = None,
        api_url: Optional[str] = None,
        prefer_api: bool = False,
    ) -> None:
        """Initialize UI client.

        Args:
            service: Optional pre-initialized MachSensePredictionService.
            api_url: Optional base URL for FastAPI backend (e.g. 'http://127.0.0.1:8000').
            prefer_api: If True, attempts HTTP API calls first before falling back to in-process service.
        """
        self.api_url = api_url or "http://127.0.0.1:8000"
        self.prefer_api = prefer_api
        self._service = service

        # Lazy/eager in-process service fallback
        if self._service is None and not self.prefer_api:
            try:
                self._service = MachSensePredictionService()
            except Exception as e:
                logger.warning("Could not initialize local in-process service on startup: %s", str(e))
                self._service = None

    def _ensure_service(self) -> MachSensePredictionService:
        """Ensure in-process prediction service is available."""
        if self._service is None:
            self._service = MachSensePredictionService()
        return self._service

    @property
    def service_info(self) -> dict[str, Any]:
        """Get model metadata and service operational information."""
        if self.prefer_api:
            try:
                import httpx

                resp = httpx.get(f"{self.api_url}/health/ready", timeout=2.0)
                if resp.status_code == 200:
                    return resp.json()
            except Exception:
                pass

        try:
            return self._ensure_service().get_service_info()
        except Exception as e:
            return {
                "model_name": "Unavailable",
                "model_version": "N/A",
                "optimal_threshold": 0.5,
                "explainer_ready": False,
                "error": str(e),
            }

    @property
    def optimal_threshold(self) -> float:
        """Get calibrated operational decision threshold."""
        try:
            return self._ensure_service().optimal_threshold
        except Exception:
            return 0.5608

    def predict_single(
        self,
        payload: dict[str, Any],
        explain: bool = True,
        threshold: Optional[float] = None,
    ) -> PredictionResult:
        """Execute prediction on a single telemetry dictionary.

        Attempts HTTP API call if configured, with graceful fallback to in-process service.
        """
        if self.prefer_api:
            try:
                import httpx

                req_body = {
                    "telemetry": payload,
                    "explain": explain,
                    "threshold": threshold,
                }
                resp = httpx.post(
                    f"{self.api_url}/api/v1/predict",
                    json=req_body,
                    timeout=5.0,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    return PredictionResult(
                        status=InferenceStatus(data.get("status", "SUCCESS")),
                        failure_probability=data.get("failure_probability"),
                        predicted_class=data.get("predicted_class"),
                        predicted_label=data.get("predicted_label"),
                        risk_level=RiskLevel(data.get("risk_level")) if data.get("risk_level") else None,
                        threshold_used=data.get("threshold_used"),
                        model_version=data.get("model_version"),
                        top_risk_escalators=data.get("top_risk_escalators", []),
                        top_stabilizers=data.get("top_stabilizers", []),
                        operator_summary=data.get("operator_summary"),
                        non_causal_disclaimer=data.get("non_causal_disclaimer", ""),
                        latency_ms=data.get("latency_ms", 0.0),
                    )
                elif resp.status_code in (400, 422):
                    err_data = resp.json()
                    errors = err_data.get("details", []) or [err_data.get("message", "Validation error")]
                    return PredictionResult(
                        status=InferenceStatus.VALIDATION_ERROR,
                        errors=errors,
                    )
            except Exception as api_exc:
                logger.warning("HTTP API call failed (%s). Falling back to in-process inference...", str(api_exc))

        # In-process execution fallback
        try:
            return self._ensure_service().predict(payload, explain=explain, threshold=threshold)
        except Exception as exc:
            logger.error("In-process prediction failed: %s", str(exc), exc_info=True)
            return PredictionResult(
                status=InferenceStatus.PROCESSING_ERROR,
                errors=["A processing error occurred while evaluating machine telemetry."],
            )

    def predict_batch(
        self,
        df: pd.DataFrame,
        explain: bool = False,
        threshold: Optional[float] = None,
    ) -> BatchPredictionResult:
        """Execute vectorized batch prediction on a DataFrame."""
        try:
            return self._ensure_service().predict_batch(df, explain=explain, threshold=threshold)
        except Exception as exc:
            logger.error("Batch prediction failed: %s", str(exc), exc_info=True)
            return BatchPredictionResult(
                status=InferenceStatus.PROCESSING_ERROR,
                total_records=len(df) if isinstance(df, pd.DataFrame) else 0,
                successful_predictions_count=0,
                errors=["A processing error occurred while scoring the fleet batch telemetry."],
            )

    @classmethod
    def get_presets(cls) -> dict[str, dict[str, Any]]:
        """Return available physical failure presets."""
        return TELEMETRY_PRESETS
