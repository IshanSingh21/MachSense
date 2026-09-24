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
    """Client providing a clean interface for the Streamlit UI without exposing ML internals."""

    def __init__(self, service: Optional[MachSensePredictionService] = None) -> None:
        """Initialize UI client."""
        if service is not None:
            self._service = service
        else:
            logger.info("Initializing MachSensePredictionService for UI client...")
            self._service = MachSensePredictionService()

    @property
    def service_info(self) -> dict[str, Any]:
        """Get model metadata and service operational information."""
        return self._service.get_service_info()

    @property
    def optimal_threshold(self) -> float:
        """Get calibrated operational decision threshold."""
        return self._service.optimal_threshold

    def predict_single(
        self,
        payload: dict[str, Any],
        explain: bool = True,
        threshold: Optional[float] = None,
    ) -> PredictionResult:
        """Execute prediction on a single telemetry dictionary."""
        return self._service.predict(payload, explain=explain, threshold=threshold)

    def predict_batch(
        self,
        df: pd.DataFrame,
        explain: bool = False,
        threshold: Optional[float] = None,
    ) -> BatchPredictionResult:
        """Execute vectorized batch prediction on a DataFrame."""
        return self._service.predict_batch(df, explain=explain, threshold=threshold)

    @classmethod
    def get_presets(cls) -> dict[str, dict[str, Any]]:
        """Return available physical failure presets."""
        return TELEMETRY_PRESETS
