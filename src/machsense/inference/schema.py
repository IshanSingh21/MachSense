"""Pydantic and dataclass schemas for MachSense online and batch inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field, field_validator

from machsense.data.schema import MachineType, SensorBoundaries
from machsense.models.explainability import NON_CAUSAL_DISCLAIMER, FeatureAttribution


class InferenceStatus(str, Enum):
    """Execution status of an inference request."""

    SUCCESS = "SUCCESS"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    PROCESSING_ERROR = "PROCESSING_ERROR"


class RiskLevel(str, Enum):
    """Categorical risk assessment level."""

    NOMINAL = "NOMINAL"
    MODERATE_WARNING = "MODERATE_WARNING"
    ELEVATED = "ELEVATED"
    CRITICAL = "CRITICAL"


class SensorPayload(BaseModel):
    """Validated raw sensor telemetry record for single-instance prediction."""

    type: MachineType = Field(
        description="Machine quality variant type (L, M, H)",
        examples=["M"],
    )
    air_temperature_k: float = Field(
        ge=SensorBoundaries.AIR_TEMP_MIN_K,
        le=SensorBoundaries.AIR_TEMP_MAX_K,
        description="Air temperature in Kelvin",
        examples=[300.2],
    )
    process_temperature_k: float = Field(
        ge=SensorBoundaries.PROCESS_TEMP_MIN_K,
        le=SensorBoundaries.PROCESS_TEMP_MAX_K,
        description="Process temperature in Kelvin",
        examples=[310.5],
    )
    rotational_speed_rpm: float = Field(
        ge=SensorBoundaries.ROTATIONAL_SPEED_MIN_RPM,
        le=SensorBoundaries.ROTATIONAL_SPEED_MAX_RPM,
        description="Spindle rotational speed in RPM",
        examples=[1500.0],
    )
    torque_nm: float = Field(
        ge=SensorBoundaries.TORQUE_MIN_NM,
        le=SensorBoundaries.TORQUE_MAX_NM,
        description="Mechanical applied torque in Nm",
        examples=[42.0],
    )
    tool_wear_min: float = Field(
        ge=SensorBoundaries.TOOL_WEAR_MIN_MIN,
        le=SensorBoundaries.TOOL_WEAR_MAX_MIN,
        description="Cumulative tool wear duration in minutes",
        examples=[120.0],
    )
    udi: Optional[int] = Field(default=None, description="Optional telemetry identifier")
    product_id: Optional[str] = Field(default=None, description="Optional product serial code")

    @field_validator("process_temperature_k")
    @classmethod
    def validate_process_temp(cls, v: float, info: Any) -> float:
        """Physical rule: process temperature must not be colder than ambient air temp."""
        air_temp = info.data.get("air_temperature_k")
        if air_temp is not None and v < (air_temp - 0.5):
            raise ValueError(
                f"Thermodynamic violation: Process temperature ({v:.2f} K) is below ambient air temperature ({air_temp:.2f} K)."
            )
        return v


@dataclass
class ValidationResult:
    """Detailed outcome of telemetry payload validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cleaned_data: Optional[dict[str, Any]] = None


@dataclass
class PredictionResult:
    """Comprehensive structured prediction output for a single telemetry record."""

    status: InferenceStatus
    failure_probability: Optional[float] = None
    predicted_class: Optional[int] = None
    predicted_label: Optional[str] = None  # "HEALTHY" or "FAILURE_IMMINENT"
    risk_level: Optional[RiskLevel] = None
    threshold_used: Optional[float] = None
    model_version: Optional[str] = None
    top_risk_escalators: list[dict[str, Any]] = field(default_factory=list)
    top_stabilizers: list[dict[str, Any]] = field(default_factory=list)
    operator_summary: Optional[str] = None
    non_causal_disclaimer: str = NON_CAUSAL_DISCLAIMER
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert result dataclass to dictionary."""
        d = asdict(self)
        if self.status:
            d["status"] = self.status.value
        if self.risk_level:
            d["risk_level"] = self.risk_level.value
        return d


@dataclass
class BatchPredictionResult:
    """Structured response for high-throughput batch prediction."""

    status: InferenceStatus
    total_records: int
    successful_predictions_count: int
    predictions: list[PredictionResult] = field(default_factory=list)
    failure_count: int = 0
    mean_failure_probability: float = 0.0
    errors: list[str] = field(default_factory=list)
    total_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert batch result to dictionary."""
        return {
            "status": self.status.value,
            "total_records": self.total_records,
            "successful_predictions_count": self.successful_predictions_count,
            "failure_count": self.failure_count,
            "mean_failure_probability": round(self.mean_failure_probability, 4),
            "errors": self.errors,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "predictions": [p.to_dict() for p in self.predictions],
        }
