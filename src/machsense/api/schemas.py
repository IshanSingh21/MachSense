"""Pydantic schemas for the MachSense FastAPI application layer."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from machsense.data.schema import MachineType, SensorBoundaries
from machsense.inference.schema import RiskLevel


class HealthResponse(BaseModel):
    """Liveness check response."""

    status: str = Field(default="healthy", description="API operational health status")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Current server UTC timestamp",
    )
    version: str = Field(default="0.1.0", description="MachSense application version")
    environment: str = Field(default="development", description="Current deployment environment")


class ReadinessResponse(BaseModel):
    """Readiness probe response verifying model availability."""

    status: str = Field(description="Readiness status ('ready' or 'unavailable')")
    model_loaded: bool = Field(description="Whether the champion model is loaded in memory")
    model_name: Optional[str] = Field(default=None, description="Name of loaded champion model")
    model_version: Optional[str] = Field(default=None, description="Version tag of active model")
    optimal_threshold: Optional[float] = Field(default=None, description="Operational decision threshold")
    explainer_ready: bool = Field(default=False, description="Whether SHAP TreeExplainer is initialized")


class RootResponse(BaseModel):
    """Root metadata response."""

    name: str = Field(default="MachSense API", description="Application name")
    description: str = Field(
        default="Industrial AI-Powered Predictive Maintenance and Real-Time Machine Health Diagnostics API"
    )
    version: str = Field(default="0.1.0", description="Application version")
    documentation_url: str = Field(default="/docs", description="Interactive OpenAPI documentation URL")
    redoc_url: str = Field(default="/redoc", description="ReDoc documentation URL")
    health_url: str = Field(default="/health", description="Health check endpoint URL")
    predict_url: str = Field(default="/api/v1/predict", description="Single prediction endpoint URL")
    batch_predict_url: str = Field(default="/api/v1/predict/batch", description="Batch prediction endpoint URL")


class TelemetryItem(BaseModel):
    """Manufacturing machine sensor telemetry record."""

    type: MachineType = Field(
        description="Machine quality variant type (L, M, H)",
        examples=["L"],
    )
    air_temperature_k: float = Field(
        ge=SensorBoundaries.AIR_TEMP_MIN_K,
        le=SensorBoundaries.AIR_TEMP_MAX_K,
        description="Air / Ambient temperature in Kelvin",
        examples=[302.5],
    )
    process_temperature_k: float = Field(
        ge=SensorBoundaries.PROCESS_TEMP_MIN_K,
        le=SensorBoundaries.PROCESS_TEMP_MAX_K,
        description="Process chamber temperature in Kelvin",
        examples=[311.8],
    )
    rotational_speed_rpm: float = Field(
        ge=SensorBoundaries.ROTATIONAL_SPEED_MIN_RPM,
        le=SensorBoundaries.ROTATIONAL_SPEED_MAX_RPM,
        description="Spindle rotational speed in RPM",
        examples=[1380.0],
    )
    torque_nm: float = Field(
        ge=SensorBoundaries.TORQUE_MIN_NM,
        le=SensorBoundaries.TORQUE_MAX_NM,
        description="Applied mechanical torque in Nm",
        examples=[58.5],
    )
    tool_wear_min: float = Field(
        ge=SensorBoundaries.TOOL_WEAR_MIN_MIN,
        le=SensorBoundaries.TOOL_WEAR_MAX_MIN,
        description="Cumulative tool wear duration in minutes",
        examples=[210.0],
    )
    udi: Optional[int] = Field(default=None, description="Optional telemetry ID")
    product_id: Optional[str] = Field(default=None, description="Optional product serial code")


class PredictRequest(BaseModel):
    """Single machine prediction request."""

    telemetry: TelemetryItem = Field(description="Raw sensor readings for a single machine cycle")
    explain: bool = Field(default=True, description="Whether to compute local SHAP feature attributions")
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional custom decision threshold override",
    )


class BatchPredictRequest(BaseModel):
    """High-throughput batch prediction request."""

    items: list[TelemetryItem] = Field(
        min_length=1,
        max_length=5000,
        description="List of telemetry records to evaluate",
    )
    explain: bool = Field(
        default=False,
        description="Whether to compute SHAP attributions for every batch record",
    )
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional custom decision threshold override",
    )


class FeatureAttributionDTO(BaseModel):
    """Individual feature contribution record."""

    feature: str
    display_name: str
    feature_value: float
    shap_value: float
    contribution_type: str
    percent_contribution: float


class PredictionResponse(BaseModel):
    """Structured response for a single prediction request."""

    status: str = Field(description="Execution status ('SUCCESS', 'VALIDATION_ERROR', 'PROCESSING_ERROR')")
    failure_probability: float = Field(description="Predicted probability of imminent machine failure (0.0 to 1.0)")
    predicted_class: int = Field(description="Binary classification (0 = Healthy, 1 = Imminent Failure)")
    predicted_label: str = Field(description="Human-readable label ('HEALTHY' or 'FAILURE_IMMINENT')")
    risk_level: str = Field(description="Categorical risk assessment ('NOMINAL', 'MODERATE_WARNING', 'ELEVATED', 'CRITICAL')")
    threshold_used: float = Field(description="Decision threshold applied for classification")
    model_version: str = Field(description="Version tag of active model")
    top_risk_escalators: list[dict[str, Any]] = Field(default_factory=list, description="Top features pushing toward failure")
    top_stabilizers: list[dict[str, Any]] = Field(default_factory=list, description="Top features maintaining safe margins")
    operator_summary: Optional[str] = Field(default=None, description="Plain-English shop-floor maintenance guidance")
    non_causal_disclaimer: str = Field(description="Non-causal statistical attribution safety notice")
    latency_ms: float = Field(description="Inference execution latency in milliseconds")


class BatchPredictionResponse(BaseModel):
    """Structured response for batch prediction request."""

    status: str
    total_records: int
    successful_predictions_count: int
    failure_count: int
    mean_failure_probability: float
    predictions: list[dict[str, Any]]
    total_latency_ms: float


class ErrorResponse(BaseModel):
    """Standardized API error response envelope."""

    status: str = Field(default="ERROR")
    error_code: str = Field(description="Machine-readable error category")
    message: str = Field(description="Human-readable error description")
    details: list[str] = Field(default_factory=list, description="Detailed validation error list")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
