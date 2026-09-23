"""Real-time and batch machine health prediction endpoints."""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
import pandas as pd

from machsense.api.dependencies import get_prediction_service
from machsense.api.schemas import (
    BatchPredictionResponse,
    BatchPredictRequest,
    ErrorResponse,
    PredictionResponse,
    PredictRequest,
)
from machsense.inference.pipeline import MachSensePredictionService
from machsense.inference.schema import InferenceStatus
from machsense.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Predictive Maintenance"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        200: {"description": "Successful prediction with failure probability and SHAP attribution."},
        400: {"model": ErrorResponse, "description": "Domain physical boundary or thermodynamic constraint violation."},
        422: {"description": "Schema or data type validation error."},
        500: {"model": ErrorResponse, "description": "Internal prediction processing error."},
    },
    summary="Real-Time Machine Failure Prediction",
    description=(
        "Evaluates a single telemetry payload against the champion machine health model. "
        "Returns failure probability, binary classification, operational risk category, "
        "top risk-escalating features, stabilizing factors, and plain-English operator guidance."
    ),
)
async def predict_single(
    request: PredictRequest,
    service: MachSensePredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    """Execute real-time prediction on a single telemetry record."""
    payload_dict = request.telemetry.model_dump()

    result = service.predict(
        payload=payload_dict,
        explain=request.explain,
        threshold=request.threshold,
    )

    if result.status == InferenceStatus.VALIDATION_ERROR:
        logger.warning("Telemetry domain validation failed: %s", result.errors)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "VALIDATION_ERROR",
                "message": "Input sensor telemetry violated physical boundary or thermodynamic constraints.",
                "errors": result.errors,
            },
        )

    if result.status == InferenceStatus.PROCESSING_ERROR:
        logger.error("Inference processing error: %s", result.errors)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "PROCESSING_ERROR",
                "message": "Internal error occurred while executing inference pipeline.",
                "errors": result.errors,
            },
        )

    return PredictionResponse(
        status=result.status.value,
        failure_probability=result.failure_probability or 0.0,
        predicted_class=result.predicted_class if result.predicted_class is not None else 0,
        predicted_label=result.predicted_label or "UNKNOWN",
        risk_level=result.risk_level.value if result.risk_level else "UNKNOWN",
        threshold_used=result.threshold_used if result.threshold_used is not None else service.optimal_threshold,
        model_version=result.model_version or service.model_version,
        top_risk_escalators=result.top_risk_escalators,
        top_stabilizers=result.top_stabilizers,
        operator_summary=result.operator_summary,
        non_causal_disclaimer=result.non_causal_disclaimer,
        latency_ms=result.latency_ms,
    )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses={
        200: {"description": "Batch prediction successfully executed."},
        400: {"model": ErrorResponse, "description": "Batch validation error."},
        422: {"description": "Pydantic payload schema validation error."},
        500: {"model": ErrorResponse, "description": "Internal batch processing error."},
    },
    summary="High-Throughput Batch Failure Prediction",
    description="Scores up to 5,000 telemetry records in a vectorized batch inference pass.",
)
async def predict_batch(
    request: BatchPredictRequest,
    service: MachSensePredictionService = Depends(get_prediction_service),
) -> BatchPredictionResponse:
    """Execute high-throughput batch prediction on a list of telemetry records."""
    records = [item.model_dump() for item in request.items]
    df = pd.DataFrame(records)

    batch_result = service.predict_batch(
        df=df,
        explain=request.explain,
        threshold=request.threshold,
    )

    if batch_result.status == InferenceStatus.VALIDATION_ERROR:
        logger.warning("Batch validation failed: %s", batch_result.errors)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "VALIDATION_ERROR",
                "message": "Batch DataFrame violated schema or domain physical boundaries.",
                "errors": batch_result.errors,
            },
        )

    if batch_result.status == InferenceStatus.PROCESSING_ERROR:
        logger.error("Batch processing error: %s", batch_result.errors)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "PROCESSING_ERROR",
                "message": "Internal error occurred during batch inference execution.",
                "errors": batch_result.errors,
            },
        )

    predictions_list = [p.to_dict() for p in batch_result.predictions]

    return BatchPredictionResponse(
        status=batch_result.status.value,
        total_records=batch_result.total_records,
        successful_predictions_count=batch_result.successful_predictions_count,
        failure_count=batch_result.failure_count,
        mean_failure_probability=round(batch_result.mean_failure_probability, 4),
        predictions=predictions_list,
        total_latency_ms=batch_result.total_latency_ms,
    )
