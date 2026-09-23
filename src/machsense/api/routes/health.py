"""Health, liveness, readiness, and metadata endpoints for MachSense API."""

from __future__ import annotations

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from machsense import __version__
from machsense.api.dependencies import get_prediction_service
from machsense.api.schemas import HealthResponse, ReadinessResponse, RootResponse
from machsense.config.settings import get_settings
from machsense.inference.pipeline import MachSensePredictionService
from machsense.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Health & System"])


@router.get(
    "/",
    response_model=RootResponse,
    summary="MachSense API Root Metadata",
    description="Returns high-level system information, documentation links, and available endpoint URLs.",
)
async def get_root() -> RootResponse:
    """Root endpoint providing service overview and discovery links."""
    return RootResponse(
        version=__version__,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness Health Check",
    description="Returns standard 200 OK if the FastAPI process is alive and accepting traffic.",
)
async def get_health() -> HealthResponse:
    """Liveness probe verifying that the backend server is running."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=__version__,
        environment=settings.project.environment,
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness Probe",
    description="Verifies that the ML model, preprocessor, and SHAP explainer are loaded in memory and ready to serve inference.",
)
async def get_readiness(
    request: Request,
) -> ReadinessResponse:
    """Readiness probe checking memory-loaded ML artifacts."""
    service: MachSensePredictionService | None = getattr(request.app.state, "prediction_service", None)

    if service is None or getattr(service, "model", None) is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unavailable",
                "model_loaded": False,
                "model_name": None,
                "model_version": None,
                "optimal_threshold": None,
                "explainer_ready": False,
            },
        )

    info = service.get_service_info()
    return ReadinessResponse(
        status="ready",
        model_loaded=True,
        model_name=info["model_name"],
        model_version=info["model_version"],
        optimal_threshold=info["optimal_threshold"],
        explainer_ready=info["explainer_ready"],
    )
