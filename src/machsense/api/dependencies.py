"""Dependency injection providers for MachSense FastAPI application."""

from __future__ import annotations

from typing import Optional

from fastapi import HTTPException, Request, status

from machsense.inference.pipeline import MachSensePredictionService
from machsense.utils.logger import get_logger

logger = get_logger(__name__)


_prediction_service_singleton: Optional[MachSensePredictionService] = None


def get_prediction_service(request: Request) -> MachSensePredictionService:
    """Retrieve the singleton MachSensePredictionService instance from app state or module cache.

    Args:
        request: Incoming FastAPI request.

    Returns:
        MachSensePredictionService instance.

    Raises:
        HTTPException: 503 if prediction service cannot be initialized.
    """
    global _prediction_service_singleton

    service: Optional[MachSensePredictionService] = getattr(request.app.state, "prediction_service", None)
    if service is not None and getattr(service, "model", None) is not None:
        return service

    if _prediction_service_singleton is not None and getattr(_prediction_service_singleton, "model", None) is not None:
        request.app.state.prediction_service = _prediction_service_singleton
        return _prediction_service_singleton

    logger.warning("MachSensePredictionService not found in application state. Initializing on-demand...")
    try:
        service = MachSensePredictionService(eager_load_explainer=True)
        _prediction_service_singleton = service
        request.app.state.prediction_service = service
        logger.info(
            "MachSensePredictionService lazily initialized successfully: %s (%s)",
            service.model_name,
            service.model_version,
        )
        return service
    except Exception as exc:
        logger.error("Failed to lazily initialize MachSensePredictionService: %s", str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MachSense prediction service is initializing or currently unavailable.",
        )
