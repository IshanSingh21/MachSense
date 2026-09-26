"""Dependency injection providers for MachSense FastAPI application."""

from __future__ import annotations

from typing import Optional

from fastapi import HTTPException, Request, status

from machsense.inference.pipeline import MachSensePredictionService
from machsense.utils.logger import get_logger

logger = get_logger(__name__)


def get_prediction_service(request: Request) -> MachSensePredictionService:
    """Retrieve the singleton MachSensePredictionService instance from app state.

    Args:
        request: Incoming FastAPI request.

    Returns:
        MachSensePredictionService instance.

    Raises:
        HTTPException: 503 if prediction service is not initialized.
    """
    service: Optional[MachSensePredictionService] = getattr(request.app.state, "prediction_service", None)

    if service is None:
        logger.error("MachSensePredictionService is not available in application state.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MachSense prediction service is initializing or currently unavailable.",
        )

    return service
