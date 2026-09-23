"""API routes aggregator package."""

from fastapi import APIRouter

from machsense.api.routes.health import router as health_router
from machsense.api.routes.predict import router as predict_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(predict_router)

__all__ = ["api_router", "health_router", "predict_router"]
