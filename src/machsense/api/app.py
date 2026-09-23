"""FastAPI application factory, middleware configuration, and lifecycle management for MachSense."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from machsense import __version__
from machsense.api.routes import api_router
from machsense.config.settings import AppConfig, get_settings
from machsense.inference.pipeline import MachSensePredictionService
from machsense.utils.logger import get_logger, setup_logging

logger = get_logger("machsense.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle events."""
    setup_logging()
    logger.info("Initializing MachSense API backend (v%s)...", __version__)

    # Eagerly load model artifacts into memory once at startup
    try:
        service = MachSensePredictionService(eager_load_explainer=True)
        app.state.prediction_service = service
        logger.info(
            "MachSensePredictionService loaded successfully into application state: %s (%s)",
            service.model_name,
            service.model_version,
        )
    except Exception as exc:
        logger.error("Failed to initialize MachSensePredictionService on startup: %s", str(exc), exc_info=True)
        app.state.prediction_service = None

    yield

    logger.info("Shutting down MachSense API backend...")
    app.state.prediction_service = None


def create_app(config: Optional[AppConfig] = None) -> FastAPI:
    """Construct and configure the FastAPI application instance.

    Args:
        config: Optional AppConfig override. Defaults to singleton get_settings().

    Returns:
        Configured FastAPI application.
    """
    settings = config or get_settings()

    app = FastAPI(
        title="MachSense Predictive Maintenance API",
        description=(
            "Industrial AI backend for real-time machine telemetry validation, "
            "failure risk scoring, calibrated decision thresholding, and TreeSHAP explainability diagnostics."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # 1. Configure CORS Middleware
    origins = settings.serving.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins if origins else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Custom Exception Handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Standardize 422 Unprocessable Entity responses."""
        error_messages = []
        for err in exc.errors():
            loc = " -> ".join(str(item) for item in err["loc"])
            msg = err["msg"]
            error_messages.append(f"{loc}: {msg}")

        logger.warning("Request schema validation failure on %s: %s", request.url.path, error_messages)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "status": "ERROR",
                "error_code": "SCHEMA_VALIDATION_ERROR",
                "message": "The request payload failed Pydantic schema validation.",
                "details": error_messages,
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """Standardize HTTP exception envelopes."""
        if isinstance(exc.detail, dict):
            content = {
                "status": "ERROR",
                "error_code": exc.detail.get("status", "HTTP_ERROR"),
                "message": exc.detail.get("message", "An HTTP error occurred."),
                "details": exc.detail.get("errors", []),
            }
        else:
            content = {
                "status": "ERROR",
                "error_code": "HTTP_ERROR",
                "message": str(exc.detail),
                "details": [],
            }

        return JSONResponse(
            status_code=exc.status_code,
            content=content,
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch unhandled runtime exceptions without leaking stack traces."""
        logger.error("Unhandled server exception on %s: %s", request.url.path, str(exc), exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "ERROR",
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An internal server error occurred while processing the request.",
                "details": [str(exc)],
            },
        )

    # 3. Include API Routers
    app.include_router(api_router)

    return app


# Default application instance for Uvicorn ASGI runner
app = create_app()
