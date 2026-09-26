"""Application server entry point placeholder for MachSense API / UI."""

from __future__ import annotations

from machsense import __version__
from machsense.config.settings import get_settings
from machsense.utils.logger import get_logger, setup_logging


def start_app() -> None:
    """Initialize and start the application server."""
    setup_logging()
    logger = get_logger("machsense.app")
    settings = get_settings()

    logger.info(
        "MachSense Serving Interface v%s initialized for environment '%s'",
        __version__,
        settings.project.environment,
    )
    logger.info(
        "Configured host: %s:%d (Workers: %d)",
        settings.serving.host,
        settings.serving.port,
        settings.serving.workers,
    )
    print(f"MachSense App ready at http://{settings.serving.host}:{settings.serving.port}")


if __name__ == "__main__":
    start_app()
