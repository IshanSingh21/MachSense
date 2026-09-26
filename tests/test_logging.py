"""Unit tests for MachSense logging infrastructure."""

import logging
from pathlib import Path

from machsense.utils.logger import get_logger, setup_logging


def test_setup_logging_creates_log_directory(tmp_path: Path):
    """Verify setup_logging automatically creates the target logs directory."""
    test_logs_dir = tmp_path / "custom_logs"
    assert not test_logs_dir.exists()

    setup_logging(logs_dir=test_logs_dir)
    assert test_logs_dir.exists()


def test_get_logger_returns_configured_logger():
    """Verify get_logger returns an active logger instance."""
    logger = get_logger("machsense.test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "machsense.test_module"


def test_logger_writes_messages_without_errors(caplog):
    """Verify logging calls capture log messages properly."""
    logger = get_logger("machsense.test_messages")
    with caplog.at_level(logging.INFO):
        logger.info("Test telemetry message: %s", "status_normal")
        logger.warning("Test anomaly warning: %s", "temperature_spike")

    assert "Test telemetry message: status_normal" in caplog.text
    assert "Test anomaly warning: temperature_spike" in caplog.text
