"""Production logging infrastructure for MachSense."""

from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path
from typing import Optional
import yaml

from machsense.config.settings import get_project_root

_LOGGING_INITIALIZED = False


def setup_logging(
    logging_config_path: Optional[Path | str] = None,
    default_level: int = logging.INFO,
    logs_dir: Optional[Path | str] = None,
) -> None:
    """Initialize logging configuration from YAML configuration or fallback defaults.

    Args:
        logging_config_path: Optional path to logging configuration YAML.
        default_level: Fallback log level if config file not found.
        logs_dir: Directory where log files should be written.
    """
    global _LOGGING_INITIALIZED

    root_dir = get_project_root()
    target_logs_dir = Path(logs_dir) if logs_dir else (root_dir / "logs")
    target_logs_dir.mkdir(parents=True, exist_ok=True)

    if logging_config_path is None:
        config_file = root_dir / "configs" / "logging.yaml"
    else:
        config_file = Path(logging_config_path)
        if not config_file.is_absolute():
            config_file = root_dir / config_file

    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)

            # Ensure all file handler destinations point to the correct absolute directory
            handlers = config_dict.get("handlers", {})
            for handler in handlers.values():
                if "filename" in handler:
                    filename = handler["filename"]
                    if not Path(filename).is_absolute():
                        handler["filename"] = str((root_dir / filename).resolve())

            logging.config.dictConfig(config_dict)
            _LOGGING_INITIALIZED = True
            return
        except Exception as e:
            # Fallback to basic configuration if dictConfig fails
            logging.basicConfig(
                level=default_level,
                format="[%(asctime)s] [%(levelname)-8s] [%(name)s:%(lineno)d] - %(message)s",
            )
            logging.warning("Failed to load logging configuration from %s: %s. Using basic config.", config_file, e)
            _LOGGING_INITIALIZED = True
            return

    # Fallback configuration
    logging.basicConfig(
        level=default_level,
        format="[%(asctime)s] [%(levelname)-8s] [%(name)s:%(lineno)d] - %(message)s",
    )
    _LOGGING_INITIALIZED = True


def get_logger(name: str = "machsense") -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Name of the logger (typically __name__ of calling module).

    Returns:
        logging.Logger instance.
    """
    global _LOGGING_INITIALIZED
    if not _LOGGING_INITIALIZED:
        setup_logging()
    return logging.getLogger(name)
