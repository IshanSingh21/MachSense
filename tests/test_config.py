"""Unit tests for configuration management and dynamic path resolution."""

import os
from pathlib import Path
import pytest
from machsense.config.settings import (
    AppConfig,
    get_project_root,
    get_settings,
    load_config,
)


def test_get_project_root_resolves_valid_dir(project_root: Path):
    """Verify that get_project_root discovers the repository directory."""
    root = get_project_root()
    assert root.exists()
    assert (root / "configs").exists()
    assert (root / "pyproject.toml").exists()


def test_load_default_config():
    """Verify loading default configs/config.yaml."""
    config = load_config()
    assert isinstance(config, AppConfig)
    assert config.project.name == "MachSense"
    assert config.project.environment in ["development", "testing", "production"]
    assert len(config.features.numerical_features) > 0
    assert config.data.target_column == "machine_failure"


def test_custom_config_loading(temp_config_file: Path):
    """Verify loading configuration from a custom YAML file path."""
    config = load_config(temp_config_file)
    assert config.project.name == "MachSenseTest"
    assert config.project.environment == "testing"
    assert config.project.random_seed == 99
    assert config.features.numerical_features == ["temp", "speed"]
    assert config.model.name == "test_model"


def test_env_variable_override(temp_config_file: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify environment variables override configuration."""
    monkeypatch.setenv("MACHSENSE_ENV", "staging")
    config = load_config(temp_config_file)
    assert config.project.environment == "staging"


def test_resolve_path_helper():
    """Verify resolve_path returns valid absolute paths without hardcoding."""
    config = get_settings()
    root = get_project_root()

    raw_path = config.resolve_path("raw_data_dir")
    assert raw_path.is_absolute()
    assert raw_path == (root / "data" / "raw").resolve()

    models_path = config.resolve_path("models_dir")
    assert models_path.is_absolute()
    assert models_path == (root / "models").resolve()


def test_resolve_path_invalid_attribute():
    """Verify resolving a non-existent path field raises AttributeError."""
    config = get_settings()
    with pytest.raises(AttributeError):
        config.resolve_path("non_existent_path_key")
