"""Configuration management and dynamic path resolution for MachSense."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field


def get_project_root() -> Path:
    """Dynamically determine the project root directory.

    Resolution order:
    1. MACHSENSE_ROOT environment variable (if set).
    2. Ascending search from current file for 'pyproject.toml' or 'configs'.
    3. Fallback to 3 levels up from this file (src/machsense/config -> root).
    """
    env_root = os.environ.get("MACHSENSE_ROOT")
    if env_root:
        return Path(env_root).resolve()

    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / "pyproject.toml").exists() or (parent / "configs" / "config.yaml").exists():
            return parent

    return current_path.parents[2]


class ProjectConfig(BaseModel):
    name: str = "MachSense"
    version: str = "0.1.0"
    environment: str = "development"
    random_seed: int = 42


class PathConfig(BaseModel):
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    interim_data_dir: str = "data/interim"
    processed_data_dir: str = "data/processed"
    models_dir: str = "models"
    logs_dir: str = "logs"
    configs_dir: str = "configs"

    def get_absolute_path(self, relative_field: str, base_dir: Optional[Path] = None) -> Path:
        """Resolve a relative path field to an absolute Path against project root."""
        root = base_dir or get_project_root()
        rel_path = getattr(self, relative_field, None)
        if rel_path is None:
            raise AttributeError(f"Path field '{relative_field}' does not exist on PathConfig.")
        return (root / rel_path).resolve()


class DataConfig(BaseModel):
    target_column: str = "machine_failure"
    id_column: str = "udi"
    timestamp_column: str = "timestamp"
    sampling_rate_hz: int = 10
    validation_split: float = 0.2
    test_split: float = 0.1


class FeaturesConfig(BaseModel):
    numerical_features: List[str] = Field(
        default_factory=lambda: [
            "air_temperature_k",
            "process_temperature_k",
            "rotational_speed_rpm",
            "torque_nm",
            "tool_wear_min",
        ]
    )
    categorical_features: List[str] = Field(default_factory=lambda: ["type"])
    scaling_method: str = "standard"


class ModelParamsConfig(BaseModel):
    n_estimators: int = 100
    max_depth: Optional[int] = 10
    random_state: int = 42
    class_weight: str = "balanced"


class ExplainabilityConfig(BaseModel):
    enabled: bool = True
    method: str = "shap"


class ModelConfig(BaseModel):
    name: str = "random_forest_baseline"
    type: str = "classifier"
    params: ModelParamsConfig = Field(default_factory=ModelParamsConfig)
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)


class ServingConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    reload: bool = True


class AppConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)

    def resolve_path(self, path_attr: str) -> Path:
        """Helper to get resolved absolute path for a configured path attribute."""
        return self.paths.get_absolute_path(path_attr, get_project_root())


def load_config(config_path: Optional[Path | str] = None) -> AppConfig:
    """Load configuration from YAML file and environment overrides."""
    root_dir = get_project_root()

    if config_path is None:
        env_config = os.environ.get("MACHSENSE_CONFIG_PATH")
        if env_config:
            config_file = Path(env_config)
        else:
            config_file = root_dir / "configs" / "config.yaml"
    else:
        config_file = Path(config_path)

    if not config_file.is_absolute():
        config_file = (root_dir / config_file).resolve()

    if not config_file.exists():
        # Return default config if file is not found
        return AppConfig()

    with open(config_file, "r", encoding="utf-8") as f:
        raw_dict: Dict[str, Any] = yaml.safe_load(f) or {}

    # Environment overrides
    env_name = os.environ.get("MACHSENSE_ENV")
    if env_name and "project" in raw_dict:
        raw_dict["project"]["environment"] = env_name

    return AppConfig(**raw_dict)


@lru_cache(maxsize=1)
def get_settings() -> AppConfig:
    """Cached singleton getter for active application settings."""
    return load_config()
