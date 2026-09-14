"""Pytest fixtures for MachSense test suite."""

import os
import sys
from pathlib import Path
import pytest

# Ensure src is in python path for test runs
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def project_root() -> Path:
    """Fixture returning the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Fixture creating a temporary valid YAML config file."""
    config_content = """
project:
  name: "MachSenseTest"
  version: "0.1.0-test"
  environment: "testing"
  random_seed: 99

paths:
  data_dir: "test_data"
  raw_data_dir: "test_data/raw"
  interim_data_dir: "test_data/interim"
  processed_data_dir: "test_data/processed"
  models_dir: "test_models"
  logs_dir: "test_logs"
  configs_dir: "configs"

data:
  target_column: "machine_failure"
  id_column: "udi"
  timestamp_column: "timestamp"
  sampling_rate_hz: 1
  validation_split: 0.2
  test_split: 0.1

features:
  numerical_features:
    - "temp"
    - "speed"
  categorical_features:
    - "type"
  scaling_method: "standard"

model:
  name: "test_model"
  type: "classifier"
  params:
    n_estimators: 10
    max_depth: 2
    random_state: 99
    class_weight: "balanced"
  explainability:
    enabled: false
    method: "shap"

serving:
  host: "127.0.0.1"
  port: 9000
  workers: 1
  reload: false
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content.strip(), encoding="utf-8")
    return config_file
