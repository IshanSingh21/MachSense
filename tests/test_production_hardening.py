"""Production hardening and security verification test suite for MachSense (Day 13)."""

import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from machsense.api.app import app, create_app
from machsense.config.settings import AppConfig, ServingConfig
from machsense.inference.pipeline import MachSensePredictionService
from machsense.inference.schema import InferenceStatus
from machsense.inference.validator import InferenceValidator
from machsense.models.registry import ModelRegistry, validate_version_tag
from machsense.ui.client import TELEMETRY_PRESETS


@pytest.fixture(scope="module")
def prediction_service():
    """Prediction service fixture."""
    return MachSensePredictionService()


@pytest.fixture(scope="module")
def api_client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# 1. Path Traversal & SemVer Defense Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "malicious_version",
    [
        "../evil",
        "../../etc/passwd",
        "..\\..\\windows\\system32",
        "v1.0.0; rm -rf /",
        "v1.0.0/../../../",
        "invalid_version",
        "1.0",
        "v1",
        "v1.0.0.0",
    ],
)
def test_model_registry_path_traversal_defense(malicious_version):
    """Verify that malicious or non-semver version strings are rejected immediately."""
    registry = ModelRegistry()

    # validate_version_tag directly
    with pytest.raises(ValueError) as excinfo:
        validate_version_tag(malicious_version)
    assert "Invalid semantic version tag" in str(excinfo.value)

    # load_versioned_artifacts
    with pytest.raises(ValueError):
        registry.load_versioned_artifacts(version=malicious_version)


def test_valid_version_tags():
    """Verify valid semantic version formats normalize correctly."""
    assert validate_version_tag("v1.0.0") == "v1.0.0"
    assert validate_version_tag("1.0.0") == "v1.0.0"
    assert validate_version_tag("V2.1.3") == "v2.1.3"


# ---------------------------------------------------------------------------
# 2. CORS Security Configuration Tests
# ---------------------------------------------------------------------------


def test_cors_wildcard_credentials_disallowed():
    """Verify that wildcard CORS origins securely disallow credentials."""
    cfg = AppConfig(serving=ServingConfig(cors_origins=["*"]))
    test_app = create_app(config=cfg)

    # Check CORS middleware kwargs in test app
    cors_mw = [mw for mw in test_app.user_middleware if "CORSMiddleware" in str(mw.cls)]
    assert len(cors_mw) == 1
    assert cors_mw[0].kwargs["allow_credentials"] is False


def test_cors_explicit_origin_credentials_allowed():
    """Verify that explicit CORS origins allow credentials securely."""
    cfg = AppConfig(serving=ServingConfig(cors_origins=["http://localhost:8501", "http://127.0.0.1:8501"]))
    test_app = create_app(config=cfg)

    cors_mw = [mw for mw in test_app.user_middleware if "CORSMiddleware" in str(mw.cls)]
    assert len(cors_mw) == 1
    assert cors_mw[0].kwargs["allow_credentials"] is True


# ---------------------------------------------------------------------------
# 3. Batch Upper Bound & DoS Protection
# ---------------------------------------------------------------------------


def test_batch_upper_bound_dos_protection():
    """Verify that batch DataFrames exceeding 5,000 records are rejected to protect from memory exhaustion."""
    oversized_size = 5001
    large_df = pd.DataFrame({
        "type": ["L"] * oversized_size,
        "air_temperature_k": [300.0] * oversized_size,
        "process_temperature_k": [310.0] * oversized_size,
        "rotational_speed_rpm": [1500.0] * oversized_size,
        "torque_nm": [40.0] * oversized_size,
        "tool_wear_min": [10.0] * oversized_size,
    })

    is_valid, errors, _ = InferenceValidator.validate_dataframe(large_df)
    assert is_valid is False
    assert any("exceeds maximum permissible limit" in err for err in errors)


# ---------------------------------------------------------------------------
# 4. Error Sanitization & Information Leakage Prevention
# ---------------------------------------------------------------------------


def test_error_response_sanitization(api_client):
    """Verify that 400 and 422 error responses do not leak local file paths or passwords."""
    invalid_request = {
        "telemetry": {
            "type": "M",
            "air_temperature_k": 310.0,
            "process_temperature_k": 280.0,  # Thermodynamic violation
            "rotational_speed_rpm": 1500.0,
            "torque_nm": 40.0,
            "tool_wear_min": 10.0,
        }
    }
    response = api_client.post("/api/v1/predict", json=invalid_request)
    assert response.status_code in (400, 422)

    raw_text = response.text
    # Ensure no internal path leaks
    assert "C:\\" not in raw_text
    assert "/Users/" not in raw_text
    assert "/home/" not in raw_text
    assert "password" not in raw_text.lower()
    assert "secret" not in raw_text.lower()
    assert "Traceback" not in raw_text


# ---------------------------------------------------------------------------
# 5. Performance & Latency Benchmarking
# ---------------------------------------------------------------------------


def test_single_prediction_latency_benchmark(prediction_service):
    """Verify single instance inference latency benchmark (< 100 ms)."""
    nominal_preset = TELEMETRY_PRESETS["Nominal Healthy Machine"]

    # Warm-up pass
    prediction_service.predict(nominal_preset, explain=True)

    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        res = prediction_service.predict(nominal_preset, explain=True)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        assert res.status == InferenceStatus.SUCCESS

    mean_latency = np.mean(latencies)
    assert mean_latency < 100.0, f"Mean single inference latency {mean_latency:.2f} ms exceeded 100 ms SLA."
