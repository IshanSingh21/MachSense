"""Comprehensive End-to-End System Integration Tests for MachSense (Day 12).

Verifies full connectivity across:
Streamlit UI Client <-> FastAPI Backend <-> Prediction Pipeline <-> Preprocessing <-> ML Model <-> TreeSHAP Explainability.
"""

import os
from pathlib import Path
import tempfile
from unittest.mock import patch
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from machsense.api.app import app
from machsense.inference.pipeline import MachSensePredictionService
from machsense.inference.schema import InferenceStatus, RiskLevel
from machsense.models.registry import ModelRegistry
from machsense.ui.client import TELEMETRY_PRESETS, MachSenseUIClient


@pytest.fixture(scope="module")
def api_client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module")
def prediction_service():
    """MachSensePredictionService fixture."""
    return MachSensePredictionService()


@pytest.fixture(scope="module")
def ui_client(prediction_service):
    """MachSenseUIClient fixture."""
    return MachSenseUIClient(service=prediction_service)


# ---------------------------------------------------------------------------
# 1. Valid End-to-End Prediction Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", list(TELEMETRY_PRESETS.keys()))
def test_e2e_valid_predictions_all_failure_presets(ui_client, preset_name):
    """Test full inference flow for each physical failure preset."""
    preset = TELEMETRY_PRESETS[preset_name]
    result = ui_client.predict_single(preset, explain=True)

    assert result.status == InferenceStatus.SUCCESS
    assert 0.0 <= result.failure_probability <= 1.0
    assert result.predicted_class in (0, 1)
    assert result.predicted_label in ("HEALTHY", "FAILURE_IMMINENT")
    assert result.risk_level in (RiskLevel.NOMINAL, RiskLevel.MODERATE_WARNING, RiskLevel.ELEVATED, RiskLevel.CRITICAL)
    assert result.latency_ms > 0.0
    assert result.operator_summary is not None
    assert "DISCLAIMER" in result.non_causal_disclaimer


def test_e2e_fastapi_to_model_pipeline(api_client):
    """Test full HTTP API request reaching ML model and SHAP explainer."""
    payload = {
        "telemetry": {
            "type": "L",
            "air_temperature_k": 302.5,
            "process_temperature_k": 311.8,
            "rotational_speed_rpm": 1380.0,
            "torque_nm": 62.5,
            "tool_wear_min": 215.0,
        },
        "explain": True,
    }
    response = api_client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["predicted_class"] == 1
    assert data["predicted_label"] == "FAILURE_IMMINENT"
    assert data["risk_level"] == "CRITICAL"
    assert data["failure_probability"] >= 0.70
    assert len(data["top_risk_escalators"]) > 0


# ---------------------------------------------------------------------------
# 2. Invalid Input & Missing Value Rejection
# ---------------------------------------------------------------------------


def test_e2e_missing_sensor_fields(ui_client, api_client):
    """Verify missing required sensor fields are rejected with clean error messages."""
    incomplete = {
        "type": "M",
        "air_temperature_k": 298.1,
        # Missing process temp, torque, speed, wear
    }

    # UI Client In-Process
    ui_res = ui_client.predict_single(incomplete)
    assert ui_res.status == InferenceStatus.VALIDATION_ERROR
    assert len(ui_res.errors) > 0
    assert any("Missing required sensor fields" in e for e in ui_res.errors)

    # API Endpoint
    api_res = api_client.post("/api/v1/predict", json={"telemetry": incomplete})
    assert api_res.status_code == 422
    assert api_res.json()["error_code"] == "SCHEMA_VALIDATION_ERROR"


def test_e2e_unexpected_string_in_numeric_field(api_client):
    """Verify non-numeric strings in float fields return 422."""
    bad_type = {
        "telemetry": {
            "type": "L",
            "air_temperature_k": "NOT_A_TEMPERATURE",
            "process_temperature_k": 310.0,
            "rotational_speed_rpm": 1500.0,
            "torque_nm": 40.0,
            "tool_wear_min": 10.0,
        }
    }
    response = api_client.post("/api/v1/predict", json=bad_type)
    assert response.status_code == 422


def test_e2e_thermodynamic_violation_handling(ui_client, api_client):
    """Verify physical thermodynamic violations are caught with user-friendly notices."""
    cold_payload = {
        "type": "M",
        "air_temperature_k": 305.0,
        "process_temperature_k": 280.0,  # 25 K colder than air
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 40.0,
        "tool_wear_min": 20.0,
    }

    # UI Client
    ui_res = ui_client.predict_single(cold_payload)
    assert ui_res.status == InferenceStatus.VALIDATION_ERROR
    assert any("Thermodynamic violation" in e for e in ui_res.errors)

    # API Endpoint
    api_res = api_client.post("/api/v1/predict", json={"telemetry": cold_payload})
    assert api_res.status_code in (400, 422)


# ---------------------------------------------------------------------------
# 3. Model Loading & API Failure Resilience
# ---------------------------------------------------------------------------


def test_e2e_model_loading_failure_handling():
    """Verify handling of missing model directory."""
    with tempfile.TemporaryDirectory() as empty_dir:
        with pytest.raises(FileNotFoundError):
            MachSensePredictionService(models_dir=empty_dir)


def test_e2e_ui_client_api_failure_fallback(ui_client):
    """Verify UI client falls back cleanly to in-process service when API endpoint is unreachable."""
    # Configure unreachable API URL
    fallback_client = MachSenseUIClient(
        service=ui_client._service,
        api_url="http://127.0.0.1:9999",  # Unreachable port
        prefer_api=True,
    )

    preset = TELEMETRY_PRESETS["Nominal Healthy Machine"]
    # Must not raise connection exception; falls back to in-process
    result = fallback_client.predict_single(preset)
    assert result.status == InferenceStatus.SUCCESS
    assert result.predicted_class == 0


# ---------------------------------------------------------------------------
# 4. Error Sanitization & Security Checks
# ---------------------------------------------------------------------------


def test_e2e_error_message_sanitization(ui_client):
    """Verify error messages do not leak local file system paths or passwords."""
    invalid_input = {
        "type": "INVALID_VARIANT",
        "air_temperature_k": 300.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": -999.0,
        "torque_nm": -50.0,
        "tool_wear_min": -10.0,
    }
    res = ui_client.predict_single(invalid_input)
    assert res.status == InferenceStatus.VALIDATION_ERROR

    for err in res.errors:
        # Check no hardcoded file paths leaked
        assert "C:\\" not in err
        assert "/Users/" not in err
        assert ".joblib" not in err
        assert "Traceback" not in err


# ---------------------------------------------------------------------------
# 5. Invariance & Zero Leakage Under Repeated Execution
# ---------------------------------------------------------------------------


def test_e2e_zero_leakage_and_immutability(prediction_service):
    """Verify that scoring multiple diverse records never mutates preprocessor or model state."""
    initial_feature_names = list(prediction_service.preprocessor.feature_names_out_)
    initial_is_fitted = prediction_service.preprocessor.is_fitted_

    # Run multiple predictions across all presets
    for _ in range(10):
        for preset in TELEMETRY_PRESETS.values():
            prediction_service.predict(preset)

    # Invariants
    assert prediction_service.preprocessor.is_fitted_ == initial_is_fitted
    assert list(prediction_service.preprocessor.feature_names_out_) == initial_feature_names
