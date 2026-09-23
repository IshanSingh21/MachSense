"""Unit and integration tests for MachSense FastAPI backend application."""

import pytest
from fastapi.testclient import TestClient

from machsense.api.app import app, create_app


@pytest.fixture(scope="module")
def client():
    """Create a TestClient instance managing startup and shutdown lifespan."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def nominal_payload():
    """Nominal healthy machine telemetry payload."""
    return {
        "telemetry": {
            "type": "M",
            "air_temperature_k": 298.1,
            "process_temperature_k": 308.6,
            "rotational_speed_rpm": 1550.0,
            "torque_nm": 40.0,
            "tool_wear_min": 30.0,
        },
        "explain": True,
    }


@pytest.fixture
def critical_failure_payload():
    """Critical failure telemetry payload (Overstrain Failure regime)."""
    return {
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


def test_root_endpoint(client):
    """Verify root endpoint returns system metadata and doc URLs."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "MachSense API"
    assert "documentation_url" in data
    assert data["documentation_url"] == "/docs"
    assert data["health_url"] == "/health"
    assert data["predict_url"] == "/api/v1/predict"


def test_health_liveness_endpoint(client):
    """Verify /health returns 200 OK and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data


def test_health_readiness_endpoint(client):
    """Verify /health/ready returns 200 OK and model is loaded."""
    response = client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    assert data["model_name"] is not None
    assert data["model_version"] is not None
    assert data["optimal_threshold"] is not None
    assert data["explainer_ready"] is True


def test_predict_single_nominal(client, nominal_payload):
    """Verify /api/v1/predict on nominal healthy machine telemetry."""
    response = client.post("/api/v1/predict", json=nominal_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["predicted_class"] == 0
    assert data["predicted_label"] == "HEALTHY"
    assert data["risk_level"] in ("NOMINAL", "MODERATE_WARNING")
    assert 0.0 <= data["failure_probability"] < 0.5
    assert data["latency_ms"] > 0.0
    assert data["operator_summary"] is not None
    assert "DISCLAIMER" in data["non_causal_disclaimer"]


def test_predict_single_critical_failure(client, critical_failure_payload):
    """Verify /api/v1/predict on critical machine failure condition."""
    response = client.post("/api/v1/predict", json=critical_failure_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["predicted_class"] == 1
    assert data["predicted_label"] == "FAILURE_IMMINENT"
    assert data["risk_level"] == "CRITICAL"
    assert data["failure_probability"] >= 0.70
    assert len(data["top_risk_escalators"]) > 0
    assert "CRITICAL ALERT" in data["operator_summary"]


def test_predict_single_schema_validation_error(client):
    """Verify /api/v1/predict returns 422 for missing required fields."""
    incomplete_payload = {
        "telemetry": {
            "type": "M",
            "air_temperature_k": 298.1,
            # Missing process_temperature, speed, torque, wear
        }
    }
    response = client.post("/api/v1/predict", json=incomplete_payload)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "ERROR"
    assert data["error_code"] == "SCHEMA_VALIDATION_ERROR"
    assert len(data["details"]) > 0


def test_predict_single_thermodynamic_violation(client):
    """Verify /api/v1/predict returns 400 for physical thermodynamic violations."""
    cold_process_payload = {
        "telemetry": {
            "type": "M",
            "air_temperature_k": 305.0,
            "process_temperature_k": 285.0,  # 20 K below ambient
            "rotational_speed_rpm": 1500.0,
            "torque_nm": 40.0,
            "tool_wear_min": 30.0,
        }
    }
    response = client.post("/api/v1/predict", json=cold_process_payload)
    assert response.status_code == 422 or response.status_code == 400
    data = response.json()
    assert data["status"] == "ERROR"


def test_predict_single_custom_threshold(client, nominal_payload):
    """Verify custom decision threshold override in API request."""
    # Custom threshold 0.0001 forces classification to positive
    nominal_payload["threshold"] = 0.0001
    response = client.post("/api/v1/predict", json=nominal_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == 1
    assert data["predicted_label"] == "FAILURE_IMMINENT"
    assert data["threshold_used"] == 0.0001


def test_predict_batch_endpoint(client):
    """Verify /api/v1/predict/batch executes high-throughput batch inference."""
    items = [
        {
            "type": "L" if i % 2 == 0 else "M",
            "air_temperature_k": 298.0 + (i * 0.2),
            "process_temperature_k": 308.0 + (i * 0.2),
            "rotational_speed_rpm": 1400.0 + (i * 20),
            "torque_nm": 35.0 + (i * 1.5),
            "tool_wear_min": 10.0 + (i * 15),
        }
        for i in range(10)
    ]
    batch_request = {"items": items, "explain": False}

    response = client.post("/api/v1/predict/batch", json=batch_request)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["total_records"] == 10
    assert data["successful_predictions_count"] == 10
    assert len(data["predictions"]) == 10
    assert data["total_latency_ms"] > 0.0


def test_predict_batch_empty_items_validation(client):
    """Verify /api/v1/predict/batch rejects empty item list."""
    response = client.post("/api/v1/predict/batch", json={"items": []})
    assert response.status_code == 422


def test_openapi_json_schema(client):
    """Verify OpenAPI JSON schema generation."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "paths" in schema
    assert "/api/v1/predict" in schema["paths"]
    assert "/health" in schema["paths"]
    assert "/health/ready" in schema["paths"]


def test_cors_headers(client):
    """Verify CORS middleware headers are present on requests."""
    response = client.options(
        "/api/v1/predict",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
