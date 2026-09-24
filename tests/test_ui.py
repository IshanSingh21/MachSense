"""Unit tests for MachSense UI client, presets, and presentation data structures."""

import pandas as pd
import pytest

from machsense.inference.schema import InferenceStatus, RiskLevel
from machsense.ui.client import TELEMETRY_PRESETS, MachSenseUIClient


@pytest.fixture
def ui_client():
    """Create a MachSenseUIClient instance."""
    return MachSenseUIClient()


def test_ui_client_initialization(ui_client):
    """Test MachSenseUIClient initializes and exposes service metadata."""
    info = ui_client.service_info
    assert isinstance(info, dict)
    assert "model_name" in info
    assert "model_version" in info
    assert info["explainer_ready"] is True
    assert 0.0 < ui_client.optimal_threshold < 1.0


def test_telemetry_presets_structure():
    """Verify all domain failure presets contain valid required keys."""
    required_keys = {"type", "air_temperature_k", "process_temperature_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min"}

    assert len(TELEMETRY_PRESETS) >= 5
    for name, preset in TELEMETRY_PRESETS.items():
        assert required_keys.issubset(set(preset.keys())), f"Preset {name} is missing required sensor keys."
        assert preset["type"] in ("L", "M", "H")
        assert 270.0 <= preset["air_temperature_k"] <= 350.0
        assert 280.0 <= preset["process_temperature_k"] <= 360.0
        assert 500.0 <= preset["rotational_speed_rpm"] <= 4500.0
        assert 0.0 <= preset["torque_nm"] <= 150.0
        assert 0.0 <= preset["tool_wear_min"] <= 500.0


def test_ui_client_predict_single_nominal(ui_client):
    """Test single prediction on Nominal preset."""
    nominal_preset = TELEMETRY_PRESETS["Nominal Healthy Machine"]
    result = ui_client.predict_single(nominal_preset, explain=True)

    assert result.status == InferenceStatus.SUCCESS
    assert result.predicted_class == 0
    assert result.predicted_label == "HEALTHY"
    assert result.risk_level in (RiskLevel.NOMINAL, RiskLevel.MODERATE_WARNING)
    assert result.failure_probability < 0.50
    assert result.operator_summary is not None
    assert len(result.top_stabilizers) > 0


def test_ui_client_predict_single_overstrain_failure(ui_client):
    """Test single prediction on Overstrain Failure preset."""
    osf_preset = TELEMETRY_PRESETS["Overstrain Failure (OSF)"]
    result = ui_client.predict_single(osf_preset, explain=True)

    assert result.status == InferenceStatus.SUCCESS
    assert result.predicted_class == 1
    assert result.predicted_label == "FAILURE_IMMINENT"
    assert result.risk_level == RiskLevel.CRITICAL
    assert result.failure_probability >= 0.70
    assert len(result.top_risk_escalators) > 0
    assert "CRITICAL ALERT" in result.operator_summary


def test_ui_client_predict_batch_dataframe(ui_client):
    """Test batch prediction across multiple preset rows."""
    preset_rows = [
        TELEMETRY_PRESETS["Nominal Healthy Machine"],
        TELEMETRY_PRESETS["Heat Dissipation Failure (HDF)"],
        TELEMETRY_PRESETS["Power Failure (PWF)"],
        TELEMETRY_PRESETS["Overstrain Failure (OSF)"],
        TELEMETRY_PRESETS["Tool Wear Failure (TWF)"],
    ]
    df = pd.DataFrame(preset_rows)
    batch_res = ui_client.predict_batch(df, explain=False)

    assert batch_res.status == InferenceStatus.SUCCESS
    assert batch_res.total_records == 5
    assert batch_res.successful_predictions_count == 5
    assert batch_res.failure_count >= 3  # At least HDF, PWF, OSF flagged
    assert len(batch_res.predictions) == 5


def test_ui_client_handles_invalid_input(ui_client):
    """Test UI client handles invalid input gracefully."""
    invalid_payload = {
        "type": "INVALID",
        "air_temperature_k": 300.0,
        # missing other fields
    }
    result = ui_client.predict_single(invalid_payload)

    assert result.status == InferenceStatus.VALIDATION_ERROR
    assert len(result.errors) > 0
