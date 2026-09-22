"""Unit and integration tests for MachSense production inference pipeline."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from machsense.data.schema import MachineType
from machsense.inference.pipeline import MachSensePredictionService
from machsense.inference.schema import (
    BatchPredictionResult,
    InferenceStatus,
    PredictionResult,
    RiskLevel,
    SensorPayload,
)
from machsense.inference.validator import InferenceValidator
from machsense.models.registry import ModelRegistry


@pytest.fixture
def valid_nominal_payload():
    """Nominal healthy machine telemetry payload."""
    return {
        "type": "M",
        "air_temperature_k": 298.1,
        "process_temperature_k": 308.6,
        "rotational_speed_rpm": 1550.0,
        "torque_nm": 40.0,
        "tool_wear_min": 30.0,
    }


@pytest.fixture
def valid_critical_failure_payload():
    """Critical failure telemetry payload (high torque + high wear -> Overstrain Failure)."""
    return {
        "type": "L",
        "air_temperature_k": 302.5,
        "process_temperature_k": 311.8,
        "rotational_speed_rpm": 1380.0,
        "torque_nm": 62.5,
        "tool_wear_min": 215.0,
    }


@pytest.fixture
def prediction_service():
    """Initialize MachSensePredictionService loaded from registered champion model."""
    return MachSensePredictionService()


# ---------------------------------------------------------------------------
# 1. Validator Tests
# ---------------------------------------------------------------------------


def test_validator_valid_dictionary_payload(valid_nominal_payload):
    """Test validator accepts valid dictionary telemetry."""
    val = InferenceValidator.validate_payload(valid_nominal_payload)
    assert val.is_valid is True
    assert len(val.errors) == 0
    assert val.cleaned_data is not None
    assert val.cleaned_data["type"] == MachineType.MEDIUM or val.cleaned_data["type"] == "M"


def test_validator_valid_pydantic_payload(valid_nominal_payload):
    """Test validator accepts valid SensorPayload instance."""
    payload = SensorPayload(**valid_nominal_payload)
    val = InferenceValidator.validate_payload(payload)
    assert val.is_valid is True
    assert len(val.errors) == 0


def test_validator_missing_fields():
    """Test validator rejects incomplete payloads."""
    incomplete = {
        "type": "M",
        "air_temperature_k": 298.1,
        # missing torque, speed, wear
    }
    val = InferenceValidator.validate_payload(incomplete)
    assert val.is_valid is False
    assert any("Missing required sensor fields" in err for err in val.errors)


def test_validator_null_or_nan_values():
    """Test validator rejects null, NaN, or infinite values."""
    nan_payload = {
        "type": "M",
        "air_temperature_k": float("nan"),
        "process_temperature_k": 308.6,
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 40.0,
        "tool_wear_min": 30.0,
    }
    val = InferenceValidator.validate_payload(nan_payload)
    assert val.is_valid is False
    assert any("cannot be null, NaN, or infinite" in err for err in val.errors)


def test_validator_physical_boundaries():
    """Test validator rejects out-of-boundary sensor measurements."""
    # Negative torque
    bad_torque = {
        "type": "M",
        "air_temperature_k": 298.1,
        "process_temperature_k": 308.6,
        "rotational_speed_rpm": 1500.0,
        "torque_nm": -10.0,
        "tool_wear_min": 30.0,
    }
    val = InferenceValidator.validate_payload(bad_torque)
    assert val.is_valid is False
    assert any("torque_nm" in err for err in val.errors)

    # Excessive speed
    bad_speed = {
        "type": "M",
        "air_temperature_k": 298.1,
        "process_temperature_k": 308.6,
        "rotational_speed_rpm": 9000.0,
        "torque_nm": 40.0,
        "tool_wear_min": 30.0,
    }
    val2 = InferenceValidator.validate_payload(bad_speed)
    assert val2.is_valid is False
    assert any("rotational_speed_rpm" in err for err in val2.errors)


def test_validator_thermodynamic_violation():
    """Test validator rejects violations where process temp is colder than air temp."""
    cold_process = {
        "type": "M",
        "air_temperature_k": 305.0,
        "process_temperature_k": 290.0,  # 15 K colder than ambient
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 40.0,
        "tool_wear_min": 30.0,
    }
    val = InferenceValidator.validate_payload(cold_process)
    assert val.is_valid is False
    assert any("Thermodynamic violation" in err for err in val.errors)


# ---------------------------------------------------------------------------
# 2. Prediction Service End-to-End Tests
# ---------------------------------------------------------------------------


def test_prediction_service_nominal_prediction(prediction_service, valid_nominal_payload):
    """Test end-to-end inference on nominal healthy operating machine."""
    result = prediction_service.predict(valid_nominal_payload, explain=True)

    assert result.status == InferenceStatus.SUCCESS
    assert result.predicted_class == 0
    assert result.predicted_label == "HEALTHY"
    assert result.risk_level in (RiskLevel.NOMINAL, RiskLevel.MODERATE_WARNING)
    assert 0.0 <= result.failure_probability < 0.5
    assert result.threshold_used == prediction_service.optimal_threshold
    assert result.latency_ms > 0.0
    assert result.operator_summary is not None
    assert "NOMINAL STATE" in result.operator_summary or "MODERATE WARNING" in result.operator_summary
    assert len(result.errors) == 0


def test_prediction_service_critical_failure_prediction(prediction_service, valid_critical_failure_payload):
    """Test end-to-end inference on critical failure condition."""
    result = prediction_service.predict(valid_critical_failure_payload, explain=True)

    assert result.status == InferenceStatus.SUCCESS
    assert result.predicted_class == 1
    assert result.predicted_label == "FAILURE_IMMINENT"
    assert result.risk_level == RiskLevel.CRITICAL
    assert result.failure_probability >= 0.70
    assert len(result.top_risk_escalators) > 0
    assert result.operator_summary is not None
    assert "CRITICAL ALERT" in result.operator_summary


def test_prediction_service_invalid_input_graceful_handling(prediction_service):
    """Test prediction service handles invalid input gracefully without unhandled exceptions."""
    corrupted_payload = {
        "type": "INVALID_TYPE",
        "air_temperature_k": 300.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": -500.0,  # invalid negative speed
        "torque_nm": 40.0,
        "tool_wear_min": 10.0,
    }
    result = prediction_service.predict(corrupted_payload)

    assert result.status == InferenceStatus.VALIDATION_ERROR
    assert result.predicted_class is None
    assert result.failure_probability is None
    assert len(result.errors) > 0
    assert result.latency_ms >= 0.0


def test_prediction_service_custom_threshold(prediction_service, valid_nominal_payload):
    """Test overriding decision threshold during inference."""
    # With standard threshold (~0.56), nominal payload is class 0
    res_default = prediction_service.predict(valid_nominal_payload)
    assert res_default.predicted_class == 0

    # With an artificially ultra-low threshold (0.0001), it triggers class 1
    res_custom = prediction_service.predict(valid_nominal_payload, threshold=0.0001)
    assert res_custom.threshold_used == 0.0001
    assert res_custom.predicted_class == 1
    assert res_custom.predicted_label == "FAILURE_IMMINENT"


def test_prediction_service_batch_prediction(prediction_service):
    """Test high-throughput batch DataFrame inference."""
    # Create synthetic test batch of 20 records
    np.random.seed(42)
    batch_df = pd.DataFrame({
        "type": ["L", "M", "H", "L"] * 5,
        "air_temperature_k": np.random.uniform(295.0, 305.0, 20),
        "process_temperature_k": np.random.uniform(306.0, 315.0, 20),
        "rotational_speed_rpm": np.random.uniform(1200.0, 2000.0, 20),
        "torque_nm": np.random.uniform(20.0, 70.0, 20),
        "tool_wear_min": np.random.uniform(0.0, 250.0, 20),
    })

    batch_result = prediction_service.predict_batch(batch_df, explain=False)

    assert isinstance(batch_result, BatchPredictionResult)
    assert batch_result.status == InferenceStatus.SUCCESS
    assert batch_result.total_records == 20
    assert batch_result.successful_predictions_count == 20
    assert len(batch_result.predictions) == 20
    assert 0.0 <= batch_result.mean_failure_probability <= 1.0
    assert batch_result.total_latency_ms > 0.0


def test_prediction_service_zero_retraining_invariant(prediction_service, valid_nominal_payload):
    """Verify that calling predict does not modify preprocessor fitted attributes or retrain the model."""
    orig_features = list(prediction_service.preprocessor.feature_names_out_)
    orig_is_fitted = prediction_service.preprocessor.is_fitted_

    # Execute several predictions
    for _ in range(5):
        prediction_service.predict(valid_nominal_payload)

    # Assert invariant: preprocessor was never refitted
    assert prediction_service.preprocessor.is_fitted_ == orig_is_fitted
    assert list(prediction_service.preprocessor.feature_names_out_) == orig_features


def test_prediction_service_info_metadata(prediction_service):
    """Test get_service_info returns expected operational dictionary."""
    info = prediction_service.get_service_info()
    assert isinstance(info, dict)
    assert "model_name" in info
    assert "model_version" in info
    assert "optimal_threshold" in info
    assert info["explainer_ready"] is True
