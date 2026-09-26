"""Runtime validation engine for incoming manufacturing telemetry."""

from __future__ import annotations

from typing import Any, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from machsense.data.schema import MachineType, SensorBoundaries
from machsense.inference.schema import SensorPayload, ValidationResult
from machsense.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_SENSOR_FIELDS: set[str] = {
    "type",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
}


class InferenceValidator:
    """Validates real-time sensor payloads and batch DataFrames against domain physical boundaries."""

    @classmethod
    def validate_payload(cls, payload: Union[dict[str, Any], SensorPayload]) -> ValidationResult:
        """Validate a single incoming sensor telemetry payload.

        Args:
            payload: Raw dictionary or SensorPayload instance.

        Returns:
            ValidationResult containing status, cleaned dictionary, and any errors.
        """
        if isinstance(payload, SensorPayload):
            return ValidationResult(
                is_valid=True,
                cleaned_data=payload.model_dump(),
                errors=[],
                warnings=[],
            )

        if not isinstance(payload, dict):
            return ValidationResult(
                is_valid=False,
                errors=[f"Expected dictionary payload, got {type(payload).__name__}."],
                cleaned_data=None,
            )

        errors: list[str] = []
        warnings: list[str] = []

        # 1. Check for missing required keys
        missing_keys = REQUIRED_SENSOR_FIELDS - set(payload.keys())
        if missing_keys:
            errors.append(f"Missing required sensor fields: {sorted(list(missing_keys))}")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # 2. Check for null or NaN values in required fields
        for field_name in REQUIRED_SENSOR_FIELDS:
            val = payload.get(field_name)
            if val is None or (isinstance(val, (float, int)) and (np.isnan(val) or np.isinf(val))):
                errors.append(f"Field '{field_name}' cannot be null, NaN, or infinite.")

        if errors:
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # 3. Pydantic validation for domain constraints & physical boundaries
        try:
            validated_record = SensorPayload(**payload)
            cleaned = validated_record.model_dump()
            return ValidationResult(is_valid=True, cleaned_data=cleaned, errors=[], warnings=warnings)
        except ValidationError as e:
            for err in e.errors():
                loc = ".".join(str(item) for item in err["loc"])
                msg = err["msg"]
                errors.append(f"Validation error on '{loc}': {msg}")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        except Exception as exc:
            errors.append(f"Unexpected validation error: {str(exc)}")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    MAX_BATCH_SIZE: int = 5000

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> Tuple[bool, list[str], pd.DataFrame]:
        """Validate a batch DataFrame before feeding into batch inference.

        Args:
            df: Raw input DataFrame.

        Returns:
            Tuple of (is_valid, errors, cleaned_dataframe).
        """
        if not isinstance(df, pd.DataFrame):
            return False, [f"Expected pd.DataFrame, got {type(df).__name__}."], pd.DataFrame()

        if df.empty:
            return False, ["Input DataFrame is empty."], df

        if len(df) > cls.MAX_BATCH_SIZE:
            return (
                False,
                [f"Batch DataFrame exceeds maximum permissible limit of {cls.MAX_BATCH_SIZE:,} records (got {len(df):,})."],
                df,
            )

        errors: list[str] = []
        missing_cols = REQUIRED_SENSOR_FIELDS - set(df.columns)
        if missing_cols:
            errors.append(f"Batch DataFrame is missing required columns: {sorted(list(missing_cols))}")
            return False, errors, df

        # Check for unrecoverable nulls
        null_counts = df[list(REQUIRED_SENSOR_FIELDS)].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if not cols_with_nulls.empty:
            logger.warning("Batch DataFrame contains nulls: %s", cols_with_nulls.to_dict())

        # Check type values
        invalid_types = df[~df["type"].isin(["L", "M", "H", MachineType.LOW, MachineType.MEDIUM, MachineType.HIGH])]
        if not invalid_types.empty:
            errors.append(f"Found {len(invalid_types)} rows with invalid machine 'type' (must be 'L', 'M', or 'H').")

        # Check physical boundary ranges
        oob_speed = df[(df["rotational_speed_rpm"] < SensorBoundaries.ROTATIONAL_SPEED_MIN_RPM) |
                       (df["rotational_speed_rpm"] > SensorBoundaries.ROTATIONAL_SPEED_MAX_RPM)]
        if not oob_speed.empty:
            errors.append(f"Found {len(oob_speed)} rows with out-of-bounds 'rotational_speed_rpm' [{SensorBoundaries.ROTATIONAL_SPEED_MIN_RPM}-{SensorBoundaries.ROTATIONAL_SPEED_MAX_RPM}].")

        oob_torque = df[(df["torque_nm"] < SensorBoundaries.TORQUE_MIN_NM) |
                        (df["torque_nm"] > SensorBoundaries.TORQUE_MAX_NM)]
        if not oob_torque.empty:
            errors.append(f"Found {len(oob_torque)} rows with out-of-bounds 'torque_nm' [{SensorBoundaries.TORQUE_MIN_NM}-{SensorBoundaries.TORQUE_MAX_NM}].")

        if errors:
            return False, errors, df

        return True, [], df
