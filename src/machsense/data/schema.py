"""Data schema definitions, physical boundaries, and validation report structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class MachineType(str, Enum):
    """Machine quality variant types."""
    LOW = "L"
    MEDIUM = "M"
    HIGH = "H"


class SensorBoundaries:
    """Domain-specific physical boundaries for manufacturing sensor telemetry."""
    AIR_TEMP_MIN_K = 270.0      # ~ -3.15 deg C
    AIR_TEMP_MAX_K = 350.0      # ~ 76.85 deg C
    PROCESS_TEMP_MIN_K = 280.0  # ~ 6.85 deg C
    PROCESS_TEMP_MAX_K = 360.0  # ~ 86.85 deg C
    MIN_TEMP_DELTA_K = 0.0      # Process temp must be >= Air temp in active cutting
    ROTATIONAL_SPEED_MIN_RPM = 500
    ROTATIONAL_SPEED_MAX_RPM = 4500
    TORQUE_MIN_NM = 0.0
    TORQUE_MAX_NM = 150.0
    TOOL_WEAR_MIN_MIN = 0.0
    TOOL_WEAR_MAX_MIN = 500.0


# Standardized internal column names mapping from external headers
COLUMN_MAPPING: Dict[str, str] = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

EXPECTED_FEATURE_COLUMNS: List[str] = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "type",
]

FAILURE_MODE_COLUMNS: List[str] = ["twf", "hdf", "pwf", "osf", "rnf"]


class SensorRecord(BaseModel):
    """Pydantic model representing a single validated telemetry record."""
    udi: int = Field(gt=0, description="Unique Machine Telemetry Identifier")
    product_id: str = Field(min_length=3, description="Product serial number")
    type: MachineType = Field(description="Quality variant category")
    air_temperature_k: float = Field(
        ge=SensorBoundaries.AIR_TEMP_MIN_K,
        le=SensorBoundaries.AIR_TEMP_MAX_K,
        description="Air temperature in Kelvin",
    )
    process_temperature_k: float = Field(
        ge=SensorBoundaries.PROCESS_TEMP_MIN_K,
        le=SensorBoundaries.PROCESS_TEMP_MAX_K,
        description="Process temperature in Kelvin",
    )
    rotational_speed_rpm: float = Field(
        ge=SensorBoundaries.ROTATIONAL_SPEED_MIN_RPM,
        le=SensorBoundaries.ROTATIONAL_SPEED_MAX_RPM,
        description="Spindle rotational speed in RPM",
    )
    torque_nm: float = Field(
        ge=SensorBoundaries.TORQUE_MIN_NM,
        le=SensorBoundaries.TORQUE_MAX_NM,
        description="Applied torque in Nm",
    )
    tool_wear_min: float = Field(
        ge=SensorBoundaries.TOOL_WEAR_MIN_MIN,
        le=SensorBoundaries.TOOL_WEAR_MAX_MIN,
        description="Cumulative tool wear in minutes",
    )
    machine_failure: int = Field(ge=0, le=1, description="Binary failure target")
    twf: Optional[int] = Field(default=0, ge=0, le=1)
    hdf: Optional[int] = Field(default=0, ge=0, le=1)
    pwf: Optional[int] = Field(default=0, ge=0, le=1)
    osf: Optional[int] = Field(default=0, ge=0, le=1)
    rnf: Optional[int] = Field(default=0, ge=0, le=1)

    @field_validator("process_temperature_k")
    @classmethod
    def validate_temp_difference(cls, v: float, info: Any) -> float:
        """Physical rule: process temperature should be >= air temperature."""
        air_temp = info.data.get("air_temperature_k")
        if air_temp is not None and v < (air_temp - 0.5):  # small margin for sensor jitter
            raise ValueError(f"Process temp ({v} K) cannot be colder than air temp ({air_temp} K)")
        return v


@dataclass
class ValidationReport:
    """Comprehensive summary of dataset validation results."""
    is_valid: bool
    total_records: int
    valid_records_count: int
    missing_values_per_col: Dict[str, int] = field(default_factory=dict)
    out_of_bound_records: int = 0
    duplicate_udi_count: int = 0
    thermodynamic_inconsistencies: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable validation summary."""
        status = "PASSED" if self.is_valid else "FAILED"
        return (
            f"Validation Report [{status}]:\n"
            f"  - Total Records:               {self.total_records}\n"
            f"  - Valid Records:               {self.valid_records_count} ({self.valid_records_count/max(1, self.total_records):.1%})\n"
            f"  - Missing Values Found:        {sum(self.missing_values_per_col.values())}\n"
            f"  - Out-of-Bounds Values:        {self.out_of_bound_records}\n"
            f"  - Duplicate UDI Records:       {self.duplicate_udi_count}\n"
            f"  - Thermodynamic Violations:    {self.thermodynamic_inconsistencies}\n"
            f"  - Errors:                      {len(self.errors)}\n"
            f"  - Warnings:                    {len(self.warnings)}"
        )
