"""Data ingestion, validation, and loading package for MachSense."""

from machsense.data.cleaner import clean_dataset, validate_dataset
from machsense.data.download import download_ai4i_dataset, generate_synthetic_ai4i_benchmark
from machsense.data.loader import load_raw_data, standardize_columns
from machsense.data.pipeline import run_data_pipeline
from machsense.data.schema import (
    COLUMN_MAPPING,
    EXPECTED_FEATURE_COLUMNS,
    FAILURE_MODE_COLUMNS,
    MachineType,
    SensorBoundaries,
    SensorRecord,
    ValidationReport,
)
from machsense.data.splitter import (
    DataSplit,
    separate_features_and_target,
    split_data,
)

__all__ = [
    "download_ai4i_dataset",
    "generate_synthetic_ai4i_benchmark",
    "load_raw_data",
    "standardize_columns",
    "clean_dataset",
    "validate_dataset",
    "split_data",
    "separate_features_and_target",
    "run_data_pipeline",
    "DataSplit",
    "SensorRecord",
    "ValidationReport",
    "SensorBoundaries",
    "MachineType",
    "COLUMN_MAPPING",
    "EXPECTED_FEATURE_COLUMNS",
    "FAILURE_MODE_COLUMNS",
]
