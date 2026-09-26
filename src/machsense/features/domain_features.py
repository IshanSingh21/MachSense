"""Domain-informed physics feature extractor for predictive maintenance."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from machsense.utils.logger import get_logger

logger = get_logger("machsense.features.domain")


class DomainFeatureExtractor(BaseEstimator, TransformerMixin):
    """Transformer that derives domain-specific physical features from telemetry.

    Derived features:
    1. power_w: Mechanical rotational power = Torque * Angular Velocity
    2. temp_difference_k: Thermal gradient = Process Temp - Air Temp
    3. temp_ratio: Thermal ratio = Process Temp / Air Temp
    4. overstrain_index: Mechanical stress product = Torque * Tool Wear
    5. tool_wear_rate: Tool wear relative to speed = Tool Wear / Rotational Speed
    """

    def __init__(self, include_wear_rate: bool = True) -> None:
        self.include_wear_rate = include_wear_rate
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y: Optional[Any] = None) -> DomainFeatureExtractor:
        """Fit transformer (stateless domain calculations)."""
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Derive physical domain features and return enriched DataFrame."""
        if isinstance(X, np.ndarray):
            # Assume standard columns if passed as numpy array
            cols = [
                "type",
                "air_temperature_k",
                "process_temperature_k",
                "rotational_speed_rpm",
                "torque_nm",
                "tool_wear_min",
            ]
            df = pd.DataFrame(X, columns=cols[: X.shape[1]])
        else:
            df = X.copy()

        # 1. Mechanical Rotational Power in Watts: Power = Torque * (RPM * 2 * pi / 60)
        if "torque_nm" in df.columns and "rotational_speed_rpm" in df.columns:
            torque = pd.to_numeric(df["torque_nm"], errors="coerce").fillna(0.0)
            rpm = pd.to_numeric(df["rotational_speed_rpm"], errors="coerce").fillna(0.0)
            df["power_w"] = torque * (rpm * (2.0 * np.pi / 60.0))

        # 2. Temperature Gradient & Thermal Ratio
        if "process_temperature_k" in df.columns and "air_temperature_k" in df.columns:
            proc_temp = pd.to_numeric(df["process_temperature_k"], errors="coerce").fillna(300.0)
            air_temp = pd.to_numeric(df["air_temperature_k"], errors="coerce").fillna(300.0)
            df["temp_difference_k"] = proc_temp - air_temp
            df["temp_ratio"] = proc_temp / np.where(air_temp != 0, air_temp, 300.0)

        # 3. Mechanical Overstrain Index: Torque * Tool Wear
        if "torque_nm" in df.columns and "tool_wear_min" in df.columns:
            torque = pd.to_numeric(df["torque_nm"], errors="coerce").fillna(0.0)
            wear = pd.to_numeric(df["tool_wear_min"], errors="coerce").fillna(0.0)
            df["overstrain_index"] = torque * wear

        # 4. Tool Wear Rate per RPM
        if self.include_wear_rate and "tool_wear_min" in df.columns and "rotational_speed_rpm" in df.columns:
            wear = pd.to_numeric(df["tool_wear_min"], errors="coerce").fillna(0.0)
            rpm = pd.to_numeric(df["rotational_speed_rpm"], errors="coerce").fillna(0.0)
            df["tool_wear_rate"] = wear / (rpm + 1e-5)

        self.feature_names_ = list(df.columns)
        return df

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        """Return array of output feature names."""
        return np.array(self.feature_names_)
