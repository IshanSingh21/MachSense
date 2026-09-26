"""Production-grade preprocessing pipeline for MachSense."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from machsense.config.settings import get_settings
from machsense.features.domain_features import DomainFeatureExtractor
from machsense.utils.logger import get_logger

logger = get_logger("machsense.features.preprocessor")


class MachSensePreprocessor(BaseEstimator, TransformerMixin):
    """End-to-end reusable preprocessor combining domain engineering, encoding, and scaling.

    Designed for strict leakage prevention (fitted exclusively on training data)
    and seamless inference on batch DataFrames or single sensor payload dictionaries.
    """

    def __init__(
        self,
        scaling_method: str = "standard",  # "standard", "robust", or "none"
        include_wear_rate: bool = True,
    ) -> None:
        self.scaling_method = scaling_method
        self.include_wear_rate = include_wear_rate

        self.domain_extractor_ = DomainFeatureExtractor(include_wear_rate=include_wear_rate)
        self.column_transformer_: Optional[ColumnTransformer] = None
        self.feature_names_out_: List[str] = []
        self.numerical_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.is_fitted_: bool = False

    def _build_column_transformer(self) -> ColumnTransformer:
        """Construct scikit-learn ColumnTransformer based on configured scaling method."""
        # Categorical pipeline: Impute most frequent -> OneHotEncode
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        categories=[["L", "M", "H"]],
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        # Numerical pipeline: Impute median -> Scale
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if self.scaling_method == "standard":
            num_steps.append(("scaler", StandardScaler()))
        elif self.scaling_method == "robust":
            num_steps.append(("scaler", RobustScaler()))
        elif self.scaling_method == "none":
            pass
        else:
            raise ValueError(f"Unknown scaling_method: '{self.scaling_method}'. Choose 'standard', 'robust', or 'none'.")

        num_pipeline = Pipeline(steps=num_steps)

        transformer = ColumnTransformer(
            transformers=[
                ("cat", cat_pipeline, self.categorical_cols_),
                ("num", num_pipeline, self.numerical_cols_),
            ],
            remainder="drop",
        )
        return transformer

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> MachSensePreprocessor:
        """Fit the domain extractor and column transformer strictly on training data."""
        logger.info("Fitting MachSensePreprocessor on training data (%d samples)...", len(X))

        # 1. Apply domain feature extraction
        X_domain = self.domain_extractor_.fit_transform(X)

        # 2. Identify categorical and numerical columns dynamically
        self.categorical_cols_ = [c for c in ["type"] if c in X_domain.columns]
        self.numerical_cols_ = [
            c
            for c in X_domain.columns
            if c not in self.categorical_cols_ and c not in ["udi", "product_id", "machine_failure"]
        ]

        # 3. Build and fit column transformer
        self.column_transformer_ = self._build_column_transformer()
        self.column_transformer_.fit(X_domain)

        # 4. Extract output feature names
        encoded_cat_names = [f"type_{cat}" for cat in ["L", "M", "H"]]
        self.feature_names_out_ = encoded_cat_names + self.numerical_cols_

        self.is_fitted_ = True
        logger.info("Preprocessor fitted successfully with %d output features: %s", len(self.feature_names_out_), self.feature_names_out_)
        return self

    def transform(self, X: pd.DataFrame, return_dataframe: bool = True) -> Union[pd.DataFrame, np.ndarray]:
        """Transform input data using the pre-fitted pipeline (no data leakage)."""
        if not self.is_fitted_ or self.column_transformer_ is None:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")

        # 1. Apply domain extraction
        X_domain = self.domain_extractor_.transform(X)

        # 2. Apply fitted column transformer
        transformed_array = self.column_transformer_.transform(X_domain)

        if return_dataframe:
            return pd.DataFrame(
                transformed_array,
                columns=self.feature_names_out_,
                index=X.index if hasattr(X, "index") else None,
            )
        return transformed_array

    def fit_transform(self, X: pd.DataFrame, y: Optional[Any] = None, return_dataframe: bool = True) -> Union[pd.DataFrame, np.ndarray]:
        """Fit to training data, then transform."""
        return self.fit(X, y).transform(X, return_dataframe=return_dataframe)

    def transform_instance(self, payload: Dict[str, Any]) -> pd.DataFrame:
        """Transform a single JSON/dict sensor payload during online inference."""
        df = pd.DataFrame([payload])
        return self.transform(df, return_dataframe=True)

    def save(self, destination_path: Optional[Path | str] = None) -> Path:
        """Serialize preprocessor artifact and metadata to disk."""
        if not self.is_fitted_:
            raise RuntimeError("Cannot save an unfitted preprocessor.")

        settings = get_settings()
        if destination_path is None:
            models_dir = settings.resolve_path("models_dir")
            save_path = models_dir / "preprocessor.joblib"
        else:
            save_path = Path(destination_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path)
        logger.info("Saved serialized preprocessor to: %s", save_path)

        # Save metadata JSON
        meta_path = save_path.with_name("preprocessor_meta.json")
        meta_data = {
            "scaling_method": self.scaling_method,
            "include_wear_rate": self.include_wear_rate,
            "feature_names_out": self.feature_names_out_,
            "numerical_cols": self.numerical_cols_,
            "categorical_cols": self.categorical_cols_,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info("Saved preprocessor metadata to: %s", meta_path)

        return save_path

    @classmethod
    def load(cls, source_path: Optional[Path | str] = None) -> MachSensePreprocessor:
        """Load serialized preprocessor from disk."""
        settings = get_settings()
        if source_path is None:
            models_dir = settings.resolve_path("models_dir")
            load_path = models_dir / "preprocessor.joblib"
        else:
            load_path = Path(source_path)

        if not load_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at: {load_path}")

        preprocessor: MachSensePreprocessor = joblib.load(load_path)
        logger.info("Loaded preprocessor from: %s", load_path)
        return preprocessor
