"""Versioned model artifact registry and loading verification engine for MachSense."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd

from machsense.config.settings import get_settings
from machsense.features.preprocessor import MachSensePreprocessor
from machsense.utils.logger import get_logger

logger = get_logger("machsense.models.registry")


@dataclass
class ModelMetadata:
    """Complete metadata record for a versioned production model."""
    model_version: str
    model_name: str
    model_type: str
    created_at: str
    hyperparameters: Dict[str, Any]
    feature_names: List[str]
    optimal_threshold: float
    cv_mean_pr_auc: float
    validation_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)


class ModelRegistry:
    """Manages versioned storage, retrieval, and verification of model artifacts."""

    def __init__(self, models_dir: Optional[Path | str] = None) -> None:
        settings = get_settings()
        self.models_dir = Path(models_dir) if models_dir else settings.resolve_path("models_dir")
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_versioned_artifacts(
        self,
        model: Any,
        preprocessor: MachSensePreprocessor,
        metadata: ModelMetadata,
        version: str = "v1.0.0",
        set_as_active: bool = True,
    ) -> Dict[str, Path]:
        """Save model, preprocessor, and metadata as versioned artifacts.

        Args:
            model: Trained classifier.
            preprocessor: Fitted MachSensePreprocessor.
            metadata: ModelMetadata instance.
            version: Semantic version tag (e.g., 'v1.0.0').
            set_as_active: If True, also updates active aliases ('best_model.joblib', 'preprocessor.joblib').

        Returns:
            Dictionary of saved artifact paths.
        """
        clean_version = version.lstrip("v")
        tag = f"v{clean_version}"

        # 1. Versioned file paths
        model_path = self.models_dir / f"machsense_model_{tag}.joblib"
        preprocessor_path = self.models_dir / f"machsense_preprocessor_{tag}.joblib"
        meta_path = self.models_dir / f"model_metadata_{tag}.json"

        # 2. Save versioned files
        joblib.dump(model, model_path)
        joblib.dump(preprocessor, preprocessor_path)

        meta_dict = metadata.to_dict()
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)

        saved_paths = {
            "model_path": model_path,
            "preprocessor_path": preprocessor_path,
            "metadata_path": meta_path,
        }

        # 3. Update active alias pointers
        if set_as_active:
            active_model = self.models_dir / "best_model.joblib"
            active_prep = self.models_dir / "preprocessor.joblib"
            active_meta = self.models_dir / "model_card.json"

            joblib.dump(model, active_model)
            joblib.dump(preprocessor, active_prep)
            with open(active_meta, "w", encoding="utf-8") as f:
                json.dump(meta_dict, f, indent=2)

            saved_paths["active_model"] = active_model
            saved_paths["active_preprocessor"] = active_prep
            saved_paths["active_metadata"] = active_meta

        logger.info("Successfully registered versioned artifacts for %s (%s)", metadata.model_name, tag)
        return saved_paths

    def load_versioned_artifacts(
        self,
        version: Optional[str] = None,
    ) -> Tuple[Any, MachSensePreprocessor, Dict[str, Any]]:
        """Load model, preprocessor, and metadata from version tag or active alias.

        Args:
            version: Semantic version tag (e.g. 'v1.0.0'). If None, loads active alias.

        Returns:
            Tuple of (loaded model, loaded preprocessor, metadata dict).
        """
        if version is None:
            model_path = self.models_dir / "best_model.joblib"
            prep_path = self.models_dir / "preprocessor.joblib"
            meta_path = self.models_dir / "model_card.json"
        else:
            clean_version = version.lstrip("v")
            tag = f"v{clean_version}"
            model_path = self.models_dir / f"machsense_model_{tag}.joblib"
            prep_path = self.models_dir / f"machsense_preprocessor_{tag}.joblib"
            meta_path = self.models_dir / f"model_metadata_{tag}.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {model_path}")
        if not prep_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at: {prep_path}")

        model = joblib.load(model_path)
        preprocessor = joblib.load(prep_path)

        metadata = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        logger.info("Loaded artifacts successfully from: %s", model_path)
        return model, preprocessor, metadata

    def verify_artifacts_integrity(
        self,
        sample_payload: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Verify that saved artifacts can be loaded and produce valid predictions.

        Returns:
            Verification diagnostic report dictionary.
        """
        if sample_payload is None:
            sample_payload = {
                "type": "M",
                "air_temperature_k": 300.2,
                "process_temperature_k": 310.5,
                "rotational_speed_rpm": 1500,
                "torque_nm": 42.0,
                "tool_wear_min": 120.0,
            }

        model, preprocessor, meta = self.load_versioned_artifacts(version=version)

        # 1. Transform instance
        X_trans = preprocessor.transform_instance(sample_payload)

        # 2. Predict probability and class
        threshold = meta.get("optimal_threshold", 0.5)
        if hasattr(model, "predict_proba"):
            failure_prob = float(model.predict_proba(X_trans)[0, 1])
        else:
            failure_prob = float(model.predict(X_trans)[0])

        predicted_class = int(failure_prob >= threshold)

        report = {
            "status": "PASSED",
            "model_version": meta.get("model_version", version or "active"),
            "model_name": meta.get("model_name", type(model).__name__),
            "input_features": list(sample_payload.keys()),
            "transformed_features_count": X_trans.shape[1],
            "failure_probability": round(failure_prob, 4),
            "predicted_class": predicted_class,
            "threshold_used": threshold,
        }
        logger.info("Artifact verification result: %s", report)
        return report
