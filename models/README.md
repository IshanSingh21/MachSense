# Models Directory

This directory stores serialized model artifacts, training checkpoints, evaluation summaries, and explainability artifacts.

## Artifact Types
- `*.joblib` / `*.pkl`: Serialized Scikit-Learn or XGBoost models and preprocessors.
- `*.onnx`: Optimized inference graph artifacts.
- `*.json` / `*.yaml`: Training metadata, feature importances, hyperparameters, and evaluation metrics per version.

## Registry & Versioning
- Models follow semantic tagging: `<model_type>_<timestamp_or_version>.<ext>` (e.g., `rf_baseline_v0.1.0.joblib`).
- Binary model artifacts are excluded from Git via `.gitignore`.
- Production deployments should fetch validated artifacts from a centralized model registry (e.g. MLflow / W&B / S3).
