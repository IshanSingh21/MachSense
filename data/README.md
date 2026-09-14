# Data Directory Architecture

This directory stores all dataset artifacts across the lifecycle stages.

## Structure
- `raw/`: Unaltered, immutable raw sensor telemetry data files (e.g., CSV, Parquet, or JSON dumps from IoT sensors). Never edit files in this directory.
- `interim/`: Intermediate transformed data that has undergone cleaning, missing value imputation, and timestamp alignment, but has not yet been formatted into feature matrices.
- `processed/`: Final, feature-engineered matrices ready for ML model training and evaluation (e.g., scaled arrays, train/val/test splits).

## Data Governance Guidelines
1. **Never commit raw or processed telemetry datasets to Git.** The `.gitignore` is configured to prevent tracking dataset files.
2. Large files should be tracked with DVC (Data Version Control) or stored in an enterprise object store (GCS / AWS S3 / Azure Blob Storage).
3. Ensure all telemetry columns comply with standard naming conventions: lowercase with snake_case (`air_temperature_k`, `rotational_speed_rpm`).
