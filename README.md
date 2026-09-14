# MachSense: AI-Powered Predictive Maintenance System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MachSense** is an enterprise-grade machine learning application that forecasts equipment failures from sensor telemetry and delivers explainable insights (SHAP/attribution) to maintenance teams before costly breakdowns occur.

---

## Key Capabilities
- **Multi-Sensor Telemetry Processing**: Real-time & batch ingestion of thermal, rotational, torque, and strain sensor data.
- **Explainable Predictions**: High-precision failure probability with local feature attribution (SHAP) to isolate root causes (e.g. heat dissipation vs. tool wear).
- **Production-Ready Foundation**: Modular architecture, dynamic configuration management, rotating logging infrastructure, and automated test suite.

---

## Repository Structure

```
machsense/
├── .gitignore                   # Comprehensive ignore rules for ML/Python
├── README.md                    # Project overview and quickstart
├── pyproject.toml               # Package build, metadata, and tool configuration
├── requirements.txt             # Production dependencies
├── requirements-dev.txt         # Dev, testing, and linting tools
├── configs/                     # Centralized configuration management
│   ├── config.yaml              # Core application, path, data, and model settings
│   └── logging.yaml             # Structured logging and rotation policies
├── data/                        # Data directory structure
│   ├── raw/                     # Immutable raw sensor telemetry
│   ├── interim/                 # Cleaned and aligned intermediate data
│   ├── processed/               # Feature-engineered matrices for training
│   └── README.md                # Data governance guidelines
├── docs/                        # Project documentation
│   ├── architecture.md          # Architectural blueprints and diagrams
│   └── setup_guide.md           # Developer onboarding guide
├── models/                      # Serialized model artifacts & evaluation summaries
│   └── README.md                # Model versioning guidelines
├── notebooks/                   # Exploratory data analysis and prototyping
├── src/machsense/               # Core source package
│   ├── __init__.py
│   ├── main.py                  # CLI and execution orchestrator
│   ├── config/                  # Dynamic path resolution & Pydantic config
│   ├── data/                    # Ingestion & validation (Day 2)
│   ├── features/                # Feature engineering pipelines (Day 2)
│   ├── models/                  # Predictive models & explainers (Day 3)
│   └── utils/                   # Structured logger and shared helpers
├── app/                         # Serving layer (REST API / UI Dashboard)
│   ├── __init__.py
│   └── main.py                  # API server entry point
└── tests/                       # Automated test suite
    ├── conftest.py              # Test fixtures and isolated environments
    ├── test_config.py           # Configuration and path resolution tests
    ├── test_logging.py          # Logging behavior tests
    └── test_cli.py              # CLI integration tests
```

---

## Quickstart

### 1. Installation
```bash
# Clone or navigate to the repository
cd machsense

# Create and activate environment
python -m venv .venv
.venv\Scripts\Activate.ps1   # On Windows
# or: source .venv/bin/activate # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

### 2. Verify Foundation
```bash
# Run CLI info check
python -m machsense.main --info

# Run automated test suite
pytest
```

---

## Configuration Management

MachSense avoids hardcoded paths by dynamically calculating the repository root and resolving all data, model, and log paths relative to the project environment.

Modify parameters in `configs/config.yaml` or override them via environment variables:
- `MACHSENSE_ROOT`: Base project directory
- `MACHSENSE_CONFIG_PATH`: Path to a custom YAML configuration file
- `MACHSENSE_ENV`: Active environment (`development`, `staging`, `production`)

---

## Project Roadmap

| Phase | Milestone | Status |
| :--- | :--- | :---: |
| **Day 1** | **Project Foundation, Architecture, Config, Logging & Tests** | ✅ Completed |
| **Day 2** | Data Ingestion, EDA, Preprocessing & Feature Engineering | ⏳ Next |
| **Day 3** | ML Baseline & Model Training (RandomForest/XGBoost/LightGBM) | ⏳ Upcoming |
| **Day 4** | Model Explainability (SHAP/Feature Attribution) & Evaluation | ⏳ Upcoming |
| **Day 5** | Production Serving (FastAPI/Dashboard) & Monitoring | ⏳ Upcoming |
