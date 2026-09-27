# MachSense: Enterprise AI Predictive Maintenance & Machine Health Diagnostics

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![SHAP](https://img.shields.io/badge/Explainability-TreeSHAP-brightgreen.svg)](https://github.com/slundberg/shap)
[![Code style: black/ruff](https://img.shields.io/badge/code%20style-black%2Fruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: 126 Passed](https://img.shields.io/badge/tests-126%20passed-success.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**MachSense** is an industrial-grade, AI-powered predictive maintenance and machine health monitoring system. It transforms multi-sensor telemetry (temperature, rotational speed, torque, tool wear) into real-time failure risk assessments, calibrated decision thresholds, and plain-English TreeSHAP root-cause diagnostics for shop-floor operators.

---

## Key Capabilities

1. **Physics-Informed Feature Engineering**: Extracts thermodynamic gradients ($\Delta T$), mechanical power ($P = \tau \cdot \omega$), specific torque, strain energy, and non-linear tool wear degradation rates.
2. **Imbalance-Calibrated Decision Engine**: Champion Random Forest tuned for Precision-Recall AUC (PR-AUC: 0.893) with an optimal cost-sensitive decision threshold ($\tau = 0.5608$) to minimize costly unpredicted machine failures.
3. **Local & Global Explainability (TreeSHAP)**: High-speed exact TreeSHAP attribution isolating failure triggers (Heat Dissipation, Power Overload, Overstrain, Tool Wear Exhaustion).
4. **Dual-Tier Serving Layer**:
   - **FastAPI REST Backend**: Sub-10ms inference latency, Pydantic v2 strict schema validation, health/readiness probes, and error sanitization.
   - **Streamlit Operator Dashboard**: Real-time telemetry sliders, physical failure presets, interactive Plotly SHAP waterfalls, fleet CSV batch scoring, and model cards.
5. **Production Hardened & Container Ready**: Path-traversal defense, CORS policies, DoS batch limits, Docker multi-stage builds, and multi-cloud deployment blueprints (Render, Streamlit Cloud, Hugging Face, Docker Compose).

---

## End-to-End System Architecture

```
                                    +-----------------------------------------+
                                    |     Machine Sensors / IoT Telemetry     |
                                    | (Air Temp, Process Temp, RPM, Nm, Wear) |
                                    +-----------------------------------------+
                                                         |
                                                         v
                                    +-----------------------------------------+
                                    |        Domain Boundary Validator        |
                                    | (Schema Check, Physical Bounds, Max 5k) |
                                    +-----------------------------------------+
                                                         |
                                                         v
                                    +-----------------------------------------+
                                    |      Physics Feature Preprocessor       |
                                    | (Power, Temp Delta, One-Hot, Scaler)    |
                                    +-----------------------------------------+
                                                         |
                                                         v
                                    +-----------------------------------------+
                                    |      MachSense Prediction Engine        |
                                    |  (Champion RF Model + Threshold 0.5608) |
                                    +-----------------------------------------+
                                                         |
                                                         v
                                    +-----------------------------------------+
                                    |         TreeSHAP Explainability         |
                                    |   (Feature Attribution & Root Causes)   |
                                    +-----------------------------------------+
                                                         |
                             +---------------------------+---------------------------+
                             |                                                       |
                             v                                                       v
            +--------------------------------+                     +----------------------------------+
            |      FastAPI Backend API       |                     |   Streamlit Operator Dashboard   |
            | (/health, /ready, /v1/predict) |                     | (Presets, Plotly SHAP, Fleet CSV)|
            +--------------------------------+                     +----------------------------------+
```

---

## Project Structure

```
machsense/
├── Dockerfile                   # Unified production container image
├── Dockerfile.api               # Dedicated FastAPI microservice image
├── Dockerfile.ui                # Dedicated Streamlit dashboard image
├── docker-compose.yml           # Full-stack multi-container orchestration
├── render.yaml                  # Render.com Infrastructure Blueprint
├── Procfile                     # PaaS process manager configuration
├── pyproject.toml               # Package build, dependencies & tool configuration
├── requirements.txt             # Production dependencies
├── requirements-dev.txt         # Development & testing tools
├── .streamlit/
│   └── config.toml              # Streamlit production headless configuration
├── configs/
│   ├── config.yaml              # Application, paths, model, and serving configuration
│   └── logging.yaml             # Rotating structured logging configuration
├── models/
│   ├── machsense_model_v1.0.0.joblib         # Champion Random Forest model artifact
│   ├── machsense_preprocessor_v1.0.0.joblib  # Fitted preprocessor pipeline artifact
│   ├── model_metadata_v1.0.0.json            # Model lineage and hyperparameter card
│   └── model_card.json                       # Public model governance card
├── scripts/
│   └── deploy_healthcheck.py    # Automated deployment verification & probe test
├── src/machsense/
│   ├── api/                     # FastAPI backend (lifespan, routes, error handlers)
│   ├── config/                  # Dynamic path resolution & environment management
│   ├── data/                    # Ingestion, validation, cleaning & splitters
│   ├── features/                # Physics domain extractors & preprocessor pipeline
│   ├── inference/               # Production prediction service & boundary validator
│   ├── models/                  # Trainers, CV optimizers, evaluation & SHAP explainer
│   ├── ui/                      # Streamlit dashboard & Plotly visual components
│   └── utils/                   # Rotating logger and utility helpers
└── tests/                       # 126 automated unit, integration, and security tests
```

---

## Quickstart & Local Setup

### 1. Installation
```bash
# Clone repository
git clone https://github.com/IshanSingh21/MachSense.git
cd MachSense

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1    # Windows PowerShell
# source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

### 2. Verify Installation & Run Test Suite
```bash
# Verify environment and configuration resolution
python -m machsense.main --info

# Run full test suite (126 tests)
pytest tests/ -v
```

---

## CLI Usage

MachSense provides a unified command-line interface:

| Command | Description |
| :--- | :--- |
| `python -m machsense.main --info` | Print resolved environment, model version, and paths |
| `python -m machsense.main --predict-sample` | Run sample inference pipeline with SHAP explanation |
| `python -m machsense.main --explain` | Generate SHAP summary plots and operator alerts |
| `python -m machsense.main --serve` | Launch FastAPI backend with Uvicorn on `http://127.0.0.1:8000` |
| `python -m machsense.main --ui` | Launch Streamlit interactive dashboard on `http://localhost:8501` |

---

## Production Docker Deployment

### 1. Unified Full-Stack Orchestration (Docker Compose)
Launch both the FastAPI API and Streamlit UI in isolated, healthchecked containers:

```bash
docker-compose up --build
```
- **Streamlit UI**: `http://localhost:8501`
- **FastAPI API & Docs**: `http://localhost:8000/docs`
- **Readiness Probe**: `http://localhost:8000/health/ready`

### 2. Individual Container Builds

**FastAPI Backend Container**:
```bash
docker build -t machsense-api -f Dockerfile.api .
docker run -p 8000:8000 --name machsense-api machsense-api
```

**Streamlit Dashboard Container**:
```bash
docker build -t machsense-ui -f Dockerfile.ui .
docker run -p 8501:8501 -e MACHSENSE_API_URL=http://localhost:8000 --name machsense-ui machsense-ui
```

---

## Free-Tier Cloud Deployment Guide

MachSense is architected for zero-cost, 1-click cloud deployment across multiple free-tier platforms:

### Option A: Streamlit Community Cloud (Recommended for UI)
1. Fork or push this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io).
3. Connect your repository:
   - **Main file path**: `src/machsense/ui/app.py`
   - **Python version**: `3.11`
4. Click **Deploy**. MachSense automatically runs in self-contained in-process mode with zero external API dependencies required.

### Option B: Render.com (Full-Stack Blueprint)
1. Push your code to GitHub.
2. In [Render Dashboard](https://dashboard.render.com), click **New** $\rightarrow$ **Blueprint**.
3. Select this repository. Render automatically reads `render.yaml` and provisions:
   - `machsense-api`: Free FastAPI Web Service with `/health/ready` probe.
   - `machsense-ui`: Free Streamlit Web Service.

### Option C: Hugging Face Spaces
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces) with SDK set to **Streamlit** (or **Docker**).
2. Push repository files. Set Space app file to `src/machsense/ui/app.py`.

---

## Health Checking & Verification

Verify any running local or cloud instance using the built-in deployment health check utility:

```bash
# Check local deployment
python scripts/deploy_healthcheck.py --url http://127.0.0.1:8000

# Check cloud deployment
python scripts/deploy_healthcheck.py --url https://your-deployed-api.onrender.com
```

### Probe Endpoints:
- `GET /`: API discovery metadata and service links.
- `GET /health` or `GET /health/live`: Liveness probe (HTTP 200).
- `GET /health/ready`: Readiness probe verifying memory-loaded model artifacts and explainer readiness.
- `POST /api/v1/predict`: Real-time single machine telemetry scoring.
- `POST /api/v1/predict/batch`: Fleet batch CSV scoring.

---

## Environment Variables Reference

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MACHSENSE_ENV` | `development` | Operating environment (`development`, `staging`, `production`) |
| `PORT` | `8000` / `8501` | Application listening port (injected dynamically by PaaS providers) |
| `HOST` | `127.0.0.1` / `0.0.0.0` | Server bind host address |
| `MACHSENSE_API_URL` | `http://127.0.0.1:8000` | Target FastAPI endpoint for Streamlit UI client |
| `MACHSENSE_PREFER_API` | `false` | If `true`, UI attempts HTTP API calls first with fallback to local inference |
| `MACHSENSE_CORS_ORIGINS` | `["*"]` | Allowed CORS origins (comma-separated list or JSON array) |
| `MACHSENSE_MODELS_DIR` | `models` | Custom directory path for model artifacts |
| `WEB_CONCURRENCY` | `1` | Number of Uvicorn worker processes in production |

---

## 14-Day Engineering Roadmap

| Phase / Day | Scope | Highlights & Milestones | Status |
| :--- | :--- | :--- | :---: |
| **Day 1** | Foundation & Config | Dynamic path resolution, Pydantic settings, rotating loggers | ✅ Complete |
| **Day 2** | Data Governance & EDA | Automated data download, schema validation, outlier analysis | ✅ Complete |
| **Day 3** | Physics Feature Eng | Power ($P=\tau\omega$), $\Delta T$, strain wear rate, leakage prevention | ✅ Complete |
| **Day 4** | Preprocessing Pipeline | Scikit-learn ColumnTransformer, RobustScaler, unit tests | ✅ Complete |
| **Day 5** | Baseline Modeling | Logistic Regression, Random Forest, XGBoost baselines | ✅ Complete |
| **Day 6** | Hyperparameter Tuning | Stratified 5-fold CV, PR-AUC objective, model registry | ✅ Complete |
| **Day 7** | Rigorous Evaluation | Cost-sensitive threshold tuning ($\tau = 0.5608$), error slicing | ✅ Complete |
| **Day 8** | Explainable AI (XAI) | Exact TreeSHAP attributions, operator alert generation | ✅ Complete |
| **Day 9** | Production Inference | Domain boundary validator, zero-retraining inference service | ✅ Complete |
| **Day 10** | FastAPI REST Backend | OpenAPI docs, health/readiness probes, single & batch scoring | ✅ Complete |
| **Day 11** | Streamlit Dashboard | Real-time sliders, failure presets, Plotly SHAP waterfall | ✅ Complete |
| **Day 12** | Full-Stack Integration | Dual-mode UI client (API/In-process), sanitized error envelopes | ✅ Complete |
| **Day 13** | Hardening & Security | Path traversal defense, CORS hardening, DoS batch caps, 0 lints | ✅ Complete |
| **Day 14** | Cloud Deployment | Docker multi-stage, docker-compose, Render blueprint, healthchecks | ✅ Complete |

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
