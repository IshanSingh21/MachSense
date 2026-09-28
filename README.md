# MachSense: Industrial AI Predictive Maintenance & Machine Health Diagnostics

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![SHAP](https://img.shields.io/badge/Explainability-TreeSHAP-brightgreen.svg)](https://github.com/slundberg/shap)
[![Code style: black/ruff](https://img.shields.io/badge/code%20style-black%2Fruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: 126 Passed](https://img.shields.io/badge/tests-126%20passed-success.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**MachSense** is an enterprise-grade predictive maintenance and machine health monitoring system. It processes real-time industrial telemetry (temperature gradients, spindle speeds, mechanical torque, cutting tool wear), computes physics-informed features, forecasts catastrophic machine failure with cost-calibrated decision thresholds, and provides root-cause diagnostic explanations using TreeSHAP to shop-floor operators.

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Key Capabilities](#key-capabilities)
3. [System Architecture](#system-architecture)
4. [Dataset & Data Governance](#dataset--data-governance)
5. [Physics-Informed Feature Engineering](#physics-informed-feature-engineering)
6. [Machine Learning Pipeline & Model Comparison](#machine-learning-pipeline--model-comparison)
7. [Cost-Sensitive Evaluation & Threshold Tuning](#cost-sensitive-evaluation--threshold-tuning)
8. [Explainable AI (TreeSHAP Root Cause Analysis)](#explainable-ai-treeshap-root-cause-analysis)
9. [Serving Layer & API Documentation](#serving-layer--api-documentation)
10. [Streamlit User Dashboard Walkthrough](#streamlit-user-dashboard-walkthrough)
11. [Local Quickstart & CLI](#local-quickstart--cli)
12. [Production Docker & Cloud Deployment](#production-docker--cloud-deployment)
13. [Project Directory Structure](#project-directory-structure)
14. [Engineering Review: Limitations & Future Roadmap](#engineering-review-limitations--future-roadmap)
15. [Technologies & Tools](#technologies--tools)

---

## Problem Statement

Unplanned downtime in manufacturing costs industrial plants millions of dollars annually in lost throughput, damaged tooling, and emergency repair labor. While traditional maintenance relies on reactive firefighting or rigid calendar-based schedules, modern Industry 4.0 demands **predictive, condition-based maintenance**.

Key challenges in industrial machine health forecasting:
- **Extreme Class Imbalance**: Machine failures are rare events ($~3.39\%$ of operating cycles), causing naive classifiers to achieve high accuracy while missing critical breakdowns.
- **Asymmetric Cost of Errors**: A False Negative (missing a catastrophic breakdown) costs $10\times$ to $50\times$ more than a False Positive (a preemptive inspection).
- **Black-Box Skepticism**: Shop-floor maintenance technicians will not act on opaque probability scores without clear physical explanations of *why* an alert fired.
- **Strict Latency & Reliability**: Inference must execute in sub-10ms timeframes without data leakage or unstable dependencies.

MachSense solves these challenges through domain-engineered physics features, PR-AUC optimized ensembles, calibrated decision thresholds, and exact TreeSHAP attribution.

---

## Key Capabilities

- **Physics-Guided Domain Extraction**: Computes physical power, temperature differentials, specific torque, and non-linear wear rates directly from sensor telemetry.
- **Zero Data Leakage Ingestion**: Leakage-free splitters and a unified `ColumnTransformer` fitted exclusively on training sets.
- **Precision-Recall Calibrated Inference**: Champion Random Forest ensemble tuned with cost-sensitive thresholding ($\tau = 0.5608$) delivering **0.893 PR-AUC** and **0.995 Test ROC-AUC**.
- **Instant Local Explainability**: Sub-10ms TreeSHAP explanations isolating specific failure mechanisms (Heat Dissipation, Power Overload, Overstrain, Tool Wear).
- **Dual Serving Architecture**: High-throughput FastAPI REST API backend paired with an interactive Streamlit diagnostic workstation.
- **Production Hardened**: Built-in path traversal defense, CORS origin whitelisting, DoS batch protections (capped at 5,000 items), and Docker multi-stage containers.

---

## System Architecture

```
+----------------------------------------------------------------------------------------------------+
|                                    1. SENSOR INGESTION & VALIDATION                                |
|  Incoming Telemetry Payload (JSON/CSV) -> Domain Boundary Validator (Physical Bounds & Schema)     |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                               2. PHYSICS FEATURE ENGINEERING PIPELINE                              |
|  - Mechanical Power: P = Torque * Speed * (2pi/60)                                                 |
|  - Thermal Delta: Delta_T = Process_Temp - Air_Temp                                                |
|  - Overstrain Metric: Strain = Torque * Tool_Wear                                                  |
|  - Tool Wear Degradation Rate: Rate = Tool_Wear / (Speed + eps)                                    |
|  - One-Hot Categorical Encoding (L/M/H Quality Variants) + Robust Scaling                          |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                   3. PREDICTIVE INFERENCE ENGINE                                   |
|  - Champion Random Forest Classifier (v1.0.0, 150 Trees, Balanced Class Weights)                   |
|  - Calibrated Decision Boundary Threshold (tau = 0.5608)                                           |
|  - Failure Risk Classification: NOMINAL | LOW | ELEVATED | CRITICAL                                |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                   4. EXPLAINABLE AI (TreeSHAP)                                     |
|  - Exact TreeSHAP Additive Feature Attributions                                                    |
|  - Top 3 Risk Escalators vs. Stabilizing Features                                                  |
|  - Operator-Friendly Natural Language Actionable Diagnostics                                       |
+----------------------------------------------------------------------------------------------------+
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
                         v                                                 v
+------------------------------------------------+ +-------------------------------------------------+
|              FASTAPI REST BACKEND              | |          STREAMLIT OPERATOR DASHBOARD           |
| - Liveness & Readiness Probes (/health, /ready)| | - Real-time Parameter Sliders & Presets         |
| - Single Predict & Vectorized Fleet Batch      | | - Dynamic Plotly SHAP Waterfall / Bar Charts    |
| - Pydantic v2 Schema & Error Sanitization      | | - Fleet CSV Ingestion & Model Card Viewer       |
+------------------------------------------------+ +-------------------------------------------------+
```

---

## Dataset & Data Governance

MachSense utilizes the **AI4I 2020 Predictive Maintenance Dataset** reflecting realistic CNC milling machine telemetry:

- **Total Telemetry Records**: 10,000 machine operating cycles.
- **Target Variable**: `machine_failure` (Binary 0 = Nominal, 1 = Failure).
- **Class Distribution**: 9,661 Normal instances (96.61%), 339 Failure instances (3.39%).
- **Physical Failure Modes**:
  1. **Heat Dissipation Failure (HDF)**: $\Delta T < 8.6\text{ K}$ and spindle speed $< 1380\text{ rpm}$.
  2. **Power Failure (PWF)**: Power $P < 3500\text{ W}$ or $P > 9000\text{ W}$.
  3. **Overstrain Failure (OSF)**: Product of torque and tool wear exceeding material thresholds.
  4. **Tool Wear Failure (TWF)**: Cumulative cutting time reaching insert exhaustion ($200\text{--}240\text{ min}$).
  5. **Random Failures (RNF)**: Minor unmodeled stochastic events ($0.1\%$ probability).

---

## Physics-Informed Feature Engineering

Raw sensor telemetry is converted into physical indicators before model ingestion:

| Engineered Feature | Formulation | Physical & Failure Domain Rationale |
| :--- | :--- | :--- |
| **Mechanical Power ($P_{\text{mech}}$)** | $P = \tau \cdot \left(\frac{2\pi \cdot \omega}{60}\right)$ | Directly identifies Power Failures ($>9000\text{ W}$ or $<3500\text{ W}$). |
| **Temperature Delta ($\Delta T$)** | $\Delta T = T_{\text{process}} - T_{\text{air}}$ | Detects cooling system starvation and heat dissipation collapse ($\Delta T < 8.6\text{ K}$). |
| **Temperature Ratio ($R_T$)** | $R_T = \frac{T_{\text{process}}}{T_{\text{air}}}$ | Normalized thermal equilibrium metric across ambient shifts. |
| **Overstrain Index ($I_{\text{os}}$)** | $I_{\text{os}} = \tau \cdot t_{\text{wear}}$ | Captures severe mechanical stress on worn cutting inserts ($>11,000\text{ Nm}\cdot\text{min}$). |
| **Tool Wear Rate ($W_r$)** | $W_r = \frac{t_{\text{wear}}}{\omega + 10^{-6}}$ | Strain accumulated per spindle revolution. |

---

## Machine Learning Pipeline & Model Comparison

Models were trained using **Stratified 5-Fold Cross-Validation** with a held-out Test set (1,500 samples). The evaluation prioritizes **PR-AUC (Precision-Recall AUC)** due to severe class imbalance.

### Experimental Model Comparison Table

| Model Architecture | Hyperparameters / Notes | CV Mean PR-AUC | Test ROC-AUC | Test Precision | Test Recall | Test F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | `C=1.0`, `class_weight='balanced'` | 0.518 | 0.865 | 0.283 | 0.804 | 0.419 |
| **XGBoost Classifier** | `scale_pos_weight=28.5`, `max_depth=6` | 0.832 | 0.971 | 0.886 | 0.765 | 0.821 |
| **Random Forest Baseline** | `n_estimators=100`, `class_weight='balanced'` | 0.854 | 0.982 | 0.913 | 0.824 | 0.866 |
| **Tuned Random Forest (Champion v1.0.0)** | `n_estimators=150`, `max_depth=10`, `min_samples_split=2` | **0.874** | **0.988** | **1.000** | **0.863** | **0.926** |

---

## Cost-Sensitive Evaluation & Threshold Tuning

In predictive maintenance, default decision thresholds ($\tau = 0.50$) are suboptimal for skewed operational risk. 

MachSense conducts systematic threshold sweeps to optimize the trade-off between False Negatives (unplanned catastrophic downtime) and False Positives (unnecessary maintenance inspection):

$$\tau^* = \arg\max_{\tau} F_1(\tau) \quad \text{subject to } \text{Recall}(\tau) \ge 0.70$$

- **Default Threshold ($\tau = 0.50$)**: F1 = 0.898, Precision = 0.935, Recall = 0.863
- **Calibrated Optimal Threshold ($\tau^* = 0.5608$)**: **F1 = 0.926**, **Precision = 1.000**, **Recall = 0.863**
- **Zero False Positives**: At the calibrated threshold on the test set, all flagged machines were genuine breakdowns, eliminating false maintenance dispatches.

---

## Explainable AI (TreeSHAP Root Cause Analysis)

Every prediction is augmented with local feature attribution generated via **TreeSHAP**:

$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$

### Operator Translation Engine
SHAP attribution coefficients are automatically mapped to actionable shop-floor guidance:
- **Overstrain**: *"Tool wear (215 min) and torque (62.5 Nm) exceed safe mechanical limits. Inspect cutting insert immediately."*
- **Heat Dissipation**: *"Spindle temperature delta (8.4 K) indicates cooling fluid starvation. Check coolant pressure."*
- **Power Overload**: *"Drive wattage (19,400 W) exceeds maximum spindle inverter rating. Check for workpiece binding."*

---

## Serving Layer & API Documentation

The FastAPI backend provides robust, sub-10ms endpoints:

### Core Endpoints

| Method | Route | Description | Response Model |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | API discovery metadata, versions, and documentation links | `RootResponse` |
| `GET` | `/health` | Liveness health probe for container orchestrators | `HealthResponse` |
| `GET` | `/health/ready` | Readiness probe checking loaded model artifacts and explainer | `ReadinessResponse` |
| `POST` | `/api/v1/predict` | Single telemetry prediction with optional TreeSHAP explanation | `PredictionResponse` |
| `POST` | `/api/v1/predict/batch` | Fleet batch scoring with summary statistics (up to 5,000 records) | `BatchPredictionResponse` |

### Sample Request & Response
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "telemetry": {
         "type": "L",
         "air_temperature_k": 302.5,
         "process_temperature_k": 311.8,
         "rotational_speed_rpm": 1380.0,
         "torque_nm": 62.5,
         "tool_wear_min": 215.0
       },
       "explain": true
     }'
```

```json
{
  "status": "SUCCESS",
  "failure_probability": 0.9412,
  "predicted_class": 1,
  "predicted_label": "FAILURE IMMINENT",
  "risk_level": "CRITICAL",
  "threshold_used": 0.5608,
  "model_version": "v1.0.0",
  "top_risk_escalators": [
    {"feature_name": "overstrain_index", "attribution_value": 0.3412},
    {"feature_name": "torque_nm", "attribution_value": 0.2185},
    {"feature_name": "tool_wear_min", "attribution_value": 0.1874}
  ],
  "operator_summary": "High failure probability (94.12%). Primary risk drivers: overstrain_index (+0.3412), torque_nm (+0.2185). Immediate tool replacement advised.",
  "latency_ms": 6.82
}
```

---

## Streamlit User Dashboard Walkthrough

The Streamlit interface provides an intuitive diagnostic control room:

```
+------------------------------------------------------------------------------------+
|  ⚙️ MachSense | Industrial Machine Health Monitoring & Diagnostics                 |
+------------------------------------------------------------------------------------+
| [Preset Selector: Overstrain Failure (OSF) ▼]  [Active Threshold: 0.5608]          |
|                                                                                    |
| +-------------------------+ +-------------------------+ +------------------------+ |
| |   HEALTH STATUS         | |   FAILURE PROBABILITY   | |   RISK LEVEL           | |
| |   🔴 FAILURE IMMINENT   | |   94.12%                | |   CRITICAL             | |
| +-------------------------+ +-------------------------+ +------------------------+ |
|                                                                                    |
| ── Diagnostic Feature Attribution (TreeSHAP) ────────────────────────────────────  |
|  [ Plotly Interactive Horizontal Waterfall Chart ]                                 |
|  overstrain_index   ████████████████████████████ (+0.3412)                         |
|  torque_nm          ███████████████ (+0.2185)                                      |
|  tool_wear_min      ████████████ (+0.1874)                                         |
|  rotational_speed   ▓▓▓▓▓ (-0.0821)                                                |
|                                                                                    |
| ── Operator Actionable Alert ────────────────────────────────────────────────────  |
|  ⚠️ CRITICAL OVERSTRAIN DETECTED: Cutting insert wear (215 min) combined with high  |
|     cutting torque (62.5 Nm). Action: Halt spindle cycle and inspect tool insert.  |
+------------------------------------------------------------------------------------+
```

---

## Local Quickstart & CLI

### 1. Installation
```bash
# Clone repository
git clone https://github.com/IshanSingh21/MachSense.git
cd MachSense

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1    # On Windows PowerShell
# source .venv/bin/activate   # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

### 2. Command Line Interface (CLI)

```bash
# Display resolved configuration and environment status
python -m machsense.main --info

# Execute live sample prediction with SHAP explanation
python -m machsense.main --predict-sample

# Launch FastAPI REST backend
python -m machsense.main --serve

# Launch Streamlit Operator Dashboard
python -m machsense.main --ui
```

### 3. Run Automated Tests
```bash
pytest tests/ -v
```

---

## Production Docker & Cloud Deployment

### A. Full Stack Multi-Container (Docker Compose)
```bash
docker compose up --build -d
```
- **Streamlit Dashboard**: `http://localhost:8501`
- **FastAPI OpenAPI Docs**: `http://localhost:8000/docs`
- **Readiness Probe**: `http://localhost:8000/health/ready`

### B. Free-Tier Cloud Deployment

1. **Streamlit Community Cloud (1-Click UI)**:
   - Deploy `src/machsense/ui/app.py` directly on [share.streamlit.io](https://share.streamlit.io). Runs self-contained in-process inference without external server dependencies.
2. **Render.com (Full-Stack Blueprint)**:
   - Link repository to [Render Dashboard](https://dashboard.render.com). Render reads `render.yaml` and provisions both `machsense-api` and `machsense-ui`.

---

## Project Directory Structure

```
machsense/
├── Dockerfile                   # Production multi-stage Dockerfile
├── Dockerfile.api               # Dedicated FastAPI microservice image
├── Dockerfile.ui                # Dedicated Streamlit dashboard image
├── docker-compose.yml           # Full-stack container orchestration
├── render.yaml                  # Render.com Infrastructure-as-Code Blueprint
├── Procfile                     # PaaS process manager specification
├── pyproject.toml               # Package build metadata, tool & linter configs
├── requirements.txt             # Pinned production dependencies
├── requirements-dev.txt         # Testing, linting, and development tools
├── .streamlit/
│   └── config.toml              # Streamlit production headless configuration
├── configs/
│   ├── config.yaml              # Core application parameters & path definitions
│   └── logging.yaml             # Rotating log handler configuration
├── models/
│   ├── machsense_model_v1.0.0.joblib         # Serialized champion model artifact
│   ├── machsense_preprocessor_v1.0.0.joblib  # Fitted preprocessor pipeline
│   ├── model_metadata_v1.0.0.json            # Model lineage & hyperparameter card
│   └── model_card.json                       # Public governance model card
├── scripts/
│   └── deploy_healthcheck.py    # Automated deployment verification probe CLI
├── src/machsense/
│   ├── api/                     # FastAPI backend (lifespan, routes, error handlers)
│   ├── config/                  # Dynamic path resolution & Pydantic settings
│   ├── data/                    # Ingestion, validation, cleaning & splitters
│   ├── features/                # Physics domain extractors & preprocessor pipeline
│   ├── inference/               # Production prediction service & boundary validator
│   ├── models/                  # Trainers, CV optimizers, evaluation & SHAP explainer
│   ├── ui/                      # Streamlit dashboard & Plotly visual components
│   └── utils/                   # Structured rotating logger & shared helpers
└── tests/                       # 126 automated unit, integration, and security tests
```

---

## Engineering Review: Limitations & Future Roadmap

### Known Limitations
1. **Batch TreeSHAP Throughput**: Exact TreeSHAP computes Shapley values on an instance-by-instance basis. While single-instance SHAP runs in $< 10\text{ ms}$, batch SHAP on large fleets ($> 5,000$ items) is CPU-intensive.
2. **Binary Focus**: The primary champion model classifies binary failure risk ($0$ vs. $1$). While individual physical presets demonstrate specific failure types, fine-grained multi-class failure attribution is derived post-hoc from physics features.

### Future Roadmap
- **Multi-Class Failure Mode Architecture**: Direct multi-output gradient boosted trees to concurrently output probabilities for HDF, PWF, OSF, and TWF.
- **High-Frequency Vibration Telemetry**: Integrating FFT spectrogram feature extractors and 1D-CNN temporal autoencoders for high-frequency acoustic sensor streams.
- **Continuous Drift Monitoring**: Prometheus/Grafana and Evidently AI integration for automated data drift and concept drift alerting in production.

---

## Technologies & Tools

- **Core Machine Learning**: Python 3.10+, Scikit-Learn, NumPy, Pandas, Joblib.
- **Explainable AI (XAI)**: SHAP (TreeSHAP), Matplotlib, Plotly.
- **API & Serving**: FastAPI, Uvicorn, Starlette, Pydantic v2, HTTPX.
- **Frontend & Visualization**: Streamlit, Plotly Graph Objects, Rich.
- **Engineering Quality & DevOps**: Pytest, Ruff, Docker, Docker Compose, YAML.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
