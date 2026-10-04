# MachSense: Industrial AI Predictive Maintenance & Machine Health Diagnostics

[![Next.js 16](https://img.shields.io/badge/Next.js-16.3-black?logo=next.js&logoColor=white)](https://nextjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Tailwind CSS v4](https://img.shields.io/badge/Tailwind_CSS-v4.0-38B2AC?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![SHAP](https://img.shields.io/badge/Explainability-TreeSHAP-brightgreen.svg)](https://github.com/slundberg/shap)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![Tests: 126 Passed](https://img.shields.io/badge/tests-126%20passed-success.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**MachSense** is an enterprise-grade predictive maintenance and machine health monitoring system. It processes real-time industrial telemetry (temperature gradients, spindle speeds, mechanical torque, cutting tool wear), computes physics-informed features, forecasts catastrophic machine failure with cost-calibrated decision thresholds ($\tau = 0.5608$), and provides root-cause diagnostic explanations using TreeSHAP on a modern **Next.js Industrial Control Center**.

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
9. [Next.js Industrial Control Center](#nextjs-industrial-control-center)
10. [Serving Layer & FastAPI Documentation](#serving-layer--fastapi-documentation)
11. [Local Quickstart & Execution](#local-quickstart--execution)
12. [Production Docker & Cloud Deployment](#production-docker--cloud-deployment)
13. [Environment Variables Reference](#environment-variables-reference)
14. [Project Directory Structure](#project-directory-structure)
15. [Engineering Review: Limitations & Roadmap](#engineering-review-limitations--roadmap)
16. [Technologies & Tools](#technologies--tools)

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
- **Modern Industrial SaaS Frontend**: Next.js 16 App Router interface styled in dark industrial mission-control theme with Recharts and Lucide icons.
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
|              FASTAPI REST BACKEND              | |         NEXT.JS INDUSTRIAL CONTROL CENTER       |
| - Liveness & Readiness Probes (/health, /ready)| | - Overview KPIs, Donut Health Chart, Timeline   |
| - Single Predict & Vectorized Fleet Batch      | | - Machines Directory & Machine Detail View      |
| - Pydantic v2 Schema & Error Sanitization      | | - Prediction Studio, Recharts XAI Waterfall     |
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

## Next.js Industrial Control Center

The frontend provides a mission-control industrial dashboard:

```
+------------------------------------------------------------------------------------+
| ⚙️ MACHSENSE PRO [LIVE]              14:22:05 UTC • Fleet: 95.8% Nominal • ENG-408 |
+------------------------------------------------------------------------------------+
| [Overview]  [Machines]  [Predictions]  [Analytics]  [Alerts]  [XAI]  [Settings]    |
|                                                                                    |
| +-------------------+ +-------------------+ +-------------------+ +---------------+ |
| | MONITORED FLEET   | | HEALTHY MACHINES  | | AT-RISK WARNING   | | CRITICAL      | |
| | 24 Units          | | 20 (95.8%)        | | 3 Approaching     | | 1 Action Req  | |
| +-------------------+ +-------------------+ +-------------------+ +---------------+ |
|                                                                                    |
| ── Featured Machine Telemetry (CNC Heavy Lathe Alpha - CNC-002) ────────────────── |
|  Air Temp: 302.5 K   Process Temp: 311.8 K   Torque: 62.5 Nm   Speed: 1380 RPM     |
|  Failure Risk: 🔴 94.12% CRITICAL (Threshold: 0.5608)                              |
|                                                                                    |
| ── TreeSHAP Feature Attribution Waterfall (Recharts) ────────────────────────────  |
|  overstrain_index   ████████████████████████████ (+0.3412)                         |
|  torque_nm          ███████████████ (+0.2185)                                      |
|  tool_wear_min      ████████████ (+0.1874)                                         |
|  rotational_speed   ▓▓▓▓▓ (-0.0821)                                                |
|                                                                                    |
| ── Operator Diagnostic Guidance ─────────────────────────────────────────────────  |
|  ⚠️ CRITICAL OVERSTRAIN DETECTED: Cutting insert wear (215 min) combined with high  |
|     cutting torque (62.5 Nm). Action: Halt spindle cycle and inspect tool insert.  |
+------------------------------------------------------------------------------------+
```

---

## Serving Layer & FastAPI Documentation

The FastAPI backend provides robust, sub-10ms endpoints:
- **FastAPI Backend API**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **FastAPI Interactive Docs (Swagger UI)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **FastAPI Alternative Docs (ReDoc)**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

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

## Local Quickstart & Execution

### 1. Backend & ML Environment Setup
```bash
# Clone repository
git clone https://github.com/IshanSingh21/MachSense.git
cd MachSense

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1    # On Windows PowerShell
# source .venv/bin/activate   # On Linux/macOS

# Install backend dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .

# Start FastAPI server
python -m machsense.main --serve
```
* **FastAPI Server**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **FastAPI Interactive Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### 2. Next.js Frontend Execution
```bash
# Start frontend from repository root (or cd frontend && npm run dev)
npm.cmd run dev
```
* **Next.js Mission Control**: Open **[http://localhost:3000](http://localhost:3000)**

---

### 3. Run Automated Tests
```bash
pytest tests/ -v
```

---

## Production Docker & Cloud Deployment

### A. Full Stack Orchestration (Docker Compose)
Launch the FastAPI backend and Next.js Industrial Frontend in isolated containers:

```bash
docker compose up --build -d
```
- **Next.js Frontend**: [http://localhost:3000](http://localhost:3000)
- **FastAPI Backend**: [http://localhost:8000](http://localhost:8000)
- **FastAPI Interactive Docs (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Environment Variables Reference

| Variable | Scope | Default | Description |
| :--- | :--- | :--- | :--- |
| `NEXT_PUBLIC_API_URL` | Frontend | `http://127.0.0.1:8000` | Target FastAPI backend URL for Next.js API client |
| `MACHSENSE_ENV` | Backend | `development` | Operating environment (`development`, `staging`, `production`) |
| `PORT` | Backend | `8000` | Application listening port (injected dynamically by PaaS providers) |
| `HOST` | Backend | `127.0.0.1` / `0.0.0.0` | Server bind host address |
| `MACHSENSE_CORS_ORIGINS` | Backend | `["*"]` | Allowed CORS origins (e.g. `http://localhost:3000`) |
| `MACHSENSE_MODELS_DIR` | Backend | `models` | Custom directory path for model artifacts |

---

## Project Directory Structure

```
machsense/
├── frontend/                    # Next.js 16 Industrial SaaS Control Center
│   ├── src/
│   │   ├── app/                 # App Router pages (/, /machines, /predictions, etc.)
│   │   ├── components/          # Reusable UI & Recharts components
│   │   ├── lib/                 # Typed API client & realistic mock data
│   │   └── types/               # TypeScript interfaces
│   ├── Dockerfile               # Multi-stage production container for Next.js
│   ├── package.json             # Frontend dependencies (React 19, Recharts, Lucide)
│   └── .env.example             # Frontend environment variable template
├── package.json                 # Root workspace scripts (npm run dev/build)
├── Dockerfile                   # FastAPI Python production image
├── Dockerfile.api               # Dedicated FastAPI microservice image
├── docker-compose.yml           # Full-stack container orchestration (API + Frontend)
├── pyproject.toml               # Package build metadata, tool & linter configs
├── requirements.txt             # Pinned production dependencies
├── configs/                     # Application & logging configuration
├── models/                      # Serialized champion model artifacts (v1.0.0)
├── src/machsense/               # Core Python package (API, Data, Models, XAI)
└── tests/                       # Automated unit, integration, and security tests
```

---

## Technologies & Tools

- **Frontend & Visualizations**: Next.js 16, React 19, TypeScript, Tailwind CSS v4, Recharts, Lucide Icons.
- **Core Machine Learning**: Python 3.10+, Scikit-Learn, NumPy, Pandas, Joblib.
- **Explainable AI (XAI)**: TreeSHAP, Additive Feature Attribution.
- **Backend & Serving**: FastAPI, Uvicorn, Starlette, Pydantic v2, HTTPX.
- **DevOps & Quality**: Pytest, Ruff, Docker, Docker Compose.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
