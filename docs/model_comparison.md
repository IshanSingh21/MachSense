# MachSense - ML Baselines & Model Comparison Report

## Executive Summary
This report benchmarks 5 diverse classification model families on the MachSense predictive maintenance dataset ($10,000$ machine cycles with $13$ engineered features).

Given the **$28.5 : 1$ class imbalance** ($3.39\%$ failure rate), models are evaluated strictly on **PR-AUC (Average Precision)**, **Recall (Failure Catch Rate)**, **Minority F1-Score**, and **False Negative count ($FN$)**.

---

## 1. Multi-Model Leaderboard (Validation Set)

| Rank | Model Architecture | PR-AUC (Primary) | Recall (Caught Failures) | Precision | F1-Score (Minority) | ROC-AUC | Missed Failures (FN) | FN Rate | Accuracy (Deceptive) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Random Forest (`class_weight='balanced'`)** | **0.8842** | **84.31%** (43/51) | **82.69%** | **0.8350** | **0.9856** | **8** | **15.69%** | 98.87% |
| 🥈 | **Hist Gradient Boosting** | **0.8710** | **86.27%** (44/51) | **78.57%** | **0.8224** | **0.9812** | **7** | **13.73%** | 98.67% |
| 🥉 | **Decision Tree (Max Depth 6)** | **0.7895** | **80.39%** (41/51) | **73.21%** | **0.7664** | **0.9520** | **10** | **19.61%** | 98.20% |
| 4 | **Logistic Regression (`balanced`)** | **0.5840** | **84.31%** (43/51) | **28.10%** | **0.4216** | **0.9234** | **8** | **15.69%** | 91.47% |
| 5 | **Dummy (Most Frequent)** | **0.0340** | **0.00%** (0/51) | **0.00%** | **0.0000** | **0.5000** | **51** | **100.00%** | 96.60% |

---

## 2. In-Depth False Negative ($FN$) Analysis

### The Cost Asymmetry of Industrial Failures
In manufacturing plant operations:
- **False Negative ($FN$) = Undetected Machine Failure**:
  - The spindle snaps, bearing seizes, or thermal overheating burns out the motor during high-speed milling.
  - Estimated impact: **\$5,000 – \$50,000+** in ruined workpieces, emergency technician dispatch, and hours of factory downtime.
- **False Positive ($FP$) = False Alarm**:
  - Operator inspects machine during scheduled break and verifies sensor calibration.
  - Estimated impact: **\$50 – \$150** in technician inspection time.

### Model Behavior on False Negatives:
1. **Dummy Baseline**: Achieves $96.60\%$ accuracy, but misses **$100\%$** of failures ($51/51$ missed). Demonstrates why accuracy is forbidden as a primary metric.
2. **Logistic Regression**: Linear hyperplanes capture high-torque and low-speed zones ($84.3\%$ recall), but generate $110$ False Positives ($28.1\%$ precision), leading to alarm fatigue.
3. **Random Forest & Gradient Boosting**: Non-linear tree splits effectively isolate multi-variable failure envelopes (such as overstrain $\tau \times \text{wear} > 11,000$ and thermal dissipation $\Delta T < 8.6\text{ K}$), reducing missed failures to $\le 8$ while maintaining $>80\%$ precision.

---

## 3. Decision Threshold Tuning for Operational Needs

By adjusting the classification probability threshold away from the default $0.50$, plant managers can calibrate the model to match specific operational risk tolerances:

- **Balanced F1 Configuration (Threshold = 0.52)**:
  - F1-Score: **0.84** | Recall: **84.3%** | Precision: **84.0%**
  - Best for general plant monitoring with balanced maintenance schedules.
- **High-Sensitivity / Mission-Critical Configuration (Threshold = 0.30)**:
  - Recall: **94.1%** ($48/51$ failures caught) | Precision: **68.0%** | FN: **3**
  - Best for high-value production lines where any catastrophic failure must be avoided at all costs.

---

## 4. Champion Model Selection

**Champion Model**: `RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)`

### Selection Rationale:
1. Highest PR-AUC (**0.8842**) and F1-Score (**0.8350**) across validation splits.
2. Excellent generalization on the unseen Test partition (**0.8790 PR-AUC**, **86.3% Recall**).
3. Highly compatible with tree explainability methods (**SHAP TreeExplainer**) for Day 6 interpretability.
4. Serialized and saved to `models/best_model.joblib`.
