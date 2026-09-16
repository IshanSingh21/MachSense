# MachSense - Exploratory Data Analysis (EDA) Findings Report

## Executive Summary
This document consolidates the findings from the comprehensive Exploratory Data Analysis (EDA) conducted on the AI4I 2020 predictive maintenance benchmark dataset ($10,000$ machine telemetry cycles).

---

## 1. Target Distribution & Class Imbalance Analysis

### 1.1 Class Imbalance Severity
- **Total Operational Records**: $10,000$
- **Non-Failure Records ($y=0$)**: $9,661$ ($96.61\%$)
- **Machine Failures ($y=1$)**: $339$ ($3.39\%$)
- **Class Imbalance Ratio**: **$28.5 : 1$**

### 1.2 Breakdown by Specific Failure Modes
The 339 total failure events originate from 5 distinct physical failure modes:

| Failure Mode | Abbreviation | Total Occurrences | % of Total Records | Physical Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| **Heat Dissipation Failure** | `HDF` | 115 | 1.15% | Insufficient heat transfer ($\Delta T < 8.6\text{ K}$ and Speed $< 1380\text{ rpm}$) |
| **Power Failure** | `PWF` | 95 | 0.95% | Electrical/mechanical power ($P = \tau \cdot \omega$) out of bounds ($<3500\text{ W}$ or $>9000\text{ W}$) |
| **Overstrain Failure** | `OSF` | 98 | 0.98% | Mechanical product of tool wear $\times$ torque exceeds material yield limit |
| **Tool Wear Failure** | `TWF` | 46 | 0.46% | Direct tool replacement wear threshold reached ($>200\text{ min}$) |
| **Random Failure** | `RNF` | 19 | 0.19% | Uncorrelated background stochastic hardware fault ($0.1\%$ probability) |

### 1.3 Evaluation Metric Strategy
Due to the 28.5:1 imbalance:
1. **Accuracy is prohibited as an optimization metric**: A dummy model outputting all zeros achieves $96.61\%$ accuracy while failing completely at industrial prevention.
2. **Primary Metric**: **Precision-Recall Area Under Curve (PR-AUC / Average Precision)**.
3. **Secondary Metrics**: **F1-Score (Minority Class)**, **Recall@Top-K**, and **Cost-Weighted Loss** (where false negatives carry heavy unplanned downtime costs vs minor false positive inspection costs).

---

## 2. Feature Distributions & Outlier Analysis

| Feature | Type | Mean | Std | Skewness | Outlier Count (1.5*IQR) | Key Distribution Characteristics |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `air_temperature_k` | Float | $300.0\text{ K}$ | $2.0\text{ K}$ | $+0.11$ | 0 | Normal Gaussian ambient thermal distribution. |
| `process_temperature_k` | Float | $310.0\text{ K}$ | $1.5\text{ K}$ | $+0.02$ | 0 | Normal Gaussian, strictly higher than air temp. |
| `rotational_speed_rpm` | Int/Float | $1538.8\text{ rpm}$ | $179.3\text{ rpm}$ | $+1.99$ | 418 | Right-skewed; long tail at high speeds ($>2000\text{ rpm}$). |
| `torque_nm` | Float | $40.0\text{ Nm}$ | $9.97\text{ Nm}$ | $-0.00$ | 69 | Symmetric bell curve centered at $40\text{ Nm}$. |
| `tool_wear_min` | Int/Float | $107.9\text{ min}$ | $63.7\text{ min}$ | $+0.03$ | 0 | Uniformly distributed across $0$ to $253\text{ min}$. |
| `type` | Categorical | - | - | - | - | 50% `L` (Low), 30% `M` (Medium), 20% `H` (High). |

---

## 3. Correlation & Physical Interaction Signals

1. **Torque vs. Rotational Speed ($\rho = -0.88$)**:
   - Strong inverse relationship reflecting power conservation: $P = \tau \times \omega$. High speed corresponds to low cutting resistance, and heavy torque causes spindle deceleration.
2. **Process vs. Air Temperature ($\rho = +0.88$)**:
   - Ambient thermal conditions dictate baseline machine equilibrium temperature.
3. **Product Quality Variant Impact**:
   - Variant `L`: $3.92\%$ failure rate (highest risk).
   - Variant `M`: $2.77\%$ failure rate.
   - Variant `H`: $2.00\%$ failure rate (highest durability).

---

## 4. Observations vs. Assumptions

| Domain Area | Direct Data Observation (Fact) | Engineering Assumption (Hypothesis) |
| :--- | :--- | :--- |
| **Overstrain** | Failures cluster strongly when $\text{Torque} > 50\text{ Nm}$ AND $\text{Tool Wear} > 180\text{ min}$. | Creating an explicit interaction feature $\text{overstrain} = \text{torque} \times \text{tool\_wear}$ will simplify decision boundaries for tree ensembles. |
| **Heat Dissipation** | All HDF events occur when $\Delta T = T_{\text{proc}} - T_{\text{air}} < 8.6\text{ K}$ at low RPM. | Engineering $\Delta T$ and thermal ratio ($T_{\text{proc}} / T_{\text{air}}$) will serve as direct indicators of cooling system degradation. |
| **Power Faults** | Extreme high/low mechanical power envelope ($<3.5\text{ kW}$ or $>9.0\text{ kW}$) triggers PWF. | Engineering exact mechanical wattage $P = \tau \cdot \omega$ will allow models to capture electrical bounds linearly. |
| **Data Cleaning** | 0 missing values, 0 duplicate IDs in official benchmark. | Production IoT pipelines will require median/mode imputation fallback when sensor packets drop in real-time. |

---

## 5. Feature Engineering Roadmap (For Day 4)

Based on these physical findings, we will implement the following domain features in `machsense.features`:
1. `power_w`: $\tau \times (\text{RPM} \times \frac{2\pi}{60})$
2. `temp_difference_k`: $T_{\text{process}} - T_{\text{air}}$
3. `temp_ratio`: $T_{\text{process}} / T_{\text{air}}$
4. `overstrain_index`: $\tau \times \text{tool\_wear}$
5. `type_encoded`: One-hot or ordinal encoding of quality variant (`L`=0, `M`=1, `H`=2).
