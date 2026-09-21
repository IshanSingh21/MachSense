# MachSense Explainability Guide: SHAP (SHapley Additive exPlanations)

This guide documents the theoretical framework, interpretation methodology, operational translation rules, and safety boundaries for model explainability in **MachSense**.

---

## 1. Executive Summary & Purpose

In industrial predictive maintenance, high statistical accuracy is insufficient on its own. Shop-floor maintenance technicians, reliability engineers, and plant managers cannot act on "black-box" failure probabilities without understanding:
1. **Which specific machine subsystems are driving the elevated risk** (e.g., thermal dissipation breakdown, excessive mechanical torque, or cumulative tool wear).
2. **Which operational parameters are stabilizing the machine** within a nominal safe regime.
3. **What specific physical inspection should be performed** before shutting down a production line.

MachSense integrates **TreeSHAP** (SHapley Additive exPlanations) directly on tree ensemble models (such as the champion `RandomForestClassifier` v1.0.0) to compute mathematically exact, game-theoretically grounded feature attributions.

---

## 2. Theoretical Foundations: Shapley Values

SHAP unifies cooperative game theory with machine learning interpretability. In MachSense:
- **Game**: The machine failure probability prediction task for a given sensor telemetry observation $x$.
- **Players**: The $M$ sensor features and engineered physics indices ($x_1, x_2, \dots, x_M$).
- **Payout**: The model's prediction score $f(x)$ relative to the baseline expected value $\mathbb{E}[f(X)]$.

The marginal contribution of feature $j$ across all possible feature subsets $S \subseteq F \setminus \{j\}$ is defined by the classical Shapley value formula:

$$\phi_j(x) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \Big[ f_{x}(S \cup \{j\}) - f_{x}(S) \Big]$$

### Four Core Axiomatic Guarantees
1. **Efficiency (Additivity)**: The sum of all feature attributions exactly equals the difference between the model prediction and the base expected value:
   $$\sum_{j=1}^{M} \phi_j(x) = f(x) - \mathbb{E}[f(X)]$$
2. **Symmetry**: If two features contribute identically across all possible subsets, their SHAP values are equal ($\phi_j = \phi_k$).
3. **Dummy (Null Player)**: A feature that does not change the model's prediction in any coalition receives a SHAP value of zero ($\phi_j = 0$).
4. **Additivity (Linearity)**: For an ensemble of trees (like a Random Forest), the SHAP value of the ensemble is the weighted average of the SHAP values across the individual trees:
   $$\phi_j^{\text{ensemble}} = \frac{1}{T} \sum_{t=1}^{T} \phi_{j}^{(t)}$$

---

## 3. Global vs. Local Explainability

### A. Global Feature Importance
Global importance quantifies the overall impact of each feature across the entire historical operational fleet:

$$I_j = \frac{1}{N} \sum_{i=1}^{N} |\phi_{i,j}|$$

In MachSense evaluation on test telemetry:
1. **`power_w` (Rotational Power)** & **`rotational_speed_rpm`**: Lead global importance (~17.7% and ~17.3% relative contribution) due to their sensitivity to Power Failures (PWF) and spindle stalling.
2. **`torque_nm`**: Third (~14.9% relative contribution), critical for mechanical overload detection.
3. **`tool_wear_min`** & **`overstrain_index`**: Primary discriminators for cumulative Tool Wear Failures (TWF) and Overstrain Failures (OSF).
4. **`temp_ratio`** & **`temp_difference_k`**: Decisive for Heat Dissipation Failures (HDF).
5. **Product Type One-Hot Encodings (`type_L`, `type_M`, `type_H`)**: Low global importance (<1% each), demonstrating that physical dynamics dominate categorical metadata.

### B. Local Instance Attribution
For any single machine telemetry cycle $x$:
- **Base Value $\mathbb{E}[f(x)]$**: The average failure probability across the training distribution (~49.9% under balanced subsampling).
- **Risk Escalators ($\phi_j > 0$)**: Sensor signals pushing the probability toward a critical failure alarm.
- **Stabilizers ($\phi_j < 0$)**: Sensor signals indicating nominal, safe operational bounds holding the risk down.

---

## 4. Translating SHAP into Shop-Floor Operator Diagnostics

Raw numerical SHAP values (e.g. `+0.2194`) are unhelpful to a machine operator under pressure. MachSense provides an automated translation layer that maps mathematical attributions to actionable maintenance guidance.

### Feature Mapping Table

| Internal Feature Name | Operator Display Name | Typical Failure Association |
|---|---|---|
| `power_w` | Electric/Mechanical Power [W] | Power Failure (PWF) - Spindle stall or excessive speed |
| `rotational_speed_rpm` | Spindle Rotational Speed [rpm] | Power Failure / Heat Dissipation |
| `torque_nm` | Mechanical Torque [Nm] | Overstrain / Tool Jamming |
| `temp_difference_k` | Process-Air Temp Difference [K] | Heat Dissipation Failure (HDF) - Cooling breakdown |
| `temp_ratio` | Process/Air Temp Ratio | Heat Dissipation Failure (HDF) |
| `overstrain_index` | Mechanical Overstrain Index [Nm * min] | Overstrain Failure (OSF) - High torque with worn tool |
| `tool_wear_min` | Tool Wear Accumulated [min] | Tool Wear Failure (TWF) - Worn cutting insert |
| `tool_wear_rate` | Tool Wear Rate [min / rpm] | Dynamic tool degradation |

### Operator Alert Format Example
```text
[CRITICAL ALERT] Failure Probability: 93.7% (Baseline: 49.9%)
Immediate inspection recommended. Machine is exhibiting severe failure precursor signatures.

Key Risk Escalators (Factors Increasing Failure Likelihood):
  - Spindle Rotational Speed [rpm]: +0.2194 SHAP impact (31.0% attribution)
  - Process/Air Temp Ratio: +0.1624 SHAP impact (22.9% attribution)
  - Process-Air Temp Difference [K]: +0.1357 SHAP impact (19.2% attribution)

Key Stabilizing Factors (Factors Maintaining Safety Margin):
  - Electric/Mechanical Power [W]: -0.0541 SHAP impact (7.6% attribution)
  - Mechanical Overstrain Index [Nm * min]: -0.0306 SHAP impact (4.3% attribution)

Safety Note: DISCLAIMER (NON-CAUSAL ATTRIBUTION): SHAP values reflect the mathematical contribution of each feature to the model's output probability under the learned decision trees. Attribution indicates correlation and predictive association within the historical training distribution, not necessarily a direct physical root cause. Always verify physical machinery before taking destructive action.
```

---

## 5. Non-Causal Safety Boundaries & Critical Limitations

> [!WARNING]
> **Correlation $\neq$ Physical Causation**: A high positive SHAP value indicates that a feature value was strongly associated with machine failure in the training data. It does NOT prove that intervening on that variable alone will prevent the failure if an underlying unobserved physical defect exists.

### Key Operational Constraints:
1. **Multicollinearity & Feature Interaction**:
   - `power_w` is mathematically derived from `rotational_speed_rpm` and `torque_nm`.
   - In correlated features, TreeSHAP attributes credit across the correlated coalition. Operators should evaluate the combined subsystem (e.g. electrical drive train) rather than isolating a single variable.
2. **Random Hardware Failures (RNF)**:
   - As established in Day 7 Error Analysis, Random Failures have 0% precursor correlation in sensor data. SHAP explanations for RNF instances will show near-zero attributions or noise-level attributions.
3. **Out-of-Distribution (OOD) Telemetry**:
   - If a machine operates outside the training envelope (e.g. extreme ambient temperature exceeding 315 K), TreeSHAP attributions follow the deepest tree splits, which may not reflect real physical failure modes.

---

## 6. How to Use in MachSense

```python
import joblib
import pandas as pd
from machsense.models.explainability import MachSenseExplainer

# Load production model and data
model = joblib.load("models/best_model.joblib")
X_test = pd.read_csv("data/processed/X_test_transformed.csv")

# Initialize explainer
explainer = MachSenseExplainer(model=model, feature_names=list(X_test.columns))

# Compute global ranking
global_importance = explainer.compute_global_importance(X_test)
print(global_importance.head())

# Compute local explanation for machine cycle #11
local_diag = explainer.explain_instance(X_test.iloc[11], threshold=0.525)
print(local_diag.operator_summary)

# Generate visualization
fig = explainer.plot_waterfall(X_test.iloc[11], save_path="reports/figures/waterfall_sample_11.png")
```
