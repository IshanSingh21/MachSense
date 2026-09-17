# MachSense - Domain Feature Engineering Guide

## 1. Overview
In industrial predictive maintenance, purely statistical features often fail to capture the physical degradation dynamics of machinery. MachSense implements a domain-informed feature pipeline based on thermodynamic laws, rotational mechanical power equations, and tool cutting mechanics.

---

## 2. Engineered Features & Physical Justification

| Feature Name | Mathematical Formula | Physical Units | Target Failure Mode | Physical Justification |
| :--- | :--- | :---: | :---: | :--- |
| `power_w` | $\tau \times \left(\text{RPM} \times \frac{2\pi}{60}\right)$ | Watts ($W$) | **PWF** (Power Failure) | Real-time mechanical cutting power. Active power draw is bounded ($<3.5\text{ kW}$ or $>9.0\text{ kW}$). Rather than forcing models to learn the non-linear hyperbolic product $\tau \cdot \omega$, this feature directly linearizes power boundaries. |
| `temp_difference_k` | $T_{\text{process}} - T_{\text{air}}$ | Kelvin ($K$) | **HDF** (Heat Dissipation Failure) | Measures heat transfer efficiency between the cutting interface and ambient environment. Failures occur when $\Delta T < 8.6\text{ K}$ due to coolant breakdown or fan degradation. |
| `temp_ratio` | $T_{\text{process}} / T_{\text{air}}$ | Dimensionless | **HDF** / Thermal Strain | Dimensionless relative thermal coefficient. Normalizes thermal elevation relative to baseline ambient shifts. |
| `overstrain_index` | $\tau \times \text{tool\_wear}$ | $N\cdot m \cdot \min$ | **OSF** (Overstrain Failure) | Measures cumulative stress product on the cutting spindle. When heavily worn tools encounter high torque resistance, mechanical strain exceeds tensile thresholds, resulting in tool breakage. |
| `tool_wear_rate` | $\frac{\text{tool\_wear}}{\text{RPM} + 1\times 10^{-5}}$ | $\min / \text{rpm}$ | **TWF** (Tool Wear Failure) | Normalizes tool degradation relative to rotational cycles. |

---

## 3. Preprocessing Architecture & Leakage Prevention

```
Raw Telemetry (X_train) 
   │
   ├─► DomainFeatureExtractor (Physics features: power_w, temp_difference_k, overstrain_index...)
   │      │
   │      ▼
   ├─► Categorical Pipeline (type)
   │      ├── SimpleImputer(strategy='most_frequent')
   │      └── OneHotEncoder(handle_unknown='ignore', categories=[['L', 'M', 'H']])
   │
   └─► Numerical Pipeline (Sensors + Domain Features)
          ├── SimpleImputer(strategy='median')  <-- Fitted ONLY on X_train
          └── StandardScaler() or RobustScaler() <-- Fitted ONLY on X_train
```

### Key Principles:
1. **Fit Exclusively on Training Data**: Imputation medians and scaling means/variances are calculated on `X_train` only. Validation, test, and live inference samples are transformed using these frozen training statistics.
2. **Inference Consistency**: The serialized `MachSensePreprocessor` artifact in `models/preprocessor.joblib` ensures that batch jobs and single-sample online API requests use the identical transformation graph.
