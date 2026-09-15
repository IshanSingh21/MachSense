# Dataset Card: AI4I 2020 Predictive Maintenance Dataset

## 1. Overview
- **Dataset Name**: AI4I 2020 Predictive Maintenance Dataset
- **Origin**: UCI Machine Learning Repository / Matan et al.
- **Task Type**: Binary and Multi-Class Anomaly / Equipment Failure Classification
- **Domain**: Smart Manufacturing, Milling Machines, Industrial IoT Telemetry
- **Record Count**: 10,000 observations
- **Feature Count**: 14 attributes (Identifier, Product Quality, 5 Physical Sensor Streams, 1 Overall Failure Target, 5 Failure Mode Indicators)

---

## 2. Feature Definitions & Physical Units

| Column Name (Standardized) | Original Header | Data Type | Physical Unit | Description / Constraints |
| :--- | :--- | :--- | :--- | :--- |
| `udi` | `UDI` | `int` | - | Unique Identifier ($1$ to $10,000$) |
| `product_id` | `Product ID` | `string` | - | Variant letter (`L`/`M`/`H`) + serial number |
| `type` | `Type` | `category` | - | Quality variant: `L` (50%), `M` (30%), `H` (20%) |
| `air_temperature_k` | `Air temperature [K]` | `float` | Kelvin ($K$) | Ambient air temperature, around $300\text{ K} \pm 2\text{ K}$ |
| `process_temperature_k` | `Process temperature [K]` | `float` | Kelvin ($K$) | Internal machine process temperature, around $310\text{ K}$ |
| `rotational_speed_rpm` | `Rotational speed [rpm]` | `float` / `int` | RPM | Spindle speed calculated for ~2860 W power |
| `torque_nm` | `Torque [Nm]` | `float` | Newton-meters ($N\cdot m$) | Torque applied to tool, normally distributed ($\sim 40\text{ N}\cdot\text{m}$) |
| `tool_wear_min` | `Tool wear [min]` | `float` | Minutes | Cumulative tool usage time before replacement |
| `machine_failure` | `Machine failure` | `int` (0/1) | - | Target indicator: $1$ if machine failed, $0$ otherwise ($339$ failures, ~3.39%) |
| `twf` | `TWF` | `int` (0/1) | - | Tool Wear Failure mode |
| `hdf` | `HDF` | `int` (0/1) | - | Heat Dissipation Failure mode ($\Delta T < 8.6\text{ K}$ and speed $< 1380\text{ rpm}$) |
| `pwf` | `PWF` | `int` (0/1) | - | Power Failure mode ($\text{Power} = \text{Torque} \times \omega < 3500\text{ W}$ or $> 9000\text{ W}$) |
| `osf` | `OSF` | `int` (0/1) | - | Overstrain Failure mode (tool wear $\times$ torque exceeds threshold) |
| `rnf` | `RNF` | `int` (0/1) | - | Random Failure mode ($0.1\%$ background probability) |

---

## 3. Physical Consistency Constraints

1. **Second Law of Thermodynamics (Heat Flow)**:
   - Under normal active cutting operation, internal process temperature must exceed ambient air temperature:
     $$T_{\text{process}} \ge T_{\text{air}}$$
2. **Positive Torque & Velocity**:
   - $\text{Rotational Speed} > 0\text{ rpm}$
   - $\text{Torque} > 0\text{ N}\cdot\text{m}$
3. **Monotonic Tool Wear**:
   - $\text{Tool Wear} \ge 0\text{ min}$

---

## 4. Class Imbalance & Evaluation Considerations
- The dataset has severe class imbalance: ~**3.39%** positive failure rate ($339 / 10,000$).
- Standard accuracy is an inappropriate metric. Evaluation must prioritize **Precision-Recall AUC (PR-AUC)**, **F1-Score (macro/minority)**, and **Recall@Top-K**.
- Data splitting must strictly use **stratified sampling** on `machine_failure` to avoid unrepresentative validation/test distributions.
