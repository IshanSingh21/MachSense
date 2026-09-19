# MachSense - Model Optimization Report (Day 6)

## Executive Summary
This report presents the hyperparameter tuning and model optimization results for the **MachSense** predictive maintenance platform.

Using **Stratified 5-Fold Cross-Validation** strictly on the training partition ($7,000$ machine cycles with $13$ engineered features), we tuned our leading tree ensembles (**Random Forest** and **Histogram Gradient Boosting**) to maximize **PR-AUC (Average Precision)** and minimize **False Negatives ($FN$)**.

---

## 1. Hyperparameter Search Spaces & CV Strategy

All search procedures used `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` scored by `average_precision` (PR-AUC) to address the $28.5 : 1$ class imbalance without test-set leakage.

### 1.1 Random Forest Search Space
- `n_estimators`: `[100, 200]`
- `max_depth`: `[8, 12, 16]`
- `min_samples_split`: `[2, 5]`
- `min_samples_leaf`: `[1, 2]`
- `class_weight`: `["balanced", "balanced_subsample"]`

### 1.2 Histogram Gradient Boosting Search Space
- `max_iter`: `[100, 150]`
- `learning_rate`: `[0.03, 0.08, 0.1]`
- `max_leaf_nodes`: `[15, 31, 63]`
- `min_samples_leaf`: `[10, 20]`
- `class_weight`: `["balanced", None]`

---

## 2. Baseline vs. Tuned Performance Comparison

| Model Architecture | Status | 5-Fold CV PR-AUC | Val PR-AUC | Val Recall (Catch Rate) | Val Precision | Val Minority F1 | Val Missed (FN) | Test PR-AUC | Test Recall | Test FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | Baseline (Day 5) | - | 0.8449 | 72.55% | 94.87% | 0.8222 | 14 | 0.9443 | 88.24% | 6 |
| **Random Forest** | **Tuned (Day 6)** | **0.8687** | **0.8652** | **78.43%** | **95.24%** | **0.8602** | **11** | **0.9585** | **90.20%** | **5** |
| **Hist Gradient Boosting** | Baseline (Day 5) | - | 0.8183 | 74.51% | 82.61% | 0.7835 | 13 | 0.9411 | 90.20% | 5 |
| **Hist Gradient Boosting** | **Tuned (Day 6)** | **0.8412** | **0.8385** | **76.47%** | **84.78%** | **0.8041** | **12** | **0.9482** | **90.20%** | **5** |

---

## 3. Final Champion Model Selection & Rationale

**Champion Model**: `tuned_random_forest` (Registered as Version **`v1.0.0`**)

### Optimal Hyperparameters:
```json
{
  "n_estimators": 100,
  "max_depth": 16,
  "min_samples_split": 2,
  "min_samples_leaf": 1,
  "class_weight": "balanced_subsample",
  "random_state": 42
}
```

### Business & ML Selection Rationale:
1. **Highest PR-AUC and F1-Score**: Outperformed all architectures with **0.8687** cross-validation PR-AUC and **0.8652** validation PR-AUC.
2. **False Negative Reduction**: Reduced validation missed failures from $14$ down to $11$ ($78.43\%$ recall) at the default threshold, and achieves **$90.20\%$ recall** ($46/51$ caught) on the held-out test split.
3. **Subsample Class Balancing (`balanced_subsample`)**: Re-estimates class weights dynamically for each bootstrap tree sample, preventing majority class overfitting in deep branches.
4. **Interpretability & XAI Readiness**: Readily compatible with **SHAP TreeExplainer** for real-time feature attribution.

---

## 4. Versioned Artifact Registry Structure

Artifacts are registered under `models/` following semantic versioning (`v1.0.0`):

```
models/
├── machsense_model_v1.0.0.joblib        # Versioned champion model binary
├── machsense_preprocessor_v1.0.0.joblib # Versioned preprocessor binary
├── model_metadata_v1.0.0.json           # Versioned metadata & threshold config
├── best_model.joblib                    # Active production model pointer (Gitignored)
├── preprocessor.joblib                  # Active preprocessor pointer (Gitignored)
├── model_card.json                      # Active model card metadata
└── tuning_experiments.json              # Full cross-validation tuning logs
```
