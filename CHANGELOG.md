# Changelog


## v0.3
- Upgraded model to **RandomForestRegressor**
- Tuned parameters: `n_estimators=300`, `max_depth=5`, `min_samples_split=5`
- RMSE improved slightly from **53.78 → 53.72**
- Demonstrates transition from linear → ensemble-based model while maintaining reproducibility

| Version | RMSE | Δ (Improvement) |
|----------|------|-----------------|
| v0.1 | 53.85 | — |
| v0.2 | 53.78 | ↓ 0.08 |
| v0.3 | 53.72 | ↓ 0.06 |


## v0.2
- Upgraded model from **LinearRegression** to **Ridge Regression (α=1.0)**.
- Added 5-fold cross-validation for hyperparameter tuning.
- Slight improvement in generalization performance.

| Metric | v0.1 | v0.2 | Δ |
|---------|------|------|---|
| RMSE (holdout) | 53.85 | 53.78 | ↓ 0.07 |

**Rationale:**  
Ridge regression adds L2 regularization, helping reduce coefficient variance and overfitting.  
The dataset is already well-behaved, so improvement is marginal but consistent with expectations.


## v0.1
**Date:** 2025-10-18  
**Changes:**
- Initial baseline model using `StandardScaler + LinearRegression`
- RMSE metric evaluated on holdout set
- FastAPI service with `/health` and `/predict` endpoints
- Dockerized, tested locally, and reproducible via GitHub Actions

**Metrics (holdout set):**
| Metric | Value |
|---------|--------|
| RMSE | 53.85 | 

**Rationale:**  
This baseline establishes a reproducible reference for future improvements in model accuracy and pipeline automation.
