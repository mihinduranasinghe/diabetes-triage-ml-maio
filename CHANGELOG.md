# Changelog

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
