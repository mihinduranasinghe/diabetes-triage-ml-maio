# Changelog

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
