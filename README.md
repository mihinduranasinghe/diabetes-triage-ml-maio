# Virtual Diabetes Clinic Triage – MLOps Pipeline

This project implements an **end-to-end MLOps pipeline** for a *Virtual Diabetes Clinic* that predicts short-term diabetes disease progression risk.  
It simulates a triage system where nurses can prioritize patients based on model-generated risk scores.

---

## 🚀 Overview

The service is built using **FastAPI**, **scikit-learn**, **Docker**, and **GitHub Actions**, following MLOps best practices for reproducibility, automation, and deployment.

| Version | Model | Description |
|----------|--------|--------------|
| **v0.1** | StandardScaler + LinearRegression | Baseline model |
| **v0.2** | StandardScaler + Ridge (α=1.0, CV tuned) | Regularized & cross-validated improvement |
| **v0.3** | StandardScaler + RandomForestRegressor (tuned) | Non-linear ensemble model optimization |

All versions are fully containerized, tested, and released via **GitHub Actions** to **GitHub Container Registry (GHCR)**.

---

## Prediction Target

- **Dataset:** `sklearn.datasets.load_diabetes()`  
- **Target:** “disease progression index” — a continuous value (higher = higher deterioration risk)  
- **Goal:** Predict the risk score so the clinic can prioritize follow-ups.

---

## 📊 Model Performance

| Version  | Algorithm | RMSE (Holdout) | Δ (Improvement) |
|-----------|------------|----------------|-----------------|
| **v0.1** | StandardScaler + LinearRegression | 53.8534 | — |
| **v0.2** | StandardScaler + Ridge(α=1.0) | 53.7774 | ↓ 0.076 |
| **v0.3** | StandardScaler + RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_split=5) | **53.7187** | ↓ 0.059 |

**Interpretation:**  
- Each iteration provides incremental improvements in model generalization.  
- Ridge regression reduced overfitting slightly, and Random Forest introduced non-linearity with stable generalization.  
- The dataset’s small size and mostly linear nature limit further gains, but the pipeline demonstrates scalability and reproducibility.

---

## ⚙️ Setup and Usage

### 1. Local setup
```bash
python -m venv .venv
source .venv/bin/activate       
pip install -r requirements.txt
```

### 2. Local setup
```bash
# Baseline (v0.1)
python training/train_v01.py

# Ridge improvement (v0.2)
python training/train_v02.py

# Random Forest (v0.3)
python training/train_v03.py
```

### 3. Run the API locally
```bash
uvicorn src.app:app --port 8080
```

### 4. Testing and Quality Checks
```bash
pytest -q              # Run unit tests
ruff check .           # Lint
black --check .        # Code style
```

### 5. Build and run (local)
```bash
# Build v0.3
docker build -t diabetes-triage:v0.3 .
# Run v0.3
docker run --rm -p 8080:8080 diabetes-triage:v0.3
```

## ⚙️ CI/CD Workflow Summary

| Stage | Trigger | Actions Performed | Output / Purpose |
|--------|----------|-------------------|------------------|
| **Continuous Integration (CI)** | On every **push** or **pull request** to `master` | - Install dependencies<br>- Run linting (`ruff`, `black`)<br>- Execute unit tests (`pytest`)<br>- Run smoke model training (if needed)<br>- Upload model artifacts (`.joblib`, `metrics.json`) | Ensures code quality, reproducibility, and that all tests pass before merging. |
| **Continuous Delivery (Release)** | On **tag push** (e.g., `v0.1`, `v0.2`) | - Build Docker image from current tag<br>- Run container smoke test (`/health` endpoint)<br>- Push image to **GitHub Container Registry (GHCR)**<br>- Create a GitHub Release with metrics & changelog | Automates release packaging and guarantees a deployable container is published. |
| **Container Registry (GHCR)** | Triggered automatically by **Release workflow** | - Stores versioned Docker images (`ghcr.io/mihinduranasinghe/diabetes-triage-mlops:vX.Y`)<br>- Provides reproducible, portable builds for local or production use | Allows anyone (including graders) to pull and run the exact versioned image. |
| **Manual Verification** | Optional (after release) | - Pull the image locally using `docker pull ghcr.io/...`<br>- Run container and verify `/health` and `/predict` endpoints | Confirms that the released image runs identically outside CI/CD environment. |

If you want to add a new ML version, 

---

### 6. GitHub Container Registry (GHCR)
#### Available images:
```bash
ghcr.io/mihindu-ilangakoon/diabetes-triage-ml-maio:v0.1
ghcr.io/mihindu-ilangakoon/diabetes-triage-ml-maio:v0.2
ghcr.io/mihindu-ilangakoon/diabetes-triage-ml-maio:v0.3
```

### 7. Pull and Run from GitHub Container Registry (GHCR)
```bash
# After each tagged release (`v0.1`, `v0.2`), the GitHub Actions workflow automatically builds and pushes a Docker image to GitHub Container Registry (GHCR)

# To verify or use the released image:

# Login to GHCR (first time only)
echo $GITHUB_TOKEN | docker login ghcr.io -u <your-github-username> --password-stdin

# Pull v1
ghcr.io/mihindu-ilangakoon/diabetes-triage-ml-maio:v0.1

# Run v1
docker run --rm -p 8080:8080 ghcr.io/mihindu-ilangakoon/diabetes-triage-ml-maio:v0.1

# Pull v2
ghcr.io/mihindu-ilangakoon/diabetes-triage-ml-maio:v0.2

# Run v2
docker run --rm -p 8080:8080 ghcr.io/mihindu-ilangakoon/diabetes-triage-ml-maio:v0.2

# Pull v0.3
docker pull ghcr.io/mihindu-ilangakoon/diabetes-triage-ml-maio:v0.3

# Run v0.3
docker run --rm -p 8080:8080 ghcr.io/mihindu-ilangakoon/diabetes-triage-ml-maio:v0.3

# Access & Swagger UI :

- http://localhost:8080/health
- http://localhost:8080/docs
```

### 8. Test endpoints
```bash
# Health check
curl http://localhost:8080/health

# Prediction
curl -X POST http://localhost:8080/predict \
 -H "Content-Type: application/json" \
 -d '{"age":0.02,"sex":-0.044,"bmi":0.07,"bp":-0.03,"s1":-0.05,"s2":0.03,"s3":-0.03,"s4":0.02,"s5":0.02,"s6":-0.001}'
```

#### expected response
```bash
{
    "prediction": 192.88626022707265
}
```

---

### 9. Adding a New Model Version

To extend the pipeline with a new ML version :
```bash

# 1 Create a new training script
- Copy an existing script (e.g., `training/train_v03.py`)
- Name it `train_v04.py`
- Implement your new model (e.g., GradientBoosting, XGBoost, NeuralNet, etc.)

# 2 Train and evaluate
python training/train_v04.py

# A new folder models/v0.4/ will be created with:
model.joblib – saved trained model
metrics.json – logged RMSE and metadata

# Update FastAPI in src/app.py
MODEL_VERSION = "v0.4"

# Test locally, 
# commit and push to a feature branch 
# create a PR to master
# Tag and release
```



