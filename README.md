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

Both versions are fully containerized, tested, and released via **GitHub Actions** to **GitHub Container Registry (GHCR)**.

---

## Prediction Target

- **Dataset:** `sklearn.datasets.load_diabetes()`  
- **Target:** “disease progression index” — a continuous value (higher = higher deterioration risk)  
- **Goal:** Predict the risk score so the clinic can prioritize follow-ups.

---

Ridge regression adds L2 regularization, slightly reducing overfitting and improving generalization.
Although the gain is small (~0.14%), it’s consistent and statistically meaningful on this small dataset.

| Version  | Algorithm                         | RMSE (Holdout) | Δ (Improvement) |
| -------- | --------------------------------- | -------------- | --------------- |
| **v0.1** | StandardScaler + LinearRegression | 53.8534        | —               |
| **v0.2** | StandardScaler + Ridge(α=1.0)     | 53.7774        | ↓ 0.076         |

---

## ⚙️ Setup and Usage

### 1 Local setup
```bash
python -m venv .venv
source .venv/bin/activate       
pip install -r requirements.txt
```

### 2 Local setup
```bash
# Baseline (v0.1)
python training/train_v01.py

# Improved (v0.2)
python training/train_v02.py
```

### 3 Run the API locally
```bash
uvicorn src.app:app --port 8080
```

### 4 Testing and Quality Checks
```bash
pytest -q              # Run unit tests
ruff check .           # Lint
black --check .        # Code style
```

### 5 Build and run (local)
```bash
docker build -t diabetes-triage:v0.2 .
docker run --rm -p 8080:8080 diabetes-triage:v0.2
```

## ⚙️ CI/CD Workflow Summary

| Stage | Trigger | Actions Performed | Output / Purpose |
|--------|----------|-------------------|------------------|
| **Continuous Integration (CI)** | On every **push** or **pull request** to `master` | - Install dependencies<br>- Run linting (`ruff`, `black`)<br>- Execute unit tests (`pytest`)<br>- Run smoke model training (if needed)<br>- Upload model artifacts (`.joblib`, `metrics.json`) | Ensures code quality, reproducibility, and that all tests pass before merging. |
| **Continuous Delivery (Release)** | On **tag push** (e.g., `v0.1`, `v0.2`) | - Build Docker image from current tag<br>- Run container smoke test (`/health` endpoint)<br>- Push image to **GitHub Container Registry (GHCR)**<br>- Create a GitHub Release with metrics & changelog | Automates release packaging and guarantees a deployable container is published. |
| **Container Registry (GHCR)** | Triggered automatically by **Release workflow** | - Stores versioned Docker images (`ghcr.io/mihinduranasinghe/diabetes-triage-mlops:vX.Y`)<br>- Provides reproducible, portable builds for local or production use | Allows anyone (including graders) to pull and run the exact versioned image. |
| **Manual Verification** | Optional (after release) | - Pull the image locally using `docker pull ghcr.io/...`<br>- Run container and verify `/health` and `/predict` endpoints | Confirms that the released image runs identically outside CI/CD environment. |

---

### 7 GitHub Container Registry (GHCR)
```bash
ghcr.io/mihinduranasinghe/diabetes-triage-ml-maio:v0.1  
ghcr.io/mihinduranasinghe/diabetes-triage-ml-maio:v0.2
```

### 8 Pull and Run from GitHub Container Registry (GHCR)
```bash
# After each tagged release (`v0.1`, `v0.2`), the GitHub Actions workflow automatically builds and pushes a Docker image to GitHub Container Registry (GHCR)

# To verify or use the released image:

# Login to GHCR (first time only)
echo $GITHUB_TOKEN | docker login ghcr.io -u <your-github-username> --password-stdin

# Pull v1
ghcr.io/mihindu-ilangakoon/diabetes-triage-mlops:v0.1

# Run v1
docker run --rm -p 8080:8080 ghcr.io/mihindu-ilangakoon/diabetes-triage-mlops:v0.1

# Pull v2
ghcr.io/mihindu-ilangakoon/diabetes-triage-mlops:v0.2

# Run v2
docker run --rm -p 8080:8080 ghcr.io/mihindu-ilangakoon/diabetes-triage-mlops:v0.2

# Access & Swagger UI :

- http://localhost:8080/health
- http://localhost:8080/docs
```

### 9 Test endpoints
```bash
# Health check
curl http://localhost:8080/health

# Prediction
curl -X POST http://localhost:8080/predict \
 -H "Content-Type: application/json" \
 -d '{"age":0.02,"sex":-0.044,"bmi":0.07,"bp":-0.03,"s1":-0.05,"s2":0.03,"s3":-0.03,"s4":0.02,"s5":0.02,"s6":-0.001}'
```



