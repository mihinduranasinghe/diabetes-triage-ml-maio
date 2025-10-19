from fastapi import FastAPI, HTTPException
from src.schema import PredictRequest, PredictResponse, HealthResponse
from src.model_io import load_model

MODEL_VERSION = "v0.3"
model, meta = load_model(MODEL_VERSION)

app = FastAPI(title="Diabetes Triage ML Service", version=MODEL_VERSION)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_version=MODEL_VERSION)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        X = [
            [
                payload.age,
                payload.sex,
                payload.bmi,
                payload.bp,
                payload.s1,
                payload.s2,
                payload.s3,
                payload.s4,
                payload.s5,
                payload.s6,
            ]
        ]
        yhat = float(model.predict(X)[0])
        return PredictResponse(prediction=yhat)
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})
