from pydantic import BaseModel, Field


# scikit-learn diabetes feature names (already normalized in dataset)
class PredictRequest(BaseModel):
    age: float = Field(..., description="Standardized age feature")
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


class PredictResponse(BaseModel):
    prediction: float


class HealthResponse(BaseModel):
    status: str
    model_version: str
