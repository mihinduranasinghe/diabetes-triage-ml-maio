import json
from pathlib import Path
import joblib


def load_model(version: str):
    base = Path("models") / version
    model = joblib.load(base / "model.joblib")
    with open(base / "metrics.json") as f:
        meta = json.load(f)
    return model, meta
