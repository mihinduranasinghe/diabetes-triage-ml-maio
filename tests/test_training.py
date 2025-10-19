import json
from pathlib import Path


def test_metrics_exist():
    p = Path("models/v0.1/metrics.json")
    assert p.exists()
    meta = json.loads(p.read_text())
    assert meta["model_version"] == "v0.1"
    assert "rmse_holdout" in meta["metrics"]
