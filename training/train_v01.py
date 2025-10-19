from pathlib import Path
import joblib
import json
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import set_seeds, holdout_split, rmse, write_metrics

OUT_DIR = Path("models/v0.1")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    set_seeds(42)
    Xy = load_diabetes(as_frame=True)
    X = Xy.frame.drop(columns=["target"])
    y = Xy.frame["target"]

    X_train, X_test, y_train, y_test = holdout_split(X, y)

    pipe = Pipeline(
        [("scaler", StandardScaler(with_mean=True, with_std=True)), ("reg", LinearRegression())]
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    metric_rmse = rmse(y_test, y_pred)

    joblib.dump(pipe, OUT_DIR / "model.joblib")
    meta = {
        "model_version": "v0.1",
        "algorithm": "StandardScaler + LinearRegression",
        "seed": 42,
        "metrics": {"rmse_holdout": metric_rmse},
    }
    write_metrics(OUT_DIR / "metrics.json", meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
