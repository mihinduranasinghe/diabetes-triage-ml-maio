from pathlib import Path
import json
import joblib
from sklearn.datasets import load_diabetes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV, KFold

from utils import set_seeds, rmse, write_metrics

OUT_DIR = Path("models/v0.2")


def main():
    set_seeds(42)
    Xy = load_diabetes(as_frame=True)
    X = Xy.frame.drop(columns=["target"])
    y = Xy.frame["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(random_state=42))])
    params = {"ridge__alpha": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(pipe, params, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    metric_rmse = rmse(y_test, y_pred)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, OUT_DIR / "model.joblib")
    meta = {
        "model_version": "v0.2",
        "algorithm": f"StandardScaler + Ridge(alpha={grid.best_params_['ridge__alpha']})",
        "seed": 42,
        "metrics": {"rmse_holdout": metric_rmse},
    }

    write_metrics(OUT_DIR / "metrics.json", meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
