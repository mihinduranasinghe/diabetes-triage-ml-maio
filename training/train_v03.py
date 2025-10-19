from pathlib import Path
import json
import joblib
from sklearn.datasets import load_diabetes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold

from utils import set_seeds, rmse, write_metrics

OUT_DIR = Path("models/v0.3")


def main():
    set_seeds(42)
    Xy = load_diabetes(as_frame=True)
    X = Xy.frame.drop(columns=["target"])
    y = Xy.frame["target"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define pipeline
    pipe = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestRegressor(random_state=42))])

    # Define hyperparameter grid
    params = {
        "rf__n_estimators": [100, 200, 300],
        "rf__max_depth": [None, 5, 10, 20],
        "rf__min_samples_split": [2, 5, 10],
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Grid search
    grid = GridSearchCV(
        pipe, param_grid=params, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    metric_rmse = rmse(y_test, y_pred)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, OUT_DIR / "model.joblib")

    meta = {
        "model_version": "v0.3",
        "algorithm": (
            f"StandardScaler + RandomForestRegressor("
            f"n_estimators={grid.best_params_['rf__n_estimators']}, "
            f"max_depth={grid.best_params_['rf__max_depth']}, "
            f"min_samples_split={grid.best_params_['rf__min_samples_split']})"
        ),
        "seed": 42,
        "metrics": {"rmse_holdout": metric_rmse},
    }

    write_metrics(OUT_DIR / "metrics.json", meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
