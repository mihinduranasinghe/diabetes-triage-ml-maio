import json
import os
import random
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

SEED = 42


def set_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)


def holdout_split(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=SEED)


def rmse(y_true, y_pred) -> float:
    return mean_squared_error(y_true, y_pred, squared=False)


def write_metrics(path, metrics: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
