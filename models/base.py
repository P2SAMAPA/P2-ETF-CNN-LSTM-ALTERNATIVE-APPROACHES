"""
Base Models Module — TUNED for speed
Changes vs original:
  - n_estimators reduced from 300 to 100 for RF/XGB/LGB
  - RandomForest grid reduced to 4 combinations (was 12)
  - TimeSeriesSplit n_splits: 3 (was 5)
  - n_jobs=-1 on GridSearchCV for parallel folds
"""

import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb


def get_base_models():
    return {
        "RandomForest": {
            "model": RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=100),  # was 300
            "params": {
                "max_depth":          [10, None],        # was [5,10,15,None]
                "min_samples_split":  [2, 5],            # was [2,5,10]
            },
        },
        "XGBoost": {
            "model": xgb.XGBRegressor(random_state=42, n_jobs=-1, n_estimators=100,   # was 300
                                       verbosity=0),
            "params": {
                "max_depth":    [3, 5, 7],
                "learning_rate":[0.05, 0.1],
                "subsample":    [0.8, 1.0],
            },
        },
        "LightGBM": {
            "model": lgb.LGBMRegressor(random_state=42, n_jobs=-1, n_estimators=100,   # was 300
                                        verbose=-1),
            "params": {
                "max_depth":    [5, 10, 15],
                "learning_rate":[0.05, 0.1],
                "num_leaves":   [31, 63],
            },
        },
        "AdaBoost": {
            "model": AdaBoostRegressor(random_state=42, n_estimators=150),
            "params": {
                "learning_rate": [0.05, 0.1, 0.5, 1.0],
            },
        },
        "DecisionTree": {
            "model": DecisionTreeRegressor(random_state=42),
            "params": {
                "max_depth":         [5, 10, 15],
                "min_samples_split": [2, 5, 10],
            },
        },
    }


def train_base_models(X_train, y_train, artifact_path="artifacts"):
    """
    Train all base models with GridSearchCV.
    TimeSeriesSplit(n_splits=3) for faster walk-forward CV.
    """
    os.makedirs(artifact_path, exist_ok=True)

    tscv           = TimeSeriesSplit(n_splits=3)   # was 5
    models_config  = get_base_models()
    trained_models = {}

    for name, cfg in models_config.items():
        print(f"    [{name}] fitting GridSearchCV …")
        gs = GridSearchCV(
            estimator  = cfg["model"],
            param_grid = cfg["params"],
            cv         = tscv,
            scoring    = "neg_mean_squared_error",
            n_jobs     = -1,
            verbose    = 0,
            refit      = True,
        )
        gs.fit(X_train, y_train)
        trained_models[name] = {
            "model":       gs.best_estimator_,
            "best_params": gs.best_params_,
            "best_score":  gs.best_score_,
        }
        print(f"      best_params={gs.best_params_}  "
              f"cv_mse={-gs.best_score_:.6f}")

    return trained_models


def save_base_models(trained_models, ma_window, etf_name, artifact_path="artifacts"):
    path = f"{artifact_path}/base_models_{etf_name}_MA{ma_window}.pkl"
    joblib.dump(trained_models, path)
    return path


def load_base_models(ma_window, etf_name, artifact_path="artifacts"):
    path = f"{artifact_path}/base_models_{etf_name}_MA{ma_window}.pkl"
    return joblib.load(path)


def predict_base_models(trained_models, X):
    return {name: d["model"].predict(X)
            for name, d in trained_models.items()}
