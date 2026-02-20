"""
models/base.py
Shared utilities for all three CNN-LSTM variants.
Key fix: class_weight support to prevent majority-class collapse.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


# ── Sequence builder ──────────────────────────────────────────────────────────

def build_sequences(features: np.ndarray, targets: np.ndarray, lookback: int):
    """
    Build supervised sequences for CNN-LSTM input.
    X[i] = features[i : i+lookback]  →  predicts  y[i+lookback]
    """
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i - lookback: i])
        y.append(targets[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── Train / val / test split ──────────────────────────────────────────────────

def train_val_test_split(X, y, train_pct=0.70, val_pct=0.15):
    n  = len(X)
    t1 = int(n * train_pct)
    t2 = int(n * (train_pct + val_pct))
    return (
        X[:t1],  y[:t1],
        X[t1:t2], y[t1:t2],
        X[t2:],  y[t2:],
    )


# ── Feature scaling ───────────────────────────────────────────────────────────

def scale_features(X_train, X_val, X_test):
    n_feat  = X_train.shape[2]
    scaler  = RobustScaler()
    scaler.fit(X_train.reshape(-1, n_feat))

    def _t(X):
        s = X.shape
        return scaler.transform(X.reshape(-1, n_feat)).reshape(s)

    return _t(X_train), _t(X_val), _t(X_test), scaler


# ── Label builder ─────────────────────────────────────────────────────────────

def returns_to_labels(y_raw, include_cash=True, cash_threshold=0.0):
    """
    Assign label = argmax(returns).
    If include_cash and best return < cash_threshold → label = n_etfs (CASH).
    """
    best        = np.argmax(y_raw, axis=1)
    if include_cash:
        best_ret = y_raw[np.arange(len(y_raw)), best]
        cash_idx = y_raw.shape[1]
        labels   = np.where(best_ret < cash_threshold, cash_idx, best)
    else:
        labels = best
    return labels.astype(np.int32)


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(y_labels: np.ndarray, n_classes: int) -> dict:
    """
    Compute balanced class weights to counteract majority-class collapse.
    Returns dict {class_index: weight} for use in model.fit().
    """
    classes = np.arange(n_classes)
    present = np.unique(y_labels)

    try:
        weights = compute_class_weight(
            class_weight="balanced",
            classes=present,
            y=y_labels,
        )
        weight_dict = {int(c): float(w) for c, w in zip(present, weights)}
    except Exception:
        weight_dict = {}

    # Fill any missing classes with weight 1.0
    for c in classes:
        if c not in weight_dict:
            weight_dict[c] = 1.0

    return weight_dict


# ── Callbacks ─────────────────────────────────────────────────────────────────

def get_callbacks(patience_es=20, patience_lr=10, min_lr=1e-6):
    """Longer patience to allow models time to learn past majority class."""
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_es,
            restore_best_weights=True,
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=min_lr,
            verbose=0,
        ),
    ]


# ── Output head ───────────────────────────────────────────────────────────────

def classification_head(x, n_classes: int, dropout: float = 0.3):
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(dropout / 2)(x)
    x = keras.layers.Dense(n_classes, activation="softmax")(x)
    return x


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_classes(model, X_test: np.ndarray) -> tuple:
    proba = model.predict(X_test, verbose=0)
    return np.argmax(proba, axis=1), proba


# ── Metrics ───────────────────────────────────────────────────────────────────

def evaluate_returns(
    preds, proba, y_raw_test, target_etfs, tbill_rate, fee_bps, include_cash=True,
):
    n_etfs     = len(target_etfs)
    daily_tbill = tbill_rate / 252
    strat_rets  = []

    for i, cls in enumerate(preds):
        if include_cash and cls == n_etfs:
            net = daily_tbill - fee_bps / 10000
        else:
            cls = min(int(cls), n_etfs - 1)
            net = float(y_raw_test[i][cls]) - fee_bps / 10000
        strat_rets.append(net)

    strat_rets  = np.array(strat_rets)
    cum_returns = np.cumprod(1 + strat_rets)
    ann_return  = cum_returns[-1] ** (252 / len(strat_rets)) - 1

    last_proba  = proba[-1]
    next_cls    = int(np.argmax(last_proba))
    next_etf    = (
        "CASH" if (include_cash and next_cls == n_etfs)
        else target_etfs[min(next_cls, n_etfs - 1)].replace("_Ret", "")
    )

    return strat_rets, ann_return, cum_returns, last_proba, next_etf
