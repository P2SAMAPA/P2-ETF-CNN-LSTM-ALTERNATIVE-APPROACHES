"""
models/base.py
Shared utilities for all three CNN-LSTM variants:
  - Data preparation (sequences, train/val/test split)
  - Common Keras layers / callbacks
  - Predict + evaluate helpers
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow import keras

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


# ── Sequence builder ──────────────────────────────────────────────────────────

def build_sequences(features: np.ndarray, targets: np.ndarray, lookback: int):
    """
    Build supervised sequences for CNN-LSTM input.

    Args:
        features : 2-D array [n_days, n_features]
        targets  : 2-D array [n_days, n_etfs]  (raw returns)
        lookback : number of past days per sample

    Returns:
        X : [n_samples, lookback, n_features]
        y : [n_samples, n_etfs]   (raw returns for the next day)
    """
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i - lookback: i])
        y.append(targets[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── Train / val / test split ──────────────────────────────────────────────────

def train_val_test_split(X, y, train_pct=0.70, val_pct=0.15):
    """Split sequences into train / val / test preserving temporal order."""
    n = len(X)
    t1 = int(n * train_pct)
    t2 = int(n * (train_pct + val_pct))

    return (
        X[:t1],  y[:t1],
        X[t1:t2], y[t1:t2],
        X[t2:],  y[t2:],
    )


# ── Feature scaling ───────────────────────────────────────────────────────────

def scale_features(X_train, X_val, X_test):
    """
    Fit RobustScaler on training data only, apply to val and test.
    Operates on the flattened feature dimension.

    Returns scaled arrays with same shape as inputs.
    """
    n_train, lb, n_feat = X_train.shape
    scaler = RobustScaler()

    # Fit on train
    scaler.fit(X_train.reshape(-1, n_feat))

    def _transform(X):
        shape = X.shape
        return scaler.transform(X.reshape(-1, n_feat)).reshape(shape)

    return _transform(X_train), _transform(X_val), _transform(X_test), scaler


# ── Label builder (classification: argmax of returns) ────────────────────────

def returns_to_labels(y_raw, include_cash=True, cash_threshold=0.0):
    """
    Convert raw return matrix to integer class labels.

    If include_cash=True, adds a CASH class (index = n_etfs) when
    the best ETF return is below cash_threshold.

    Args:
        y_raw           : [n_samples, n_etfs]
        include_cash    : whether to allow CASH class
        cash_threshold  : minimum ETF return to prefer over CASH

    Returns:
        labels : [n_samples] integer class indices
    """
    best = np.argmax(y_raw, axis=1)
    if include_cash:
        best_return = y_raw[np.arange(len(y_raw)), best]
        cash_idx    = y_raw.shape[1]
        labels      = np.where(best_return < cash_threshold, cash_idx, best)
    else:
        labels = best
    return labels.astype(np.int32)


# ── Common Keras callbacks ────────────────────────────────────────────────────

def get_callbacks(patience_es=15, patience_lr=8, min_lr=1e-6):
    """Standard early stopping + reduce-LR callbacks shared by all models."""
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


# ── Common output head ────────────────────────────────────────────────────────

def classification_head(x, n_classes: int, dropout: float = 0.3):
    """
    Shared dense output head for all three CNN-LSTM variants.

    Args:
        x         : input tensor
        n_classes : number of ETF classes (+ 1 for CASH if applicable)
        dropout   : dropout rate

    Returns:
        output tensor with softmax activation
    """
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(n_classes, activation="softmax")(x)
    return x


# ── Prediction helper ─────────────────────────────────────────────────────────

def predict_classes(model, X_test: np.ndarray) -> np.ndarray:
    """Return integer class predictions from a Keras model."""
    proba = model.predict(X_test, verbose=0)
    return np.argmax(proba, axis=1), proba


# ── Metrics helper ────────────────────────────────────────────────────────────

def evaluate_returns(
    preds: np.ndarray,
    proba: np.ndarray,
    y_raw_test: np.ndarray,
    target_etfs: list,
    tbill_rate: float,
    fee_bps: int,
    include_cash: bool = True,
):
    """
    Given integer class predictions and raw return matrix,
    compute strategy returns and summary metrics.

    Returns:
        strat_rets      : np.ndarray of daily net returns
        ann_return      : annualised return (float)
        cum_returns     : cumulative return series
        last_proba      : probability vector for the last prediction
        next_etf        : name of ETF predicted for next session
    """
    n_etfs     = len(target_etfs)
    strat_rets = []

    for i, cls in enumerate(preds):
        if include_cash and cls == n_etfs:
            # CASH: earn daily T-bill rate
            daily_tbill = tbill_rate / 252
            net = daily_tbill - (fee_bps / 10000)
        else:
            ret = y_raw_test[i][cls]
            net = ret - (fee_bps / 10000)
        strat_rets.append(net)

    strat_rets  = np.array(strat_rets)
    cum_returns = np.cumprod(1 + strat_rets)
    ann_return  = (cum_returns[-1] ** (252 / len(strat_rets))) - 1

    last_proba  = proba[-1]
    next_cls    = int(np.argmax(last_proba))
    next_etf    = "CASH" if (include_cash and next_cls == n_etfs) else target_etfs[next_cls].replace("_Ret", "")

    return strat_rets, ann_return, cum_returns, last_proba, next_etf
