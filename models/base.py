"""
models/base.py
Shared utilities for all CNN-LSTM variants.
Optimised for CPU training on HF Spaces.
"""

import numpy as np
import hashlib
import pickle
import os
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight

SEED     = 42
CACHE_DIR = Path("/tmp/p2_model_cache")
CACHE_DIR.mkdir(exist_ok=True)

np.random.seed(SEED)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def make_cache_key(last_date: str, start_yr: int, fee_bps: int,
                   epochs: int, split: str, include_cash: bool,
                   lookback: int) -> str:
    raw = f"{last_date}_{start_yr}_{fee_bps}_{epochs}_{split}_{include_cash}_{lookback}"
    return hashlib.md5(raw.encode()).hexdigest()


def save_cache(key: str, payload: dict):
    path = CACHE_DIR / f"{key}.pkl"
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_cache(key: str) -> dict | None:
    path = CACHE_DIR / f"{key}.pkl"
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            path.unlink(missing_ok=True)
    return None


# ── Sequence builder ──────────────────────────────────────────────────────────

def build_sequences(features: np.ndarray, targets: np.ndarray, lookback: int):
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
    return X[:t1], y[:t1], X[t1:t2], y[t1:t2], X[t2:], y[t2:]


# ── Feature scaling ───────────────────────────────────────────────────────────

def scale_features(X_train, X_val, X_test):
    n_feat = X_train.shape[2]
    scaler = RobustScaler()
    scaler.fit(X_train.reshape(-1, n_feat))
    def _t(X):
        s = X.shape
        return scaler.transform(X.reshape(-1, n_feat)).reshape(s)
    return _t(X_train), _t(X_val), _t(X_test), scaler


# ── Label builder ─────────────────────────────────────────────────────────────

def returns_to_labels(y_raw, include_cash=True, cash_threshold=0.0):
    best = np.argmax(y_raw, axis=1)
    if include_cash:
        best_ret = y_raw[np.arange(len(y_raw)), best]
        cash_idx = y_raw.shape[1]
        labels   = np.where(best_ret < cash_threshold, cash_idx, best)
    else:
        labels = best
    return labels.astype(np.int32)


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(y_labels: np.ndarray, n_classes: int) -> dict:
    present = np.unique(y_labels)
    try:
        weights = compute_class_weight("balanced", classes=present, y=y_labels)
        weight_dict = {int(c): float(w) for c, w in zip(present, weights)}
    except Exception:
        weight_dict = {}
    for c in range(n_classes):
        if c not in weight_dict:
            weight_dict[c] = 1.0
    return weight_dict


# ── Callbacks ─────────────────────────────────────────────────────────────────

def get_callbacks(patience_es=15, patience_lr=8, min_lr=1e-6):
    from tensorflow import keras
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


# ── Lightweight output head (CPU-optimised) ───────────────────────────────────

def classification_head(x, n_classes: int, dropout: float = 0.3):
    """Smaller head than original — faster on CPU, less overfitting risk."""
    from tensorflow import keras
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(n_classes, activation="softmax")(x)
    return x


# ── Auto lookback selection ───────────────────────────────────────────────────

def find_best_lookback(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    y_labels_fn,
    train_pct: float,
    val_pct: float,
    n_classes: int,
    include_cash: bool,
    candidates: list = None,
):
    """
    Train a fast lightweight CNN on each lookback candidate using val loss.
    Returns best lookback int.
    Uses only Approach 1 architecture (fastest) to pick the winner.
    """
    from tensorflow import keras

    if candidates is None:
        candidates = [30, 45, 60]

    best_lb   = candidates[0]
    best_loss = np.inf

    for lb in candidates:
        try:
            X_seq, y_seq = build_sequences(X_raw, y_raw, lb)
            y_lab        = y_labels_fn(y_seq)

            X_tr, y_tr, X_v, y_v, _, _ = train_val_test_split(X_seq, y_lab, train_pct, val_pct)
            X_tr_s, X_v_s, _, _        = scale_features(X_tr, X_v, X_v)

            cw = compute_class_weights(y_tr, n_classes)

            # Tiny fast model just for lookback selection
            inp = keras.Input(shape=X_tr_s.shape[1:])
            x   = keras.layers.Conv1D(16, min(3, lb), padding="causal", activation="relu")(inp)
            x   = keras.layers.GlobalAveragePooling1D()(x)
            out = keras.layers.Dense(n_classes, activation="softmax")(x)
            m   = keras.Model(inp, out)
            m.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

            hist = m.fit(
                X_tr_s, y_tr,
                validation_data=(X_v_s, y_v),
                epochs=15,
                batch_size=64,
                class_weight=cw,
                callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
                verbose=0,
            )
            val_loss = min(hist.history.get("val_loss", [np.inf]))
            if val_loss < best_loss:
                best_loss = val_loss
                best_lb   = lb

            del m
        except Exception:
            continue

    return best_lb
