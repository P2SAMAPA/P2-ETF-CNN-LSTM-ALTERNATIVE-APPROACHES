"""
models/approach3_multiscale.py
Approach 3: Multi-Scale Parallel CNN-LSTM

Pipeline:
  Raw macro signals
  → 3 parallel CNN towers: kernel 3 (short), 7 (medium), 21 (long)
  → Concatenate [96 features]
  → LSTM (128 units)
  → Dense 64 → Softmax (n_etfs + 1 CASH)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from models.base import classification_head, get_callbacks

# Kernel sizes represent: momentum (3d), weekly cycle (7d), monthly trend (21d)
KERNEL_SIZES  = [3, 7, 21]
FILTERS_EACH  = 32   # 32 × 3 towers = 96 concatenated features


# ── Model builder ─────────────────────────────────────────────────────────────

def build_multiscale_cnn_lstm(
    input_shape: tuple,
    n_classes: int,
    kernel_sizes: list = None,
    filters: int = FILTERS_EACH,
    dropout: float = 0.3,
    lstm_units: int = 128,
) -> keras.Model:
    """
    Multi-scale parallel CNN-LSTM.

    Three CNN towers with different kernel sizes run in parallel on the
    same input, capturing momentum, weekly cycle, and monthly trend
    simultaneously. Their outputs are concatenated before the LSTM.

    Args:
        input_shape  : (lookback, n_features)
        n_classes    : number of output classes (ETFs + CASH)
        kernel_sizes : list of kernel sizes for each tower
        filters      : number of Conv1D filters per tower
        dropout      : dropout rate
        lstm_units   : LSTM hidden size

    Returns:
        Compiled Keras model
    """
    if kernel_sizes is None:
        kernel_sizes = KERNEL_SIZES

    inputs = keras.Input(shape=input_shape, name="multiscale_input")

    towers = []
    for k in kernel_sizes:
        # Each tower: Conv → BN → Conv → BN → GlobalAvgPool
        t = keras.layers.Conv1D(
            filters, kernel_size=k, padding="causal", activation="relu",
            name=f"conv1_k{k}"
        )(inputs)
        t = keras.layers.BatchNormalization(name=f"bn1_k{k}")(t)
        t = keras.layers.Conv1D(
            filters, kernel_size=k, padding="causal", activation="relu",
            name=f"conv2_k{k}"
        )(t)
        t = keras.layers.BatchNormalization(name=f"bn2_k{k}")(t)
        t = keras.layers.Dropout(dropout, name=f"drop_k{k}")(t)
        towers.append(t)

    # Concatenate along the feature dimension — keeps temporal axis intact for LSTM
    if len(towers) > 1:
        merged = keras.layers.Concatenate(axis=-1, name="tower_concat")(towers)
    else:
        merged = towers[0]

    # LSTM integrates multi-scale temporal features
    x = keras.layers.LSTM(lstm_units, dropout=dropout, recurrent_dropout=0.1, name="lstm")(merged)

    # Output head
    outputs = classification_head(x, n_classes, dropout)

    model = keras.Model(inputs, outputs, name="Approach3_MultiScale_CNN_LSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Full train pipeline ───────────────────────────────────────────────────────

def train_approach3(
    X_train, y_train,
    X_val,   y_val,
    n_classes: int,
    epochs: int = 100,
    batch_size: int = 32,
    dropout: float = 0.3,
    lstm_units: int = 128,
    kernel_sizes: list = None,
):
    """
    Build and train the multi-scale CNN-LSTM.

    Args:
        X_train/val : [n, lookback, n_features]
        y_train/val : [n] integer class labels
        n_classes   : total output classes

    Returns:
        model   : trained Keras model
        history : training history
    """
    if kernel_sizes is None:
        kernel_sizes = KERNEL_SIZES

    # Guard: lookback must be >= largest kernel
    lookback = X_train.shape[1]
    valid_kernels = [k for k in kernel_sizes if k <= lookback]
    if not valid_kernels:
        valid_kernels = [min(3, lookback)]

    model = build_multiscale_cnn_lstm(
        input_shape=X_train.shape[1:],
        n_classes=n_classes,
        kernel_sizes=valid_kernels,
        dropout=dropout,
        lstm_units=lstm_units,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(),
        verbose=0,
    )

    return model, history


def predict_approach3(model, X_test: np.ndarray) -> tuple:
    """Predict on test set. Returns (class_preds, proba)."""
    proba = model.predict(X_test, verbose=0)
    preds = np.argmax(proba, axis=1)
    return preds, proba
