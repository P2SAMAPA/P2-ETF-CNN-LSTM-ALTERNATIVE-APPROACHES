"""
models/approach1_wavelet.py
Approach 1: Wavelet Decomposition CNN-LSTM

Pipeline:
  Raw macro signals
  → DWT (db4, level=3) per signal → multi-band channel stack
  → 1D CNN (64 filters, k=3) → MaxPool → (32 filters, k=3)
  → LSTM (128 units)
  → Dense 64 → Softmax (n_etfs + 1 CASH)
"""

import numpy as np
import pywt
import tensorflow as tf
from tensorflow import keras
from models.base import classification_head, get_callbacks

WAVELET   = "db4"
LEVEL     = 3


# ── Wavelet feature engineering ───────────────────────────────────────────────

def _wavelet_decompose_signal(signal: np.ndarray, wavelet: str, level: int) -> np.ndarray:
    """
    Decompose a 1-D signal into DWT subbands and return them stacked.

    For a signal of length T:
      coeffs = [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    We interpolate each subband back to length T so we can stack them.

    Returns: array of shape [T, level+1]
    """
    T      = len(signal)
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    bands  = []
    for c in coeffs:
        # Interpolate back to original length
        band = np.interp(
            np.linspace(0, len(c) - 1, T),
            np.arange(len(c)),
            c,
        )
        bands.append(band)
    return np.stack(bands, axis=-1)   # [T, level+1]


def apply_wavelet_transform(X: np.ndarray, wavelet: str = WAVELET, level: int = LEVEL) -> np.ndarray:
    """
    Apply DWT to every feature channel across all samples.

    Args:
        X : [n_samples, lookback, n_features]

    Returns:
        X_wt : [n_samples, lookback, n_features * (level+1)]
    """
    n_samples, lookback, n_features = X.shape
    n_bands   = level + 1
    X_wt      = np.zeros((n_samples, lookback, n_features * n_bands), dtype=np.float32)

    for s in range(n_samples):
        for f in range(n_features):
            decomposed = _wavelet_decompose_signal(X[s, :, f], wavelet, level)   # [T, n_bands]
            start = f * n_bands
            X_wt[s, :, start: start + n_bands] = decomposed

    return X_wt


# ── Model builder ─────────────────────────────────────────────────────────────

def build_wavelet_cnn_lstm(
    input_shape: tuple,
    n_classes: int,
    dropout: float = 0.3,
    lstm_units: int = 128,
) -> keras.Model:
    """
    Build Wavelet CNN-LSTM model.

    Args:
        input_shape : (lookback, n_features * n_bands)  — post-DWT shape
        n_classes   : number of output classes (ETFs + CASH)
        dropout     : dropout rate
        lstm_units  : LSTM hidden size

    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=input_shape, name="wavelet_input")

    # CNN block 1
    x = keras.layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)

    # CNN block 2
    x = keras.layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout)(x)

    # LSTM
    x = keras.layers.LSTM(lstm_units, dropout=dropout, recurrent_dropout=0.1)(x)

    # Output head
    outputs = classification_head(x, n_classes, dropout)

    model = keras.Model(inputs, outputs, name="Approach1_Wavelet_CNN_LSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Full train pipeline ───────────────────────────────────────────────────────

def train_approach1(
    X_train, y_train,
    X_val,   y_val,
    n_classes: int,
    epochs: int = 100,
    batch_size: int = 32,
    dropout: float = 0.3,
    lstm_units: int = 128,
):
    """
    Apply wavelet transform then train the CNN-LSTM.

    Args:
        X_train/val : [n, lookback, n_features]  (scaled, pre-wavelet)
        y_train/val : [n] integer class labels
        n_classes   : total output classes

    Returns:
        model    : trained Keras model
        history  : training history
        wt_shape : post-DWT input shape (for inference)
    """
    # Apply DWT
    X_train_wt = apply_wavelet_transform(X_train)
    X_val_wt   = apply_wavelet_transform(X_val)

    input_shape = X_train_wt.shape[1:]   # (lookback, n_features * n_bands)
    model       = build_wavelet_cnn_lstm(input_shape, n_classes, dropout, lstm_units)

    history = model.fit(
        X_train_wt, y_train,
        validation_data=(X_val_wt, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(),
        verbose=0,
    )

    return model, history, input_shape


def predict_approach1(model, X_test: np.ndarray) -> tuple:
    """Apply DWT to test set then predict. Returns (class_preds, proba)."""
    X_test_wt = apply_wavelet_transform(X_test)
    proba     = model.predict(X_test_wt, verbose=0)
    preds     = np.argmax(proba, axis=1)
    return preds, proba
