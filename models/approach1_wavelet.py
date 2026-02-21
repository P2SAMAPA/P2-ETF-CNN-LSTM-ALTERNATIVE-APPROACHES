"""
models/approach1_wavelet.py
Approach 1: Wavelet Decomposition CNN-LSTM
- Dynamic wavelet level based on sequence length (no boundary warnings)
- CPU-optimised smaller architecture
- Class weights to prevent majority-class collapse
"""

import numpy as np
import pywt

WAVELET = "db4"


def _safe_wavelet_level(lookback: int, wavelet: str = WAVELET) -> int:
    """Compute max safe wavelet level for the given sequence length."""
    max_level = pywt.dwt_max_level(lookback, wavelet)
    return min(2, max_level)   # cap at 2 to avoid boundary effects


def _wavelet_decompose_signal(signal: np.ndarray, wavelet: str, level: int) -> np.ndarray:
    T      = len(signal)
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    bands  = []
    for c in coeffs:
        band = np.interp(np.linspace(0, len(c) - 1, T), np.arange(len(c)), c)
        bands.append(band)
    return np.stack(bands, axis=-1)


def apply_wavelet_transform(X: np.ndarray, wavelet: str = WAVELET) -> np.ndarray:
    n_samples, lookback, n_features = X.shape
    level   = _safe_wavelet_level(lookback, wavelet)
    n_bands = level + 1
    X_wt    = np.zeros((n_samples, lookback, n_features * n_bands), dtype=np.float32)
    for s in range(n_samples):
        for f in range(n_features):
            decomposed = _wavelet_decompose_signal(X[s, :, f], wavelet, level)
            start = f * n_bands
            X_wt[s, :, start: start + n_bands] = decomposed
    return X_wt


def build_wavelet_cnn_lstm(input_shape, n_classes, dropout=0.3, lstm_units=64):
    from tensorflow import keras
    from models.base import classification_head

    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv1D(32, 3, padding="causal", activation="relu")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Conv1D(16, 3, padding="causal", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.LSTM(lstm_units, dropout=dropout)(x)
    outputs = classification_head(x, n_classes, dropout)

    model = keras.Model(inputs, outputs, name="Approach1_Wavelet")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_approach1(
    X_train, y_train, X_val, y_val,
    n_classes, epochs=80, batch_size=64, dropout=0.3, lstm_units=64,
):
    from models.base import get_callbacks, compute_class_weights

    X_train_wt  = apply_wavelet_transform(X_train)
    X_val_wt    = apply_wavelet_transform(X_val)
    input_shape = X_train_wt.shape[1:]
    model       = build_wavelet_cnn_lstm(input_shape, n_classes, dropout, lstm_units)
    cw          = compute_class_weights(y_train, n_classes)

    history = model.fit(
        X_train_wt, y_train,
        validation_data=(X_val_wt, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=cw,
        callbacks=get_callbacks(),
        verbose=0,
    )
    return model, history, input_shape


def predict_approach1(model, X_test: np.ndarray) -> tuple:
    X_test_wt = apply_wavelet_transform(X_test)
    proba     = model.predict(X_test_wt, verbose=0)
    return np.argmax(proba, axis=1), proba
