"""
models/approach3_multiscale.py
Approach 3: Multi-Scale Parallel CNN-LSTM
- CPU-optimised smaller architecture
- Class weights to prevent majority-class collapse
- Lazy imports to prevent module-level failures
"""

import numpy as np

KERNEL_SIZES = [3, 7, 21]
FILTERS_EACH = 16    # reduced from 32 for CPU speed


def build_multiscale_cnn_lstm(
    input_shape, n_classes,
    kernel_sizes=None, filters=FILTERS_EACH,
    dropout=0.3, lstm_units=64,
):
    from tensorflow import keras
    from models.base import classification_head

    if kernel_sizes is None:
        kernel_sizes = KERNEL_SIZES

    inputs  = keras.Input(shape=input_shape, name="multiscale_input")
    towers  = []
    for k in kernel_sizes:
        t = keras.layers.Conv1D(filters, k, padding="causal", activation="relu",
                                name=f"conv1_k{k}")(inputs)
        t = keras.layers.BatchNormalization(name=f"bn1_k{k}")(t)
        t = keras.layers.Dropout(dropout, name=f"drop_k{k}")(t)
        towers.append(t)

    merged  = keras.layers.Concatenate(axis=-1)(towers) if len(towers) > 1 else towers[0]
    x       = keras.layers.LSTM(lstm_units, dropout=dropout)(merged)
    outputs = classification_head(x, n_classes, dropout)

    model = keras.Model(inputs, outputs, name="Approach3_MultiScale")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_approach3(
    X_train, y_train, X_val, y_val,
    n_classes, epochs=80, batch_size=64,
    dropout=0.3, lstm_units=64, kernel_sizes=None,
):
    from models.base import get_callbacks, compute_class_weights

    if kernel_sizes is None:
        kernel_sizes = KERNEL_SIZES

    lookback      = X_train.shape[1]
    valid_kernels = [k for k in kernel_sizes if k <= lookback] or [min(3, lookback)]
    model         = build_multiscale_cnn_lstm(
        X_train.shape[1:], n_classes, valid_kernels,
        dropout=dropout, lstm_units=lstm_units,
    )
    cw = compute_class_weights(y_train, n_classes)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=cw,
        callbacks=get_callbacks(),
        verbose=0,
    )
    return model, history


def predict_approach3(model, X_test: np.ndarray) -> tuple:
    proba = model.predict(X_test, verbose=0)
    return np.argmax(proba, axis=1), proba
