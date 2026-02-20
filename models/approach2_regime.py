"""
models/approach2_regime.py
Approach 2: Regime-Conditioned CNN-LSTM

Pipeline:
  Raw macro signals
  -> CNN Tower (64 filters, k=3) -> feature vector
  -> Regime Classifier (HMM on VIX + HY spread + T10Y2Y) -> one-hot [4]
  -> Concatenate CNN features + regime embedding
  -> LSTM (128 units)
  -> Dense 64 -> Softmax (n_etfs + 1 CASH)

NOTE: tensorflow and hmmlearn are imported lazily inside functions
to prevent module-level import failures from making this module
appear broken to Python's import system.
"""

import numpy as np

N_REGIMES    = 4
REGIME_HINTS = ["VIX", "HY", "Spread", "T10Y2Y", "T10Y3M", "Credit"]


# ---------------------------------------------------------------------------
# Regime detection helpers
# ---------------------------------------------------------------------------

def _get_regime_cols(feature_names: list) -> list:
    return [
        f for f in feature_names
        if any(hint.lower() in f.lower() for hint in REGIME_HINTS)
    ]


def fit_regime_model(X_flat: np.ndarray, feature_names: list,
                     n_regimes: int = N_REGIMES):
    """
    Fit a Gaussian HMM on regime-relevant macro features.
    Returns (hmm_model, regime_cols_idx).
    hmm_model is None if hmmlearn is unavailable or fitting fails.
    """
    regime_col_names = _get_regime_cols(feature_names)
    if not regime_col_names:
        regime_col_names = feature_names[:min(3, len(feature_names))]

    regime_cols_idx = [
        feature_names.index(c) for c in regime_col_names
        if c in feature_names
    ]
    X_regime = X_flat[:, regime_cols_idx]

    try:
        from hmmlearn.hmm import GaussianHMM
        hmm = GaussianHMM(
            n_components=n_regimes,
            covariance_type="diag",
            n_iter=100,
            random_state=42,
        )
        hmm.fit(X_regime)
        return hmm, regime_cols_idx
    except Exception as e:
        print(f"[Approach 2] HMM fitting failed: {e}. Using fallback.")
        return None, regime_cols_idx


def predict_regimes(hmm_model, X_flat: np.ndarray,
                    regime_cols_idx: list,
                    n_regimes: int = N_REGIMES) -> np.ndarray:
    """Predict integer regime label for each day."""
    X_regime = X_flat[:, regime_cols_idx]

    if hmm_model is not None:
        try:
            return hmm_model.predict(X_regime)
        except Exception:
            pass

    # Fallback: quantile binning on first regime feature
    feat      = X_regime[:, 0]
    quantiles = np.percentile(feat, np.linspace(0, 100, n_regimes + 1))
    return np.digitize(feat, quantiles[1:-1]).astype(int)


def regimes_to_onehot(regimes: np.ndarray,
                      n_regimes: int = N_REGIMES) -> np.ndarray:
    one_hot = np.zeros((len(regimes), n_regimes), dtype=np.float32)
    for i, r in enumerate(regimes):
        one_hot[i, min(int(r), n_regimes - 1)] = 1.0
    return one_hot


def build_regime_sequences(X_seq: np.ndarray,
                            regimes_flat: np.ndarray,
                            lookback: int) -> np.ndarray:
    n_samples = X_seq.shape[0]
    aligned   = regimes_flat[lookback: lookback + n_samples]
    return regimes_to_onehot(aligned)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_regime_cnn_lstm(seq_input_shape: tuple,
                           n_classes: int,
                           n_regimes: int = N_REGIMES,
                           dropout: float = 0.3,
                           lstm_units: int = 128):
    """Build and compile the regime-conditioned CNN-LSTM model."""
    from tensorflow import keras
    from models.base import classification_head

    seq_input = keras.Input(shape=seq_input_shape, name="seq_input")
    x = keras.layers.Conv1D(64, kernel_size=3, padding="causal",
                            activation="relu")(seq_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)
    x = keras.layers.Conv1D(32, kernel_size=3, padding="causal",
                            activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout)(x)
    cnn_out = keras.layers.GlobalAveragePooling1D()(x)

    regime_input = keras.Input(shape=(n_regimes,), name="regime_input")
    regime_emb   = keras.layers.Dense(8, activation="relu")(regime_input)

    merged = keras.layers.Concatenate()([cnn_out, regime_emb])
    x      = keras.layers.Reshape((1, merged.shape[-1]))(merged)
    x      = keras.layers.LSTM(lstm_units, dropout=dropout)(x)

    outputs = classification_head(x, n_classes, dropout)

    model = keras.Model(
        inputs=[seq_input, regime_input],
        outputs=outputs,
        name="Approach2_Regime_CNN_LSTM",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def train_approach2(
    X_train, y_train,
    X_val,   y_val,
    X_flat_all: np.ndarray,
    feature_names: list,
    lookback: int,
    train_size: int,
    val_size: int,
    n_classes: int,
    epochs: int = 100,
    batch_size: int = 32,
    dropout: float = 0.3,
    lstm_units: int = 128,
):
    """
    Fit HMM regime model then train the regime-conditioned CNN-LSTM.
    Returns: model, history, hmm_model, regime_cols_idx
    """
    from models.base import get_callbacks, compute_class_weights

    X_flat_train = X_flat_all[:train_size + lookback]
    hmm_model, regime_cols_idx = fit_regime_model(X_flat_train, feature_names)

    regimes_all = predict_regimes(hmm_model, X_flat_all, regime_cols_idx)

    R_train = build_regime_sequences(X_train, regimes_all, lookback)
    R_val   = build_regime_sequences(X_val,   regimes_all, lookback + train_size)

    model = build_regime_cnn_lstm(
        X_train.shape[1:], n_classes,
        dropout=dropout, lstm_units=lstm_units,
    )

    cw = compute_class_weights(y_train, n_classes)

    history = model.fit(
        [X_train, R_train], y_train,
        validation_data=([X_val, R_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=cw,
        callbacks=get_callbacks(),
        verbose=0,
    )

    return model, history, hmm_model, regime_cols_idx


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_approach2(
    model,
    X_test: np.ndarray,
    X_flat_all: np.ndarray,
    regime_cols_idx: list,
    hmm_model,
    lookback: int,
    train_size: int,
    val_size: int,
) -> tuple:
    """Predict on test set with regime conditioning. Returns (preds, proba)."""
    regimes_all = predict_regimes(hmm_model, X_flat_all, regime_cols_idx)
    offset      = lookback + train_size + val_size
    R_test      = build_regime_sequences(X_test, regimes_all, offset)

    proba = model.predict([X_test, R_test], verbose=0)
    preds = np.argmax(proba, axis=1)
    return preds, proba
