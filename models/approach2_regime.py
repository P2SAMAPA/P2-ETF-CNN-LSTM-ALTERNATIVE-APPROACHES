"""
models/approach2_regime.py
Approach 2: Regime-Conditioned CNN-LSTM
- Fixed HMM convergence settings
- CPU-optimised smaller architecture
- Lazy imports to prevent module-level failures
- Class weights to prevent majority-class collapse
"""

import numpy as np

N_REGIMES    = 3          # reduced from 4 to improve HMM convergence
REGIME_HINTS = ["VIX", "HY", "Spread", "T10Y2Y", "T10Y3M", "Credit",
                 "IG_SPREAD", "HY_SPREAD"]


def _get_regime_cols(feature_names: list) -> list:
    return [
        f for f in feature_names
        if any(hint.lower() in f.lower() for hint in REGIME_HINTS)
    ]


def fit_regime_model(X_flat: np.ndarray, feature_names: list,
                     n_regimes: int = N_REGIMES):
    regime_col_names = _get_regime_cols(feature_names)
    if not regime_col_names:
        regime_col_names = feature_names[:min(3, len(feature_names))]

    regime_cols_idx = [feature_names.index(c) for c in regime_col_names
                       if c in feature_names]
    X_regime = X_flat[:, regime_cols_idx]

    try:
        from hmmlearn.hmm import GaussianHMM
        hmm = GaussianHMM(
            n_components=n_regimes,
            covariance_type="diag",
            n_iter=50,           # reduced from 100
            tol=1e-2,            # looser tolerance — avoids non-convergence warning
            random_state=42,
        )
        hmm.fit(X_regime)
        return hmm, regime_cols_idx
    except Exception as e:
        print(f"[Approach 2] HMM failed: {e}. Using quantile fallback.")
        return None, regime_cols_idx


def predict_regimes(hmm_model, X_flat: np.ndarray,
                    regime_cols_idx: list,
                    n_regimes: int = N_REGIMES) -> np.ndarray:
    X_regime = X_flat[:, regime_cols_idx]
    if hmm_model is not None:
        try:
            return hmm_model.predict(X_regime)
        except Exception:
            pass
    feat      = X_regime[:, 0]
    quantiles = np.percentile(feat, np.linspace(0, 100, n_regimes + 1))
    return np.digitize(feat, quantiles[1:-1]).astype(int)


def regimes_to_onehot(regimes: np.ndarray, n_regimes: int = N_REGIMES) -> np.ndarray:
    one_hot = np.zeros((len(regimes), n_regimes), dtype=np.float32)
    for i, r in enumerate(regimes):
        one_hot[i, min(int(r), n_regimes - 1)] = 1.0
    return one_hot


def build_regime_sequences(X_seq: np.ndarray, regimes_flat: np.ndarray,
                            lookback: int) -> np.ndarray:
    n_samples = X_seq.shape[0]
    aligned   = regimes_flat[lookback: lookback + n_samples]
    return regimes_to_onehot(aligned)


def build_regime_cnn_lstm(seq_input_shape, n_classes,
                           n_regimes=N_REGIMES, dropout=0.3, lstm_units=64):
    from tensorflow import keras
    from models.base import classification_head

    seq_input = keras.Input(shape=seq_input_shape, name="seq_input")
    x = keras.layers.Conv1D(32, 3, padding="causal", activation="relu")(seq_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Conv1D(16, 3, padding="causal", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout)(x)
    cnn_out = keras.layers.GlobalAveragePooling1D()(x)

    regime_input = keras.Input(shape=(n_regimes,), name="regime_input")
    regime_emb   = keras.layers.Dense(8, activation="relu")(regime_input)

    merged = keras.layers.Concatenate()([cnn_out, regime_emb])
    x      = keras.layers.Reshape((1, merged.shape[-1]))(merged)
    x      = keras.layers.LSTM(lstm_units, dropout=dropout)(x)
    outputs = classification_head(x, n_classes, dropout)

    model = keras.Model(inputs=[seq_input, regime_input], outputs=outputs,
                        name="Approach2_Regime")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_approach2(
    X_train, y_train, X_val, y_val,
    X_flat_all, feature_names, lookback,
    train_size, val_size, n_classes,
    epochs=80, batch_size=64, dropout=0.3, lstm_units=64,
):
    from models.base import get_callbacks, compute_class_weights

    X_flat_train = X_flat_all[:train_size + lookback]
    hmm_model, regime_cols_idx = fit_regime_model(X_flat_train, feature_names)
    regimes_all = predict_regimes(hmm_model, X_flat_all, regime_cols_idx)

    R_train = build_regime_sequences(X_train, regimes_all, lookback)
    R_val   = build_regime_sequences(X_val,   regimes_all, lookback + train_size)

    model = build_regime_cnn_lstm(X_train.shape[1:], n_classes,
                                   dropout=dropout, lstm_units=lstm_units)
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


def predict_approach2(model, X_test, X_flat_all, regime_cols_idx,
                       hmm_model, lookback, train_size, val_size) -> tuple:
    regimes_all = predict_regimes(hmm_model, X_flat_all, regime_cols_idx)
    offset      = lookback + train_size + val_size
    R_test      = build_regime_sequences(X_test, regimes_all, offset)
    proba       = model.predict([X_test, R_test], verbose=0)
    return np.argmax(proba, axis=1), proba
