"""
models/mc_dropout.py
MC Dropout uncertainty wrapper for all three CNN-LSTM approaches.

Works by running N stochastic forward passes with dropout ACTIVE at inference
time (model kept in training mode). The variance across passes yields a free
epistemic uncertainty signal per ETF class.

Approach-specific notes
-----------------------
Approach 1 (Wavelet):   apply_wavelet_transform is called once before the loop;
                         the same transformed array is reused across all passes.
Approach 2 (Regime):    regime one-hot is deterministic — only the neural-network
                         dropout varies across passes.
Approach 3 (MultiScale): plain forward pass, three parallel CNN towers each with
                         independent dropout masks.

Usage (drop-in replacement for existing predict_approachN calls)
----------------------------------------------------------------
from models.mc_dropout import mc_predict_approach1, mc_predict_approach2, mc_predict_approach3

preds, proba, unc = mc_predict_approach1(model, X_test, n_passes=50)
preds, proba, unc = mc_predict_approach2(model, X_test, X_flat_all,
                                          regime_cols_idx, hmm_model,
                                          lookback, train_size, val_size,
                                          n_passes=50)
preds, proba, unc = mc_predict_approach3(model, X_test, n_passes=50)
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from typing import Tuple

# ── Type alias ────────────────────────────────────────────────────────────────
MCResult = Tuple[np.ndarray, np.ndarray, np.ndarray]
# (predicted_class [N], mean_proba [N, C], uncertainty [N, C])

# ── Default passes ────────────────────────────────────────────────────────────
DEFAULT_N_PASSES = 50   # sweet spot: variance stabilises by ~30, diminishing returns after 60


# ── Core stochastic forward pass ─────────────────────────────────────────────

def _mc_forward_single(model: tf.keras.Model,
                        inputs,
                        n_passes: int) -> np.ndarray:
    """
    Run `n_passes` stochastic forward passes on a Keras model.

    The model is called with `training=True` to activate dropout layers.
    No gradient tape is opened — inference only.

    Args:
        model   : compiled Keras model with Dropout layers
        inputs  : numpy array OR list of arrays (for multi-input models)
        n_passes: number of stochastic forward passes

    Returns:
        stack : float32 array [n_passes, N, C]
    """
    all_proba = []
    for _ in range(n_passes):
        if isinstance(inputs, (list, tuple)):
            # Multi-input model (Approach 2: [seq_input, regime_input])
            tensor_inputs = [tf.constant(x, dtype=tf.float32) for x in inputs]
            proba = model(tensor_inputs, training=True).numpy()
        else:
            proba = model(tf.constant(inputs, dtype=tf.float32),
                          training=True).numpy()
        all_proba.append(proba)

    return np.stack(all_proba, axis=0).astype(np.float32)  # [n_passes, N, C]


# ── Uncertainty metrics ───────────────────────────────────────────────────────

def _summarise_passes(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Derive mean, uncertainty, and predictions from a [n_passes, N, C] stack.

    Uncertainty is the standard deviation of per-class softmax probabilities
    across passes, averaged over classes to give one scalar per sample.
    This is the *epistemic* uncertainty — it is high when dropout consistently
    changes the predicted distribution, i.e. the model is genuinely uncertain.

    Returns
    -------
    preds       : int32   [N]       argmax of mean_proba
    mean_proba  : float32 [N, C]    mean softmax across passes
    uncertainty : float32 [N, C]    std across passes per class
    """
    mean_proba  = stack.mean(axis=0)          # [N, C]
    uncertainty = stack.std(axis=0)           # [N, C]  per-class std
    preds       = np.argmax(mean_proba, axis=1).astype(np.int32)
    return preds, mean_proba, uncertainty


# ── Approach 1: Wavelet ───────────────────────────────────────────────────────

def mc_predict_approach1(model: tf.keras.Model,
                          X_test: np.ndarray,
                          n_passes: int = DEFAULT_N_PASSES) -> MCResult:
    """
    MC Dropout inference for Approach 1 (Wavelet CNN-LSTM).

    Wavelet transform is deterministic — computed once, reused each pass.
    Only the CNN/LSTM Dropout layers vary.

    Args:
        model    : trained Approach 1 Keras model
        X_test   : raw (unprocessed) test sequences [N, lookback, features]
        n_passes : stochastic forward passes (default 50)

    Returns:
        preds, mean_proba, uncertainty
    """
    from models.approach1_wavelet import apply_wavelet_transform

    X_wt  = apply_wavelet_transform(X_test)   # deterministic, done once
    stack = _mc_forward_single(model, X_wt, n_passes)
    return _summarise_passes(stack)


# ── Approach 2: Regime-Conditioned ───────────────────────────────────────────

def mc_predict_approach2(model: tf.keras.Model,
                          X_test: np.ndarray,
                          X_flat_all: np.ndarray,
                          regime_cols_idx: list,
                          hmm_model,
                          lookback: int,
                          train_size: int,
                          val_size: int,
                          n_passes: int = DEFAULT_N_PASSES) -> MCResult:
    """
    MC Dropout inference for Approach 2 (Regime-Conditioned CNN-LSTM).

    The regime one-hot vector is deterministic (HMM predict is fixed).
    Only the neural-network Dropout layers vary across passes.

    Args:
        model           : trained Approach 2 Keras model
        X_test          : scaled test sequences [N, lookback, features]
        X_flat_all      : full unsequenced feature matrix [T, features]
        regime_cols_idx : indices of regime feature columns
        hmm_model       : fitted HMM (or None → quantile fallback)
        lookback        : sequence lookback used during training
        train_size      : number of training samples
        val_size        : number of validation samples
        n_passes        : stochastic forward passes (default 50)

    Returns:
        preds, mean_proba, uncertainty
    """
    from models.approach2_regime import predict_regimes, build_regime_sequences

    # Regime is deterministic — build once
    regimes_all = predict_regimes(hmm_model, X_flat_all, regime_cols_idx)
    offset      = lookback + train_size + val_size
    R_test      = build_regime_sequences(X_test, regimes_all, offset)

    stack = _mc_forward_single(model, [X_test, R_test], n_passes)
    return _summarise_passes(stack)


# ── Approach 3: Multi-Scale ───────────────────────────────────────────────────

def mc_predict_approach3(model: tf.keras.Model,
                          X_test: np.ndarray,
                          n_passes: int = DEFAULT_N_PASSES) -> MCResult:
    """
    MC Dropout inference for Approach 3 (Multi-Scale Parallel CNN-LSTM).

    Three parallel CNN towers each have independent Dropout masks per pass,
    producing the richest uncertainty signal of the three approaches.

    Args:
        model    : trained Approach 3 Keras model
        X_test   : scaled test sequences [N, lookback, features]
        n_passes : stochastic forward passes (default 50)

    Returns:
        preds, mean_proba, uncertainty
    """
    stack = _mc_forward_single(model, X_test, n_passes)
    return _summarise_passes(stack)
