"""
signals/mc_conviction.py
MC Dropout-aware conviction scoring.

Extends the existing signals/conviction.py with uncertainty dimensions.
The existing compute_conviction() is preserved and unchanged — this module
wraps it and adds uncertainty-adjusted metrics from the MC Dropout pass stack.

Key concepts
------------
- mean_proba    : mean softmax across N passes  → replaces single-pass proba
- uncertainty   : per-class std across passes   → epistemic uncertainty
- conviction    : Z-score of mean_proba[best]   → same formula as before
- unc_score     : scalar [0,1], 1=certain, 0=uncertain
- adjusted_z    : z_score * unc_score           → conviction penalised by uncertainty
- cash_flag     : True if adjusted_z < cash_threshold → go CASH

The UI can call either compute_mc_conviction() (full dict) or
get_cash_flag() (simple bool) independently.
"""

from __future__ import annotations

import numpy as np
from signals.conviction import (
    compute_conviction,
    conviction_color,
    conviction_icon,
    CONVICTION_THRESHOLDS,
)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_CASH_THRESHOLD = 0.4   # adjusted_z below this → CASH
                                # Start at 0.4, tune via backtest


# ── Core function ─────────────────────────────────────────────────────────────

def compute_mc_conviction(
    mean_proba: np.ndarray,
    uncertainty: np.ndarray,
    target_etfs: list,
    include_cash: bool = False,
    cash_threshold: float = DEFAULT_CASH_THRESHOLD,
) -> dict:
    """
    Full MC Dropout conviction dict.

    Wraps compute_conviction() (Z-score on mean_proba) and adds
    uncertainty-penalised metrics.

    Args:
        mean_proba      : [C] mean softmax probabilities across MC passes
        uncertainty     : [C] per-class std across MC passes
        target_etfs     : list of ETF return column names
        include_cash    : whether CASH is the last class
        cash_threshold  : adjusted_z below this → recommend CASH

    Returns dict with all keys from compute_conviction() PLUS:
        uncertainty         : np.ndarray [C]   per-class std
        mean_uncertainty    : float            mean uncertainty across classes
        unc_score           : float [0,1]      1 = certain, 0 = uncertain
        adjusted_z          : float            z_score * unc_score
        cash_flag           : bool             True → model recommends CASH
        cash_reason         : str              human-readable explanation
        n_passes_implied    : None             (informational placeholder)
        unc_label           : str              "Certain" / "Uncertain" / "Very Uncertain"
        unc_color           : str              hex colour for UI
        uncertainty_pairs   : list[(name, unc)] sorted high-uncertainty first
    """
    # Base conviction from existing scorer (uses mean_proba as the probability)
    base = compute_conviction(mean_proba, target_etfs, include_cash)

    # Uncertainty scalar: mean std across all classes, normalised to [0,1]
    # Lower uncertainty → higher unc_score
    mean_unc  = float(np.mean(uncertainty))
    # A fully uncertain model has std ≈ 1/sqrt(C*(C-1)) for uniform dist;
    # practical max is ~0.5 for binary, less for many classes.
    # We clip to [0, 0.5] then invert.
    unc_score = float(1.0 - min(mean_unc / 0.5, 1.0))   # [0, 1]

    # Adjusted conviction: penalise Z by uncertainty
    adjusted_z = base["z_score"] * unc_score

    # Cash flag
    cash_flag   = adjusted_z < cash_threshold
    if cash_flag:
        cash_reason = (
            f"Adjusted conviction {adjusted_z:.2f} < threshold {cash_threshold:.2f} "
            f"(Z={base['z_score']:.2f}, uncertainty={mean_unc:.3f})"
        )
    else:
        cash_reason = ""

    # Uncertainty label + colour
    if mean_unc < 0.05:
        unc_label = "Certain"
        unc_color = "#00b894"
    elif mean_unc < 0.12:
        unc_label = "Moderate uncertainty"
        unc_color = "#fdcb6e"
    else:
        unc_label = "Very Uncertain"
        unc_color = "#d63031"

    # Per-ETF uncertainty pairs (for UI bar chart)
    etf_names = [e.replace("_Ret", "") for e in target_etfs]
    if include_cash:
        etf_names = etf_names + ["CASH"]
    n = min(len(etf_names), len(uncertainty))
    uncertainty_pairs = sorted(
        zip(etf_names[:n], uncertainty[:n].tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        # ── from base compute_conviction ────────────────────────────────────
        **base,
        # ── MC-specific additions ───────────────────────────────────────────
        "uncertainty":        uncertainty,
        "mean_uncertainty":   mean_unc,
        "unc_score":          unc_score,
        "adjusted_z":         adjusted_z,
        "cash_flag":          cash_flag,
        "cash_reason":        cash_reason,
        "n_passes_implied":   None,
        "unc_label":          unc_label,
        "unc_color":          unc_color,
        "uncertainty_pairs":  uncertainty_pairs,
    }


# ── Simple cash flag helper ───────────────────────────────────────────────────

def get_cash_flag(
    mean_proba: np.ndarray,
    uncertainty: np.ndarray,
    target_etfs: list,
    include_cash: bool = False,
    cash_threshold: float = DEFAULT_CASH_THRESHOLD,
) -> bool:
    """
    Lightweight helper: returns True if MC Dropout recommends CASH.
    Does not require computing the full conviction dict.
    """
    result = compute_mc_conviction(
        mean_proba, uncertainty, target_etfs, include_cash, cash_threshold
    )
    return result["cash_flag"]


# ── Uncertainty summary for display ──────────────────────────────────────────

def uncertainty_summary_text(mc_conv: dict) -> str:
    """
    One-line text summary of uncertainty state for UI display.
    e.g. "Uncertainty: Moderate (σ̄=0.087) · Adjusted conviction: 1.23"
    """
    return (
        f"Uncertainty: {mc_conv['unc_label']} "
        f"(σ̄={mc_conv['mean_uncertainty']:.3f}) · "
        f"Adjusted conviction: {mc_conv['adjusted_z']:.2f}"
    )
