"""
signals/conviction.py
Signal conviction scoring via Z-score of model probabilities.
"""

import numpy as np


CONVICTION_THRESHOLDS = {
    "Very High": 2.0,
    "High":      1.0,
    "Moderate":  0.0,
    # Below 0.0 → "Low"
}


def compute_conviction(proba: np.ndarray, target_etfs: list, include_cash: bool = True) -> dict:
    """
    Compute Z-score conviction for the selected signal.

    Args:
        proba       : 1-D softmax probability vector [n_classes]
        target_etfs : list of ETF return column names (e.g. ["TLT_Ret", ...])
        include_cash: whether CASH is the last class

    Returns:
        dict with keys:
            best_idx        : int
            best_name       : str  (ETF ticker or "CASH")
            z_score         : float
            label           : str  ("Very High" / "High" / "Moderate" / "Low")
            scores          : np.ndarray (raw proba)
            etf_names       : list of display names
            sorted_pairs    : list of (name, score) sorted high→low
    """
    scores    = np.array(proba, dtype=float)
    best_idx  = int(np.argmax(scores))
    n_etfs    = len(target_etfs)

    # Display names
    etf_names = [e.replace("_Ret", "") for e in target_etfs]
    if include_cash:
        etf_names = etf_names + ["CASH"]

    best_name = etf_names[best_idx] if best_idx < len(etf_names) else "CASH"

    # Z-score
    mean = np.mean(scores)
    std  = np.std(scores)
    z    = float((scores[best_idx] - mean) / std) if std > 1e-9 else 0.0

    # Label
    label = "Low"
    for lbl, threshold in CONVICTION_THRESHOLDS.items():
        if z >= threshold:
            label = lbl
            break

    # Sorted pairs for UI bar chart
    sorted_pairs = sorted(
        zip(etf_names, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "best_idx":     best_idx,
        "best_name":    best_name,
        "z_score":      z,
        "label":        label,
        "scores":       scores,
        "etf_names":    etf_names,
        "sorted_pairs": sorted_pairs,
    }


def conviction_color(label: str) -> str:
    """Return hex accent colour for a conviction label."""
    return {
        "Very High": "#00b894",
        "High":      "#00cec9",
        "Moderate":  "#fdcb6e",
        "Low":       "#d63031",
    }.get(label, "#888888")


def conviction_icon(label: str) -> str:
    return {
        "Very High": "🟢",
        "High":      "🟢",
        "Moderate":  "🟡",
        "Low":       "🔴",
    }.get(label, "⚪")
