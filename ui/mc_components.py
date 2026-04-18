"""
ui/mc_components.py
Streamlit UI components for MC Dropout uncertainty display.

Designed to slot into the existing app.py display flow alongside the
existing show_conviction_panel() and show_signal_banner() calls.
All functions are self-contained — no changes to existing ui/components.py needed.

Usage in app.py
---------------
Replace the existing predict + conviction block in run_module():

    # OLD
    preds, proba = predict_approach1(model, X_test_s)
    conviction   = compute_conviction(proba[-1], target_etfs)

    # NEW (drop-in)
    from models.mc_dropout import mc_predict_approach1
    from signals.mc_conviction import compute_mc_conviction
    from ui.mc_components import show_mc_conviction_panel, show_mc_uncertainty_bar

    preds, mean_proba, uncertainty = mc_predict_approach1(model, X_test_s, n_passes=50)
    mc_conv = compute_mc_conviction(mean_proba[-1], uncertainty[-1], target_etfs)
    show_mc_conviction_panel(mc_conv)
    show_mc_uncertainty_bar(mc_conv)
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from signals.mc_conviction import uncertainty_summary_text


# ── MC Conviction Panel ────────────────────────────────────────────────────────

def show_mc_conviction_panel(mc_conv: dict, n_passes: int = 50) -> None:
    """
    Full MC Dropout conviction panel.
    Replaces show_conviction_panel() when MC Dropout is enabled.

    Displays:
    - Best ETF + Z-score conviction label (same as existing panel)
    - CASH override banner if cash_flag is True
    - Uncertainty label + mean σ
    - Adjusted conviction score
    - Per-ETF probability bars (mean_proba)
    - Per-ETF uncertainty bars (std across passes)
    """
    best_name  = mc_conv["best_name"]
    z          = mc_conv["z_score"]
    adj_z      = mc_conv["adjusted_z"]
    label      = mc_conv["label"]
    unc_label  = mc_conv["unc_label"]
    unc_color  = mc_conv["unc_color"]
    cash_flag  = mc_conv["cash_flag"]
    mean_unc   = mc_conv["mean_uncertainty"]
    unc_score  = mc_conv["unc_score"]

    from signals.conviction import conviction_color, conviction_icon
    conv_color = conviction_color(label)
    icon       = conviction_icon(label)

    st.markdown("### 🎯 MC Dropout Signal Conviction")
    st.caption(f"Based on {n_passes} stochastic forward passes · Dropout active at inference")

    # ── CASH override ──────────────────────────────────────────────────────────
    if cash_flag:
        st.error(
            f"⚠️ **CASH OVERRIDE** — Uncertainty too high to commit.  \n"
            f"{mc_conv['cash_reason']}"
        )

    # ── Top metrics row ────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Best ETF",
            value=f"{icon} {best_name}",
            delta=None,
        )
    with col2:
        st.metric(
            label="Conviction (Z)",
            value=f"{z:.2f}",
            delta=label,
        )
    with col3:
        st.metric(
            label="Adjusted Conviction",
            value=f"{adj_z:.2f}",
            help="Z-score × uncertainty score. Lower when model is uncertain.",
        )
    with col4:
        st.metric(
            label="Uncertainty",
            value=f"{mean_unc:.3f} σ̄",
            delta=unc_label,
            delta_color="inverse",
        )

    st.divider()

    # ── Side-by-side bars: probability vs uncertainty ─────────────────────────
    col_prob, col_unc = st.columns(2)

    with col_prob:
        st.markdown("**Mean Probability (across passes)**")
        sorted_pairs = mc_conv["sorted_pairs"]
        for name, score in sorted_pairs:
            pct = float(score) * 100
            bar = "█" * int(pct / 5)
            color = conv_color if name == best_name else "#888888"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px'>"
                f"<span style='min-width:55px;font-size:13px;font-weight:600'>{name}</span>"
                f"<span style='flex:1;background:#f0f0f0;border-radius:4px;overflow:hidden'>"
                f"<span style='display:block;width:{pct:.1f}%;background:{color};"
                f"height:18px;border-radius:4px'></span></span>"
                f"<span style='font-size:12px;min-width:42px;text-align:right'>{pct:.1f}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    with col_unc:
        st.markdown("**Uncertainty per ETF (σ across passes)**")
        unc_pairs = mc_conv["uncertainty_pairs"]
        max_unc   = max(u for _, u in unc_pairs) if unc_pairs else 1.0
        for name, unc in unc_pairs:
            pct  = (unc / max(max_unc, 1e-9)) * 100
            # Colour: red if this ETF has the highest uncertainty
            bar_color = "#d63031" if unc == max_unc else "#fdcb6e"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px'>"
                f"<span style='min-width:55px;font-size:13px;font-weight:600'>{name}</span>"
                f"<span style='flex:1;background:#f0f0f0;border-radius:4px;overflow:hidden'>"
                f"<span style='display:block;width:{pct:.1f}%;background:{bar_color};"
                f"height:18px;border-radius:4px'></span></span>"
                f"<span style='font-size:12px;min-width:48px;text-align:right'>{unc:.4f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.caption(uncertainty_summary_text(mc_conv))


# ── Compact uncertainty badge ─────────────────────────────────────────────────

def show_mc_uncertainty_badge(mc_conv: dict) -> None:
    """
    Compact one-line badge for use in the approach comparison table.
    Shows: [unc_label] σ̄=0.087 | adj_z=1.23
    """
    color = mc_conv["unc_color"]
    label = mc_conv["unc_label"]
    mean_unc = mc_conv["mean_uncertainty"]
    adj_z    = mc_conv["adjusted_z"]
    cash_str = " ⚠️ CASH" if mc_conv["cash_flag"] else ""

    st.markdown(
        f"<span style='background:{color}20;color:{color};border:1px solid {color};"
        f"border-radius:4px;padding:2px 8px;font-size:12px;font-weight:600'>"
        f"{label}{cash_str}</span>"
        f"<span style='font-size:12px;margin-left:8px;color:#888'>"
        f"σ̄={mean_unc:.3f} | adj_z={adj_z:.2f}</span>",
        unsafe_allow_html=True,
    )


# ── N-passes selector ─────────────────────────────────────────────────────────

def mc_passes_selector(key: str = "mc_n_passes") -> int:
    """
    Sidebar widget for selecting number of MC passes.
    Call from within `with st.sidebar:` block in app.py.

    Returns: int n_passes
    """
    return st.slider(
        "🎲 MC Dropout passes",
        min_value=10,
        max_value=100,
        value=50,
        step=10,
        key=key,
        help=(
            "Number of stochastic forward passes for uncertainty estimation. "
            "50 is the recommended default: variance stabilises around 30 passes, "
            "and returns above 60 are marginal on CPU."
        ),
    )


# ── All-signals panel with uncertainty ────────────────────────────────────────

def show_mc_all_signals_panel(
    mc_results: dict,
    target_etfs: list,
    include_cash: bool,
    next_date: str,
    optimal_lookback: int,
    n_passes: int = 50,
) -> None:
    """
    Replacement for show_all_signals_panel() when MC Dropout is enabled.

    mc_results: dict keyed by approach name, each value:
        {
            "signal":    str,
            "mc_conv":   dict (from compute_mc_conviction),
            "is_winner": bool,
        }
    """
    st.subheader(f"📡 All Approach Signals — {next_date}")
    st.caption(
        f"Lookback: {optimal_lookback}d · MC Dropout: {n_passes} passes · "
        f"Dropout active at inference"
    )

    cols = st.columns(len(mc_results))

    for col, (approach_name, info) in zip(cols, mc_results.items()):
        with col:
            mc_conv   = info.get("mc_conv")
            signal    = info.get("signal", "N/A")
            is_winner = info.get("is_winner", False)

            winner_badge = " 🏆" if is_winner else ""
            st.markdown(f"**{approach_name}{winner_badge}**")

            if mc_conv is None:
                st.warning("No signal")
                continue

            # Signal ETF
            cash_str = " → ⚠️ **CASH**" if mc_conv["cash_flag"] else ""
            st.markdown(f"📌 **{signal}**{cash_str}")

            # Conviction badge
            show_mc_uncertainty_badge(mc_conv)

            # Mini probability bars
            st.caption("Probabilities")
            for name, score in mc_conv["sorted_pairs"][:3]:
                pct = float(score) * 100
                st.markdown(
                    f"<div style='font-size:11px;display:flex;justify-content:space-between'>"
                    f"<span>{name}</span><span>{pct:.1f}%</span></div>",
                    unsafe_allow_html=True,
                )
