"""
ui/components.py
Reusable Streamlit UI blocks.
Changes:
- Metrics row: Ann Return compared vs SPY (not T-bill)
- Max Daily DD: shows date it happened
- Conviction panel: compact single-line ETF list (no big bars)
- applymap → map (deprecation fix)
"""

import streamlit as st
import pandas as pd
import numpy as np
from signals.conviction import conviction_color, conviction_icon


# ── Freshness status ──────────────────────────────────────────────────────────

def show_freshness_status(freshness: dict):
    if freshness.get("fresh"):
        st.success(freshness["message"])
    else:
        st.warning(freshness["message"])


# ── Winner signal banner ──────────────────────────────────────────────────────

def show_signal_banner(next_signal: str, next_date, approach_name: str):
    is_cash = next_signal == "CASH"
    bg = ("linear-gradient(135deg, #2d3436 0%, #1a1a2e 100%)" if is_cash
          else "linear-gradient(135deg, #00d1b2 0%, #00a896 100%)")
    label = ("⚠️ DRAWDOWN PROTECTION ACTIVE — CASH"
             if is_cash else f"🎯 {next_date.strftime('%Y-%m-%d')} → {next_signal}")
    st.markdown(f"""
    <div style="background:{bg}; padding:25px; border-radius:15px;
                text-align:center; box-shadow:0 8px 16px rgba(0,0,0,0.3); margin:16px 0;">
      <div style="color:rgba(255,255,255,0.7); font-size:12px;
                  letter-spacing:3px; margin-bottom:6px;">
        {approach_name.upper()} · NEXT TRADING DAY SIGNAL
      </div>
      <h1 style="color:white; font-size:40px; margin:0; font-weight:800;
                 text-shadow:2px 2px 4px rgba(0,0,0,0.3);">
        {label}
      </h1>
    </div>
    """, unsafe_allow_html=True)


# ── All models signals panel ──────────────────────────────────────────────────

def show_all_signals_panel(all_signals: dict, target_etfs: list,
                            include_cash: bool, next_date, optimal_lookback: int):
    COLORS = {"Approach 1": "#00ffc8", "Approach 2": "#7c6aff", "Approach 3": "#ff6b6b"}

    st.subheader(f"🗓️ All Models — {next_date.strftime('%Y-%m-%d')} Signals")
    st.caption(f"📐 Lookback **{optimal_lookback}d** found optimal (auto-selected from 30 / 45 / 60d)")

    cols = st.columns(len(all_signals))
    for col, (name, info) in zip(cols, all_signals.items()):
        color    = COLORS.get(name, "#888")
        signal   = info["signal"]
        top_prob = float(np.max(info["proba"])) * 100
        badge    = " ⭐" if info["is_winner"] else ""
        sig_col  = "#aaa" if signal == "CASH" else "white"

        col.markdown(f"""
        <div style="border:2px solid {color}; border-radius:12px; padding:18px 16px;
                    background:#111118; text-align:center; margin-bottom:8px;">
            <div style="color:{color}; font-size:10px; font-weight:700;
                        letter-spacing:2px; margin-bottom:6px;">{name.upper()}{badge}</div>
            <div style="color:{sig_col}; font-size:30px; font-weight:800; margin:8px 0;">{signal}</div>
            <div style="color:#aaa; font-size:12px;">
                Confidence: <span style="color:{color}; font-weight:700;">{top_prob:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Signal conviction panel ───────────────────────────────────────────────────

def show_conviction_panel(conviction: dict):
    label        = conviction["label"]
    z_score      = conviction["z_score"]
    best_name    = conviction["best_name"]
    sorted_pairs = conviction["sorted_pairs"]
    color        = conviction_color(label)
    icon         = conviction_icon(label)

    z_clipped = max(-3.0, min(3.0, z_score))
    bar_pct   = int((z_clipped + 3) / 6 * 100)

    # ── Header with Z-score gauge ─────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:#ffffff; border:1px solid #ddd;
                border-left:5px solid {color}; border-radius:12px;
                padding:18px 24px 16px 24px; margin:12px 0;
                box-shadow:0 2px 8px rgba(0,0,0,0.07);">
      <div style="display:flex; align-items:center; gap:12px; margin-bottom:14px; flex-wrap:wrap;">
        <span style="font-size:20px;">{icon}</span>
        <span style="font-size:18px; font-weight:700; color:#1a1a1a;">Signal Conviction</span>
        <span style="background:#f0f0f0; border:1px solid {color}; color:{color};
                     font-weight:700; font-size:14px; padding:3px 12px; border-radius:8px;">
          Z = {z_score:.2f} &sigma;
        </span>
        <span style="margin-left:auto; background:{color}; color:#fff;
                     font-weight:700; padding:4px 16px; border-radius:20px; font-size:13px;">
          {label}
        </span>
      </div>
      <div style="display:flex; justify-content:space-between;
                  font-size:11px; color:#999; margin-bottom:4px;">
        <span>Weak &minus;3&sigma;</span><span>Neutral 0&sigma;</span><span>Strong +3&sigma;</span>
      </div>
      <div style="background:#f0f0f0; border-radius:8px; height:10px; overflow:hidden;
                  position:relative; border:1px solid #e0e0e0; margin-bottom:16px;">
        <div style="position:absolute; left:50%; top:0; width:2px; height:100%; background:#ccc;"></div>
        <div style="width:{bar_pct}%; height:100%;
                    background:linear-gradient(90deg,#fab1a0,{color}); border-radius:8px;"></div>
      </div>
      <div style="font-size:11px; color:#999; margin-bottom:8px; font-weight:600; letter-spacing:1px;">
        MODEL PROBABILITY BY ETF
      </div>
      <div style="display:flex; flex-wrap:wrap; gap:8px;">
        {"".join([
            f'<span style="background:{"#e8fdf7" if n == best_name else "#f8f8f8"}; '
            f'border:1px solid {"' + color + '" if n == best_name else "#ddd"}; '
            f'border-radius:6px; padding:4px 10px; font-size:13px; '
            f'color:{"' + color + '" if n == best_name else "#555"}; font-weight:{"700" if n == best_name else "400"};">'
            f'{"★ " if n == best_name else ""}{n} {s:.3f}</span>'
            for n, s in sorted_pairs
        ])}
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "Z-score = std deviations the top ETF's probability sits above the mean of all ETF probabilities. "
        "Higher → model is more decisive.  "
        "⚠️ CASH override triggers if 2-day cumulative return ≤ −15%, exits when Z ≥ 1.0."
    )


# ── Metrics row ───────────────────────────────────────────────────────────────

def show_metrics_row(result: dict, tbill_rate: float, spy_ann_return: float = None):
    c1, c2, c3, c4, c5 = st.columns(5)

    # Ann return vs SPY
    if spy_ann_return is not None:
        diff = (result['ann_return'] - spy_ann_return) * 100
        sign = "+" if diff >= 0 else ""
        delta_str = f"vs SPY: {sign}{diff:.2f}%"
    else:
        delta_str = f"vs T-bill: {(result['ann_return'] - tbill_rate)*100:.2f}%"

    c1.metric("📈 Ann. Return", f"{result['ann_return']*100:.2f}%", delta=delta_str)
    c2.metric("📊 Sharpe",      f"{result['sharpe']:.2f}",
              delta="Strong" if result['sharpe'] > 1 else "Weak")
    c3.metric("🎯 Hit Ratio 15d", f"{result['hit_ratio']*100:.0f}%",
              delta="Good" if result['hit_ratio'] > 0.55 else "Weak")
    c4.metric("📉 Max Drawdown", f"{result['max_dd']*100:.2f}%",
              delta="Peak to Trough")

    # Max daily DD with date
    worst_date = result.get("max_daily_date", "N/A")
    c5.metric("⚠️ Max Daily DD", f"{result['max_daily_dd']*100:.2f}%",
              delta=f"on {worst_date}")


# ── Comparison table ──────────────────────────────────────────────────────────

def show_comparison_table(comparison_df: pd.DataFrame):
    def _highlight(row):
        if "WINNER" in str(row.get("Winner", "")):
            return ["background-color: rgba(0,200,150,0.15); font-weight:bold"] * len(row)
        return [""] * len(row)

    styled = (
        comparison_df.style
        .apply(_highlight, axis=1)
        .set_properties(**{"text-align": "center", "font-size": "14px"})
        .set_table_styles([
            {"selector": "th", "props": [("font-size", "14px"),
                                          ("font-weight", "bold"),
                                          ("text-align", "center")]},
            {"selector": "td", "props": [("padding", "10px")]},
        ])
    )
    st.dataframe(styled, use_container_width=True)


# ── Audit trail ───────────────────────────────────────────────────────────────

def show_audit_trail(audit_trail: list):
    if not audit_trail:
        st.info("No audit trail data available.")
        return

    df   = pd.DataFrame(audit_trail).tail(20)
    cols = [c for c in ["Date", "Signal", "Net_Return", "Z_Score"] if c in df.columns]
    df   = df[cols]

    def _color_ret(val):
        return ("color: #00c896; font-weight:bold" if val > 0
                else "color: #ff4b4b; font-weight:bold")

    fmt = {"Net_Return": "{:.2%}"}
    if "Z_Score" in df.columns:
        fmt["Z_Score"] = "{:.2f}"

    styled = (
        df.style
        .map(_color_ret, subset=["Net_Return"])
        .format(fmt)
        .set_properties(**{"font-size": "14px", "text-align": "center"})
        .set_table_styles([
            {"selector": "th", "props": [("font-size", "14px"),
                                          ("font-weight", "bold"),
                                          ("text-align", "center")]},
            {"selector": "td", "props": [("padding", "10px")]},
        ])
    )
    st.dataframe(styled, use_container_width=True, height=500)
