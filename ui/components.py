"""
ui/components.py
Reusable Streamlit UI blocks.
- Fixed applymap → map deprecation
- Removed debug expanders
- Added show_all_signals_panel
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
    bg      = ("linear-gradient(135deg, #2d3436 0%, #1a1a2e 100%)" if is_cash
               else "linear-gradient(135deg, #00d1b2 0%, #00a896 100%)")
    st.markdown(f"""
    <div style="background:{bg}; padding:25px; border-radius:15px;
                text-align:center; box-shadow:0 8px 16px rgba(0,0,0,0.3); margin:16px 0;">
      <div style="color:rgba(255,255,255,0.7); font-size:12px;
                  letter-spacing:3px; margin-bottom:6px;">
        {approach_name.upper()} · NEXT TRADING DAY SIGNAL
      </div>
      <h1 style="color:white; font-size:44px; margin:0 0 8px 0;
                 font-weight:800; text-shadow:2px 2px 4px rgba(0,0,0,0.3);">
        🎯 {next_date.strftime('%Y-%m-%d')} → {next_signal}
      </h1>
    </div>
    """, unsafe_allow_html=True)


# ── All models signals panel ──────────────────────────────────────────────────

def show_all_signals_panel(all_signals: dict, target_etfs: list,
                            include_cash: bool, next_date, optimal_lookback: int):
    APPROACH_COLORS = {
        "Approach 1": "#00ffc8",
        "Approach 2": "#7c6aff",
        "Approach 3": "#ff6b6b",
    }

    st.subheader(f"🗓️ All Models — {next_date.strftime('%Y-%m-%d')} Signals")
    st.caption(f"📐 Optimal lookback: **{optimal_lookback}d** (auto-selected from 30/45/60)")

    cols = st.columns(len(all_signals))
    for col, (name, info) in zip(cols, all_signals.items()):
        color     = APPROACH_COLORS.get(name, "#888888")
        signal    = info["signal"]
        proba     = info["proba"]
        top_prob  = float(np.max(proba)) * 100
        is_winner = info["is_winner"]
        badge     = " ⭐" if is_winner else ""

        col.markdown(f"""
        <div style="border:2px solid {color}; border-radius:12px; padding:18px 16px;
                    background:#111118; text-align:center; margin-bottom:8px;">
            <div style="color:{color}; font-size:10px; font-weight:700;
                        letter-spacing:2px; margin-bottom:6px;">
                {name.upper()}{badge}
            </div>
            <div style="color:white; font-size:30px; font-weight:800; margin:8px 0;">
                {signal}
            </div>
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
    max_score = max((s for _, s in sorted_pairs), default=1.0)
    if max_score <= 0:
        max_score = 1.0

    st.markdown(f"""
    <div style="background:#ffffff; border:1px solid #ddd;
                border-left:5px solid {color}; border-radius:12px 12px 0 0;
                padding:18px 24px 12px 24px; margin:12px 0 0 0;
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
      <div style="background:#f0f0f0; border-radius:8px; height:14px; overflow:hidden;
                  position:relative; border:1px solid #e0e0e0; margin-bottom:14px;">
        <div style="position:absolute; left:50%; top:0; width:2px; height:100%; background:#ccc;"></div>
        <div style="width:{bar_pct}%; height:100%;
                    background:linear-gradient(90deg,#fab1a0,{color}); border-radius:8px;"></div>
      </div>
      <div style="font-size:12px; color:#999; margin-bottom:2px;">
        Model probability by ETF (ranked high &rarr; low):
      </div>
    </div>
    """, unsafe_allow_html=True)

    for i, (name, score) in enumerate(sorted_pairs):
        is_winner = (name == best_name)
        is_last   = (i == len(sorted_pairs) - 1)
        bar_w     = int(score / max_score * 100)
        name_style = "font-weight:700; color:#00897b;" if is_winner else "color:#444;"
        bar_color  = color if is_winner else "#b2dfdb" if score > max_score * 0.5 else "#e0e0e0"
        star       = " ★" if is_winner else ""
        bottom_r   = "0 0 12px 12px" if is_last else "0"
        border_bot = "border-bottom:1px solid #f0f0f0;" if not is_last else ""

        st.markdown(f"""
        <div style="background:#ffffff; border:1px solid #ddd; border-top:none;
                    border-radius:{bottom_r}; padding:7px 24px; {border_bot}
                    box-shadow:0 2px 8px rgba(0,0,0,0.07);">
          <div style="display:flex; align-items:center; gap:12px;">
            <span style="width:44px; text-align:right; font-size:13px; {name_style}">{name}{star}</span>
            <div style="flex:1; background:#f5f5f5; border-radius:4px; height:14px;
                        overflow:hidden; border:1px solid #e8e8e8;">
              <div style="width:{bar_w}%; height:100%; background:{bar_color}; border-radius:4px;"></div>
            </div>
            <span style="width:56px; font-size:12px; color:#888; text-align:right;">{score:.4f}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "Z-score = std deviations the top ETF's probability sits above the mean. "
        "Higher → model is more decisive."
    )


# ── Metrics row ───────────────────────────────────────────────────────────────

def show_metrics_row(result: dict, tbill_rate: float):
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📈 Ann. Return",   f"{result['ann_return']*100:.2f}%",
                delta=f"vs T-bill: {(result['ann_return'] - tbill_rate)*100:.2f}%")
    col2.metric("📊 Sharpe",        f"{result['sharpe']:.2f}",
                delta="Strong" if result['sharpe'] > 1 else "Weak")
    col3.metric("🎯 Hit Ratio 15d", f"{result['hit_ratio']*100:.0f}%",
                delta="Good" if result['hit_ratio'] > 0.55 else "Weak")
    col4.metric("📉 Max Drawdown",  f"{result['max_dd']*100:.2f}%",
                delta="Peak to Trough")
    col5.metric("⚠️ Max Daily DD",  f"{result['max_daily_dd']*100:.2f}%",
                delta="Worst Day")


# ── Comparison table ──────────────────────────────────────────────────────────

def show_comparison_table(comparison_df: pd.DataFrame):
    def highlight_winner(row):
        if "WINNER" in str(row.get("Winner", "")):
            return ["background-color: rgba(0,200,150,0.15); font-weight:bold"] * len(row)
        return [""] * len(row)

    styled = (
        comparison_df.style
        .apply(highlight_winner, axis=1)
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

    df = pd.DataFrame(audit_trail).tail(20)[["Date", "Signal", "Net_Return"]]

    def color_return(val):
        return ("color: #00c896; font-weight:bold" if val > 0
                else "color: #ff4b4b; font-weight:bold")

    styled = (
        df.style
        .map(color_return, subset=["Net_Return"])
        .format({"Net_Return": "{:.2%}"})
        .set_properties(**{"font-size": "14px", "text-align": "center"})
        .set_table_styles([
            {"selector": "th", "props": [("font-size", "14px"),
                                          ("font-weight", "bold"),
                                          ("text-align", "center")]},
            {"selector": "td", "props": [("padding", "10px")]},
        ])
    )
    st.dataframe(styled, use_container_width=True, height=500)
