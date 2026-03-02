"""
ui/multiyear.py
Multi-Year Consensus Sweep — runs Approach 2 (regime-conditioned) across
multiple start years and aggregates signals into a vote tally + comparison table.

Design principles:
- Reuses the existing cache wherever possible (no redundant retraining)
- Only Approach 2 is used for the sweep (it's the regime-aware model, most
  sensitive to start-year choice, and typically the winner)
- Each year runs independently; failures are soft (skipped with a warning)
- Results are shown as: (1) vote tally bar chart, (2) full per-year table
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import Counter

from data.loader   import get_features_and_targets
from models.base   import (build_sequences, train_val_test_split,
                            scale_features, returns_to_labels,
                            find_best_lookback, make_cache_key,
                            save_cache, load_cache)
from models.approach2_regime import train_approach2, predict_approach2
from strategy.backtest       import execute_strategy, select_winner
from signals.conviction      import compute_conviction


# ── ETF display colours ───────────────────────────────────────────────────────
ETF_COLOURS = {
    "TLT":  "#4fc3f7",
    "VNQ":  "#aed581",
    "SLV":  "#b0bec5",
    "GLD":  "#ffd54f",
    "LQD":  "#7986cb",
    "HYG":  "#ff8a65",
    "VCIT": "#a1887f",
    "CASH": "#78909c",
}
DEFAULT_COLOUR = "#90caf9"


def _etf_colour(name: str) -> str:
    return ETF_COLOURS.get(name, DEFAULT_COLOUR)


# ── Core sweep runner ─────────────────────────────────────────────────────────

def run_multiyear_sweep(
    df_raw:        pd.DataFrame,
    sweep_years:   list,
    fee_bps:       int,
    epochs:        int,
    split_option:  str,
    last_date_str: str,
    train_pct:     float,
    val_pct:       float,
) -> list:
    """
    For each year in sweep_years, train/load Approach 2 and collect:
      - next_signal
      - Z-score conviction
      - ann_return, sharpe, max_dd
      - lookback used
      - whether result came from cache

    Returns list of dicts, one per year (None-safe).
    """
    sweep_results = []
    progress_bar  = st.progress(0, text="Starting sweep...")
    status_area   = st.empty()

    for idx, yr in enumerate(sweep_years):
        pct  = int((idx / len(sweep_years)) * 100)
        progress_bar.progress(pct, text=f"Processing start year {yr}…")
        status_area.info(f"🔄 Year {yr} ({idx+1}/{len(sweep_years)})")

        row = {"start_year": yr, "signal": None, "z_score": None,
               "conviction": None, "ann_return": None, "sharpe": None,
               "max_dd": None, "lookback": None, "from_cache": False,
               "error": None}

        try:
            df = df_raw[df_raw.index.year >= yr].copy()
            if len(df) < 300:
                row["error"] = "Insufficient data (<300 rows)"
                sweep_results.append(row)
                continue

            input_features, target_etfs, tbill_rate, df, _ = get_features_and_targets(df)
            n_etfs    = len(target_etfs)
            n_classes = n_etfs

            X_raw = df[input_features].values.astype(np.float32)
            y_raw = np.clip(df[target_etfs].values.astype(np.float32), -0.5, 0.5)

            for j in range(X_raw.shape[1]):
                mask = np.isnan(X_raw[:, j])
                if mask.any():
                    X_raw[mask, j] = np.nanmean(X_raw[:, j])
            for j in range(y_raw.shape[1]):
                mask = np.isnan(y_raw[:, j])
                if mask.any():
                    y_raw[mask, j] = 0.0

            # ── Lookback ──────────────────────────────────────────────────────
            lb_key    = make_cache_key(last_date_str, yr, fee_bps, epochs,
                                       split_option, False, 0)
            lb_cached = load_cache(f"lb_{lb_key}")

            if lb_cached is not None:
                optimal_lookback = lb_cached["optimal_lookback"]
            else:
                optimal_lookback = find_best_lookback(
                    X_raw, y_raw, train_pct, val_pct, n_classes,
                    candidates=[30, 45, 60],
                )
                save_cache(f"lb_{lb_key}", {"optimal_lookback": optimal_lookback})

            lookback = optimal_lookback
            row["lookback"] = lookback

            # ── Model cache ───────────────────────────────────────────────────
            # Use a sweep-specific cache key so it doesn't clash with 3-approach runs
            cache_key   = make_cache_key(
                f"sweep2_{last_date_str}", yr, fee_bps, epochs,
                split_option, False, lookback
            )
            cached_data = load_cache(cache_key)

            if cached_data is not None:
                result   = cached_data["result"]
                proba    = cached_data["proba"]
                row["from_cache"] = True
            else:
                X_seq, y_seq = build_sequences(X_raw, y_raw, lookback)
                y_labels     = returns_to_labels(y_seq)

                (X_train, y_train_r, X_val, y_val_r,
                 X_test,  y_test_r) = train_val_test_split(X_seq, y_seq,    train_pct, val_pct)
                (_,       y_train_l,  _,    y_val_l,
                 _,       _)        = train_val_test_split(X_seq, y_labels, train_pct, val_pct)

                X_train_s, X_val_s, X_test_s, _ = scale_features(X_train, X_val, X_test)

                train_size = len(X_train)
                val_size   = len(X_val)
                test_start = lookback + train_size + val_size
                test_dates = df.index[test_start: test_start + len(X_test)]

                model_out    = train_approach2(
                    X_train_s, y_train_l, X_val_s, y_val_l,
                    X_flat_all=X_raw, feature_names=input_features,
                    lookback=lookback, train_size=train_size,
                    val_size=val_size, n_classes=n_classes, epochs=epochs,
                )
                preds, proba = predict_approach2(
                    model_out[0], X_test_s, X_raw, model_out[3], model_out[2],
                    lookback, train_size, val_size,
                )
                result = execute_strategy(
                    preds, proba, y_test_r, test_dates,
                    target_etfs, fee_bps, tbill_rate,
                )
                save_cache(cache_key, {"result": result, "proba": proba})

            # ── Conviction ────────────────────────────────────────────────────
            conviction = compute_conviction(proba[-1], target_etfs, include_cash=False)

            row.update({
                "signal":     result["next_signal"],
                "z_score":    conviction["z_score"],
                "conviction": conviction["label"],
                "ann_return": result["ann_return"],
                "sharpe":     result["sharpe"],
                "max_dd":     result["max_dd"],
            })

        except Exception as e:
            row["error"] = str(e)

        sweep_results.append(row)

    progress_bar.progress(100, text="Sweep complete ✅")
    status_area.empty()
    progress_bar.empty()

    return sweep_results


# ── Display helpers ───────────────────────────────────────────────────────────

def _vote_tally_chart(sweep_results: list) -> go.Figure:
    """Bar chart of how many start years voted for each ETF."""
    signals = [r["signal"] for r in sweep_results if r["signal"] is not None]
    counts  = Counter(signals)

    # Sort by count desc
    etfs   = sorted(counts.keys(), key=lambda e: -counts[e])
    values = [counts[e] for e in etfs]
    colors = [_etf_colour(e) for e in etfs]
    total  = sum(values)
    pcts   = [f"{v/total*100:.0f}%" for v in values]

    fig = go.Figure(go.Bar(
        x=etfs,
        y=values,
        text=[f"{v} votes<br>{p}" for v, p in zip(values, pcts)],
        textposition="outside",
        marker_color=colors,
        marker_line_color="rgba(255,255,255,0.3)",
        marker_line_width=1.5,
    ))
    fig.update_layout(
        template="plotly_dark",
        height=340,
        title=dict(text="Signal Vote Tally Across Start Years", font=dict(size=15)),
        xaxis_title="ETF",
        yaxis_title="Number of Start Years",
        yaxis=dict(dtick=1, range=[0, max(values) + 1.5]),
        margin=dict(l=40, r=30, t=50, b=40),
        showlegend=False,
        bargap=0.35,
    )
    return fig


def _conviction_scatter(sweep_results: list) -> go.Figure:
    """Scatter: start year vs Z-score, coloured by ETF signal."""
    valid = [r for r in sweep_results if r["signal"] is not None and r["z_score"] is not None]
    if not valid:
        return None

    years   = [r["start_year"] for r in valid]
    zscores = [r["z_score"]    for r in valid]
    signals = [r["signal"]     for r in valid]
    colors  = [_etf_colour(s)  for s in signals]

    fig = go.Figure()

    # One trace per unique ETF so we get a legend
    seen = set()
    for r in valid:
        etf = r["signal"]
        if etf in seen:
            continue
        seen.add(etf)
        subset = [v for v in valid if v["signal"] == etf]
        fig.add_trace(go.Scatter(
            x    = [v["start_year"] for v in subset],
            y    = [v["z_score"]    for v in subset],
            mode = "markers+text",
            name = etf,
            text = [etf for _ in subset],
            textposition = "top center",
            marker = dict(size=14, color=_etf_colour(etf),
                          line=dict(color="white", width=1.5)),
        ))

    # Neutral line
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                  annotation_text="Neutral 0σ", annotation_position="right")

    fig.update_layout(
        template="plotly_dark",
        height=320,
        title=dict(text="Conviction Z-Score by Start Year", font=dict(size=15)),
        xaxis=dict(title="Start Year", dtick=1),
        yaxis=dict(title="Z-Score (σ)"),
        margin=dict(l=40, r=30, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
    )
    return fig


def _build_full_table(sweep_results: list) -> pd.DataFrame:
    """Build the full per-year comparison DataFrame."""
    rows = []
    for r in sweep_results:
        if r["error"]:
            rows.append({
                "Start Year":  r["start_year"],
                "Signal":      "ERROR",
                "Conviction":  "—",
                "Z-Score":     "—",
                "Ann. Return": "—",
                "Sharpe":      "—",
                "Max Drawdown":"—",
                "Lookback":    "—",
                "Cache":       "—",
                "Note":        r["error"][:40],
            })
        else:
            rows.append({
                "Start Year":   r["start_year"],
                "Signal":       r["signal"]      or "—",
                "Conviction":   r["conviction"]  or "—",
                "Z-Score":      f"{r['z_score']:.2f}σ" if r["z_score"] is not None else "—",
                "Ann. Return":  f"{r['ann_return']*100:.2f}%" if r["ann_return"] is not None else "—",
                "Sharpe":       f"{r['sharpe']:.2f}"          if r["sharpe"]     is not None else "—",
                "Max Drawdown": f"{r['max_dd']*100:.2f}%"     if r["max_dd"]     is not None else "—",
                "Lookback":     f"{r['lookback']}d"           if r["lookback"]   is not None else "—",
                "Cache":        "⚡" if r["from_cache"] else "🆕",
                "Note":         "",
            })
    return pd.DataFrame(rows)


def _consensus_banner(sweep_results: list):
    """Show the consensus signal with vote count and strength."""
    signals = [r["signal"] for r in sweep_results if r["signal"] is not None]
    if not signals:
        st.warning("No valid signals collected.")
        return

    counts     = Counter(signals)
    total      = len(signals)
    top_signal = counts.most_common(1)[0][0]
    top_votes  = counts.most_common(1)[0][1]
    pct        = top_votes / total * 100

    # Strength label
    if pct >= 75:
        strength, bg = "🔥 Strong Consensus", "linear-gradient(135deg,#00b894,#00cec9)"
    elif pct >= 50:
        strength, bg = "✅ Majority Signal", "linear-gradient(135deg,#0984e3,#6c5ce7)"
    else:
        strength, bg = "⚠️ Split Signal", "linear-gradient(135deg,#636e72,#2d3436)"

    # Average Z-score for the top signal
    z_vals = [r["z_score"] for r in sweep_results
              if r["signal"] == top_signal and r["z_score"] is not None]
    avg_z  = np.mean(z_vals) if z_vals else 0.0

    st.markdown(f"""
    <div style="background:{bg}; padding:24px 28px; border-radius:16px;
                box-shadow:0 8px 20px rgba(0,0,0,0.3); margin:16px 0;">
      <div style="color:rgba(255,255,255,0.75); font-size:12px;
                  letter-spacing:3px; margin-bottom:6px; text-align:center;">
        CONSENSUS SIGNAL · APPROACH 2 · ALL START YEARS
      </div>
      <h1 style="color:white; font-size:44px; font-weight:900; text-align:center;
                 margin:4px 0; text-shadow:2px 2px 6px rgba(0,0,0,0.4);">
        🎯 {top_signal}
      </h1>
      <div style="text-align:center; color:rgba(255,255,255,0.85); font-size:16px; margin-top:8px;">
        {strength} &nbsp;·&nbsp; <b>{top_votes}/{total}</b> years agree &nbsp;·&nbsp;
        avg Z = <b>{avg_z:.2f}σ</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Runner-up signals
    if len(counts) > 1:
        others = counts.most_common()[1:]
        parts  = " &nbsp;|&nbsp; ".join(
            f'<span style="color:{_etf_colour(e)}; font-weight:600;">{e}</span> '
            f'<span style="color:#aaa;">({c} vote{"s" if c>1 else ""})</span>'
            for e, c in others
        )
        st.markdown(
            f'<div style="text-align:center; font-size:13px; color:#ccc; margin-top:4px;">'
            f'Also picked: {parts}</div>',
            unsafe_allow_html=True,
        )


# ── Main display entry point ──────────────────────────────────────────────────

def show_multiyear_results(sweep_results: list, sweep_years: list):
    """Render the full multi-year consensus UI."""

    valid = [r for r in sweep_results if r["signal"] is not None]
    failed = [r for r in sweep_results if r["error"] is not None]

    if failed:
        with st.expander(f"⚠️ {len(failed)} year(s) failed — click to see details"):
            for r in failed:
                st.warning(f"**{r['start_year']}**: {r['error']}")

    if not valid:
        st.error("No valid results from any start year.")
        return

    # ── Consensus banner ──────────────────────────────────────────────────────
    _consensus_banner(sweep_results)

    st.divider()

    # ── Charts row ────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        tally_fig = _vote_tally_chart(valid)
        st.plotly_chart(tally_fig, use_container_width=True)

    with col_right:
        scatter_fig = _conviction_scatter(valid)
        if scatter_fig:
            st.plotly_chart(scatter_fig, use_container_width=True)

    st.divider()

    # ── Full comparison table ─────────────────────────────────────────────────
    st.subheader("📋 Full Per-Year Breakdown")
    st.caption(
        "Each row = Approach 2 trained from that start year forward. "
        "⚡ = loaded from cache (no retraining). 🆕 = freshly trained."
    )

    table_df = _build_full_table(sweep_results)

    def _style_table(df: pd.DataFrame):
        def _row_style(row):
            styles = [""] * len(row)
            sig = row.get("Signal", "")
            if sig and sig not in ("—", "ERROR"):
                col = _etf_colour(sig)
                # Softer tint on signal cell
                styles[list(df.columns).index("Signal")] = (
                    f"background-color: {col}22; color: {col}; font-weight: 700;"
                )
            if row.get("Note", ""):
                styles = [
                    "color: #ff6b6b; font-style: italic;"
                ] * len(row)
            return styles

        return (
            df.style
            .apply(_row_style, axis=1)
            .set_properties(**{"text-align": "center", "font-size": "14px"})
            .set_table_styles([
                {"selector": "th", "props": [
                    ("font-size", "13px"), ("font-weight", "bold"),
                    ("text-align", "center"), ("background-color", "#1e1e2e"),
                    ("color", "#e0e0e0"),
                ]},
                {"selector": "td", "props": [("padding", "10px 14px")]},
            ])
        )

    st.dataframe(_style_table(table_df), use_container_width=True, hide_index=True)

    # ── How to read this ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("📖 How to Read These Results")
    st.markdown("""
**Why does the signal change by start year?**
Each start year defines the *training regime* the model learns from.
- **2010**: includes GFC recovery, euro crisis, QE era
- **2016+**: post-taper, Trump era, COVID shock
- **2021+**: rate-hike cycle, inflation regime

A different data window = a different view of which ETF leads in risk-off or momentum environments.

**How to use the consensus:**
| Votes for top ETF | Interpretation | Action |
|---|---|---|
| 7–8 / 8 | Very strong consensus | High confidence signal |
| 5–6 / 8 | Majority agreement | Moderate confidence |
| 3–4 / 8 | Split market | Consider waiting or splitting position |
| ≤ 2 / 8 | No consensus | Regime is unstable — treat with caution |

**Z-Score** measures how decisively the model chose the top ETF over alternatives.
A Z > 1.5σ is considered High conviction regardless of which ETF was chosen.

> 💡 **Best practice:** Weight the consensus signal most heavily when 
> *both* the vote count and the average Z-score are high simultaneously.
    """)
