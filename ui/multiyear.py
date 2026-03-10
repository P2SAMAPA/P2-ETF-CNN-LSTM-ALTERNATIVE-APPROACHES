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
- [NEW] force_retrain=True bypasses all cache and retrains every year fresh
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
    force_retrain: bool = False,
) -> list:
    """
    For each year in sweep_years, train/load Approach 2 and collect:
      - next_signal
      - Z-score conviction
      - ann_return, sharpe, max_dd
      - lookback used
      - whether result came from cache

    Parameters
    ----------
    force_retrain : bool
        When True, completely ignores all existing cache entries and retrains
        every year from scratch. Fresh results are still saved to cache so
        subsequent normal runs can use them.
        When False (default), loads from cache wherever available.

    Returns list of dicts, one per year (None-safe).
    """
    sweep_results = []
    progress_bar  = st.progress(0, text="Starting sweep...")
    status_area   = st.empty()

    if force_retrain:
        st.caption("🔄 Force retrain mode — ignoring all cached results, training from scratch.")

    for idx, yr in enumerate(sweep_years):
        pct  = int((idx / len(sweep_years)) * 100)
        progress_bar.progress(pct, text=f"Processing start year {yr}…")
        status_area.info(f"{'🔄 Retraining' if force_retrain else '🔍 Loading'} year {yr} ({idx+1}/{len(sweep_years)})")

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
            # force_retrain also clears lookback cache so we get a fresh
            # auto-selection rather than a potentially stale window choice
            lb_key    = make_cache_key(last_date_str, yr, fee_bps, epochs,
                                       split_option, False, 0)
            lb_cached = None if force_retrain else load_cache(f"lb_{lb_key}")

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
            # Use a sweep-specific cache key so it doesn't clash with 3-approach runs.
            # force_retrain=True → skip load_cache entirely; still save at the end
            # so subsequent normal runs benefit from the fresh result.
            cache_key   = make_cache_key(
                f"sweep2_{last_date_str}", yr, fee_bps, epochs,
                split_option, False, lookback
            )
            cached_data = None if force_retrain else load_cache(cache_key)

            if cached_data is not None:
                result          = cached_data["result"]
                proba           = cached_data["proba"]
                row["from_cache"] = True
                row["run_date"]   = cached_data.get("run_date", last_date_str)
            else:
                # ── Full retrain ──────────────────────────────────────────────
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

                from datetime import datetime as _dt2, timezone as _tz2, timedelta as _td2
                _run_date = (_dt2.now(_tz2.utc) - _td2(hours=5)).strftime("%Y-%m-%d")

                # Always save to cache (even after force retrain) so the next
                # normal run can use these fresh results without retraining again
                save_cache(cache_key, {
                    "result":   result,
                    "proba":    proba,
                    "run_date": _run_date,
                })

                row["from_cache"] = False   # freshly trained this run

            # ── Conviction ────────────────────────────────────────────────────
            conviction = compute_conviction(proba[-1], target_etfs, include_cash=False)

            from datetime import datetime as _dt, timezone as _tz, timedelta as _td
            row.update({
                "run_date":   (_dt.now(_tz.utc) - _td(hours=5)).strftime("%Y-%m-%d"),
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


# ── Weighted scoring ──────────────────────────────────────────────────────────
#
# Per-year composite score:
#   40% Ann. Return  (higher = better)
#   20% Z-Score      (higher = better)
#   20% Sharpe       (higher = better, positive preferred)
#   20% Max Drawdown (lower magnitude = better, i.e. score on -max_dd)
#
# Each metric is min-max normalised across all valid years before weighting,
# so no single metric dominates due to scale differences.

W_RETURN = 0.40
W_ZSCORE = 0.20
W_SHARPE = 0.20
W_DD     = 0.20


def _compute_weighted_scores(valid: list) -> list:
    """
    Returns a copy of valid rows, each augmented with:
      - 'weighted_score'  : float in [0, 1]
      - 'score_breakdown' : dict of normalised component scores
    """
    def _minmax(vals):
        arr = np.array(vals, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.ones(len(arr)) * 0.5
        return (arr - mn) / (mx - mn)

    returns = [r["ann_return"] if r["ann_return"] is not None else 0.0 for r in valid]
    zscores = [r["z_score"]    if r["z_score"]    is not None else 0.0 for r in valid]
    sharpes = [r["sharpe"]     if r["sharpe"]     is not None else 0.0 for r in valid]
    dds     = [-(r["max_dd"]   if r["max_dd"]     is not None else -1.0) for r in valid]

    n_ret = _minmax(returns)
    n_z   = _minmax(zscores)
    n_sh  = _minmax(sharpes)
    n_dd  = _minmax(dds)

    scored = []
    for i, r in enumerate(valid):
        composite = (W_RETURN * n_ret[i] +
                     W_ZSCORE * n_z[i]   +
                     W_SHARPE * n_sh[i]  +
                     W_DD     * n_dd[i])
        scored.append({
            **r,
            "weighted_score": float(composite),
            "score_breakdown": {
                "Return (40%)":   float(n_ret[i]),
                "Z-Score (20%)":  float(n_z[i]),
                "Sharpe (20%)":   float(n_sh[i]),
                "Max DD (20%)":   float(n_dd[i]),
            },
        })
    return scored


# ── Display helpers ───────────────────────────────────────────────────────────

def _vote_tally_chart(scored: list) -> go.Figure:
    """
    Bar chart: weighted score accumulated per ETF across all start years.
    """
    from collections import defaultdict
    etf_scores = defaultdict(float)
    etf_counts = Counter()

    for r in scored:
        etf = r["signal"]
        etf_scores[etf] += r["weighted_score"]
        etf_counts[etf] += 1

    etfs        = sorted(etf_scores.keys(), key=lambda e: -etf_scores[e])
    values      = [etf_scores[e] for e in etfs]
    counts      = [etf_counts[e] for e in etfs]
    colors      = [_etf_colour(e) for e in etfs]
    total_score = sum(values)
    pcts        = [f"{v/total_score*100:.0f}%" for v in values]

    fig = go.Figure(go.Bar(
        x=etfs,
        y=values,
        text=[f"{c} yr{'s' if c>1 else ''} · {p}<br>score {v:.2f}"
              for c, p, v in zip(counts, pcts, values)],
        textposition="outside",
        marker_color=colors,
        marker_line_color="rgba(255,255,255,0.3)",
        marker_line_width=1.5,
    ))
    fig.update_layout(
        template="plotly_dark",
        height=340,
        title=dict(
            text="Weighted Score per ETF  (40% Return · 20% Z · 20% Sharpe · 20% -MaxDD)",
            font=dict(size=13),
        ),
        xaxis_title="ETF",
        yaxis_title="Cumulative Weighted Score",
        yaxis=dict(range=[0, max(values) * 1.25]),
        margin=dict(l=40, r=30, t=55, b=40),
        showlegend=False,
        bargap=0.35,
    )
    return fig


def _conviction_scatter(sweep_results: list) -> go.Figure:
    """Scatter: start year vs Z-score, coloured by ETF signal."""
    valid = [r for r in sweep_results if r["signal"] is not None and r["z_score"] is not None]
    if not valid:
        return None

    fig = go.Figure()
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


def _build_full_table(scored: list) -> pd.DataFrame:
    """Build the full per-year comparison DataFrame, including weighted score."""
    rows = []
    for r in scored:
        if r.get("error"):
            rows.append({
                "Start Year":   r["start_year"],
                "Signal":       "ERROR",
                "Wtd Score":    "—",
                "Conviction":   "—",
                "Z-Score":      "—",
                "Ann. Return":  "—",
                "Sharpe":       "—",
                "Max Drawdown": "—",
                "Lookback":     "—",
                "Cache":        "—",
                "Note":         r["error"][:40],
            })
        else:
            ws = r.get("weighted_score")
            rows.append({
                "Start Year":   r["start_year"],
                "Signal":       r["signal"]      or "—",
                "Wtd Score":    f"{ws:.3f}"       if ws   is not None else "—",
                "Conviction":   r["conviction"]  or "—",
                "Z-Score":      f"{r['z_score']:.2f}σ"       if r["z_score"]    is not None else "—",
                "Ann. Return":  f"{r['ann_return']*100:.2f}%" if r["ann_return"] is not None else "—",
                "Sharpe":       f"{r['sharpe']:.2f}"          if r["sharpe"]     is not None else "—",
                "Max Drawdown": f"{r['max_dd']*100:.2f}%"     if r["max_dd"]     is not None else "—",
                "Lookback":     f"{r['lookback']}d"           if r["lookback"]   is not None else "—",
                "Cache":        "⚡" if r["from_cache"] else "🆕",
                "Note":         "",
            })
    return pd.DataFrame(rows)


def _consensus_banner(scored: list, run_date_str: str = ""):
    """
    Show the consensus signal selected by highest cumulative weighted score.
    """
    if not scored:
        st.warning("No valid signals collected.")
        return

    from collections import defaultdict
    etf_total_score = defaultdict(float)
    etf_counts      = Counter()
    for r in scored:
        etf = r["signal"]
        etf_total_score[etf] += r["weighted_score"]
        etf_counts[etf]      += 1

    top_signal  = max(etf_total_score, key=lambda e: etf_total_score[e])
    top_score   = etf_total_score[top_signal]
    total_score = sum(etf_total_score.values())
    score_pct   = top_score / total_score * 100
    top_votes   = etf_counts[top_signal]
    total_years = len(scored)
    avg_ws      = top_score / top_votes

    if score_pct >= 60:
        strength, bg = "🔥 Strong Consensus", "linear-gradient(135deg,#00b894,#00cec9)"
    elif score_pct >= 40:
        strength, bg = "✅ Majority Signal",   "linear-gradient(135deg,#0984e3,#6c5ce7)"
    else:
        strength, bg = "⚠️ Split Signal",      "linear-gradient(135deg,#636e72,#2d3436)"

    winners = [r for r in scored if r["signal"] == top_signal]
    avg_ret = np.mean([r["ann_return"] for r in winners if r["ann_return"] is not None]) * 100
    avg_z   = np.mean([r["z_score"]   for r in winners if r["z_score"]   is not None])
    avg_sh  = np.mean([r["sharpe"]    for r in winners if r["sharpe"]    is not None])
    avg_dd  = np.mean([r["max_dd"]    for r in winners if r["max_dd"]    is not None]) * 100

    st.markdown(f"""
    <div style="background:{bg}; padding:24px 28px; border-radius:16px;
                box-shadow:0 8px 20px rgba(0,0,0,0.3); margin:16px 0;">
      <div style="color:rgba(255,255,255,0.75); font-size:12px;
                  letter-spacing:3px; margin-bottom:6px; text-align:center;">
        WEIGHTED CONSENSUS · APPROACH 2 · ALL START YEARS · {run_date_str}
      </div>
      <h1 style="color:white; font-size:44px; font-weight:900; text-align:center;
                 margin:4px 0; text-shadow:2px 2px 6px rgba(0,0,0,0.4);">
        🎯 {top_signal}
      </h1>
      <div style="text-align:center; color:rgba(255,255,255,0.85); font-size:15px; margin-top:8px;">
        {strength} &nbsp;·&nbsp;
        Score share <b>{score_pct:.0f}%</b> &nbsp;·&nbsp;
        <b>{top_votes}/{total_years}</b> years &nbsp;·&nbsp;
        avg score <b>{avg_ws:.2f}</b>
      </div>
      <div style="display:flex; justify-content:center; gap:28px; margin-top:14px;
                  flex-wrap:wrap; font-size:13px; color:rgba(255,255,255,0.7);">
        <span>📈 Avg Return <b style="color:white">{avg_ret:+.1f}%</b></span>
        <span>⚡ Avg Z <b style="color:white">{avg_z:.2f}σ</b></span>
        <span>📊 Avg Sharpe <b style="color:white">{avg_sh:.2f}</b></span>
        <span>📉 Avg MaxDD <b style="color:white">{avg_dd:.1f}%</b></span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    others = sorted(
        [(e, s) for e, s in etf_total_score.items() if e != top_signal],
        key=lambda x: -x[1]
    )
    if others:
        parts = " &nbsp;|&nbsp; ".join(
            f'<span style="color:{_etf_colour(e)}; font-weight:600;">{e}</span> '
            f'<span style="color:#aaa;">(score {s:.2f} · {etf_counts[e]} yr{"s" if etf_counts[e]>1 else ""})</span>'
            for e, s in others
        )
        st.markdown(
            f'<div style="text-align:center; font-size:13px; color:#ccc; margin-top:6px;">'
            f'Also ranked: {parts}</div>',
            unsafe_allow_html=True,
        )


# ── Main display entry point ──────────────────────────────────────────────────

def show_multiyear_results(sweep_results: list, sweep_years: list):
    """Render the full multi-year consensus UI."""

    valid  = [r for r in sweep_results if r["signal"] is not None]
    failed = [r for r in sweep_results if r["error"]  is not None]

    if failed:
        with st.expander(f"⚠️ {len(failed)} year(s) failed — click to see details"):
            for r in failed:
                st.warning(f"**{r['start_year']}**: {r['error']}")

    if not valid:
        st.error("No valid results from any start year.")
        return

    scored       = _compute_weighted_scores(valid)
    scored_by_yr = {r["start_year"]: r for r in scored}
    full_scored  = [scored_by_yr.get(r["start_year"], r) for r in sweep_results]

    run_dates    = [r.get("run_date", "") for r in scored if r.get("run_date")]
    run_date_str = max(run_dates) if run_dates else ""
    _consensus_banner(scored, run_date_str=run_date_str)

    st.divider()

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.plotly_chart(_vote_tally_chart(scored), use_container_width=True)
    with col_right:
        scatter_fig = _conviction_scatter(scored)
        if scatter_fig:
            st.plotly_chart(scatter_fig, use_container_width=True)

    st.divider()

    st.subheader("📋 Full Per-Year Breakdown")
    st.caption(
        "**Wtd Score** = 40% Ann. Return + 20% Z-Score + 20% Sharpe + 20% (−Max DD), "
        "each metric min-max normalised across all years.  "
        "⚡ = loaded from cache (no retraining). 🆕 = freshly trained."
    )

    table_df = _build_full_table(full_scored)

    def _style_table(df: pd.DataFrame):
        def _row_style(row):
            styles = [""] * len(row)
            sig = row.get("Signal", "")
            if sig and sig not in ("—", "ERROR"):
                col = _etf_colour(sig)
                styles[list(df.columns).index("Signal")] = (
                    f"background-color: {col}22; color: {col}; font-weight: 700;"
                )
                if "Wtd Score" in df.columns:
                    styles[list(df.columns).index("Wtd Score")] = (
                        f"color: {col}; font-weight: 700;"
                    )
            if row.get("Note", ""):
                styles = ["color: #ff6b6b; font-style: italic;"] * len(row)
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

    st.divider()
    st.subheader("📖 How to Read These Results")
    st.markdown("""
**Why does the signal change by start year?**
Each start year defines the *training regime* the model learns from.
- **2010**: includes GFC recovery, euro crisis, QE era
- **2016+**: post-taper, Trump era, COVID shock
- **2021+**: rate-hike cycle, inflation regime

A different data window = a different view of which ETF leads in risk-off or momentum environments.

**How the weighted consensus works:**
Each year's result gets a composite score (0–1) based on four normalised metrics:

| Metric | Weight | Logic |
|---|---|---|
| Ann. Return | **40%** | Higher is better |
| Z-Score | **20%** | Higher = more decisive model |
| Sharpe Ratio | **20%** | Higher and positive is better |
| Max Drawdown | **20%** | Lower magnitude is better |

The ETF with the highest **total cumulative score** across all start years wins.

**Score share interpretation:**
| Score share | Interpretation |
|---|---|
| ≥ 60% | Strong consensus — high confidence |
| 40–60% | Majority signal — moderate confidence |
| < 40% | Split signal — regime unstable, consider caution |

**Force Retrain vs Run Consensus Sweep:**
| Button | Behaviour |
|---|---|
| 🚀 Run Consensus Sweep | Uses cached results where available; only retrains years with no cache hit |
| 🔄 Force Retrain All | Ignores all cache; retrains every year from scratch; saves fresh results to cache |

> 💡 **Best practice:** Use **Run Consensus Sweep** daily (fast, cache-aware). Use **Force Retrain All** when you change epochs, fee, or split settings and want results that reflect the new configuration.
    """)
