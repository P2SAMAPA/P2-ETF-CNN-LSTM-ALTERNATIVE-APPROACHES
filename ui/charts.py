"""
ui/charts.py
Equity curve: winner vs SPY and AGG only.
Y-axis: % cumulative growth from 0 (not raw multiplier).
SPY/AGG returns are verified as pct returns (clipped) before compounding.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

WINNER_COLOUR     = "#00ffc8"
BENCHMARK_COLOURS = {"SPY": "#ff4b4b", "AGG": "#ffa500"}


def equity_curve_chart(
    results: dict,
    winner_name: str,
    plot_dates: pd.DatetimeIndex,
    df: pd.DataFrame,
    test_slice: slice,
    tbill_rate: float,
) -> go.Figure:
    from strategy.backtest import compute_benchmark_metrics

    fig = go.Figure()

    # ── Winner strategy ───────────────────────────────────────────────────────
    winner_res = results.get(winner_name)
    if winner_res is not None:
        cum = winner_res["cum_returns"]
        # Sanity: if cum[-1] > 10x (1000%), something is wrong — skip render
        if cum[-1] < 10:
            n   = min(len(cum), len(plot_dates))
            pct = (cum[:n] - 1) * 100
            fig.add_trace(go.Scatter(
                x=plot_dates[:n], y=pct,
                mode="lines",
                name=f"{winner_name} ★",
                line=dict(color=WINNER_COLOUR, width=2.5),
                fill="tozeroy",
                fillcolor="rgba(0,255,200,0.07)",
            ))

    # ── SPY ───────────────────────────────────────────────────────────────────
    spy_ann = None
    if "SPY_Ret" in df.columns:
        raw = df["SPY_Ret"].iloc[test_slice].values.copy().astype(float)
        raw = raw[~np.isnan(raw)]
        # If values look like prices (median > 1), convert to returns
        if len(raw) > 0 and np.median(np.abs(raw)) > 1:
            raw = np.diff(raw) / raw[:-1]
        raw = np.clip(raw, -0.5, 0.5)
        if len(raw) > 0:
            n     = min(len(raw), len(plot_dates))
            spy_m = compute_benchmark_metrics(raw[:n], tbill_rate)
            spy_ann = spy_m.get("ann_return")
            fig.add_trace(go.Scatter(
                x=plot_dates[:n],
                y=(spy_m["cum_returns"] - 1) * 100,
                mode="lines", name="SPY",
                line=dict(color=BENCHMARK_COLOURS["SPY"], width=1.5, dash="dot"),
            ))

    # ── AGG ───────────────────────────────────────────────────────────────────
    if "AGG_Ret" in df.columns:
        raw = df["AGG_Ret"].iloc[test_slice].values.copy().astype(float)
        raw = raw[~np.isnan(raw)]
        if len(raw) > 0 and np.median(np.abs(raw)) > 1:
            raw = np.diff(raw) / raw[:-1]
        raw = np.clip(raw, -0.5, 0.5)
        if len(raw) > 0:
            n     = min(len(raw), len(plot_dates))
            agg_m = compute_benchmark_metrics(raw[:n], tbill_rate)
            fig.add_trace(go.Scatter(
                x=plot_dates[:n],
                y=(agg_m["cum_returns"] - 1) * 100,
                mode="lines", name="AGG",
                line=dict(color=BENCHMARK_COLOURS["AGG"], width=1.5, dash="dot"),
            ))

    fig.update_layout(
        template="plotly_dark",
        height=420,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(size=11)),
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        yaxis=dict(ticksuffix="%"),
        margin=dict(l=50, r=30, t=20, b=50),
    )
    return fig, spy_ann
