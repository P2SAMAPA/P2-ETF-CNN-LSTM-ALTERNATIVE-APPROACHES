"""
ui/charts.py
Plotly chart builders.
Equity curve: winner + SPY + AGG only. Y-axis as % growth (not raw multiplier).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

WINNER_COLOUR    = "#00ffc8"
BENCHMARK_COLOURS = {"SPY": "#ff4b4b", "AGG": "#ffa500"}


def equity_curve_chart(
    results: dict,
    winner_name: str,
    plot_dates: pd.DatetimeIndex,
    df: pd.DataFrame,
    test_slice: slice,
    tbill_rate: float,
) -> go.Figure:
    """
    Equity curve: winner strategy vs SPY and AGG.
    Y-axis shows % growth (cum_return - 1) * 100 for readability.
    """
    from strategy.backtest import compute_benchmark_metrics

    fig = go.Figure()

    # ── Winner strategy ───────────────────────────────────────────────────────
    winner_res = results.get(winner_name)
    if winner_res is not None:
        cum = winner_res["cum_returns"]
        n   = min(len(cum), len(plot_dates))
        fig.add_trace(go.Scatter(
            x=plot_dates[:n],
            y=(cum[:n] - 1) * 100,
            mode="lines",
            name=f"{winner_name} ★",
            line=dict(color=WINNER_COLOUR, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(0,255,200,0.07)",
        ))

    # ── SPY benchmark ─────────────────────────────────────────────────────────
    if "SPY_Ret" in df.columns:
        spy_rets = df["SPY_Ret"].iloc[test_slice].values.copy()
        spy_rets = np.clip(spy_rets, -0.5, 0.5)   # sanity clip
        spy_rets = spy_rets[~np.isnan(spy_rets)]
        n        = min(len(spy_rets), len(plot_dates))
        spy_m    = compute_benchmark_metrics(spy_rets[:n], tbill_rate)
        fig.add_trace(go.Scatter(
            x=plot_dates[:n],
            y=(spy_m["cum_returns"] - 1) * 100,
            mode="lines",
            name="SPY",
            line=dict(color=BENCHMARK_COLOURS["SPY"], width=1.5, dash="dot"),
        ))

    # ── AGG benchmark ─────────────────────────────────────────────────────────
    if "AGG_Ret" in df.columns:
        agg_rets = df["AGG_Ret"].iloc[test_slice].values.copy()
        agg_rets = np.clip(agg_rets, -0.5, 0.5)
        agg_rets = agg_rets[~np.isnan(agg_rets)]
        n        = min(len(agg_rets), len(plot_dates))
        agg_m    = compute_benchmark_metrics(agg_rets[:n], tbill_rate)
        fig.add_trace(go.Scatter(
            x=plot_dates[:n],
            y=(agg_m["cum_returns"] - 1) * 100,
            mode="lines",
            name="AGG",
            line=dict(color=BENCHMARK_COLOURS["AGG"], width=1.5, dash="dot"),
        ))

    fig.update_layout(
        template="plotly_dark",
        height=420,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(size=11)),
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        margin=dict(l=50, r=30, t=20, b=50),
        yaxis=dict(ticksuffix="%"),
    )
    return fig


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"
