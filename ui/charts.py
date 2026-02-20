"""
ui/charts.py
All Plotly chart builders for the Streamlit UI.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


APPROACH_COLOURS = {
    "Approach 1": "#00ffc8",
    "Approach 2": "#7c6aff",
    "Approach 3": "#ff6b6b",
}
BENCHMARK_COLOURS = {
    "SPY": "#ff4b4b",
    "AGG": "#ffa500",
}


def equity_curve_chart(
    results: dict,
    winner_name: str,
    plot_dates: pd.DatetimeIndex,
    df: pd.DataFrame,
    test_slice: slice,
    tbill_rate: float,
) -> go.Figure:
    """
    Equity curve chart showing all three approaches + SPY + AGG benchmarks.

    Args:
        results     : {approach_name: result_dict}
        winner_name : highlighted approach
        plot_dates  : DatetimeIndex for x-axis
        df          : full DataFrame (for benchmark columns)
        test_slice  : slice object to extract test-period benchmark returns
        tbill_rate  : for benchmark metric calculation
    """
    from strategy.backtest import compute_benchmark_metrics

    fig = go.Figure()

    # ── Strategy lines ────────────────────────────────────────────────────────
    for name, res in results.items():
        if res is None:
            continue
        colour = APPROACH_COLOURS.get(name, "#aaaaaa")
        width  = 3 if name == winner_name else 1.5
        dash   = "solid" if name == winner_name else "dot"

        n = min(len(res["cum_returns"]), len(plot_dates))

        fig.add_trace(go.Scatter(
            x=plot_dates[:n],
            y=res["cum_returns"][:n],
            mode="lines",
            name=f"{name} {'★' if name == winner_name else ''}",
            line=dict(color=colour, width=width, dash=dash),
            fill="tozeroy" if name == winner_name else None,
            fillcolor=f"rgba({_hex_to_rgb(colour)},0.07)" if name == winner_name else None,
        ))

    # ── Benchmark: SPY ────────────────────────────────────────────────────────
    if "SPY_Ret" in df.columns:
        spy_rets = df["SPY_Ret"].iloc[test_slice].values
        n        = min(len(spy_rets), len(plot_dates))
        spy_m    = compute_benchmark_metrics(spy_rets[:n], tbill_rate)
        fig.add_trace(go.Scatter(
            x=plot_dates[:n],
            y=spy_m["cum_returns"],
            mode="lines",
            name="SPY (Equity BM)",
            line=dict(color=BENCHMARK_COLOURS["SPY"], width=1.5, dash="dot"),
        ))

    # ── Benchmark: AGG ────────────────────────────────────────────────────────
    if "AGG_Ret" in df.columns:
        agg_rets = df["AGG_Ret"].iloc[test_slice].values
        n        = min(len(agg_rets), len(plot_dates))
        agg_m    = compute_benchmark_metrics(agg_rets[:n], tbill_rate)
        fig.add_trace(go.Scatter(
            x=plot_dates[:n],
            y=agg_m["cum_returns"],
            mode="lines",
            name="AGG (Bond BM)",
            line=dict(color=BENCHMARK_COLOURS["AGG"], width=1.5, dash="dot"),
        ))

    fig.update_layout(
        template="plotly_dark",
        height=460,
        hovermode="x unified",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(size=11)),
        xaxis_title="Date",
        yaxis_title="Cumulative Return (×)",
        margin=dict(l=50, r=30, t=20, b=50),
    )
    return fig


def comparison_bar_chart(results: dict, winner_name: str) -> go.Figure:
    """
    Horizontal bar chart comparing annualised returns across all three approaches.
    """
    names   = []
    returns = []
    colours = []

    for name, res in results.items():
        if res is None:
            continue
        names.append(name)
        returns.append(res["ann_return"] * 100)
        colours.append(APPROACH_COLOURS.get(name, "#aaaaaa"))

    fig = go.Figure(go.Bar(
        x=returns,
        y=names,
        orientation="h",
        marker_color=colours,
        text=[f"{r:.1f}%" for r in returns],
        textposition="auto",
    ))

    fig.update_layout(
        template="plotly_dark",
        height=200,
        xaxis_title="Annualised Return (%)",
        margin=dict(l=100, r=30, t=10, b=40),
        showlegend=False,
    )
    return fig


# ── Helper ────────────────────────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> str:
    """Convert #rrggbb to 'r,g,b' string for rgba()."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"
