"""
strategy/backtest.py
Strategy execution, performance metrics, and benchmark calculations.
Supports CASH as a class (earns T-bill rate when selected).
"""

import numpy as np
import pandas as pd
from datetime import datetime


# ── Strategy execution ────────────────────────────────────────────────────────

def execute_strategy(
    preds: np.ndarray,
    proba: np.ndarray,
    y_raw_test: np.ndarray,
    test_dates: pd.DatetimeIndex,
    target_etfs: list,
    fee_bps: int,
    tbill_rate: float,
    include_cash: bool = True,
) -> dict:
    """
    Execute strategy from model predictions.

    Args:
        preds       : [n] integer class predictions
        proba       : [n, n_classes] softmax probabilities
        y_raw_test  : [n, n_etfs] actual next-day ETF returns
        test_dates  : DatetimeIndex aligned with y_raw_test
        target_etfs : list of ETF return column names e.g. ["TLT_Ret", ...]
        fee_bps     : transaction fee in basis points
        tbill_rate  : annualised 3m T-bill rate (e.g. 0.045)
        include_cash: whether CASH is a valid class (index = n_etfs)

    Returns:
        dict with keys:
            strat_rets, cum_returns, ann_return, sharpe,
            hit_ratio, max_dd, max_daily_dd, cum_max,
            audit_trail, next_signal, next_proba
    """
    n_etfs      = len(target_etfs)
    daily_tbill = tbill_rate / 252
    today       = datetime.now().date()

    strat_rets  = []
    audit_trail = []

    for i, cls in enumerate(preds):
        if include_cash and cls == n_etfs:
            signal_etf   = "CASH"
            realized_ret = daily_tbill
        else:
            cls          = min(cls, n_etfs - 1)
            signal_etf   = target_etfs[cls].replace("_Ret", "")
            realized_ret = float(y_raw_test[i][cls])

        net_ret = realized_ret - (fee_bps / 10000)
        strat_rets.append(net_ret)

        trade_date = test_dates[i]
        if trade_date.date() < today:
            audit_trail.append({
                "Date":       trade_date.strftime("%Y-%m-%d"),
                "Signal":     signal_etf,
                "Realized":   realized_ret,
                "Net_Return": net_ret,
            })

    strat_rets = np.array(strat_rets, dtype=np.float64)

    # Next signal (last prediction)
    last_cls   = int(preds[-1])
    next_proba = proba[-1]

    if include_cash and last_cls == n_etfs:
        next_signal = "CASH"
    else:
        last_cls    = min(last_cls, n_etfs - 1)
        next_signal = target_etfs[last_cls].replace("_Ret", "")

    metrics = _compute_metrics(strat_rets, tbill_rate)

    return {
        **metrics,
        "strat_rets":  strat_rets,
        "audit_trail": audit_trail,
        "next_signal": next_signal,
        "next_proba":  next_proba,
    }


# ── Performance metrics ───────────────────────────────────────────────────────

def _compute_metrics(strat_rets: np.ndarray, tbill_rate: float) -> dict:
    if len(strat_rets) == 0:
        return {}

    cum_returns = np.cumprod(1 + strat_rets)
    n           = len(strat_rets)
    ann_return  = float(cum_returns[-1] ** (252 / n) - 1)

    excess      = strat_rets - tbill_rate / 252
    sharpe      = float(np.mean(excess) / (np.std(strat_rets) + 1e-9) * np.sqrt(252))

    recent      = strat_rets[-15:]
    hit_ratio   = float(np.mean(recent > 0))

    cum_max     = np.maximum.accumulate(cum_returns)
    drawdown    = (cum_returns - cum_max) / cum_max
    max_dd      = float(np.min(drawdown))
    max_daily   = float(np.min(strat_rets))

    return {
        "cum_returns": cum_returns,
        "ann_return":  ann_return,
        "sharpe":      sharpe,
        "hit_ratio":   hit_ratio,
        "max_dd":      max_dd,
        "max_daily_dd":max_daily,
        "cum_max":     cum_max,
    }


def compute_benchmark_metrics(returns: np.ndarray, tbill_rate: float) -> dict:
    """Compute metrics for a benchmark return series."""
    return _compute_metrics(returns, tbill_rate)


# ── Winner selection ──────────────────────────────────────────────────────────

def select_winner(results: dict) -> str:
    """
    Given a dict of {approach_name: result_dict}, return the approach name
    with the highest annualised return (raw, not risk-adjusted).

    Args:
        results : {"Approach 1": {...}, "Approach 2": {...}, "Approach 3": {...}}

    Returns:
        winner_name : str
    """
    best_name   = None
    best_return = -np.inf

    for name, res in results.items():
        if res is None:
            continue
        ret = res.get("ann_return", -np.inf)
        if ret > best_return:
            best_return = ret
            best_name   = name

    return best_name


# ── Comparison table ──────────────────────────────────────────────────────────

def build_comparison_table(results: dict, winner_name: str) -> pd.DataFrame:
    """
    Build a summary DataFrame comparing all three approaches.

    Args:
        results     : {name: result_dict}
        winner_name : name of the winner

    Returns:
        pd.DataFrame with one row per approach
    """
    rows = []
    for name, res in results.items():
        if res is None:
            rows.append({
                "Approach":       name,
                "Ann. Return":    "N/A",
                "Sharpe":         "N/A",
                "Hit Ratio (15d)":"N/A",
                "Max Drawdown":   "N/A",
                "Winner":         "",
            })
            continue

        rows.append({
            "Approach":        name,
            "Ann. Return":     f"{res['ann_return']*100:.2f}%",
            "Sharpe":          f"{res['sharpe']:.2f}",
            "Hit Ratio (15d)": f"{res['hit_ratio']*100:.0f}%",
            "Max Drawdown":    f"{res['max_dd']*100:.2f}%",
            "Winner":          "⭐ WINNER" if name == winner_name else "",
        })

    return pd.DataFrame(rows)
