"""
strategy/backtest.py
Strategy execution, performance metrics, and benchmark calculations.

CASH logic (drawdown risk overlay — not a model class):
  ENTER : 2-day cumulative return <= -15%
  EXIT  : model conviction Z-score >= 1.0 (model decisively picks an ETF again)
"""

import numpy as np
import pandas as pd
from datetime import datetime

CASH_DRAWDOWN_TRIGGER = -0.15   # 2-day cumulative return threshold
CASH_EXIT_Z           =  1.0   # Z-score required to exit CASH


def _zscore(proba: np.ndarray) -> float:
    std = np.std(proba)
    return float((np.max(proba) - np.mean(proba)) / std) if std > 1e-9 else 0.0


def execute_strategy(
    preds: np.ndarray,
    proba: np.ndarray,
    y_raw_test: np.ndarray,
    test_dates: pd.DatetimeIndex,
    target_etfs: list,
    fee_bps: int,
    tbill_rate: float,
    include_cash: bool = True,   # kept for API compat but CASH is now overlay-only
) -> dict:
    n_etfs      = len(target_etfs)
    daily_tbill = tbill_rate / 252
    fee         = fee_bps / 10000
    today       = datetime.now().date()

    strat_rets  = []
    audit_trail = []
    date_index  = []

    in_cash       = False
    recent_rets   = []   # rolling 2-day window

    for i, cls in enumerate(preds):
        cls      = min(int(cls), n_etfs - 1)
        etf_name = target_etfs[cls].replace("_Ret", "")
        etf_ret  = float(np.clip(y_raw_test[i][cls], -0.5, 0.5))
        z        = _zscore(proba[i])

        # ── 2-day drawdown check ──────────────────────────────────────────────
        recent_rets.append(etf_ret)
        if len(recent_rets) > 2:
            recent_rets.pop(0)
        two_day = ((1 + recent_rets[0]) * (1 + recent_rets[-1]) - 1
                   if len(recent_rets) >= 2 else 0.0)

        if two_day <= CASH_DRAWDOWN_TRIGGER:
            in_cash = True
        if in_cash and z >= CASH_EXIT_Z:
            in_cash = False

        # ── Execute ───────────────────────────────────────────────────────────
        if in_cash:
            signal_etf   = "CASH"
            realized_ret = daily_tbill
        else:
            signal_etf   = etf_name
            realized_ret = etf_ret

        net_ret = realized_ret - fee
        strat_rets.append(net_ret)
        date_index.append(test_dates[i])

        if test_dates[i].date() < today:
            audit_trail.append({
                "Date":       test_dates[i].strftime("%Y-%m-%d"),
                "Signal":     signal_etf,
                "Net_Return": net_ret,
                "Z_Score":    round(z, 2),
            })

    strat_rets = np.array(strat_rets, dtype=np.float64)

    # Next signal
    last_cls  = min(int(preds[-1]), n_etfs - 1)
    last_z    = _zscore(proba[-1])
    last_ret  = float(np.clip(y_raw_test[-1][last_cls], -0.5, 0.5))
    prev_ret  = float(np.clip(y_raw_test[-2][last_cls], -0.5, 0.5)) if len(y_raw_test) > 1 else 0.0
    last_2d   = (1 + prev_ret) * (1 + last_ret) - 1
    next_cash = last_2d <= CASH_DRAWDOWN_TRIGGER and last_z < CASH_EXIT_Z
    next_signal = "CASH" if next_cash else target_etfs[last_cls].replace("_Ret", "")

    metrics = _compute_metrics(strat_rets, tbill_rate, date_index)

    return {
        **metrics,
        "strat_rets":  strat_rets,
        "audit_trail": audit_trail,
        "next_signal": next_signal,
        "next_proba":  proba[-1],
    }


def _compute_metrics(strat_rets: np.ndarray, tbill_rate: float,
                     date_index: list = None) -> dict:
    if len(strat_rets) == 0:
        return {}

    cum_returns = np.cumprod(1 + strat_rets)
    n           = len(strat_rets)
    ann_return  = float(cum_returns[-1] ** (252 / n) - 1)

    excess = strat_rets - tbill_rate / 252
    sharpe = float(np.mean(excess) / (np.std(strat_rets) + 1e-9) * np.sqrt(252))

    hit_ratio = float(np.mean(strat_rets[-15:] > 0))

    cum_max  = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - cum_max) / cum_max
    max_dd   = float(np.min(drawdown))

    worst_idx  = int(np.argmin(strat_rets))
    max_daily  = float(strat_rets[worst_idx])
    worst_date = (date_index[worst_idx].strftime("%Y-%m-%d")
                  if date_index and worst_idx < len(date_index) else "N/A")

    return {
        "cum_returns":    cum_returns,
        "ann_return":     ann_return,
        "sharpe":         sharpe,
        "hit_ratio":      hit_ratio,
        "max_dd":         max_dd,
        "max_daily_dd":   max_daily,
        "max_daily_date": worst_date,
        "cum_max":        cum_max,
    }


def compute_benchmark_metrics(returns: np.ndarray, tbill_rate: float) -> dict:
    return _compute_metrics(np.array(returns, dtype=np.float64), tbill_rate)


def select_winner(results: dict) -> str:
    best_name, best_ret = None, -np.inf
    for name, res in results.items():
        if res is None:
            continue
        r = res.get("ann_return", -np.inf)
        if r > best_ret:
            best_ret, best_name = r, name
    return best_name


def build_comparison_table(results: dict, winner_name: str) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        if res is None:
            rows.append({"Approach": name, "Ann. Return": "N/A",
                         "Sharpe": "N/A", "Hit Ratio (15d)": "N/A",
                         "Max Drawdown": "N/A", "Winner": ""})
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
