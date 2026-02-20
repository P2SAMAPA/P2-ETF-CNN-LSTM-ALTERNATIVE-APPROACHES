"""
data/loader.py
Loads master_data.parquet from HF Dataset.
Validates freshness against the last NYSE trading day.
No external pings — all data comes from HF Dataset only.

Actual dataset columns (confirmed from parquet inspection):
  ETFs    : AGG, GLD, SLV, SPY, TBT, TLT, VNQ
  Macro   : VIX, DXY, T10Y2Y, TBILL_3M, IG_SPREAD, HY_SPREAD
"""

import pandas as pd
import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download
from datetime import datetime, timedelta
import pytz

try:
    import pandas_market_calendars as mcal
    NYSE_CAL_AVAILABLE = True
except ImportError:
    NYSE_CAL_AVAILABLE = False

DATASET_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
PARQUET_FILE = "master_data.parquet"

TARGET_ETF_COLS = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
BENCHMARK_COLS  = ["SPY", "AGG"]
TBILL_COL       = "TBILL_3M"
MACRO_COLS      = ["VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD"]


# ── NYSE calendar ─────────────────────────────────────────────────────────────

def get_last_nyse_trading_day(as_of=None):
    est = pytz.timezone("US/Eastern")
    if as_of is None:
        as_of = datetime.now(est)
    today = as_of.date()
    if NYSE_CAL_AVAILABLE:
        try:
            nyse  = mcal.get_calendar("NYSE")
            sched = nyse.schedule(start_date=today - timedelta(days=10), end_date=today)
            if len(sched) > 0:
                return sched.index[-1].date()
        except Exception:
            pass
    candidate = today
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_dataset(hf_token: str) -> pd.DataFrame:
    try:
        path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=PARQUET_FILE,
            repo_type="dataset",
            token=hf_token,
        )
        df = pd.read_parquet(path)

        if not isinstance(df.index, pd.DatetimeIndex):
            for col in ["Date", "date", "DATE"]:
                if col in df.columns:
                    df = df.set_index(col)
                    break
            df.index = pd.to_datetime(df.index)

        return df.sort_index()

    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        return pd.DataFrame()


# ── Freshness check ───────────────────────────────────────────────────────────

def check_data_freshness(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"fresh": False, "message": "Dataset is empty."}

    last   = df.index[-1].date()
    expect = get_last_nyse_trading_day()
    fresh  = last >= expect

    msg = (
        f"✅ Dataset up to date through **{last}**." if fresh else
        f"⚠️ **{expect}** data not yet updated. Latest: **{last}**. "
        f"Dataset updates daily after market close."
    )
    return {"fresh": fresh, "last_date_in_data": last,
            "expected_date": expect, "message": msg}


# ── Detect whether a column holds prices or returns ───────────────────────────

def _is_price_series(series: pd.Series) -> bool:
    """
    Heuristic: a price series has abs(median) > 2 and std/mean < 0.5.
    A return series has abs(median) < 0.1 and many values near zero.
    """
    clean = series.dropna()
    if len(clean) == 0:
        return False
    med = abs(clean.median())
    # Strong price signal: median > 2 (e.g. TLT ~ 90, TBT ~ 20)
    if med > 2:
        return True
    # Strong return signal: most values between -0.2 and 0.2
    if (clean.abs() < 0.2).mean() > 0.9:
        return False
    return med > 0.5


# ── Feature / target extraction ───────────────────────────────────────────────

def get_features_and_targets(df: pd.DataFrame):
    """
    Build return columns for target ETFs and benchmarks.
    Auto-detects whether source columns are prices or already returns.

    Returns:
        input_features : list[str]
        target_etfs    : list[str]  e.g. ["TLT_Ret", ...]
        tbill_rate     : float
        df             : DataFrame with _Ret columns added
        col_info       : dict of diagnostics for sidebar display
    """
    missing = [c for c in TARGET_ETF_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing ETF columns: {missing}. "
            f"Found in dataset: {list(df.columns)}"
        )

    col_info = {}

    # ── Build _Ret columns ────────────────────────────────────────────────────
    def make_ret(col):
        ret_col = f"{col}_Ret"
        if ret_col in df.columns:
            col_info[col] = "pre-computed _Ret"
            return ret_col
        if _is_price_series(df[col]):
            df[ret_col] = df[col].pct_change()
            col_info[col] = f"price→pct_change (median={df[col].median():.2f})"
        else:
            df[ret_col] = df[col]
            col_info[col] = f"used as-is (median={df[col].median():.4f})"
        return ret_col

    target_etfs    = [make_ret(c) for c in TARGET_ETF_COLS]
    benchmark_rets = [make_ret(c) for c in BENCHMARK_COLS if c in df.columns]

    # Drop NaN rows (first row from pct_change)
    df = df.dropna(subset=target_etfs).copy()

    # Sanity check: target returns should be small daily values
    for ret_col in target_etfs:
        med = df[ret_col].abs().median()
        if med > 0.1:
            st.warning(
                f"⚠️ {ret_col} has median absolute value {med:.4f} — "
                f"these may not be daily returns. Check dataset column '{ret_col.replace('_Ret','')}'. "
                f"Sample values: {df[ret_col].tail(3).values}"
            )

    # ── Input features ────────────────────────────────────────────────────────
    exclude = set(
        TARGET_ETF_COLS + BENCHMARK_COLS + target_etfs + benchmark_rets +
        [f"{c}_Ret" for c in BENCHMARK_COLS] + [TBILL_COL]
    )

    # First try known macro columns
    input_features = [c for c in MACRO_COLS if c in df.columns and c not in exclude]

    # Then add any engineered signal columns
    extra = [
        c for c in df.columns
        if c not in exclude
        and c not in input_features
        and any(k in c for k in ["_Z", "_Vol", "Regime", "YC_", "Credit_",
                                  "Rates_", "VIX_", "Spread", "DXY", "T10Y",
                                  "TBILL", "SOFR", "MOVE"])
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    input_features += extra

    # Fallback: all numeric non-excluded columns
    if not input_features:
        input_features = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]

    # ── T-bill rate ───────────────────────────────────────────────────────────
    tbill_rate = 0.045
    if TBILL_COL in df.columns:
        raw = df[TBILL_COL].dropna()
        if len(raw) > 0:
            v = float(raw.iloc[-1])
            tbill_rate = v / 100 if v > 1 else v

    return input_features, target_etfs, tbill_rate, df, col_info


# ── Dataset summary ───────────────────────────────────────────────────────────

def dataset_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    return {
        "rows":        len(df),
        "columns":     len(df.columns),
        "start_date":  df.index[0].strftime("%Y-%m-%d"),
        "end_date":    df.index[-1].strftime("%Y-%m-%d"),
        "etfs_found":  [c for c in TARGET_ETF_COLS if c in df.columns],
        "benchmarks":  [c for c in BENCHMARK_COLS  if c in df.columns],
        "macro_found": [c for c in MACRO_COLS       if c in df.columns],
        "tbill_found": TBILL_COL in df.columns,
        "all_cols":    list(df.columns),
    }
