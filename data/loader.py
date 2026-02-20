"""
data/loader.py
Loads master_data.parquet from HF Dataset.
Validates freshness against the last NYSE trading day.
No external pings — all data comes from HF Dataset only.

Actual dataset columns (from parquet inspection):
  ETFs    : AGG, GLD, SLV, SPY, TBT, TLT, VNQ
  Macro   : VIX, DXY, T10Y2Y, TBILL_3M, IG_SPREAD, HY_SPREAD
"""

import pandas as pd
import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download
from datetime import datetime, timedelta
import pytz
import os

try:
    import pandas_market_calendars as mcal
    NYSE_CAL_AVAILABLE = True
except ImportError:
    NYSE_CAL_AVAILABLE = False

DATASET_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
PARQUET_FILE = "master_data.parquet"

# ── Actual column names in the dataset ───────────────────────────────────────
TARGET_ETF_COLS  = ["TLT", "TBT", "VNQ", "SLV", "GLD"]   # traded ETFs
BENCHMARK_COLS   = ["SPY", "AGG"]                           # chart only
TBILL_COL        = "TBILL_3M"                               # 3m T-bill rate
MACRO_COLS       = ["VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD"]


# ── NYSE calendar helpers ─────────────────────────────────────────────────────

def get_last_nyse_trading_day(as_of=None):
    """Return the most recent NYSE trading day on or before as_of (default: today EST)."""
    est = pytz.timezone("US/Eastern")
    if as_of is None:
        as_of = datetime.now(est)
    today = as_of.date()

    if NYSE_CAL_AVAILABLE:
        try:
            nyse  = mcal.get_calendar("NYSE")
            start = today - timedelta(days=10)
            sched = nyse.schedule(start_date=start, end_date=today)
            if len(sched) > 0:
                return sched.index[-1].date()
        except Exception:
            pass

    # Fallback: skip weekends
    candidate = today
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_dataset(hf_token: str) -> pd.DataFrame:
    """
    Download master_data.parquet from HF Dataset and return as DataFrame.
    Cached for 1 hour. Index is parsed as DatetimeIndex.
    """
    try:
        path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=PARQUET_FILE,
            repo_type="dataset",
            token=hf_token,
        )
        df = pd.read_parquet(path)

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            for col in ["Date", "date", "DATE"]:
                if col in df.columns:
                    df = df.set_index(col)
                    break
            df.index = pd.to_datetime(df.index)

        df = df.sort_index()
        return df

    except Exception as e:
        st.error(f"❌ Failed to load dataset from HuggingFace: {e}")
        return pd.DataFrame()


# ── Freshness check ───────────────────────────────────────────────────────────

def check_data_freshness(df: pd.DataFrame) -> dict:
    """
    Check whether the dataset contains data for the last NYSE trading day.
    """
    if df.empty:
        return {
            "fresh": False,
            "last_date_in_data": None,
            "expected_date": None,
            "message": "Dataset is empty.",
        }

    last_date_in_data = df.index[-1].date()
    expected_date     = get_last_nyse_trading_day()
    fresh             = last_date_in_data >= expected_date

    if fresh:
        message = f"✅ Dataset is up to date through **{last_date_in_data}**."
    else:
        message = (
            f"⚠️ **{expected_date}** data not yet updated in dataset. "
            f"Latest available: **{last_date_in_data}**. "
            f"Please check back later — the dataset updates daily after market close."
        )

    return {
        "fresh": fresh,
        "last_date_in_data": last_date_in_data,
        "expected_date": expected_date,
        "message": message,
    }


# ── Feature / target extraction ───────────────────────────────────────────────

def get_features_and_targets(df: pd.DataFrame):
    """
    Extract input feature columns and target ETF return columns.

    The dataset stores raw price or return values directly under ticker names.
    We compute daily log returns for target ETFs if they are not already returns.

    Returns:
        input_features : list of column names to use as model inputs
        target_etfs    : list of ETF column names (after return computation)
        tbill_rate     : latest 3m T-bill rate as float (annualised, e.g. 0.045)
        df             : DataFrame (possibly with new _Ret columns added)
    """

    # ── Confirm target ETFs exist ─────────────────────────────────────────────
    missing = [c for c in TARGET_ETF_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing ETF columns: {missing}. "
            f"Found in dataset: {list(df.columns)}"
        )

    # ── Build return columns ──────────────────────────────────────────────────
    # If values look like prices (>5), compute pct returns.
    # If they already look like small returns (<1 in abs), use as-is.
    target_etfs = []
    for col in TARGET_ETF_COLS:
        ret_col = f"{col}_Ret"
        if ret_col not in df.columns:
            sample = df[col].dropna()
            if len(sample) > 0 and abs(sample.median()) > 1:
                # Looks like price — compute pct change
                df[ret_col] = df[col].pct_change()
            else:
                # Already returns
                df[ret_col] = df[col]
        target_etfs.append(ret_col)

    # Same for benchmarks
    for col in BENCHMARK_COLS:
        ret_col = f"{col}_Ret"
        if ret_col not in df.columns and col in df.columns:
            sample = df[col].dropna()
            if len(sample) > 0 and abs(sample.median()) > 1:
                df[ret_col] = df[col].pct_change()
            else:
                df[ret_col] = df[col]

    # Drop rows with NaN in target columns (first row after pct_change)
    df = df.dropna(subset=target_etfs)

    # ── Input features ────────────────────────────────────────────────────────
    # Use macro columns directly; exclude ETF price/return cols and benchmarks
    exclude = set(
        TARGET_ETF_COLS + BENCHMARK_COLS +
        target_etfs +
        [f"{c}_Ret" for c in BENCHMARK_COLS] +
        [TBILL_COL]
    )

    input_features = [
        c for c in df.columns
        if c not in exclude
        and c in (MACRO_COLS + [
            col for col in df.columns
            if any(k in col for k in ["_Z", "_Vol", "Regime", "YC_", "Credit_",
                                       "Rates_", "VIX_", "Spread", "DXY", "T10Y"])
        ])
    ]

    # Fallback: if none matched, use all non-excluded numeric columns
    if not input_features:
        input_features = [
            c for c in df.columns
            if c not in exclude
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    # ── T-bill rate ───────────────────────────────────────────────────────────
    tbill_rate = 0.045   # default
    if TBILL_COL in df.columns:
        raw = df[TBILL_COL].dropna()
        if len(raw) > 0:
            last_val   = float(raw.iloc[-1])
            tbill_rate = last_val / 100 if last_val > 1 else last_val

    return input_features, target_etfs, tbill_rate, df


# ── Dataset summary ───────────────────────────────────────────────────────────

def dataset_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    return {
        "rows":        len(df),
        "columns":     len(df.columns),
        "start_date":  df.index[0].strftime("%Y-%m-%d"),
        "end_date":    df.index[-1].strftime("%Y-%m-%d"),
        "etfs_found":  [c for c in TARGET_ETF_COLS  if c in df.columns],
        "benchmarks":  [c for c in BENCHMARK_COLS   if c in df.columns],
        "macro_found": [c for c in MACRO_COLS        if c in df.columns],
        "tbill_found": TBILL_COL in df.columns,
    }
