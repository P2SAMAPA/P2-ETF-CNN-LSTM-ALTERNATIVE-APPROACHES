"""
data/loader.py
Loads master_data.parquet from HF Dataset.
Validates freshness against the last NYSE trading day.
No external pings — all data comes from HF Dataset only.
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

# Columns expected in the dataset
REQUIRED_ETF_COLS   = ["TLT_Ret", "TBT_Ret", "VNQ_Ret", "SLV_Ret", "GLD_Ret"]
BENCHMARK_COLS      = ["SPY_Ret", "AGG_Ret"]
TBILL_COL           = "DTB3"          # 3m T-bill column in HF dataset
TARGET_ETFS         = REQUIRED_ETF_COLS   # 5 targets (no CASH in returns, CASH handled in strategy)


# ── NYSE calendar helpers ─────────────────────────────────────────────────────

def get_last_nyse_trading_day(as_of: datetime = None) -> datetime.date:
    """Return the most recent NYSE trading day before or on as_of (default: today EST)."""
    est = pytz.timezone("US/Eastern")
    if as_of is None:
        as_of = datetime.now(est)

    today = as_of.date()

    if NYSE_CAL_AVAILABLE:
        try:
            nyse = mcal.get_calendar("NYSE")
            # Look back up to 10 days to find last trading day
            start = today - timedelta(days=10)
            schedule = nyse.schedule(start_date=start, end_date=today)
            if len(schedule) > 0:
                return schedule.index[-1].date()
        except Exception:
            pass

    # Fallback: skip weekends
    candidate = today
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


def is_nyse_trading_day(date) -> bool:
    """Return True if date is a NYSE trading day."""
    if NYSE_CAL_AVAILABLE:
        try:
            nyse = mcal.get_calendar("NYSE")
            schedule = nyse.schedule(start_date=date, end_date=date)
            return len(schedule) > 0
        except Exception:
            pass
    return date.weekday() < 5


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
            if "Date" in df.columns:
                df = df.set_index("Date")
            elif "date" in df.columns:
                df = df.set_index("date")
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

    Returns a dict:
        {
            "fresh": bool,
            "last_date_in_data": date,
            "expected_date": date,
            "message": str
        }
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

    fresh = last_date_in_data >= expected_date

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
    Extract input feature columns and target ETF return columns from the dataset.

    Returns:
        input_features : list of column names
        target_etfs    : list of ETF return column names (e.g. TLT_Ret)
        tbill_rate     : latest 3m T-bill rate as a float (annualised, e.g. 0.045)
    """
    # Target ETF return columns
    target_etfs = [c for c in REQUIRED_ETF_COLS if c in df.columns]

    if not target_etfs:
        raise ValueError(
            f"No target ETF columns found. Expected: {REQUIRED_ETF_COLS}. "
            f"Found in dataset: {list(df.columns)}"
        )

    # Input features: Z-scores, vol, regime, yield curve, credit, rates, VIX terms
    exclude = set(target_etfs + BENCHMARK_COLS + [TBILL_COL])
    input_features = [
        c for c in df.columns
        if c not in exclude
        and (
            c.endswith("_Z")
            or c.endswith("_Vol")
            or "Regime" in c
            or "YC_"    in c
            or "Credit_" in c
            or "Rates_"  in c
            or "VIX_"    in c
            or "Spread"  in c
            or "DXY"     in c
            or "VIX"     in c
            or "T10Y"    in c
        )
    ]

    # 3m T-bill rate (for CASH return & Sharpe)
    tbill_rate = 0.045   # default fallback
    if TBILL_COL in df.columns:
        raw = df[TBILL_COL].dropna()
        if len(raw) > 0:
            last_val = raw.iloc[-1]
            # DTB3 is typically in percent (e.g. 5.25 means 5.25%)
            tbill_rate = float(last_val) / 100 if last_val > 1 else float(last_val)

    return input_features, target_etfs, tbill_rate


# ── Column info helper (for sidebar display) ──────────────────────────────────

def dataset_summary(df: pd.DataFrame) -> dict:
    """Return a brief summary dict for sidebar display."""
    if df.empty:
        return {}
    return {
        "rows":       len(df),
        "columns":    len(df.columns),
        "start_date": df.index[0].strftime("%Y-%m-%d"),
        "end_date":   df.index[-1].strftime("%Y-%m-%d"),
        "etfs_found": [c for c in REQUIRED_ETF_COLS if c in df.columns],
        "benchmarks": [c for c in BENCHMARK_COLS     if c in df.columns],
        "tbill_found": TBILL_COL in df.columns,
    }
