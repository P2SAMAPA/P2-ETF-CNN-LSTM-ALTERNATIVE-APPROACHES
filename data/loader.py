"""
data/loader.py
Loads master_data.parquet from HF Dataset.
Engineers rich feature set from raw price/macro columns.
No external pings — all data from HF Dataset only.
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

DATASET_REPO    = "P2SAMAPA/fi-etf-macro-signal-master-data"
PARQUET_FILE    = "master_data.parquet"
TARGET_ETF_COLS = ["TLT", "VNQ", "SLV", "GLD", "LQD", "HYG", "VCIT"]
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
        f"✅ Dataset up to date through {last}." if fresh else
        f"⚠️ {expect} data not yet updated. Latest: {last}. "
        f"Dataset updates daily after market close."
    )
    return {"fresh": fresh, "last_date_in_data": last,
            "expected_date": expect, "message": msg}

# ── Price → returns ───────────────────────────────────────────────────────────
def _to_returns(series: pd.Series) -> pd.Series:
    """Convert price series to daily pct returns. If already returns, pass through."""
    clean = series.dropna()
    if len(clean) == 0:
        return series
    if abs(clean.median()) > 2:          # price series
        return series.pct_change()
    return series                         # already returns

# ── Feature engineering ───────────────────────────────────────────────────────
def _engineer_features(df: pd.DataFrame, ret_cols: list) -> pd.DataFrame:
    """
    Build a rich feature set from raw macro + ETF return columns.
    Features added per ETF return:
      - 1d, 5d, 21d lagged returns
      - 5d, 21d rolling volatility
      - 5d, 21d momentum (cumulative return)

    Features added per macro column:
      - raw value (z-scored over rolling 252d window)
      - 5d change
      - 1d lag

    Also adds:
      - TBILL_3M as a feature (rate level)
      - VIX regime flag (VIX > 25)
      - Yield curve slope (already T10Y2Y)
      - Cross-asset momentum: spread between TLT_ret and AGG_ret
    """
    feat = pd.DataFrame(index=df.index)

    # ── ETF return features ───────────────────────────────────────────────────
    for col in ret_cols:
        r = df[col]
        feat[f"{col}_lag1"]  = r.shift(1)
        feat[f"{col}_lag5"]  = r.shift(5)
        feat[f"{col}_lag21"] = r.shift(21)
        feat[f"{col}_vol5"]  = r.rolling(5).std()
        feat[f"{col}_vol21"] = r.rolling(21).std()
        feat[f"{col}_mom5"]  = r.rolling(5).sum()
        feat[f"{col}_mom21"] = r.rolling(21).sum()

    # ── Macro features ────────────────────────────────────────────────────────
    for col in MACRO_COLS:
        if col not in df.columns:
            continue
        s = df[col]
        # Z-score over rolling 252-day window
        roll_mean = s.rolling(252, min_periods=63).mean()
        roll_std  = s.rolling(252, min_periods=63).std()
        feat[f"{col}_z"]     = (s - roll_mean) / (roll_std + 1e-9)
        feat[f"{col}_chg5"]  = s.diff(5)
        feat[f"{col}_lag1"]  = s.shift(1)

    # ── TBILL level ───────────────────────────────────────────────────────────
    if TBILL_COL in df.columns:
        tbill = df[TBILL_COL]
        feat["TBILL_level"] = tbill
        feat["TBILL_chg5"]  = tbill.diff(5)

    # ── Derived cross-asset signals ───────────────────────────────────────────
    if "TLT_Ret" in df.columns and "AGG_Ret" in df.columns:
        feat["TLT_AGG_spread_mom5"] = (
            df["TLT_Ret"].rolling(5).sum() - df["AGG_Ret"].rolling(5).sum()
        )

    if "VIX" in df.columns:
        feat["VIX_regime"] = (df["VIX"] > 25).astype(float)
        feat["VIX_mom5"]   = df["VIX"].diff(5)

    if "T10Y2Y" in df.columns:
        feat["YC_inverted"] = (df["T10Y2Y"] < 0).astype(float)

    if "IG_SPREAD" in df.columns and "HY_SPREAD" in df.columns:
        feat["credit_ratio"] = df["HY_SPREAD"] / (df["IG_SPREAD"] + 1e-9)

    return feat

# ── Main extraction function ──────────────────────────────────────────────────
def get_features_and_targets(df: pd.DataFrame):
    """
    Build return columns for target ETFs and engineer a rich feature set.
    Returns:
        input_features : list[str]
        target_etfs    : list[str]  e.g. ["TLT_Ret", ...]
        tbill_rate     : float
        df_out         : DataFrame with all columns
        col_info       : dict of diagnostics
    """
    missing = [c for c in TARGET_ETF_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing ETF columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    col_info = {}

    # ── Build ETF return columns ──────────────────────────────────────────────
    target_etfs = []
    for col in TARGET_ETF_COLS:
        ret_col = f"{col}_Ret"
        df[ret_col] = _to_returns(df[col])
        med = abs(df[col].dropna().median())
        col_info[col] = f"price→pct_change (median={med:.2f})" if med > 2 else f"used as-is (median={med:.4f})"
        target_etfs.append(ret_col)

    # ── Build benchmark return columns ────────────────────────────────────────
    for col in BENCHMARK_COLS:
        if col in df.columns:
            df[f"{col}_Ret"] = _to_returns(df[col])

    # ── Drop NaN from first pct_change row ────────────────────────────────────
    df = df.dropna(subset=target_etfs).copy()

    # ── Engineer features ─────────────────────────────────────────────────────
    feat_df = _engineer_features(df, target_etfs)

    # Merge features into df
    for col in feat_df.columns:
        df[col] = feat_df[col].values

    # Drop rows with NaN in features (from lags/rolling)
    feat_cols = list(feat_df.columns)
    df = df.dropna(subset=feat_cols).copy()

    # ── T-bill rate ───────────────────────────────────────────────────────────
    tbill_rate = 0.045
    if TBILL_COL in df.columns:
        raw = df[TBILL_COL].dropna()
        if len(raw) > 0:
            v = float(raw.iloc[-1])
            tbill_rate = v / 100 if v > 1 else v

    # Input features = all engineered feature columns
    exclude = set(
        TARGET_ETF_COLS + BENCHMARK_COLS + target_etfs +
        [f"{c}_Ret" for c in BENCHMARK_COLS] + [TBILL_COL] +
        list(MACRO_COLS)
    )
    input_features = [c for c in feat_cols if c not in exclude]

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
