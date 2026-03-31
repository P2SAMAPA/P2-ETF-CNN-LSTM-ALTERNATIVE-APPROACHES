"""
data/loader.py
Loads master_data.parquet from HF Dataset.
Engineers rich feature set from raw price/macro columns.
No external pings — all data from HF Dataset only.
Supports both FI and Equity ETF modules.
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

# ── Module-specific ETF definitions ───────────────────────────────────────────
FI_ETF_COLS = ["TLT", "VNQ", "SLV", "GLD", "LQD", "HYG", "VCIT"]
EQUITY_ETF_COLS = ["QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XME", "GDX", "IWM"]

BENCHMARK_COLS = ["SPY", "AGG"]
TBILL_COL = "TBILL_3M"
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD"]

# Minimum non-NaN fraction a feature column must have to be included in dropna.
MIN_COVERAGE = 0.80

# ── NYSE calendar ─────────────────────────────────────────────────────────────
def get_last_nyse_trading_day(as_of=None):
    est = pytz.timezone("US/Eastern")
    if as_of is None:
        as_of = datetime.now(est)
    today = as_of.date()
    if NYSE_CAL_AVAILABLE:
        try:
            nyse = mcal.get_calendar("NYSE")
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
            for col in ["Date", "date", "DATE", "__index_level_0__"]:
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
    last = df.index[-1].date()
    expect = get_last_nyse_trading_day()
    fresh = last >= expect
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
    if abs(clean.median()) > 2:  # price series
        return series.pct_change()
    return series  # already returns

# ── Feature engineering ───────────────────────────────────────────────────────
def _engineer_features(df: pd.DataFrame, ret_cols: list, module_type: str = "fi") -> pd.DataFrame:
    """
    Build a rich feature set from raw macro + ETF return columns.
    
    Args:
        df: DataFrame with raw data
        ret_cols: List of return column names to use as targets
        module_type: "fi" or "equity" - determines feature emphasis
    """
    feat = pd.DataFrame(index=df.index)

    # ── ETF return features ───────────────────────────────────────────────────
    for col in ret_cols:
        r = df[col]
        feat[f"{col}_lag1"] = r.shift(1)
        feat[f"{col}_lag5"] = r.shift(5)
        feat[f"{col}_lag21"] = r.shift(21)
        feat[f"{col}_vol5"] = r.rolling(5).std()
        feat[f"{col}_vol21"] = r.rolling(21).std()
        feat[f"{col}_mom5"] = r.rolling(5).sum()
        feat[f"{col}_mom21"] = r.rolling(21).sum()

    # ── Macro features ────────────────────────────────────────────────────────
    for col in MACRO_COLS:
        if col not in df.columns:
            continue
        s = df[col].ffill()
        roll_mean = s.rolling(252, min_periods=63).mean()
        roll_std = s.rolling(252, min_periods=63).std()
        feat[f"{col}_z"] = (s - roll_mean) / (roll_std + 1e-9)
        feat[f"{col}_chg5"] = s.diff(5)
        feat[f"{col}_lag1"] = s.shift(1)

    # ── TBILL level ───────────────────────────────────────────────────────────
    if TBILL_COL in df.columns:
        tbill = df[TBILL_COL].ffill()
        feat["TBILL_level"] = tbill
        feat["TBILL_chg5"] = tbill.diff(5)

    # ── Derived cross-asset signals (module-specific) ─────────────────────────
    if module_type == "fi":
        # FI-specific spreads
        if "TLT_Ret" in df.columns and "AGG_Ret" in df.columns:
            feat["TLT_AGG_spread_mom5"] = (
                df["TLT_Ret"].rolling(5).sum() - df["AGG_Ret"].rolling(5).sum()
            )
        if "IG_SPREAD" in df.columns and "HY_SPREAD" in df.columns:
            feat["credit_ratio"] = (
                df["HY_SPREAD"].ffill() / (df["IG_SPREAD"].ffill() + 1e-9)
            )
    else:
        # Equity-specific signals
        if "QQQ_Ret" in df.columns and "SPY_Ret" in df.columns:
            feat["QQQ_SPY_spread_mom5"] = (
                df["QQQ_Ret"].rolling(5).sum() - df["SPY_Ret"].rolling(5).sum()
            )
        if "XLE_Ret" in df.columns and "XLK_Ret" in df.columns:
            feat["XLE_XLK_rotation"] = (
                df["XLE_Ret"].rolling(5).sum() - df["XLK_Ret"].rolling(5).sum()
            )
        if "XLF_Ret" in df.columns and "XLV_Ret" in df.columns:
            feat["XLF_XLV_spread"] = (
                df["XLF_Ret"].rolling(5).sum() - df["XLV_Ret"].rolling(5).sum()
            )

    # ── Universal regime indicators ───────────────────────────────────────────
    if "VIX" in df.columns:
        vix = df["VIX"].ffill()
        feat["VIX_regime"] = (vix > 25).astype(float)
        feat["VIX_mom5"] = vix.diff(5)

    if "T10Y2Y" in df.columns:
        feat["YC_inverted"] = (df["T10Y2Y"].ffill() < 0).astype(float)

    return feat

# ── Main extraction function (module-aware) ───────────────────────────────────
def get_features_and_targets(df: pd.DataFrame, module_type: str = "fi"):
    """
    Build return columns for target ETFs and engineer a rich feature set.
    
    Args:
        df: Raw DataFrame
        module_type: "fi" or "equity" - determines which ETFs to use
    
    Returns:
        input_features, target_etfs, tbill_rate, df_out, col_info
    """
    target_cols = FI_ETF_COLS if module_type == "fi" else EQUITY_ETF_COLS
    
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing {module_type.upper()} ETF columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    col_info = {}
    rows_start = len(df)

    # ── Build ETF return columns ──────────────────────────────────────────────
    target_etfs = []
    for col in target_cols:
        ret_col = f"{col}_Ret"
        df[ret_col] = _to_returns(df[col])
        med = abs(df[col].dropna().median())
        col_info[col] = (
            f"price→pct_change (median={med:.2f})" if med > 2
            else f"used as-is (median={med:.4f})"
        )
        target_etfs.append(ret_col)

    # ── Build benchmark return columns ────────────────────────────────────────
    for col in BENCHMARK_COLS:
        if col in df.columns:
            df[f"{col}_Ret"] = _to_returns(df[col])

    # ── Drop NaN from first pct_change row ────────────────────────────────────
    df = df.dropna(subset=target_etfs).copy()
    rows_after_ret = len(df)

    # ── Engineer features ─────────────────────────────────────────────────────
    feat_df = _engineer_features(df, target_etfs, module_type=module_type)

    for col in feat_df.columns:
        df[col] = feat_df[col].values

    feat_cols = list(feat_df.columns)

    # ── Smart dropna ──────────────────────────────────────────────────────────
    n = len(df)
    strict_cols = []
    ffill_cols = []
    for col in feat_cols:
        coverage = df[col].notna().sum() / n if n > 0 else 0
        if coverage >= MIN_COVERAGE:
            strict_cols.append(col)
        else:
            ffill_cols.append(col)

    if ffill_cols:
        df[ffill_cols] = df[ffill_cols].ffill()

    if strict_cols:
        df = df.dropna(subset=strict_cols).copy()

    rows_after_feat = len(df)

    col_info["_diagnostics"] = (
        f"rows: {rows_start} raw → {rows_after_ret} after ret dropna → "
        f"{rows_after_feat} after feature dropna | "
        f"strict_cols={len(strict_cols)} ffill_cols={len(ffill_cols)}"
    )

    # ── T-bill rate ───────────────────────────────────────────────────────────
    tbill_rate = 0.045
    if TBILL_COL in df.columns:
        raw = df[TBILL_COL].dropna()
        if len(raw) > 0:
            v = float(raw.iloc[-1])
            tbill_rate = v / 100 if v > 1 else v

    # ── Input features ────────────────────────────────────────────────────────
    exclude = set(
        target_cols + BENCHMARK_COLS + target_etfs +
        [f"{c}_Ret" for c in BENCHMARK_COLS] + [TBILL_COL] +
        list(MACRO_COLS)
    )
    input_features = [c for c in feat_cols if c not in exclude]

    if len(df) == 0:
        raise ValueError(
            f"No rows remain after feature engineering. "
            f"Diagnostics: {col_info['_diagnostics']}"
        )

    return input_features, target_etfs, tbill_rate, df, col_info

# ── Dataset summary (module-aware) ────────────────────────────────────────────
def dataset_summary(df: pd.DataFrame, module_type: str = "fi") -> dict:
    if df.empty:
        return {}
    
    target_cols = FI_ETF_COLS if module_type == "fi" else EQUITY_ETF_COLS
    
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "start_date": df.index[0].strftime("%Y-%m-%d"),
        "end_date": df.index[-1].strftime("%Y-%m-%d"),
        "etfs_found": [c for c in target_cols if c in df.columns],
        "benchmarks": [c for c in BENCHMARK_COLS if c in df.columns],
        "macro_found": [c for c in MACRO_COLS if c in df.columns],
        "tbill_found": TBILL_COL in df.columns,
        "all_cols": list(df.columns),
        "module": module_type,
    }
