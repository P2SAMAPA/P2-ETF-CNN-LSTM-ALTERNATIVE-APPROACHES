"""
app.py
P2-ETF-CNN-LSTM-ALTERNATIVE-APPROACHES
Clean dual-module version with Single-Year and Multi-Year tabs inside each module
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

from data.loader import (load_dataset, check_data_freshness,
                         get_features_and_targets, dataset_summary,
                         FI_ETF_COLS, EQUITY_ETF_COLS)
from utils.calendar import get_est_time, get_next_signal_date
from models.base import (build_sequences, train_val_test_split,
                         scale_features, returns_to_labels,
                         find_best_lookback, make_cache_key,
                         save_cache, load_cache)
from models.approach1_wavelet import train_approach1, predict_approach1
from models.approach2_regime import train_approach2, predict_approach2
from models.approach3_multiscale import train_approach3, predict_approach3
from strategy.backtest import execute_strategy, select_winner, build_comparison_table
from signals.conviction import compute_conviction
from ui.components import (
    show_freshness_status, show_signal_banner, show_conviction_panel,
    show_metrics_row, show_comparison_table, show_audit_trail,
    show_all_signals_panel,
)
from ui.multiyear import run_multiyear_sweep, show_multiyear_results

st.set_page_config(page_title="P2-ETF-CNN-LSTM", page_icon="🧠", layout="wide")

HF_TOKEN = os.getenv("HF_TOKEN", "")

# ── Initialize session state with module prefixes ───────────────────────────
def init_module_state(prefix):
    """Initialize all state keys for a given module prefix."""
    defaults = {
        f"{prefix}_output_ready": False,
        f"{prefix}_results": None,
        f"{prefix}_trained_info": None,
        f"{prefix}_test_dates": None,
        f"{prefix}_test_slice": None,
        f"{prefix}_optimal_lookback": None,
        f"{prefix}_df_for_chart": None,
        f"{prefix}_target_etfs": None,
        # Multi-year sweep state (per module)
        f"{prefix}_multiyear_ready": False,
        f"{prefix}_multiyear_results": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

# Initialize both modules
init_module_state("fi")
init_module_state("eq")

# Shared state
if "tbill_rate" not in st.session_state:
    st.session_state["tbill_rate"] = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.write(f"🕒 **EST:** {get_est_time().strftime('%H:%M:%S')}")
    st.divider()

    start_yr = st.slider("📅 Start Year", 2010, 2024, 2016)
    fee_bps = st.slider("💰 Fee (bps)", 0, 50, 10)
    epochs = st.number_input("🔁 Max Epochs", 20, 150, 80, step=10)

    st.divider()
    split_option = st.selectbox("📊 Train/Val/Test Split", ["70/15/15", "80/10/10"], index=0)
    train_pct, val_pct = {"70/15/15": (0.70, 0.15), "80/10/10": (0.80, 0.10)}[split_option]

    st.caption("💡 CASH triggered automatically on 2-day drawdown ≤ −15%")
    st.divider()

if not HF_TOKEN:
    st.error("❌ HF_TOKEN secret not found.")
    st.stop()

# ── Load dataset ──────────────────────────────────────────────────────────────
with st.spinner("📡 Loading dataset from HuggingFace..."):
    df_raw = load_dataset(HF_TOKEN)

if df_raw.empty:
    st.stop()

freshness = check_data_freshness(df_raw)
last_date_str = str(freshness.get("last_date_in_data", "unknown"))

# ── Dataset info sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.subheader("📦 Dataset Info")
    
    fi_summary = dataset_summary(df_raw, module_type="fi")
    eq_summary = dataset_summary(df_raw, module_type="equity")
    
    st.write(f"**Data Range:** {fi_summary['start_date']} → {fi_summary['end_date']}")
    st.write(f"**Rows:** {fi_summary['rows']:,}")
    
    with st.expander("📊 Fixed Income ETFs"):
        st.write(f"Available: {', '.join(fi_summary['etfs_found'])}")
        
    with st.expander("📈 Equity ETFs"):
        st.write(f"Available: {', '.join(eq_summary['etfs_found'])}")
        
    st.write(f"**Macro Signals:** {', '.join(fi_summary['macro_found'])}")
    st.write(f"**T-bill col:** {'✅' if fi_summary['tbill_found'] else '❌'}")

# ── Main Title ─────────────────────────────────────────────────────────────────
st.title("🧠 P2-ETF-CNN-LSTM")
st.caption("Multi-Asset ETF Rotation using CNN-LSTM | Fixed Income & Equity Modules")

show_freshness_status(freshness)

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE RUNNER FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════
def run_module(module_type: str, df_raw: pd.DataFrame, start_yr: int, fee_bps: int, 
               epochs: int, train_pct: float, val_pct: float, last_date_str: str):
    """Execute all 3 approaches for a given module type (fi or equity)."""
    prefix = module_type
    
    st.session_state[f"{prefix}_output_ready"] = False
    
    df = df_raw[df_raw.index.year >= start_yr].copy()
    n_rows = len(df)
    
    if n_rows < 100:
        st.error(f"❌ Insufficient data: only {n_rows} rows available from {start_yr}.")
        return False
    
    st.write(f"📅 **Data:** {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')} "
             f"({df.index[-1].year - df.index[0].year + 1} years, {n_rows} rows)")

    try:
        input_features, target_etfs, tbill_rate, df, col_info = get_features_and_targets(
            df, module_type=module_type
        )
    except ValueError as e:
        st.error(str(e))
        return False

    n_classes = len(target_etfs)

    st.info(
        f"🎯 **Targets:** {', '.join([t.replace('_Ret','') for t in target_etfs])} · "
        f"**Features:** {len(input_features)} signals · "
        f"**T-bill:** {tbill_rate*100:.2f}% · "
        f"**Rows:** {len(df)}"
    )

    X_raw = df[input_features].values.astype(np.float32)
    y_raw = np.clip(df[target_etfs].values.astype(np.float32), -0.5, 0.5)

    for j in range(X_raw.shape[1]):
        mask = np.isnan(X_raw[:, j])
        if mask.any():
            X_raw[mask, j] = np.nanmean(X_raw[:, j])
    for j in range(y_raw.shape[1]):
        mask = np.isnan(y_raw[:, j])
        if mask.any():
            y_raw[mask, j] = 0.0

    # Auto-select lookback
    cache_prefix = f"{last_date_str}_{module_type}"
    lb_key = make_cache_key(cache_prefix, start_yr, fee_bps, int(epochs),
                            split_option, False, 0)
    lb_cached = load_cache(f"lb_{lb_key}")

    if lb_cached is not None:
        optimal_lookback = lb_cached["optimal_lookback"]
        st.success(f"⚡ Lookback cache hit: **{optimal_lookback}d**")
    else:
        with st.spinner("🔍 Auto-selecting optimal lookback (30 / 45 / 60d)..."):
            try:
                optimal_lookback = find_best_lookback(
                    X_raw, y_raw, train_pct, val_pct, n_classes,
                    candidates=[30, 45, 60],
                )
            except ValueError as e:
                st.error(f"❌ Lookback selection failed: {e}")
                return False
        save_cache(f"lb_{lb_key}", {"optimal_lookback": optimal_lookback})
        st.success(f"📐 Optimal lookback: **{optimal_lookback}d**")

    lookback = optimal_lookback

    # Check model cache
    cache_key = make_cache_key(cache_prefix, start_yr, fee_bps, int(epochs),
                               split_option, False, lookback)
    cached_data = load_cache(cache_key)

    if cached_data is not None:
        results = cached_data["results"]
        trained_info = cached_data["trained_info"]
        test_dates = pd.DatetimeIndex(cached_data["test_dates"])
        test_slice = cached_data["test_slice"]
        st.success("⚡ Model results loaded from cache")
    else:
        X_seq, y_seq = build_sequences(X_raw, y_raw, lookback)
        y_labels = returns_to_labels(y_seq)

        (X_train, y_train_r, X_val, y_val_r,
         X_test, y_test_r) = train_val_test_split(X_seq, y_seq, train_pct, val_pct)
        (_, y_train_l, _, y_val_l, _, _) = train_val_test_split(
            X_seq, y_labels, train_pct, val_pct
        )

        if len(X_train) == 0:
            st.error("❌ Training set is empty. Try an earlier Start Year.")
            return False
        if len(X_val) == 0:
            st.error("❌ Validation set is empty. Try an earlier Start Year.")
            return False
        if len(X_test) == 0:
            st.error("❌ Test set is empty. Try an earlier Start Year.")
            return False

        X_train_s, X_val_s, X_test_s, _ = scale_features(X_train, X_val, X_test)

        train_size = len(X_train)
        val_size = len(X_val)
        test_start = lookback + train_size + val_size
        test_dates = df.index[test_start: test_start + len(X_test)]
        test_slice = slice(test_start, test_start + len(X_test))

        results, trained_info = {}, {}
        progress = st.progress(0, text="Training Approach 1...")

        approach_configs = [
            ("Approach 1", 
             lambda: train_approach1(X_train_s, y_train_l, X_val_s, y_val_l,
                                     n_classes=n_classes, epochs=int(epochs)),
             lambda m: predict_approach1(m[0], X_test_s)),
            ("Approach 2",
             lambda: train_approach2(X_train_s, y_train_l, X_val_s, y_val_l,
                                     X_flat_all=X_raw, feature_names=input_features,
                                     lookback=lookback, train_size=train_size,
                                     val_size=val_size, n_classes=n_classes,
                                     epochs=int(epochs)),
             lambda m: predict_approach2(m[0], X_test_s, X_raw, m[3], m[2],
                                          lookback, train_size, val_size)),
            ("Approach 3",
             lambda: train_approach3(X_train_s, y_train_l, X_val_s, y_val_l,
                                     n_classes=n_classes, epochs=int(epochs)),
             lambda m: predict_approach3(m[0], X_test_s)),
        ]

        for idx, (approach, train_fn, predict_fn) in enumerate(approach_configs):
            try:
                model_out = train_fn()
                preds, proba = predict_fn(model_out)
                results[approach] = execute_strategy(
                    preds, proba, y_test_r, test_dates,
                    target_etfs, fee_bps, tbill_rate,
                )
                trained_info[approach] = {"proba": proba}
            except Exception as e:
                st.warning(f"⚠️ {approach} failed: {e}")
                results[approach] = None
                trained_info[approach] = {"proba": None}

            pct = int((idx + 1) / 3 * 100)
            progress.progress(pct, text=f"{approach} complete...")

        progress.empty()

        save_cache(cache_key, {
            "results": results,
            "trained_info": trained_info,
            "test_dates": list(test_dates),
            "test_slice": test_slice,
        })

    # Store results
    st.session_state[f"{prefix}_results"] = results
    st.session_state[f"{prefix}_trained_info"] = trained_info
    st.session_state[f"{prefix}_test_dates"] = test_dates
    st.session_state[f"{prefix}_test_slice"] = test_slice
    st.session_state[f"{prefix}_optimal_lookback"] = optimal_lookback
    st.session_state[f"{prefix}_df_for_chart"] = df
    st.session_state[f"{prefix}_target_etfs"] = target_etfs
    st.session_state["tbill_rate"] = tbill_rate
    st.session_state[f"{prefix}_output_ready"] = True
    
    return True

# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY SINGLE-YEAR RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
def display_single_year_results(module_type: str):
    """Display single-year results for a specific module."""
    prefix = module_type
    
    # Check if results exist
    if not st.session_state.get(f"{prefix}_output_ready"):
        st.info(f"👈 Click **🚀 Run Analysis** in the header above to see Single-Year results.")
        return
    
    results = st.session_state.get(f"{prefix}_results")
    trained_info = st.session_state.get(f"{prefix}_trained_info")
    test_dates = st.session_state.get(f"{prefix}_test_dates")
    test_slice = st.session_state.get(f"{prefix}_test_slice")
    optimal_lookback = st.session_state.get(f"{prefix}_optimal_lookback")
    df = st.session_state.get(f"{prefix}_df_for_chart")
    tbill_rate = st.session_state.get("tbill_rate")
    target_etfs = st.session_state.get(f"{prefix}_target_etfs")

    if not all([results, trained_info, test_dates is not None, df is not None]):
        st.error("❌ Missing required data. Please run the analysis again.")
        return

    winner_name = select_winner(results)
    winner_res = results.get(winner_name)

    if winner_res is None:
        st.error("❌ All approaches failed.")
        return

    st.caption("Winner selected by highest raw annualised return on out-of-sample test set.")

    next_date = get_next_signal_date()
    st.divider()

    show_signal_banner(winner_res["next_signal"], next_date, winner_name)

    winner_proba = trained_info[winner_name]["proba"]
    if winner_proba is not None:
        conviction = compute_conviction(winner_proba[-1], target_etfs, include_cash=False)
        show_conviction_panel(conviction)

    st.divider()

    all_signals = {
        name: {"signal": res["next_signal"],
               "proba": trained_info[name]["proba"][-1] if trained_info[name]["proba"] is not None else None,
               "is_winner": name == winner_name}
        for name, res in results.items() if res is not None
    }
    show_all_signals_panel(all_signals, target_etfs, False, next_date, optimal_lookback)

    st.divider()
    st.subheader(f"📊 {winner_name} — Performance Metrics")

    spy_ann = None
    if df is not None and "SPY_Ret" in df.columns and test_slice is not None:
        spy_raw = df["SPY_Ret"].iloc[test_slice].values.copy().astype(float)
        spy_raw = spy_raw[~np.isnan(spy_raw)]
        spy_raw = np.clip(spy_raw, -0.5, 0.5)
        if len(spy_raw) > 5:
            spy_cum = np.prod(1 + spy_raw)
            spy_ann = float(spy_cum ** (252 / len(spy_raw)) - 1)

    show_metrics_row(winner_res, tbill_rate, spy_ann_return=spy_ann)

    st.divider()
    st.subheader("🏆 Approach Comparison")
    show_comparison_table(build_comparison_table(results, winner_name))

    st.divider()
    st.subheader(f"📋 Audit Trail — {winner_name} (Last 20 Trading Days)")
    show_audit_trail(winner_res["audit_trail"])

# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY MULTI-YEAR SWEEP (per module)
# ═══════════════════════════════════════════════════════════════════════════════
def display_multiyear_sweep(module_type: str, last_date_str: str, fee_bps: int, epochs: int,
                            split_option: str, train_pct: float, val_pct: float, df_raw: pd.DataFrame):
    """Display multi-year sweep section for a specific module."""
    prefix = module_type
    # --- CHANGE: Now runs all years from 2008 to 2025 ---
    SWEEP_YEARS = list(range(2008, 2026))   # 2008..2025 inclusive
    # ---------------------------------------------------
    
    st.subheader("🔁 Multi-Year Consensus Sweep")
    
    st.markdown(
        "Runs **all 3 approaches** across **all years from 2008 to 2025**, picks the winner per year, "
        "and aggregates signals into a weighted consensus vote. "
        "Each year uses the same fee, epochs, and split settings as the sidebar."
    )
    
    st.caption(f"Sweep years: {', '.join(str(y) for y in SWEEP_YEARS)}")
    
    col_info, col_run, col_force = st.columns([2, 1, 1])
    
    with col_info:
        st.caption(f"Data: {last_date_str}")
    
    with col_run:
        sweep_button = st.button(
            "🚀 Run Consensus Sweep",
            type="primary",
            use_container_width=True,
            key=f"{prefix}_sweep_run"
        )
    
    with col_force:
        force_retrain_button = st.button(
            "🔄 Force Retrain All",
            type="secondary",
            use_container_width=True,
            key=f"{prefix}_sweep_force"
        )
    
    # Handle sweep execution
    if force_retrain_button:
        st.session_state[f"{prefix}_multiyear_ready"] = False
        st.session_state[f"{prefix}_multiyear_results"] = None
        with st.spinner("🗑️ Sweep cache cleared — retraining all years from scratch…"):
            try:
                sweep_results = run_multiyear_sweep(
                    df_raw=df_raw,
                    sweep_years=SWEEP_YEARS,
                    fee_bps=fee_bps,
                    epochs=int(epochs),
                    split_option=split_option,
                    last_date_str=last_date_str,
                    train_pct=train_pct,
                    val_pct=val_pct,
                    force_retrain=True,
                    module_type=module_type,
                )
                st.session_state[f"{prefix}_multiyear_results"] = sweep_results
                st.session_state[f"{prefix}_multiyear_ready"] = True
                st.rerun()
            except Exception as e:
                st.error(f"❌ Sweep failed: {e}")
    
    elif sweep_button:
        st.session_state[f"{prefix}_multiyear_ready"] = False
        with st.spinner("Running sweep..."):
            try:
                sweep_results = run_multiyear_sweep(
                    df_raw=df_raw,
                    sweep_years=SWEEP_YEARS,
                    fee_bps=fee_bps,
                    epochs=int(epochs),
                    split_option=split_option,
                    last_date_str=last_date_str,
                    train_pct=train_pct,
                    val_pct=val_pct,
                    force_retrain=False,
                    module_type=module_type,
                )
                st.session_state[f"{prefix}_multiyear_results"] = sweep_results
                st.session_state[f"{prefix}_multiyear_ready"] = True
                st.rerun()
            except Exception as e:
                st.error(f"❌ Sweep failed: {e}")
    
    # Display sweep results
    if st.session_state.get(f"{prefix}_multiyear_ready") and st.session_state.get(f"{prefix}_multiyear_results"):
        show_multiyear_results(
            st.session_state[f"{prefix}_multiyear_results"],
            sweep_years=SWEEP_YEARS,
        )
    elif not st.session_state.get(f"{prefix}_multiyear_ready"):
        st.info("Click **🚀 Run Consensus Sweep** to analyse all start years at once.")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN MODULE TAB BUILDER
# ═══════════════════════════════════════════════════════════════════════════════
def build_module_tab(module_type: str, module_name: str, etf_list: str, 
                     last_date_str: str, fee_bps: int, epochs: int,
                     split_option: str, train_pct: float, val_pct: float, df_raw: pd.DataFrame):
    """Build a complete module tab with Single-Year and Multi-Year sub-tabs."""
    
    st.header(f"{module_name} ETF Rotation")
    st.markdown(f"**ETFs:** {etf_list}")
    
    # Run button at the top of the tab
    run_button = st.button(
        f"🚀 Run {module_name} Analysis", 
        type="primary", 
        use_container_width=True,
        key=f"{module_type}_run_button"
    )
    
    if run_button:
        with st.spinner(f"Running {module_name} module..."):
            success = run_module(
                module_type, df_raw, start_yr, fee_bps, epochs, 
                train_pct, val_pct, last_date_str
            )
        if success:
            st.rerun()
    
    st.divider()
    
    # ALWAYS show sub-tabs - they handle their own "not ready" states
    tab_single, tab_multi = st.tabs(["📊 Single-Year Results", "🔁 Multi-Year Consensus"])
    
    with tab_single:
        display_single_year_results(module_type)
    
    with tab_multi:
        display_multiyear_sweep(module_type, last_date_str, fee_bps, epochs,
                               split_option, train_pct, val_pct, df_raw)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TABS: FI and Equity
# ═══════════════════════════════════════════════════════════════════════════════
tab_fi, tab_equity = st.tabs(["🏛️ Fixed Income (FI)", "📈 Equity"])

with tab_fi:
    build_module_tab(
        module_type="fi",
        module_name="Fixed Income",
        etf_list="TLT, VNQ, SLV, GLD, LQD, HYG, VCIT",
        last_date_str=last_date_str,
        fee_bps=fee_bps,
        epochs=epochs,
        split_option=split_option,
        train_pct=train_pct,
        val_pct=val_pct,
        df_raw=df_raw
    )

with tab_equity:
    build_module_tab(
        module_type="eq",
        module_name="Equity",
        etf_list="QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XME, GDX, IWM",
        last_date_str=last_date_str,
        fee_bps=fee_bps,
        epochs=epochs,
        split_option=split_option,
        train_pct=train_pct,
        val_pct=val_pct,
        df_raw=df_raw
    )
