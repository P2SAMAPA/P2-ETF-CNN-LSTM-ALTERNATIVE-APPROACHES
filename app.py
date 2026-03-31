"""
app.py
P2-ETF-CNN-LSTM-ALTERNATIVE-APPROACHES
Dual-module version: FI and Equity ETFs
- Session state persistence (results don't vanish on rerun)
- Model caching keyed by data date + config params + module_type
- Auto-lookback (30/45/60d)
- CASH is a drawdown risk overlay (not a model class)
- Ann. Return compared vs SPY in metrics row
- Two main tabs: Fixed Income (FI) and Equity
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

# ── Session state init ────────────────────────────────────────────────────────
for key, default in [
    # FI module state
    ("fi_output_ready", False), ("fi_results", None), ("fi_trained_info", None),
    ("fi_test_dates", None), ("fi_test_slice", None), ("fi_optimal_lookback", None),
    ("fi_df_for_chart", None), ("fi_target_etfs", None),
    # Equity module state
    ("eq_output_ready", False), ("eq_results", None), ("eq_trained_info", None),
    ("eq_test_dates", None), ("eq_test_slice", None), ("eq_optimal_lookback", None),
    ("eq_df_for_chart", None), ("eq_target_etfs", None),
    # Shared
    ("tbill_rate", None), ("from_cache", False),
    # Multi-year sweep state
    ("multiyear_ready", False), ("multiyear_results", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

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
    
    # Show both FI and Equity ETF availability
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

# ── MAIN TABS: FI vs Equity ───────────────────────────────────────────────────
tab_fi, tab_equity = st.tabs(["🏛️ Fixed Income (FI)", "📈 Equity"])

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTION: Run Module
# ═══════════════════════════════════════════════════════════════════════════════
def run_module(module_type: str, df_raw: pd.DataFrame, start_yr: int, fee_bps: int, 
               epochs: int, train_pct: float, val_pct: float, last_date_str: str):
    """Execute all 3 approaches for a given module type (fi or equity)."""
    
    prefix = "fi" if module_type == "fi" else "eq"
    output_ready_key = f"{prefix}_output_ready"
    results_key = f"{prefix}_results"
    trained_info_key = f"{prefix}_trained_info"
    test_dates_key = f"{prefix}_test_dates"
    test_slice_key = f"{prefix}_test_slice"
    optimal_lookback_key = f"{prefix}_optimal_lookback"
    df_chart_key = f"{prefix}_df_for_chart"
    target_etfs_key = f"{prefix}_target_etfs"
    
    st.session_state[output_ready_key] = False
    
    df = df_raw[df_raw.index.year >= start_yr].copy()
    n_rows = len(df)
    
    st.write(f"📅 **Data:** {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')} "
             f"({df.index[-1].year - df.index[0].year + 1} years, {n_rows} rows)")

    try:
        input_features, target_etfs, tbill_rate, df, col_info = get_features_and_targets(df, module_type=module_type)
    except ValueError as e:
        st.error(str(e))
        return False

    n_etfs = len(target_etfs)
    n_classes = n_etfs

    st.info(
        f"🎯 **Targets:** {', '.join([t.replace('_Ret','') for t in target_etfs])} · "
        f"**Features:** {len(input_features)} signals · "
        f"**T-bill:** {tbill_rate*100:.2f}% · "
        f"**Rows after feature engineering:** {len(df)}"
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

    # ── Auto-select lookback ──────────────────────────────────────────────────
    # Include module_type in cache key to keep FI and Equity separate
    lb_key = make_cache_key(f"{last_date_str}_{module_type}", start_yr, fee_bps, int(epochs),
                            split_option, False, 0)
    lb_cached = load_cache(f"lb_{lb_key}")

    if lb_cached is not None:
        optimal_lookback = lb_cached["optimal_lookback"]
        st.success(f"⚡ Cache hit · Optimal lookback: **{optimal_lookback}d**")
    else:
        with st.spinner("🔍 Auto-selecting optimal lookback (30 / 45 / 60d)..."):
            try:
                optimal_lookback = find_best_lookback(
                    X_raw, y_raw,
                    train_pct, val_pct, n_classes,
                    candidates=[30, 45, 60],
                )
            except ValueError as e:
                st.error(
                    f"❌ Could not find a valid lookback window.\n\n{e}\n\n"
                    f"**Try an earlier Start Year** (e.g. 2013 or earlier)."
                )
                return False
        save_cache(f"lb_{lb_key}", {"optimal_lookback": optimal_lookback})
        st.success(f"📐 Optimal lookback: **{optimal_lookback}d** (auto-selected)")

    lookback = optimal_lookback

    # ── Check model cache ─────────────────────────────────────────────────────
    cache_key = make_cache_key(f"{last_date_str}_{module_type}", start_yr, fee_bps, int(epochs),
                               split_option, False, lookback)
    cached_data = load_cache(cache_key)

    if cached_data is not None:
        results = cached_data["results"]
        trained_info = cached_data["trained_info"]
        test_dates = pd.DatetimeIndex(cached_data["test_dates"])
        test_slice = cached_data["test_slice"]
        st.success("⚡ Results loaded from cache — no retraining needed.")
    else:
        X_seq, y_seq = build_sequences(X_raw, y_raw, lookback)
        y_labels = returns_to_labels(y_seq)

        (X_train, y_train_r, X_val, y_val_r,
         X_test, y_test_r) = train_val_test_split(X_seq, y_seq, train_pct, val_pct)
        (_, y_train_l, _, y_val_l,
         _, _) = train_val_test_split(X_seq, y_labels, train_pct, val_pct)

        n_seq = len(X_seq)
        if len(X_train) == 0:
            st.error(f"❌ Training set is empty. Try earlier Start Year.")
            return False
        if len(X_val) == 0:
            st.error(f"❌ Validation set is empty. Try earlier Start Year.")
            return False
        if len(X_test) == 0:
            st.error(f"❌ Test set is empty. Try earlier Start Year.")
            return False

        X_train_s, X_val_s, X_test_s, _ = scale_features(X_train, X_val, X_test)

        train_size = len(X_train)
        val_size = len(X_val)
        test_start = lookback + train_size + val_size
        test_dates = df.index[test_start: test_start + len(X_test)]
        test_slice = slice(test_start, test_start + len(X_test))

        results, trained_info = {}, {}
        progress = st.progress(0, text=f"Training {module_type.upper()} Approach 1...")

        for approach, train_fn, predict_fn in [
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
        ]:
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

            pct = {"Approach 1": 33, "Approach 2": 66, "Approach 3": 100}[approach]
            progress.progress(pct, text=f"{approach} done...")

        progress.empty()

        save_cache(cache_key, {
            "results": results, "trained_info": trained_info,
            "test_dates": list(test_dates), "test_slice": test_slice,
        })

    st.session_state.update({
        results_key: results,
        trained_info_key: trained_info,
        test_dates_key: test_dates,
        test_slice_key: test_slice,
        optimal_lookback_key: optimal_lookback,
        df_chart_key: df,
        "tbill_rate": tbill_rate,
        target_etfs_key: target_etfs,
        output_ready_key: True,
    })
    
    return True

# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY FUNCTION: Show Module Results
# ═══════════════════════════════════════════════════════════════════════════════
def display_module_results(module_type: str):
    """Display results for a specific module."""
    prefix = "fi" if module_type == "fi" else "eq"
    
    results = st.session_state[f"{prefix}_results"]
    trained_info = st.session_state[f"{prefix}_trained_info"]
    test_dates = st.session_state[f"{prefix}_test_dates"]
    test_slice = st.session_state[f"{prefix}_test_slice"]
    optimal_lookback = st.session_state[f"{prefix}_optimal_lookback"]
    df = st.session_state[f"{prefix}_df_for_chart"]
    tbill_rate = st.session_state["tbill_rate"]
    target_etfs = st.session_state[f"{prefix}_target_etfs"]

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
    conviction = compute_conviction(winner_proba[-1], target_etfs, include_cash=False)
    show_conviction_panel(conviction)

    st.divider()

    all_signals = {
        name: {"signal": res["next_signal"],
               "proba": trained_info[name]["proba"][-1],
               "is_winner": name == winner_name}
        for name, res in results.items() if res is not None
    }
    show_all_signals_panel(all_signals, target_etfs, False, next_date, optimal_lookback)

    st.divider()
    st.subheader(f"📊 {winner_name} — Performance Metrics")

    spy_ann = None
    if "SPY_Ret" in df.columns:
        spy_raw = df["SPY_Ret"].iloc[test_slice].values.copy().astype(float)
        spy_raw = spy_raw[~np.isnan(spy_raw)]
        spy_raw = np.clip(spy_raw, -0.5, 0.5)
        if len(spy_raw) > 5:
            spy_cum = np.prod(1 + spy_raw)
            spy_ann = float(spy_cum ** (252 / len(spy_raw)) - 1)

    show_metrics_row(winner_res, tbill_rate, spy_ann_return=spy_ann)

    st.divider()
    st.subheader("🏆 Approach Comparison (Winner = Highest Raw Annualised Return)")
    show_comparison_table(build_comparison_table(results, winner_name))

    st.divider()
    st.subheader(f"📋 Audit Trail — {winner_name} (Last 20 Trading Days)")
    show_audit_trail(winner_res["audit_trail"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: FIXED INCOME (FI)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_fi:
    st.header("🏛️ Fixed Income ETF Rotation")
    st.markdown("**ETFs:** TLT, VNQ, SLV, GLD, LQD, HYG, VCIT")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        run_fi = st.button("🚀 Run FI Analysis", type="primary", use_container_width=True)
    
    if run_fi:
        with st.spinner("Running Fixed Income module..."):
            success = run_module("fi", df_raw, start_yr, fee_bps, epochs, 
                                train_pct, val_pct, last_date_str)
        if success:
            st.rerun()
    
    if st.session_state["fi_output_ready"]:
        display_module_results("fi")
    else:
        st.info("👈 Click **🚀 Run FI Analysis** to start.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: EQUITY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_equity:
    st.header("📈 Equity ETF Rotation")
    st.markdown("**ETFs:** QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XME, GDX, IWM")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        run_eq = st.button("🚀 Run Equity Analysis", type="primary", use_container_width=True)
    
    if run_eq:
        with st.spinner("Running Equity module..."):
            success = run_module("equity", df_raw, start_yr, fee_bps, epochs,
                                train_pct, val_pct, last_date_str)
        if success:
            st.rerun()
    
    if st.session_state["eq_output_ready"]:
        display_module_results("equity")
    else:
        st.info("👈 Click **🚀 Run Equity Analysis** to start.")

# ── Multi-Year Sweep (Global) ─────────────────────────────────────────────────
st.divider()
st.subheader("🔁 Multi-Year Consensus Sweep")

with st.expander("Run Multi-Year Analysis (Optional)"):
    st.markdown(
        "Runs **all 3 approaches** across **8 start years**, picks the winner per year, "
        "and aggregates signals into a weighted consensus vote."
    )
    
    # Module selection for sweep
    sweep_module = st.radio("Select Module for Sweep", ["Fixed Income", "Equity"], horizontal=True)
    sweep_module_type = "fi" if sweep_module == "Fixed Income" else "equity"
    
    SWEEP_YEARS = [2010, 2012, 2014, 2016, 2018, 2019, 2021, 2023]
    st.caption(f"Sweep years: {', '.join(str(y) for y in SWEEP_YEARS)}")
    
    # Action buttons
    col_info, col_run, col_force = st.columns([2, 1, 1])
    
    with col_info:
        st.caption(f"Data date: {last_date_str} | Module: {sweep_module}")
    
    with col_run:
        sweep_button = st.button(
            "🚀 Run Consensus Sweep",
            type="primary",
            use_container_width=True,
            help="Runs sweep — uses cache where available, retrains stale years only.",
        )
    
    with col_force:
        force_retrain_button = st.button(
            "🔄 Force Retrain All",
            type="secondary",
            use_container_width=True,
            help="Clears all cached sweep results and retrains every year from scratch.",
        )
    
    # Handle Force Retrain
    if force_retrain_button:
        st.session_state.multiyear_ready = False
        st.session_state.multiyear_results = None
        with st.spinner(f"🗑️ Sweep cache cleared — retraining all {sweep_module} years from scratch…"):
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
                module_type=sweep_module_type,  # <-- Pass module_type
            )
        st.session_state.multiyear_results = sweep_results
        st.session_state.multiyear_ready = True
    
    # Handle normal Run
    elif sweep_button:
        st.session_state.multiyear_ready = False
        with st.spinner(f"Running {sweep_module} sweep..."):
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
                module_type=sweep_module_type,  # <-- Pass module_type
            )
        st.session_state.multiyear_results = sweep_results
        st.session_state.multiyear_ready = True
    
    # Display results
    if st.session_state.multiyear_ready and st.session_state.multiyear_results:
        show_multiyear_results(
            st.session_state.multiyear_results,
            sweep_years=SWEEP_YEARS,
        )
    elif not st.session_state.multiyear_ready:
        st.info(f"Click **🚀 Run Consensus Sweep** to analyse all {sweep_module} start years at once.")
