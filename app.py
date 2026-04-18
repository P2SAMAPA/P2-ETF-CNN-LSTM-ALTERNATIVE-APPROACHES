"""
app.py  —  MC Dropout edition
P2-ETF-CNN-LSTM-ALTERNATIVE-APPROACHES

Changes from original
---------------------
1. Sidebar: added MC Dropout toggle + n_passes slider (shown only when enabled)
2. run_module(): predict_approachN calls replaced with mc_predict_approachN
   when MC Dropout is on; falls back to original predict_approachN when off.
3. display_single_year_results(): show_conviction_panel replaced with
   show_mc_conviction_panel when MC Dropout results are available.
4. show_all_signals_panel replaced with show_mc_all_signals_panel (MC mode).
5. Cache keys include mc_enabled + n_passes so toggling forces a fresh run.
6. trained_info now stores {"proba", "mean_proba", "uncertainty"} so both
   MC and non-MC paths share the same downstream display logic.

All original behaviour is preserved when MC Dropout is disabled.
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

# ── MC Dropout imports (new) ──────────────────────────────────────────────────
from models.mc_dropout import (
    mc_predict_approach1,
    mc_predict_approach2,
    mc_predict_approach3,
)
from signals.mc_conviction import compute_mc_conviction
from ui.mc_components import (
    show_mc_conviction_panel,
    show_mc_all_signals_panel,
    mc_passes_selector,
)

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


# ── Initialize session state with module prefixes ────────────────────────────
def init_module_state(prefix):
    defaults = {
        f"{prefix}_output_ready":    False,
        f"{prefix}_results":         None,
        f"{prefix}_trained_info":    None,
        f"{prefix}_test_dates":      None,
        f"{prefix}_test_slice":      None,
        f"{prefix}_optimal_lookback": None,
        f"{prefix}_df_for_chart":    None,
        f"{prefix}_target_etfs":     None,
        f"{prefix}_multiyear_ready": False,
        f"{prefix}_multiyear_results": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


init_module_state("fi")
init_module_state("eq")

if "tbill_rate" not in st.session_state:
    st.session_state["tbill_rate"] = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.write(f"🕒 **EST:** {get_est_time().strftime('%H:%M:%S')}")
    st.divider()

    start_yr = st.slider("📅 Start Year", 2010, 2024, 2016)
    fee_bps  = st.slider("💰 Fee (bps)", 0, 50, 10)
    epochs   = st.number_input("🔁 Max Epochs", 20, 150, 80, step=10)

    st.divider()
    split_option = st.selectbox("📊 Train/Val/Test Split",
                                 ["70/15/15", "80/10/10"], index=0)
    train_pct, val_pct = {"70/15/15": (0.70, 0.15),
                           "80/10/10": (0.80, 0.10)}[split_option]

    # ── MC Dropout controls (new) ─────────────────────────────────────────────
    st.divider()
    mc_enabled = st.toggle(
        "🎲 MC Dropout Uncertainty",
        value=True,
        help=(
            "Run N stochastic forward passes at inference with dropout active. "
            "Produces per-ETF uncertainty estimates and an adjusted conviction score. "
            "Automatically recommends CASH when uncertainty is too high."
        ),
    )
    if mc_enabled:
        n_passes = mc_passes_selector(key="mc_n_passes_sidebar")
        st.caption(
            f"**{n_passes} passes** · Dropout active at inference · "
            f"~{n_passes * 5}ms extra per approach on CPU"
        )
    else:
        n_passes = 0
        st.caption("MC Dropout off — single deterministic forward pass.")
    # ─────────────────────────────────────────────────────────────────────────

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

freshness     = check_data_freshness(df_raw)
last_date_str = str(freshness.get("last_date_in_data", "unknown"))

# ── Dataset info sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.subheader("📦 Dataset Info")
    fi_summary  = dataset_summary(df_raw, module_type="fi")
    eq_summary  = dataset_summary(df_raw, module_type="equity")
    st.write(f"**Data Range:** {fi_summary['start_date']} → {fi_summary['end_date']}")
    st.write(f"**Rows:** {fi_summary['rows']:,}")
    with st.expander("📊 Fixed Income ETFs"):
        st.write(f"Available: {', '.join(fi_summary['etfs_found'])}")
    with st.expander("📈 Equity ETFs"):
        st.write(f"Available: {', '.join(eq_summary['etfs_found'])}")
    st.write(f"**Macro Signals:** {', '.join(fi_summary['macro_found'])}")
    st.write(f"**T-bill col:** {'✅' if fi_summary['tbill_found'] else '❌'}")

# ── Main Title ────────────────────────────────────────────────────────────────
st.title("🧠 P2-ETF-CNN-LSTM")
st.caption("Multi-Asset ETF Rotation using CNN-LSTM | Fixed Income & Equity Modules")
show_freshness_status(freshness)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════
def run_module(module_type: str, df_raw: pd.DataFrame, start_yr: int, fee_bps: int,
               epochs: int, train_pct: float, val_pct: float, last_date_str: str,
               mc_enabled: bool = True, n_passes: int = 50):
    """Execute all 3 approaches for a given module type (fi or equity)."""
    prefix = module_type
    st.session_state[f"{prefix}_output_ready"] = False

    df    = df_raw[df_raw.index.year >= start_yr].copy()
    n_rows = len(df)

    if n_rows < 100:
        st.error(f"❌ Insufficient data: only {n_rows} rows from {start_yr}.")
        return False

    st.write(
        f"📅 **Data:** {df.index[0].strftime('%Y-%m-%d')} → "
        f"{df.index[-1].strftime('%Y-%m-%d')} "
        f"({df.index[-1].year - df.index[0].year + 1} years, {n_rows} rows)"
    )

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

    # ── Lookback auto-select (unchanged) ─────────────────────────────────────
    cache_prefix = f"{last_date_str}_{module_type}"
    lb_key       = make_cache_key(cache_prefix, start_yr, fee_bps, int(epochs),
                                   split_option, False, 0)
    lb_cached    = load_cache(f"lb_{lb_key}")

    if lb_cached is not None:
        optimal_lookback = lb_cached["optimal_lookback"]
        st.success(f"⚡ Lookback cache hit: **{optimal_lookback}d**")
    else:
        with st.spinner("🔍 Auto-selecting optimal lookback (30/45/60d)..."):
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

    # ── Model cache — key includes mc flag + n_passes so toggling forces rerun
    cache_key  = make_cache_key(
        f"{cache_prefix}_mc{int(mc_enabled)}_{n_passes}",
        start_yr, fee_bps, int(epochs), split_option, False, lookback,
    )
    cached_data = load_cache(cache_key)

    if cached_data is not None:
        results      = cached_data["results"]
        trained_info = cached_data["trained_info"]
        test_dates   = pd.DatetimeIndex(cached_data["test_dates"])
        test_slice   = cached_data["test_slice"]
        st.success("⚡ Model results loaded from cache")
    else:
        X_seq, y_seq = build_sequences(X_raw, y_raw, lookback)
        y_labels     = returns_to_labels(y_seq)

        (X_train, y_train_r, X_val, y_val_r,
         X_test,  y_test_r) = train_val_test_split(X_seq, y_seq, train_pct, val_pct)
        (_, y_train_l, _, y_val_l, _, _) = train_val_test_split(
            X_seq, y_labels, train_pct, val_pct
        )

        for name, arr in [("Training", X_train), ("Validation", X_val), ("Test", X_test)]:
            if len(arr) == 0:
                st.error(f"❌ {name} set is empty. Try an earlier Start Year.")
                return False

        X_train_s, X_val_s, X_test_s, _ = scale_features(X_train, X_val, X_test)

        train_size = len(X_train)
        val_size   = len(X_val)
        test_start = lookback + train_size + val_size
        test_dates = df.index[test_start: test_start + len(X_test)]
        test_slice = slice(test_start, test_start + len(X_test))

        results, trained_info = {}, {}
        progress = st.progress(0, text="Training Approach 1...")

        # ── Approach configs ────────────────────────────────────────────────
        # Each entry: (name, train_fn, predict_fn_standard, predict_fn_mc)
        # predict_fn_mc returns (preds, mean_proba, uncertainty)
        # predict_fn_standard returns (preds, proba)

        approach_configs = [
            (
                "Approach 1",
                lambda: train_approach1(
                    X_train_s, y_train_l, X_val_s, y_val_l,
                    n_classes=n_classes, epochs=int(epochs),
                ),
                lambda m: predict_approach1(m[0], X_test_s),
                lambda m: mc_predict_approach1(m[0], X_test_s, n_passes=n_passes),
            ),
            (
                "Approach 2",
                lambda: train_approach2(
                    X_train_s, y_train_l, X_val_s, y_val_l,
                    X_flat_all=X_raw, feature_names=input_features,
                    lookback=lookback, train_size=train_size,
                    val_size=val_size, n_classes=n_classes,
                    epochs=int(epochs),
                ),
                lambda m: predict_approach2(
                    m[0], X_test_s, X_raw, m[3], m[2],
                    lookback, train_size, val_size,
                ),
                lambda m: mc_predict_approach2(
                    m[0], X_test_s, X_raw, m[3], m[2],
                    lookback, train_size, val_size, n_passes=n_passes,
                ),
            ),
            (
                "Approach 3",
                lambda: train_approach3(
                    X_train_s, y_train_l, X_val_s, y_val_l,
                    n_classes=n_classes, epochs=int(epochs),
                ),
                lambda m: predict_approach3(m[0], X_test_s),
                lambda m: mc_predict_approach3(m[0], X_test_s, n_passes=n_passes),
            ),
        ]

        for idx, (approach, train_fn, predict_fn, mc_predict_fn) in enumerate(approach_configs):
            try:
                model_out = train_fn()

                if mc_enabled:
                    # ── MC path ───────────────────────────────────────────
                    preds, mean_proba, uncertainty = mc_predict_fn(model_out)
                    trained_info[approach] = {
                        "proba":       mean_proba,   # mean_proba is the "proba" for backtest
                        "mean_proba":  mean_proba,
                        "uncertainty": uncertainty,
                        "mc_enabled":  True,
                    }
                else:
                    # ── Standard path (unchanged) ─────────────────────────
                    preds, proba = predict_fn(model_out)
                    trained_info[approach] = {
                        "proba":       proba,
                        "mean_proba":  None,
                        "uncertainty": None,
                        "mc_enabled":  False,
                    }

                results[approach] = execute_strategy(
                    preds, trained_info[approach]["proba"],
                    y_test_r, test_dates,
                    target_etfs, fee_bps, tbill_rate,
                )

            except Exception as e:
                st.warning(f"⚠️ {approach} failed: {e}")
                results[approach]      = None
                trained_info[approach] = {
                    "proba": None, "mean_proba": None,
                    "uncertainty": None, "mc_enabled": mc_enabled,
                }

            pct = int((idx + 1) / 3 * 100)
            progress.progress(pct, text=f"{approach} complete...")

        progress.empty()

        save_cache(cache_key, {
            "results":      results,
            "trained_info": trained_info,
            "test_dates":   list(test_dates),
            "test_slice":   test_slice,
        })

    # ── Store in session state ────────────────────────────────────────────────
    st.session_state[f"{prefix}_results"]          = results
    st.session_state[f"{prefix}_trained_info"]     = trained_info
    st.session_state[f"{prefix}_test_dates"]       = test_dates
    st.session_state[f"{prefix}_test_slice"]       = test_slice
    st.session_state[f"{prefix}_optimal_lookback"] = optimal_lookback
    st.session_state[f"{prefix}_df_for_chart"]     = df
    st.session_state[f"{prefix}_target_etfs"]      = target_etfs
    st.session_state["tbill_rate"]                 = tbill_rate
    st.session_state[f"{prefix}_output_ready"]     = True
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY SINGLE-YEAR RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
def display_single_year_results(module_type: str):
    prefix = module_type

    if not st.session_state.get(f"{prefix}_output_ready"):
        st.info("👈 Click **🚀 Run Analysis** to see Single-Year results.")
        return

    results          = st.session_state.get(f"{prefix}_results")
    trained_info     = st.session_state.get(f"{prefix}_trained_info")
    test_dates       = st.session_state.get(f"{prefix}_test_dates")
    test_slice       = st.session_state.get(f"{prefix}_test_slice")
    optimal_lookback = st.session_state.get(f"{prefix}_optimal_lookback")
    df               = st.session_state.get(f"{prefix}_df_for_chart")
    tbill_rate       = st.session_state.get("tbill_rate")
    target_etfs      = st.session_state.get(f"{prefix}_target_etfs")

    if not all([results, trained_info, test_dates is not None, df is not None]):
        st.error("❌ Missing data. Please run the analysis again.")
        return

    winner_name = select_winner(results)
    winner_res  = results.get(winner_name)

    if winner_res is None:
        st.error("❌ All approaches failed.")
        return

    st.caption("Winner selected by highest raw annualised return on out-of-sample test set.")
    next_date = get_next_signal_date()
    st.divider()
    show_signal_banner(winner_res["next_signal"], next_date, winner_name)

    # ── Conviction panel: MC or standard ─────────────────────────────────────
    winner_info = trained_info[winner_name]
    _is_mc      = winner_info.get("mc_enabled", False)

    if _is_mc and winner_info.get("mean_proba") is not None:
        mean_p = winner_info["mean_proba"]
        unc    = winner_info["uncertainty"]
        mc_conv = compute_mc_conviction(mean_p[-1], unc[-1], target_etfs,
                                         include_cash=False)
        show_mc_conviction_panel(mc_conv, n_passes=n_passes)
    else:
        # Original path — unchanged
        winner_proba = winner_info["proba"]
        if winner_proba is not None:
            conviction = compute_conviction(winner_proba[-1], target_etfs,
                                             include_cash=False)
            show_conviction_panel(conviction)

    st.divider()

    # ── All-signals panel: MC or standard ────────────────────────────────────
    if _is_mc:
        mc_all_signals = {}
        for name, res in results.items():
            if res is None:
                continue
            info = trained_info[name]
            mp   = info.get("mean_proba")
            uc   = info.get("uncertainty")
            mc_all_signals[name] = {
                "signal":    res["next_signal"],
                "mc_conv":   compute_mc_conviction(mp[-1], uc[-1], target_etfs,
                                                    include_cash=False)
                             if mp is not None else None,
                "is_winner": name == winner_name,
            }
        show_mc_all_signals_panel(
            mc_all_signals, target_etfs, False, next_date,
            optimal_lookback, n_passes=n_passes,
        )
    else:
        all_signals = {
            name: {
                "signal":    res["next_signal"],
                "proba":     trained_info[name]["proba"][-1]
                             if trained_info[name]["proba"] is not None else None,
                "is_winner": name == winner_name,
            }
            for name, res in results.items() if res is not None
        }
        show_all_signals_panel(all_signals, target_etfs, False,
                                next_date, optimal_lookback)

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
# DISPLAY MULTI-YEAR SWEEP  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════════
def display_multiyear_sweep(module_type: str, last_date_str: str, fee_bps: int,
                             epochs: int, split_option: str, train_pct: float,
                             val_pct: float, df_raw: pd.DataFrame):
    prefix       = module_type
    SWEEP_YEARS  = list(range(2008, 2026))

    st.subheader("🔁 Multi-Year Consensus Sweep")
    st.markdown(
        "Runs **all 3 approaches** across **all years from 2008 to 2025**, picks the winner "
        "per year, and aggregates signals into a weighted consensus vote."
    )
    st.caption(f"Sweep years: {', '.join(str(y) for y in SWEEP_YEARS)}")

    col_info, col_run, col_force = st.columns([2, 1, 1])
    with col_info:
        st.caption(f"Data: {last_date_str}")
    with col_run:
        sweep_button = st.button("🚀 Run Consensus Sweep", type="primary",
                                  use_container_width=True,
                                  key=f"{prefix}_sweep_run")
    with col_force:
        force_retrain_button = st.button("🔄 Force Retrain All", type="secondary",
                                          use_container_width=True,
                                          key=f"{prefix}_sweep_force")

    if force_retrain_button:
        st.session_state[f"{prefix}_multiyear_ready"]   = False
        st.session_state[f"{prefix}_multiyear_results"] = None
        with st.spinner("🗑️ Cache cleared — retraining all years…"):
            try:
                sweep_results = run_multiyear_sweep(
                    df_raw=df_raw, sweep_years=SWEEP_YEARS, fee_bps=fee_bps,
                    epochs=int(epochs), split_option=split_option,
                    last_date_str=last_date_str, train_pct=train_pct,
                    val_pct=val_pct, force_retrain=True, module_type=module_type,
                )
                st.session_state[f"{prefix}_multiyear_results"] = sweep_results
                st.session_state[f"{prefix}_multiyear_ready"]   = True
                st.rerun()
            except Exception as e:
                st.error(f"❌ Sweep failed: {e}")

    elif sweep_button:
        st.session_state[f"{prefix}_multiyear_ready"] = False
        with st.spinner("Running sweep..."):
            try:
                sweep_results = run_multiyear_sweep(
                    df_raw=df_raw, sweep_years=SWEEP_YEARS, fee_bps=fee_bps,
                    epochs=int(epochs), split_option=split_option,
                    last_date_str=last_date_str, train_pct=train_pct,
                    val_pct=val_pct, force_retrain=False, module_type=module_type,
                )
                st.session_state[f"{prefix}_multiyear_results"] = sweep_results
                st.session_state[f"{prefix}_multiyear_ready"]   = True
                st.rerun()
            except Exception as e:
                st.error(f"❌ Sweep failed: {e}")

    if (st.session_state.get(f"{prefix}_multiyear_ready")
            and st.session_state.get(f"{prefix}_multiyear_results")):
        show_multiyear_results(st.session_state[f"{prefix}_multiyear_results"],
                                sweep_years=SWEEP_YEARS)
    elif not st.session_state.get(f"{prefix}_multiyear_ready"):
        st.info("Click **🚀 Run Consensus Sweep** to analyse all start years at once.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN MODULE TAB BUILDER  (unchanged structure, passes mc_enabled + n_passes)
# ═══════════════════════════════════════════════════════════════════════════════
def build_module_tab(module_type: str, module_name: str, etf_list: str,
                     last_date_str: str, fee_bps: int, epochs: int,
                     split_option: str, train_pct: float, val_pct: float,
                     df_raw: pd.DataFrame):
    st.header(f"{module_name} ETF Rotation")
    st.markdown(f"**ETFs:** {etf_list}")

    run_button = st.button(
        f"🚀 Run {module_name} Analysis", type="primary",
        use_container_width=True, key=f"{module_type}_run_button",
    )

    if run_button:
        with st.spinner(f"Running {module_name} module..."):
            success = run_module(
                module_type, df_raw, start_yr, fee_bps, epochs,
                train_pct, val_pct, last_date_str,
                mc_enabled=mc_enabled, n_passes=n_passes,
            )
        if success:
            st.rerun()

    st.divider()

    tab_single, tab_multi = st.tabs(["📊 Single-Year Results",
                                      "🔁 Multi-Year Consensus"])
    with tab_single:
        display_single_year_results(module_type)
    with tab_multi:
        display_multiyear_sweep(module_type, last_date_str, fee_bps, epochs,
                                 split_option, train_pct, val_pct, df_raw)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_fi, tab_equity = st.tabs(["🏛️ Fixed Income (FI)", "📈 Equity"])

with tab_fi:
    build_module_tab(
        module_type="fi", module_name="Fixed Income",
        etf_list="TLT, VNQ, SLV, GLD, LQD, HYG, VCIT",
        last_date_str=last_date_str, fee_bps=fee_bps,
        epochs=epochs, split_option=split_option,
        train_pct=train_pct, val_pct=val_pct, df_raw=df_raw,
    )

with tab_equity:
    build_module_tab(
        module_type="eq", module_name="Equity",
        etf_list="QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XME, GDX, IWM",
        last_date_str=last_date_str, fee_bps=fee_bps,
        epochs=epochs, split_option=split_option,
        train_pct=train_pct, val_pct=val_pct, df_raw=df_raw,
    )
