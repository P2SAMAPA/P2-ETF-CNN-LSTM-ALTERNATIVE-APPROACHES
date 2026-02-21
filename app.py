"""
app.py
P2-ETF-CNN-LSTM-ALTERNATIVE-APPROACHES
Uses st.session_state to persist results across Streamlit reruns.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

from data.loader      import (load_dataset, check_data_freshness,
                               get_features_and_targets, dataset_summary)
from utils.calendar   import get_est_time, get_next_signal_date
from models.base      import (build_sequences, train_val_test_split,
                               scale_features, returns_to_labels,
                               find_best_lookback, make_cache_key,
                               save_cache, load_cache)
from models.approach1_wavelet    import train_approach1, predict_approach1
from models.approach2_regime     import train_approach2, predict_approach2
from models.approach3_multiscale import train_approach3, predict_approach3
from strategy.backtest  import execute_strategy, select_winner, build_comparison_table
from signals.conviction import compute_conviction
from ui.components import (
    show_freshness_status, show_signal_banner, show_conviction_panel,
    show_metrics_row, show_comparison_table, show_audit_trail,
    show_all_signals_panel,
)
from ui.charts import equity_curve_chart

st.set_page_config(page_title="P2-ETF-CNN-LSTM", page_icon="🧠", layout="wide")

HF_TOKEN = os.getenv("HF_TOKEN", "")

# ── Session state init ────────────────────────────────────────────────────────
if "output_ready" not in st.session_state:
    st.session_state.output_ready   = False
if "results" not in st.session_state:
    st.session_state.results        = None
if "trained_info" not in st.session_state:
    st.session_state.trained_info   = None
if "test_dates" not in st.session_state:
    st.session_state.test_dates     = None
if "test_slice" not in st.session_state:
    st.session_state.test_slice     = None
if "optimal_lookback" not in st.session_state:
    st.session_state.optimal_lookback = None
if "df_for_chart" not in st.session_state:
    st.session_state.df_for_chart   = None
if "tbill_rate" not in st.session_state:
    st.session_state.tbill_rate     = None
if "target_etfs" not in st.session_state:
    st.session_state.target_etfs    = None
if "from_cache" not in st.session_state:
    st.session_state.from_cache     = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    now_est = get_est_time()
    st.write(f"🕒 **EST:** {now_est.strftime('%H:%M:%S')}")
    st.divider()

    start_yr     = st.slider("📅 Start Year", 2010, 2024, 2016)
    fee_bps      = st.slider("💰 Fee (bps)", 0, 50, 10)
    epochs       = st.number_input("🔁 Max Epochs", 20, 150, 80, step=10)

    st.divider()
    split_option = st.selectbox("📊 Train/Val/Test Split", ["70/15/15", "80/10/10"], index=0)
    train_pct, val_pct = {"70/15/15": (0.70, 0.15), "80/10/10": (0.80, 0.10)}[split_option]

    include_cash = st.checkbox("💵 Include CASH class", value=True,
        help="Model can select CASH (earns T-bill rate) instead of any ETF")

    st.divider()
    run_button = st.button("🚀 Run All 3 Approaches", type="primary", use_container_width=True)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🧠 P2-ETF-CNN-LSTM")
st.caption("Approach 1: Wavelet  ·  Approach 2: Regime-Conditioned  ·  Approach 3: Multi-Scale Parallel")
st.caption("Winner selected by highest raw annualised return on out-of-sample test set.")

if not HF_TOKEN:
    st.error("❌ HF_TOKEN secret not found.")
    st.stop()

# ── Load dataset (always) ─────────────────────────────────────────────────────
with st.spinner("📡 Loading dataset from HuggingFace..."):
    df_raw = load_dataset(HF_TOKEN)

if df_raw.empty:
    st.stop()

freshness = check_data_freshness(df_raw)
show_freshness_status(freshness)

# ── Dataset info sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.subheader("📦 Dataset Info")
    summary = dataset_summary(df_raw)
    if summary:
        st.write(f"**Rows:** {summary['rows']:,}")
        st.write(f"**Range:** {summary['start_date']} → {summary['end_date']}")
        st.write(f"**ETFs:** {', '.join(summary['etfs_found'])}")
        st.write(f"**Benchmarks:** {', '.join(summary['benchmarks'])}")
        st.write(f"**Macro:** {', '.join(summary['macro_found'])}")
        st.write(f"**T-bill col:** {'✅' if summary['tbill_found'] else '❌'}")

# ── Run button triggers training ──────────────────────────────────────────────
if run_button:
    # Clear previous output so it rerenders fresh
    st.session_state.output_ready = False

    # ── Filter & prepare ──────────────────────────────────────────────────────
    df = df_raw[df_raw.index.year >= start_yr].copy()
    st.write(f"📅 **Data:** {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')} "
             f"({df.index[-1].year - df.index[0].year + 1} years)")

    try:
        input_features, target_etfs, tbill_rate, df, _ = get_features_and_targets(df)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    n_etfs    = len(target_etfs)
    n_classes = n_etfs + (1 if include_cash else 0)

    st.info(
        f"🎯 **Targets:** {', '.join([t.replace('_Ret','') for t in target_etfs])}  ·  "
        f"**Features:** {len(input_features)} signals  ·  "
        f"**T-bill:** {tbill_rate*100:.2f}%"
    )

    # ── Raw arrays ────────────────────────────────────────────────────────────
    X_raw = df[input_features].values.astype(np.float32)
    y_raw = df[target_etfs].values.astype(np.float32)

    # Clip y_raw to daily return range before anything else
    y_raw = np.clip(y_raw, -0.5, 0.5)

    for j in range(X_raw.shape[1]):
        mask = np.isnan(X_raw[:, j])
        if mask.any():
            X_raw[mask, j] = np.nanmean(X_raw[:, j])
    for j in range(y_raw.shape[1]):
        mask = np.isnan(y_raw[:, j])
        if mask.any():
            y_raw[mask, j] = 0.0

    last_date_str = str(freshness.get("last_date_in_data", "unknown"))

    # ── Auto-select lookback ──────────────────────────────────────────────────
    lb_cache_key = make_cache_key(last_date_str, start_yr, fee_bps, int(epochs),
                                   split_option, include_cash, 0)
    lb_cached = load_cache(f"lb_{lb_cache_key}")

    if lb_cached is not None:
        optimal_lookback = lb_cached["optimal_lookback"]
        st.success(f"⚡ Cache hit · Optimal lookback: **{optimal_lookback}d**")
    else:
        with st.spinner("🔍 Auto-selecting optimal lookback (30 / 45 / 60d)..."):
            def _y_labels_fn(y_seq):
                return returns_to_labels(y_seq, include_cash=include_cash)
            optimal_lookback = find_best_lookback(
                X_raw, y_raw, _y_labels_fn,
                train_pct, val_pct, n_classes, include_cash,
                candidates=[30, 45, 60],
            )
        save_cache(f"lb_{lb_cache_key}", {"optimal_lookback": optimal_lookback})
        st.success(f"📐 Optimal lookback: **{optimal_lookback}d** (auto-selected from 30/45/60)")

    lookback = optimal_lookback

    # ── Check full model cache ────────────────────────────────────────────────
    cache_key   = make_cache_key(last_date_str, start_yr, fee_bps, int(epochs),
                                  split_option, include_cash, lookback)
    cached_data = load_cache(cache_key)

    if cached_data is not None:
        results      = cached_data["results"]
        trained_info = cached_data["trained_info"]
        test_dates   = pd.DatetimeIndex(cached_data["test_dates"])
        test_slice   = cached_data["test_slice"]
        from_cache   = True
        st.success("⚡ Results loaded from cache — no retraining needed.")
    else:
        from_cache = False

        # ── Build sequences ───────────────────────────────────────────────────
        X_seq, y_seq = build_sequences(X_raw, y_raw, lookback)
        y_labels     = returns_to_labels(y_seq, include_cash=include_cash)

        (X_train, y_train_r, X_val, y_val_r,
         X_test,  y_test_r)  = train_val_test_split(X_seq, y_seq,    train_pct, val_pct)
        (_,       y_train_l,  _,    y_val_l,
         _,       _)         = train_val_test_split(X_seq, y_labels, train_pct, val_pct)

        X_train_s, X_val_s, X_test_s, _ = scale_features(X_train, X_val, X_test)

        train_size = len(X_train)
        val_size   = len(X_val)
        test_start = lookback + train_size + val_size
        test_dates = df.index[test_start: test_start + len(X_test)]
        test_slice = slice(test_start, test_start + len(X_test))

        results      = {}
        trained_info = {}
        progress     = st.progress(0, text="Training Approach 1...")

        # Approach 1
        try:
            model1, _, _ = train_approach1(
                X_train_s, y_train_l, X_val_s, y_val_l,
                n_classes=n_classes, epochs=int(epochs),
            )
            preds1, proba1 = predict_approach1(model1, X_test_s)
            results["Approach 1"] = execute_strategy(
                preds1, proba1, y_test_r, test_dates,
                target_etfs, fee_bps, tbill_rate, include_cash,
            )
            trained_info["Approach 1"] = {"proba": proba1}
        except Exception as e:
            st.warning(f"⚠️ Approach 1 failed: {e}")
            results["Approach 1"] = None

        progress.progress(33, text="Training Approach 2...")

        # Approach 2
        try:
            model2, _, hmm2, regime_cols2 = train_approach2(
                X_train_s, y_train_l, X_val_s, y_val_l,
                X_flat_all=X_raw, feature_names=input_features,
                lookback=lookback, train_size=train_size, val_size=val_size,
                n_classes=n_classes, epochs=int(epochs),
            )
            preds2, proba2 = predict_approach2(
                model2, X_test_s, X_raw, regime_cols2, hmm2,
                lookback, train_size, val_size,
            )
            results["Approach 2"] = execute_strategy(
                preds2, proba2, y_test_r, test_dates,
                target_etfs, fee_bps, tbill_rate, include_cash,
            )
            trained_info["Approach 2"] = {"proba": proba2}
        except Exception as e:
            st.warning(f"⚠️ Approach 2 failed: {e}")
            results["Approach 2"] = None

        progress.progress(66, text="Training Approach 3...")

        # Approach 3
        try:
            model3, _ = train_approach3(
                X_train_s, y_train_l, X_val_s, y_val_l,
                n_classes=n_classes, epochs=int(epochs),
            )
            preds3, proba3 = predict_approach3(model3, X_test_s)
            results["Approach 3"] = execute_strategy(
                preds3, proba3, y_test_r, test_dates,
                target_etfs, fee_bps, tbill_rate, include_cash,
            )
            trained_info["Approach 3"] = {"proba": proba3}
        except Exception as e:
            st.warning(f"⚠️ Approach 3 failed: {e}")
            results["Approach 3"] = None

        progress.progress(100, text="Done!")
        progress.empty()

        save_cache(cache_key, {
            "results":      results,
            "trained_info": trained_info,
            "test_dates":   list(test_dates),
            "test_slice":   test_slice,
        })

    # ── Store everything in session_state ─────────────────────────────────────
    st.session_state.results          = results
    st.session_state.trained_info     = trained_info
    st.session_state.test_dates       = test_dates
    st.session_state.test_slice       = test_slice
    st.session_state.optimal_lookback = optimal_lookback
    st.session_state.df_for_chart     = df
    st.session_state.tbill_rate       = tbill_rate
    st.session_state.target_etfs      = target_etfs
    st.session_state.from_cache       = from_cache
    st.session_state.include_cash     = include_cash
    st.session_state.output_ready     = True

# ── Render output from session_state (persists across reruns) ─────────────────
if not st.session_state.output_ready:
    st.info("👈 Configure parameters and click **🚀 Run All 3 Approaches**.")
    st.stop()

results          = st.session_state.results
trained_info     = st.session_state.trained_info
test_dates       = st.session_state.test_dates
test_slice       = st.session_state.test_slice
optimal_lookback = st.session_state.optimal_lookback
df               = st.session_state.df_for_chart
tbill_rate       = st.session_state.tbill_rate
target_etfs      = st.session_state.target_etfs
include_cash     = st.session_state.include_cash

if st.session_state.from_cache:
    st.success("⚡ Showing cached results.")

winner_name = select_winner(results)
winner_res  = results.get(winner_name)

if winner_res is None:
    st.error("❌ All approaches failed.")
    st.stop()

next_date = get_next_signal_date()
st.divider()

# ── Winner signal banner ──────────────────────────────────────────────────────
show_signal_banner(winner_res["next_signal"], next_date, winner_name)

# ── Conviction panel ──────────────────────────────────────────────────────────
winner_proba = trained_info[winner_name]["proba"]
conviction   = compute_conviction(winner_proba[-1], target_etfs, include_cash)
show_conviction_panel(conviction)

st.divider()

# ── All models signals ────────────────────────────────────────────────────────
all_signals = {
    name: {
        "signal":    res["next_signal"],
        "proba":     trained_info[name]["proba"][-1],
        "is_winner": name == winner_name,
    }
    for name, res in results.items() if res is not None
}
show_all_signals_panel(all_signals, target_etfs, include_cash, next_date, optimal_lookback)

st.divider()

# ── Winner metrics ────────────────────────────────────────────────────────────
st.subheader(f"📊 {winner_name} — Performance Metrics")
show_metrics_row(winner_res, tbill_rate)

st.divider()

# ── Comparison table ──────────────────────────────────────────────────────────
st.subheader("🏆 Approach Comparison (Winner = Highest Raw Annualised Return)")
comparison_df = build_comparison_table(results, winner_name)
show_comparison_table(comparison_df)

st.divider()

# ── Equity curve ──────────────────────────────────────────────────────────────
st.subheader(f"📈 {winner_name} vs SPY & AGG — Out-of-Sample")
fig = equity_curve_chart(results, winner_name, test_dates, df, test_slice, tbill_rate)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Audit trail ───────────────────────────────────────────────────────────────
st.subheader(f"📋 Audit Trail — {winner_name} (Last 20 Trading Days)")
show_audit_trail(winner_res["audit_trail"])
