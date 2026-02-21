"""
app.py
P2-ETF-CNN-LSTM-ALTERNATIVE-APPROACHES
- Session state persistence (results don't vanish on rerun)
- Model caching keyed by data date + config params
- Auto-lookback (30/45/60d)
- CASH is a drawdown risk overlay (not a model class)
- Ann. Return compared vs SPY in metrics row
- Max Daily DD shows date it occurred
- Conviction panel: compact ETF probability list
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

st.set_page_config(page_title="P2-ETF-CNN-LSTM", page_icon="🧠", layout="wide")

HF_TOKEN = os.getenv("HF_TOKEN", "")

# ── Session state init ────────────────────────────────────────────────────────
for key, default in [
    ("output_ready", False), ("results", None), ("trained_info", None),
    ("test_dates", None), ("test_slice", None), ("optimal_lookback", None),
    ("df_for_chart", None), ("tbill_rate", None), ("target_etfs", None),
    ("from_cache", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.write(f"🕒 **EST:** {get_est_time().strftime('%H:%M:%S')}")
    st.divider()

    start_yr     = st.slider("📅 Start Year", 2010, 2024, 2016)
    fee_bps      = st.slider("💰 Fee (bps)", 0, 50, 10)
    epochs       = st.number_input("🔁 Max Epochs", 20, 150, 80, step=10)

    st.divider()
    split_option = st.selectbox("📊 Train/Val/Test Split", ["70/15/15", "80/10/10"], index=0)
    train_pct, val_pct = {"70/15/15": (0.70, 0.15), "80/10/10": (0.80, 0.10)}[split_option]

    st.caption("💡 CASH triggered automatically on 2-day drawdown ≤ −15%")
    st.divider()
    run_button = st.button("🚀 Run All 3 Approaches", type="primary", use_container_width=True)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🧠 P2-ETF-CNN-LSTM")
st.caption("Approach 1: Wavelet  ·  Approach 2: Regime-Conditioned  ·  Approach 3: Multi-Scale Parallel")
st.caption("Winner selected by highest raw annualised return on out-of-sample test set.")

if not HF_TOKEN:
    st.error("❌ HF_TOKEN secret not found.")
    st.stop()

# ── Load dataset ──────────────────────────────────────────────────────────────
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

# ── Run button ────────────────────────────────────────────────────────────────
if run_button:
    st.session_state.output_ready = False

    df = df_raw[df_raw.index.year >= start_yr].copy()
    st.write(f"📅 **Data:** {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')} "
             f"({df.index[-1].year - df.index[0].year + 1} years)")

    try:
        input_features, target_etfs, tbill_rate, df, _ = get_features_and_targets(df)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    n_etfs    = len(target_etfs)
    n_classes = n_etfs   # CASH is overlay only — model always picks from ETFs

    st.info(
        f"🎯 **Targets:** {', '.join([t.replace('_Ret','') for t in target_etfs])}  ·  "
        f"**Features:** {len(input_features)} signals  ·  "
        f"**T-bill:** {tbill_rate*100:.2f}%"
    )

    # ── Raw arrays ────────────────────────────────────────────────────────────
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

    last_date_str = str(freshness.get("last_date_in_data", "unknown"))

    # ── Auto-select lookback ──────────────────────────────────────────────────
    lb_key    = make_cache_key(last_date_str, start_yr, fee_bps, int(epochs),
                                split_option, False, 0)
    lb_cached = load_cache(f"lb_{lb_key}")

    if lb_cached is not None:
        optimal_lookback = lb_cached["optimal_lookback"]
        st.success(f"⚡ Cache hit · Optimal lookback: **{optimal_lookback}d**")
    else:
        with st.spinner("🔍 Auto-selecting optimal lookback (30 / 45 / 60d)..."):
            optimal_lookback = find_best_lookback(
                X_raw, y_raw,
                train_pct, val_pct, n_classes,
                candidates=[30, 45, 60],
            )
        save_cache(f"lb_{lb_key}", {"optimal_lookback": optimal_lookback})
        st.success(f"📐 Optimal lookback: **{optimal_lookback}d** (auto-selected from 30/45/60)")

    lookback = optimal_lookback

    # ── Check model cache ─────────────────────────────────────────────────────
    cache_key   = make_cache_key(last_date_str, start_yr, fee_bps, int(epochs),
                                  split_option, False, lookback)
    cached_data = load_cache(cache_key)

    if cached_data is not None:
        results      = cached_data["results"]
        trained_info = cached_data["trained_info"]
        test_dates   = pd.DatetimeIndex(cached_data["test_dates"])
        test_slice   = cached_data["test_slice"]
        st.success("⚡ Results loaded from cache — no retraining needed.")
    else:
        X_seq, y_seq = build_sequences(X_raw, y_raw, lookback)
        y_labels     = returns_to_labels(y_seq)

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

        results, trained_info = {}, {}
        progress = st.progress(0, text="Training Approach 1...")

        for approach, train_fn, predict_fn, train_kwargs in [
            ("Approach 1",
             lambda: train_approach1(X_train_s, y_train_l, X_val_s, y_val_l,
                                     n_classes=n_classes, epochs=int(epochs)),
             lambda m: predict_approach1(m[0], X_test_s),
             None),
            ("Approach 2",
             lambda: train_approach2(X_train_s, y_train_l, X_val_s, y_val_l,
                                     X_flat_all=X_raw, feature_names=input_features,
                                     lookback=lookback, train_size=train_size,
                                     val_size=val_size, n_classes=n_classes,
                                     epochs=int(epochs)),
             lambda m: predict_approach2(m[0], X_test_s, X_raw, m[3], m[2],
                                          lookback, train_size, val_size),
             None),
            ("Approach 3",
             lambda: train_approach3(X_train_s, y_train_l, X_val_s, y_val_l,
                                     n_classes=n_classes, epochs=int(epochs)),
             lambda m: predict_approach3(m[0], X_test_s),
             None),
        ]:
            try:
                model_out    = train_fn()
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

    # ── Persist to session state ──────────────────────────────────────────────
    st.session_state.update({
        "results": results, "trained_info": trained_info,
        "test_dates": test_dates, "test_slice": test_slice,
        "optimal_lookback": optimal_lookback, "df_for_chart": df,
        "tbill_rate": tbill_rate, "target_etfs": target_etfs,
        "output_ready": True,
    })

# ── Render (persists across reruns via session_state) ─────────────────────────
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

winner_name = select_winner(results)
winner_res  = results.get(winner_name)

if winner_res is None:
    st.error("❌ All approaches failed.")
    st.stop()

if st.session_state.from_cache:
    st.success("⚡ Showing cached results.")

next_date = get_next_signal_date()
st.divider()

show_signal_banner(winner_res["next_signal"], next_date, winner_name)

winner_proba = trained_info[winner_name]["proba"]
conviction   = compute_conviction(winner_proba[-1], target_etfs, include_cash=False)
show_conviction_panel(conviction)

st.divider()

all_signals = {
    name: {"signal": res["next_signal"],
           "proba":  trained_info[name]["proba"][-1],
           "is_winner": name == winner_name}
    for name, res in results.items() if res is not None
}
show_all_signals_panel(all_signals, target_etfs, False, next_date, optimal_lookback)

st.divider()
st.subheader(f"📊 {winner_name} — Performance Metrics")

# Compute SPY annualised return directly from raw returns for metrics comparison
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
