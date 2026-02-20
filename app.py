"""
app.py
P2-ETF-CNN-LSTM-ALTERNATIVE-APPROACHES
Streamlit orchestrator — UI wiring only, no business logic here.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

# ── Module imports ────────────────────────────────────────────────────────────
from data.loader      import load_dataset, check_data_freshness, get_features_and_targets, dataset_summary
from utils.calendar   import get_est_time, is_sync_window, get_next_signal_date
from models.base      import build_sequences, train_val_test_split, scale_features, returns_to_labels
from models.approach1_wavelet    import train_approach1, predict_approach1
from models.approach2_regime     import train_approach2, predict_approach2
from models.approach3_multiscale import train_approach3, predict_approach3
from strategy.backtest  import execute_strategy, select_winner, build_comparison_table
from signals.conviction import compute_conviction
from ui.components import (
    show_freshness_status, show_signal_banner, show_conviction_panel,
    show_metrics_row, show_comparison_table, show_audit_trail,
)
from ui.charts import equity_curve_chart, comparison_bar_chart

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="P2-ETF-CNN-LSTM",
    page_icon="🧠",
    layout="wide",
)

# ── Secrets ───────────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    now_est = get_est_time()
    st.write(f"🕒 **EST:** {now_est.strftime('%H:%M:%S')}")
    if is_sync_window():
        st.success("✅ Sync Window Active")
    else:
        st.info("⏸️ Sync Window Inactive")

    st.divider()

    start_yr = st.slider("📅 Start Year", 2010, 2024, 2016)
    fee_bps  = st.slider("💰 Fee (bps)", 0, 50, 10)
    lookback = st.slider("📐 Lookback (days)", 20, 60, 30, step=5)
    epochs   = st.number_input("🔁 Max Epochs", 20, 300, 100, step=10)

    st.divider()

    split_option = st.selectbox("📊 Train/Val/Test Split", ["70/15/15", "80/10/10"], index=0)
    split_map    = {"70/15/15": (0.70, 0.15), "80/10/10": (0.80, 0.10)}
    train_pct, val_pct = split_map[split_option]

    include_cash = st.checkbox("💵 Include CASH class", value=True,
                               help="Model can select CASH (earns T-bill rate) as an alternative to any ETF")

    st.divider()

    run_button = st.button("🚀 Run All 3 Approaches", type="primary", use_container_width=True)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🧠 P2-ETF-CNN-LSTM")
st.caption("Approach 1: Wavelet  ·  Approach 2: Regime-Conditioned  ·  Approach 3: Multi-Scale Parallel")
st.caption("Winner selected by highest raw annualised return on out-of-sample test set.")

# ── Load data (always, to check freshness) ────────────────────────────────────
if not HF_TOKEN:
    st.error("❌ HF_TOKEN secret not found. Please add it to your HF Space / GitHub secrets.")
    st.stop()

with st.spinner("📡 Loading dataset from HuggingFace..."):
    df = load_dataset(HF_TOKEN)

if df.empty:
    st.stop()

# ── Freshness check ───────────────────────────────────────────────────────────
freshness = check_data_freshness(df)
show_freshness_status(freshness)

# ── Dataset summary in sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.subheader("📦 Dataset Info")
    summary = dataset_summary(df)
    if summary:
        st.write(f"**Rows:** {summary['rows']:,}")
        st.write(f"**Range:** {summary['start_date']} → {summary['end_date']}")
        st.write(f"**ETFs:** {', '.join([e.replace('_Ret','') for e in summary['etfs_found']])}")
        st.write(f"**Benchmarks:** {', '.join([b.replace('_Ret','') for b in summary['benchmarks']])}")
        st.write(f"**T-bill col:** {'✅' if summary['tbill_found'] else '❌'}")

# ── Main execution ────────────────────────────────────────────────────────────
if not run_button:
    st.info("👈 Configure parameters in the sidebar and click **🚀 Run All 3 Approaches** to begin.")
    st.stop()

# ── Filter by start year ──────────────────────────────────────────────────────
df = df[df.index.year >= start_yr].copy()
st.write(f"📅 **Data:** {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}  "
         f"({df.index[-1].year - df.index[0].year + 1} years)")

# ── Feature / target extraction ───────────────────────────────────────────────
try:
    input_features, target_etfs, tbill_rate = get_features_and_targets(df)
except ValueError as e:
    st.error(str(e))
    st.stop()

st.info(f"🎯 **Targets:** {len(target_etfs)} ETFs  ·  **Features:** {len(input_features)} signals  ·  "
        f"**T-bill rate:** {tbill_rate*100:.2f}%")

# ── Prepare sequences ─────────────────────────────────────────────────────────
X_raw    = df[input_features].values.astype(np.float32)
y_raw    = df[target_etfs].values.astype(np.float32)
n_etfs   = len(target_etfs)
n_classes = n_etfs + (1 if include_cash else 0)   # +1 for CASH

# Fill NaNs with column means
col_means = np.nanmean(X_raw, axis=0)
for j in range(X_raw.shape[1]):
    mask = np.isnan(X_raw[:, j])
    X_raw[mask, j] = col_means[j]

X_seq, y_seq = build_sequences(X_raw, y_raw, lookback)
y_labels     = returns_to_labels(y_seq, include_cash=include_cash)

X_train, y_train_r, X_val, y_val_r, X_test, y_test_r = train_val_test_split(X_seq, y_seq, train_pct, val_pct)
_, y_train_l, _, y_val_l, _, y_test_l                 = train_val_test_split(X_seq, y_labels, train_pct, val_pct)

X_train_s, X_val_s, X_test_s, _ = scale_features(X_train, X_val, X_test)

train_size = len(X_train)
val_size   = len(X_val)

# Test dates (aligned with y_test)
test_start  = lookback + train_size + val_size
test_dates  = df.index[test_start: test_start + len(X_test)]
test_slice  = slice(test_start, test_start + len(X_test))

st.success(f"✅ Sequences — Train: {train_size} · Val: {val_size} · Test: {len(X_test)}")

# ── Train all three approaches ────────────────────────────────────────────────
results      = {}
trained_info = {}   # store extra info needed for conviction

progress = st.progress(0, text="Starting training...")

# ── Approach 1: Wavelet ───────────────────────────────────────────────────────
with st.spinner("🌊 Training Approach 1 — Wavelet CNN-LSTM..."):
    try:
        model1, hist1, _ = train_approach1(
            X_train_s, y_train_l,
            X_val_s,   y_val_l,
            n_classes=n_classes, epochs=int(epochs),
        )
        preds1, proba1 = predict_approach1(model1, X_test_s)
        results["Approach 1"] = execute_strategy(
            preds1, proba1, y_test_r, test_dates, target_etfs, fee_bps, tbill_rate, include_cash,
        )
        trained_info["Approach 1"] = {"proba": proba1}
        st.success("✅ Approach 1 complete")
    except Exception as e:
        st.warning(f"⚠️ Approach 1 failed: {e}")
        results["Approach 1"] = None

progress.progress(33, text="Approach 1 done...")

# ── Approach 2: Regime-Conditioned ───────────────────────────────────────────
with st.spinner("🔀 Training Approach 2 — Regime-Conditioned CNN-LSTM..."):
    try:
        model2, hist2, hmm2, regime_cols2 = train_approach2(
            X_train_s, y_train_l,
            X_val_s,   y_val_l,
            X_flat_all=X_raw,
            feature_names=input_features,
            lookback=lookback,
            train_size=train_size,
            val_size=val_size,
            n_classes=n_classes, epochs=int(epochs),
        )
        preds2, proba2 = predict_approach2(
            model2, X_test_s, X_raw, regime_cols2, hmm2,
            lookback, train_size, val_size,
        )
        results["Approach 2"] = execute_strategy(
            preds2, proba2, y_test_r, test_dates, target_etfs, fee_bps, tbill_rate, include_cash,
        )
        trained_info["Approach 2"] = {"proba": proba2}
        st.success("✅ Approach 2 complete")
    except Exception as e:
        st.warning(f"⚠️ Approach 2 failed: {e}")
        results["Approach 2"] = None

progress.progress(66, text="Approach 2 done...")

# ── Approach 3: Multi-Scale ───────────────────────────────────────────────────
with st.spinner("📡 Training Approach 3 — Multi-Scale CNN-LSTM..."):
    try:
        model3, hist3 = train_approach3(
            X_train_s, y_train_l,
            X_val_s,   y_val_l,
            n_classes=n_classes, epochs=int(epochs),
        )
        preds3, proba3 = predict_approach3(model3, X_test_s)
        results["Approach 3"] = execute_strategy(
            preds3, proba3, y_test_r, test_dates, target_etfs, fee_bps, tbill_rate, include_cash,
        )
        trained_info["Approach 3"] = {"proba": proba3}
        st.success("✅ Approach 3 complete")
    except Exception as e:
        st.warning(f"⚠️ Approach 3 failed: {e}")
        results["Approach 3"] = None

progress.progress(100, text="All approaches complete!")
progress.empty()

# ── Select winner ─────────────────────────────────────────────────────────────
winner_name = select_winner(results)
winner_res  = results.get(winner_name)

if winner_res is None:
    st.error("❌ All approaches failed. Please check your data and configuration.")
    st.stop()

# ── Next trading date ─────────────────────────────────────────────────────────
next_date = get_next_signal_date()

st.divider()

# ── Signal banner (winner) ────────────────────────────────────────────────────
show_signal_banner(winner_res["next_signal"], next_date, winner_name)

# ── Conviction panel ──────────────────────────────────────────────────────────
winner_proba = trained_info[winner_name]["proba"]
conviction   = compute_conviction(winner_proba[-1], target_etfs, include_cash)
show_conviction_panel(conviction)

st.divider()

# ── Winner metrics ────────────────────────────────────────────────────────────
st.subheader(f"📊 {winner_name} — Performance Metrics")
show_metrics_row(winner_res, tbill_rate)

st.divider()

# ── Comparison table ──────────────────────────────────────────────────────────
st.subheader("🏆 Approach Comparison (Winner = Highest Raw Annualised Return)")
comparison_df = build_comparison_table(results, winner_name)
show_comparison_table(comparison_df)

# ── Comparison bar chart ──────────────────────────────────────────────────────
st.plotly_chart(comparison_bar_chart(results, winner_name), use_container_width=True)

st.divider()

# ── Equity curves ─────────────────────────────────────────────────────────────
st.subheader("📈 Out-of-Sample Equity Curves — All Approaches vs Benchmarks")
fig = equity_curve_chart(results, winner_name, test_dates, df, test_slice, tbill_rate)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Audit trail (winner) ──────────────────────────────────────────────────────
st.subheader(f"📋 Audit Trail — {winner_name} (Last 20 Trading Days)")
show_audit_trail(winner_res["audit_trail"])
