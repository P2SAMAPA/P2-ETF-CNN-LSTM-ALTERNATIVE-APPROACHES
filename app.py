"""
app.py
P2-ETF-CNN-LSTM-ALTERNATIVE-APPROACHES
Streamlit orchestrator — UI wiring only, no business logic here.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

from data.loader      import (load_dataset, check_data_freshness,
                               get_features_and_targets, dataset_summary)
from utils.calendar   import get_est_time, is_sync_window, get_next_signal_date
from models.base      import (build_sequences, train_val_test_split,
                               scale_features, returns_to_labels)
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

st.set_page_config(page_title="P2-ETF-CNN-LSTM", page_icon="🧠", layout="wide")

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
    start_yr     = st.slider("📅 Start Year", 2010, 2024, 2016)
    fee_bps      = st.slider("💰 Fee (bps)", 0, 50, 10)
    lookback     = st.slider("📐 Lookback (days)", 20, 60, 30, step=5)
    epochs       = st.number_input("🔁 Max Epochs", 20, 300, 100, step=10)

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

        with st.expander("🔍 All columns"):
            st.write(summary["all_cols"])

if not run_button:
    st.info("👈 Configure parameters and click **🚀 Run All 3 Approaches**.")
    st.stop()

# ── Filter by start year ──────────────────────────────────────────────────────
df = df_raw[df_raw.index.year >= start_yr].copy()
st.write(f"📅 **Data:** {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')} "
         f"({df.index[-1].year - df.index[0].year + 1} years)")

# ── Features & targets ────────────────────────────────────────────────────────
try:
    input_features, target_etfs, tbill_rate, df, col_info = get_features_and_targets(df)
except ValueError as e:
    st.error(str(e))
    st.stop()

n_etfs    = len(target_etfs)
n_classes = n_etfs + (1 if include_cash else 0)

# ── Show column detection diagnostics ────────────────────────────────────────
with st.expander("🔬 Column detection diagnostics", expanded=False):
    st.write("**How each ETF column was interpreted:**")
    for col, info in col_info.items():
        st.write(f"- `{col}`: {info}")
    st.write(f"**Input features ({len(input_features)}):** {input_features}")
    st.write(f"**T-bill rate used:** {tbill_rate*100:.3f}%")

    # Show sample return values to verify correctness
    st.write("**Sample target return values (last 3 rows):**")
    st.dataframe(df[target_etfs].tail(3))

st.info(
    f"🎯 **Targets:** {', '.join([t.replace('_Ret','') for t in target_etfs])}  ·  "
    f"**Features:** {len(input_features)} signals  ·  "
    f"**T-bill:** {tbill_rate*100:.2f}%"
)

# ── Build sequences ───────────────────────────────────────────────────────────
X_raw = df[input_features].values.astype(np.float32)
y_raw = df[target_etfs].values.astype(np.float32)

# Fill NaNs
col_means = np.nanmean(X_raw, axis=0)
for j in range(X_raw.shape[1]):
    mask = np.isnan(X_raw[:, j])
    if mask.any():
        X_raw[mask, j] = col_means[j]

# Also fill NaNs in y_raw
y_means = np.nanmean(y_raw, axis=0)
for j in range(y_raw.shape[1]):
    mask = np.isnan(y_raw[:, j])
    if mask.any():
        y_raw[mask, j] = y_means[j]

X_seq, y_seq = build_sequences(X_raw, y_raw, lookback)
y_labels     = returns_to_labels(y_seq, include_cash=include_cash)

(X_train, y_train_r, X_val, y_val_r,
 X_test,  y_test_r)  = train_val_test_split(X_seq, y_seq,    train_pct, val_pct)
(_,       y_train_l,  _,    y_val_l,
 _,       y_test_l)  = train_val_test_split(X_seq, y_labels, train_pct, val_pct)

X_train_s, X_val_s, X_test_s, _ = scale_features(X_train, X_val, X_test)

train_size = len(X_train)
val_size   = len(X_val)
test_start = lookback + train_size + val_size
test_dates = df.index[test_start: test_start + len(X_test)]
test_slice = slice(test_start, test_start + len(X_test))

st.success(f"✅ Sequences — Train: {train_size:,} · Val: {val_size:,} · Test: {len(X_test):,}")

# Show class distribution to check for degenerate labels
with st.expander("🔬 Label distribution (train set)", expanded=False):
    unique, counts = np.unique(y_train_l, return_counts=True)
    label_names = [target_etfs[i].replace("_Ret","") if i < n_etfs else "CASH" for i in unique]
    dist_df = pd.DataFrame({"Class": label_names, "Count": counts,
                             "Pct": (counts / counts.sum() * 100).round(1)})
    st.dataframe(dist_df)

# ── Train all three approaches ────────────────────────────────────────────────
results      = {}
trained_info = {}
progress     = st.progress(0, text="Starting training...")

# Approach 1
with st.spinner("🌊 Training Approach 1 — Wavelet CNN-LSTM..."):
    try:
        model1, hist1, _ = train_approach1(
            X_train_s, y_train_l, X_val_s, y_val_l,
            n_classes=n_classes, epochs=int(epochs),
        )
        preds1, proba1 = predict_approach1(model1, X_test_s)
        results["Approach 1"] = execute_strategy(
            preds1, proba1, y_test_r, test_dates,
            target_etfs, fee_bps, tbill_rate, include_cash,
        )
        trained_info["Approach 1"] = {"proba": proba1}
        st.success("✅ Approach 1 complete")
    except Exception as e:
        st.warning(f"⚠️ Approach 1 failed: {e}")
        results["Approach 1"] = None

progress.progress(33, text="Approach 1 done...")

# Approach 2
with st.spinner("🔀 Training Approach 2 — Regime-Conditioned CNN-LSTM..."):
    try:
        model2, hist2, hmm2, regime_cols2 = train_approach2(
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
        st.success("✅ Approach 2 complete")
    except Exception as e:
        st.warning(f"⚠️ Approach 2 failed: {e}")
        results["Approach 2"] = None

progress.progress(66, text="Approach 2 done...")

# Approach 3
with st.spinner("📡 Training Approach 3 — Multi-Scale CNN-LSTM..."):
    try:
        model3, hist3 = train_approach3(
            X_train_s, y_train_l, X_val_s, y_val_l,
            n_classes=n_classes, epochs=int(epochs),
        )
        preds3, proba3 = predict_approach3(model3, X_test_s)
        results["Approach 3"] = execute_strategy(
            preds3, proba3, y_test_r, test_dates,
            target_etfs, fee_bps, tbill_rate, include_cash,
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
    st.error("❌ All approaches failed. Please check data and configuration.")
    st.stop()

next_date = get_next_signal_date()
st.divider()

show_signal_banner(winner_res["next_signal"], next_date, winner_name)

winner_proba = trained_info[winner_name]["proba"]
conviction   = compute_conviction(winner_proba[-1], target_etfs, include_cash)
show_conviction_panel(conviction)

st.divider()
st.subheader(f"📊 {winner_name} — Performance Metrics")
show_metrics_row(winner_res, tbill_rate)

st.divider()
st.subheader("🏆 Approach Comparison (Winner = Highest Raw Annualised Return)")
comparison_df = build_comparison_table(results, winner_name)
show_comparison_table(comparison_df)
st.plotly_chart(comparison_bar_chart(results, winner_name), use_container_width=True)

st.divider()
st.subheader("📈 Out-of-Sample Equity Curves — All Approaches vs Benchmarks")
fig = equity_curve_chart(results, winner_name, test_dates, df, test_slice, tbill_rate)
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader(f"📋 Audit Trail — {winner_name} (Last 20 Trading Days)")
show_audit_trail(winner_res["audit_trail"])
