---
title: P2-ETF-CNN-LSTM-ALTERNATIVE-APPROACHES
emoji: 🧠
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.32.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# P2-ETF-CNN-LSTM-ALTERNATIVE-APPROACHES

Macro-driven ETF rotation using three augmented CNN-LSTM variants.  
Winner selected by **highest raw annualised return** on the out-of-sample test set.

---

## Architecture Overview

| Approach | Core Idea | Key Addition |
|---|---|---|
| **1 — Wavelet** | DWT decomposes each macro signal into frequency subbands before the CNN | Separates trend / cycle / noise |
| **2 — Regime-Conditioned** | HMM detects macro regimes; one-hot regime label concatenated into the network | Removes non-stationarity |
| **3 — Multi-Scale Parallel** | Three CNN towers (kernels 3, 7, 21 days) run in parallel before the LSTM | Captures momentum + cycle + trend simultaneously |

---

## ETF Universe

| Ticker | Description |
|---|---|
| TLT | 20+ Year Treasury Bond |
| TBT | 20+ Year Treasury Short (2×) |
| VNQ | Real Estate (REIT) |
| SLV | Silver |
| GLD | Gold |
| CASH | 3m T-bill rate (from HF dataset) |

Benchmarks (chart only, not traded): **SPY**, **AGG**

---

## Data

All data sourced exclusively from:  
**`P2SAMAPA/fi-etf-macro-signal-master-data`** (HuggingFace Dataset)  
File: `master_data.parquet`

No external API calls (no yfinance, no FRED).  
The app checks daily whether the prior NYSE trading day's data is present in the dataset.

---

## Project Structure

```
├── .github/
│   └── workflows/
│       └── sync.yml            # Auto-sync GitHub → HF Space on push to main
│
├── app.py                      # Streamlit orchestrator (UI wiring only)
│
├── data/
│   └── loader.py               # HF dataset load, freshness check, column validation
│
├── models/
│   ├── base.py                 # Shared: sequences, splits, scaling, callbacks
│   ├── approach1_wavelet.py    # Wavelet CNN-LSTM
│   ├── approach2_regime.py     # Regime-Conditioned CNN-LSTM
│   └── approach3_multiscale.py # Multi-Scale Parallel CNN-LSTM
│
├── strategy/
│   └── backtest.py             # execute_strategy, metrics, winner selection
│
├── signals/
│   └── conviction.py           # Z-score conviction scoring
│
├── ui/
│   ├── components.py           # Banner, conviction panel, metrics, audit trail
│   └── charts.py               # Plotly equity curve + comparison bar chart
│
├── utils/
│   └── calendar.py             # NYSE calendar, next trading day, EST time
│
├── requirements.txt
└── README.md
```

---

## Secrets Required

| Secret | Where | Purpose |
|---|---|---|
| `HF_TOKEN` | GitHub + HF Space | Read HF dataset · Sync HF Space |

Set in:
- GitHub: `Settings → Secrets → Actions → New repository secret`
- HF Space: `Settings → Repository secrets`

---

## Deployment

Push to `main` → GitHub Actions (`sync.yml`) automatically syncs to HF Space.

### Local development

```bash
pip install -r requirements.txt
export HF_TOKEN=your_token
streamlit run app.py
```

---

## Output UI

1. **Data freshness warning** — alerts if prior NYSE trading day data is missing
2. **Next Trading Day Signal** — date + ETF from the winning approach
3. **Signal Conviction** — Z-score gauge + per-ETF probability bars
4. **Performance Metrics** — Annualised Return, Sharpe, Hit Ratio, Max DD
5. **Approach Comparison Table** — all three approaches side by side
6. **Equity Curves** — all three approaches + SPY + AGG benchmarks
7. **Audit Trail** — last 20 trading days for the winning approach

# MC Dropout Wrapper — P2-ETF-CNN-LSTM-ALTERNATIVE-APPROACHES

Adds epistemic uncertainty estimation to all three CNN-LSTM approaches via
**MC Dropout** — running N stochastic forward passes at inference time with
dropout layers kept active.

---

## Files in this patch

| File | Action | Description |
|------|--------|-------------|
| `models/mc_dropout.py` | **NEW** | Core MC inference engine for all 3 approaches |
| `signals/mc_conviction.py` | **NEW** | Uncertainty-adjusted conviction scorer |
| `ui/mc_components.py` | **NEW** | Streamlit UI panels for MC uncertainty display |
| `app.py` | **REPLACE** | Full app wired with MC toggle, falls back to original when off |

All existing files (`models/base.py`, `approach1_wavelet.py`, `approach2_regime.py`,
`approach3_multiscale.py`, `signals/conviction.py`, `strategy/`, `ui/components.py`,
`data/`, `utils/`) are **unchanged**.

---

## How to deploy

```bash
# 1. Drop the three new files into your repo
cp models/mc_dropout.py      your_repo/models/
cp signals/mc_conviction.py  your_repo/signals/
cp ui/mc_components.py       your_repo/ui/

# 2. Replace app.py
cp app.py your_repo/app.py

# 3. No new dependencies needed — uses tensorflow + numpy already in requirements.txt
```

---

## How it works

### Core mechanism (`models/mc_dropout.py`)

Standard inference calls `model.predict()` which internally calls `model(x, training=False)`,
turning dropout OFF. MC Dropout calls `model(x, training=True)` N times instead.
Each pass applies a different random dropout mask — the variance across passes is
the **epistemic uncertainty**.

```
N forward passes → [N, samples, classes] stack
                 → mean_proba  [samples, classes]   ← replaces single proba
                 → uncertainty [samples, classes]   ← std across passes
```

### Approach-specific behaviour

| Approach | What varies per pass | What is deterministic |
|----------|---------------------|-----------------------|
| **1 Wavelet** | CNN + LSTM dropout masks | Wavelet transform (computed once before loop) |
| **2 Regime** | CNN + LSTM dropout masks | HMM regime labels (deterministic) |
| **3 MultiScale** | 3 parallel CNN tower dropout masks (independent) | Nothing — richest uncertainty signal |

### Uncertainty-adjusted conviction (`signals/mc_conviction.py`)

```
unc_score   = 1 - clip(mean_σ / 0.5, 0, 1)   # 1 = certain, 0 = uncertain
adjusted_z  = z_score × unc_score             # penalised conviction
cash_flag   = adjusted_z < cash_threshold     # default threshold = 0.4
```

The `cash_threshold` default of 0.4 is a starting point — tune it by backtesting
across your 2008–2026 dataset. Higher threshold → more CASH days → lower drawdown
but potentially lower return.

---

## UI changes

### Sidebar additions
- **🎲 MC Dropout Uncertainty** toggle (default ON)
- **Passes slider** (10–100, default 50) — shown only when toggle is ON

### Conviction panel
When MC is ON, `show_conviction_panel()` is replaced by `show_mc_conviction_panel()`:
- 4 metric cards: Best ETF / Conviction Z / Adjusted Conviction / Uncertainty σ̄
- CASH override banner when `cash_flag = True`
- Side-by-side bars: mean probability (left) vs uncertainty per ETF (right)

### All-signals panel
`show_all_signals_panel()` replaced by `show_mc_all_signals_panel()` in MC mode —
shows uncertainty badge + adjusted conviction per approach.

---

## Performance on CPU (free tier)

| n_passes | Extra time per approach | Total extra (3 approaches) |
|----------|------------------------|---------------------------|
| 10       | ~50ms                  | ~150ms                     |
| 50       | ~250ms                 | ~750ms                     |
| 100      | ~500ms                 | ~1.5s                      |

Recommended: **50 passes**. Variance stabilises around 30; returns above 60 are
marginal. The 750ms overhead is negligible against training time.

---

## Cache behaviour

The cache key now includes `mc_enabled` and `n_passes`:

```python
cache_key = make_cache_key(
    f"{cache_prefix}_mc{int(mc_enabled)}_{n_passes}",
    start_yr, fee_bps, epochs, split_option, False, lookback
)
```

Toggling MC Dropout ON/OFF, or changing n_passes, automatically triggers a fresh
inference run (training is not repeated — models are retrained only when other
params change).

---

## Fallback behaviour

When MC Dropout is toggled OFF, the app reverts to the original
`predict_approach1/2/3` functions exactly as before. No behavioural change.

---

*Research only · Not financial advice · P2SAMAPA*
