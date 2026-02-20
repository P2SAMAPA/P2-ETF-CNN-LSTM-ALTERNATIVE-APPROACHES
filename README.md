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
