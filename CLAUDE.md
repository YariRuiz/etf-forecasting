# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ETF price forecasting system using a hybrid Conv1D + LSTM neural network. Originally developed as a master's thesis prototype in AI, now being refactored for production. The model predicts the next day's closing price from a 60-day sliding window of normalized prices.

**Roadmap:** refactor core → improve model → REST backend → React frontend → tests → deploy.

## Environment Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Common Commands

```bash
# Run a module directly to verify the pipeline step works
python -m src.data_loader
python -m src.preprocessor

# Launch Jupyter to explore the original notebook
jupyter notebook notebooks/Prototipo_ETF_TFE_V4.ipynb
```

## Architecture

The pipeline is: **load → normalize → window → train → denormalize → evaluate**

```
src/
  data_loader.py    — load_etf_data(filepath, etf_symbol) → np.ndarray of close prices
  preprocessor.py   — normalize() / denormalize() (min-max [0,1]) + create_windows(data, window_size)
  lstm_model.py     — LSTMModel class: Conv1D(128) → LSTM(256) → Dropout → LSTM(256) → Dropout → Dense(128) → Dense(1)
```

**Data:** `data/etf-prices.csv` (193 MB, gitignored). Columns: `fund_symbol`, `price_date`, `open`, `high`, `low`, `close`, `adj_close`, `volume`. Filter by `fund_symbol` to get a single ETF's time series.

**Model defaults:** window_size=60, lstm_units=256, dropout=0.3, lr=1e-4. Loss: Huber. Optimizer: Adam. The notebook used ETF `FEUZ` and achieved validation MAE ≈ 40.67.

**Normalization contract:** `normalize()` returns `(normalized_array, min_val, max_val)`. Always carry `min_val`/`max_val` through the pipeline — they are required by `denormalize()` to recover real price scale.

**GAN exploration:** The notebook (`Prototipo_ETF_TFE_V4.ipynb`) contains a GAN for synthetic ETF data generation (cells 21–27). This is experimental and not yet ported to `src/`.

## Known Bugs to Fix

- `lstm_model.py` line 33–35: `return_sequencies` is a typo — must be `return_sequences` (causes a `TypeError` at build time).
- `Conv1D` uses `stride=` (invalid kwarg) — correct parameter is `strides=`.

## Key Decisions from the Thesis Prototype

- **Causal padding** on Conv1D prevents data leakage from future timesteps.
- **Huber loss** instead of MSE for robustness to price outliers.
- **EarlyStopping(patience=15)** and a learning rate scheduler were used in the notebook; these are not yet in `LSTMModel`.
- Train/validation split was 80/20 (no shuffle — preserves temporal order).
