# TechnicalAnalysis

Testing whether chart-based technical analysis (TA) carries measurable signal in BTC,
using hourly OHLCV data and machine learning.

## Goal

Test TA strategies on historical BTC OHLCV data. Rather than asking "does TA predict
direction?", we ask whether price *reacts* at TA-derived levels, and whether that
reaction scales with how obvious the level is.

## Thesis

TA is self-fulfilling. The more a level on a chart is watched, the more significance it
carries, because more participants place orders around it. If that is true, the thing to
measure is not directional prediction but **reaction rate at levels, scaling with
obviousness**. A flat response across obviousness tiers means the "signal" is noise,
regardless of how the aggregate statistics look.

Prior work in this direction:

- Lo, Mamaysky & Wang (2000), *Foundations of Technical Analysis* — formalises chart
  patterns via kernel regression and tests whether they carry information.
- Osler (2000, 2003) on FX round numbers — documents the self-fulfilling mechanism
  directly in order flow: stop-loss and take-profit orders cluster at obvious levels.

## Idea

### Pivots as the primitive

We detect swing points algorithmically. A bar is a **pivot high** if its high exceeds the
highs of the N bars on either side; a **pivot low** is defined symmetrically. We run at
three scales:

| N  | Tier         |
|----|--------------|
| 5  | minor        |
| 20 | intermediate |
| 50 | major        |

The tier is our proxy for obviousness: a major pivot is visible on every chart, a minor
one only on short timeframes.

### Confirmation lag

A pivot at time *t* is not knowable until *t + N*. All pivots are timestamped at
**confirmation**, not occurrence, so no feature can see into the future.

### Strategies under test

Each strategy is built from confirmed pivots and emits numeric features per bar:

1. **Horizontal levels** (support / resistance)
2. **Trendlines**
3. **Fibonacci retracements / extensions** from OHLC swings

Features include:

- Distance to nearest support / resistance, in ATR units
- Touch count on that level
- Trendline slope
- Breakout decisiveness (close-through magnitude, volume confirmation)
- Position within the current Fibonacci range
- Confluence count across timeframes / tiers

### Model

LightGBM, chosen for speed so we can iterate on feature design quickly and get an early
read on whether the idea has merit.

### Validation

A random train/test split is invalid here: adjacent windows share ~99% of their inputs,
so the model would just memorise. We use **purged walk-forward validation with an
embargo of at least the prediction horizon** between train and test folds.

## Project layout

```
.
├── data/                        # SQLite store (gitignored)
│   └── technical_analysis.sqlite
├── src/
│   ├── api/
│   │   └── binance.py           # BTC hourly OHLCV fetch + SQLite cache
│   └── features/
│       └── pivots.py            # N-bar pivot detection, stamped at confirmation
├── tests/
│   └── test_pivots.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Hourly BTC/USDT OHLCV from the Binance public klines API, going back 10 years (Binance
spot BTCUSDT history starts August 2017, so in practice the series starts there).

```bash
python -m src.api.binance
```

The first run fetches from Binance and writes to `data/technical_analysis.sqlite`, table
`prices`. Later calls (or `load_prices()` from Python) read from the database and only
hit the API to fill gaps or extend the series to the present.

## Pivots

```bash
python -m src.features.pivots --check   # summary per tier, verified against brute force
```

`pivot_table(df)` returns one row per (swing, N) with both the occurrence bar and the
confirmation bar (`confirm_idx = idx + N`). `known_pivots(piv, t)` gives the swings a
chart-watcher could see at the close of bar `t`, with each swing's tier as it was known
*then*; because the tiers confirm at different times, a swing's tier upgrades over time.
`pivot_events(df, piv)` is the wide, time-aligned view stamped at confirmation, which is
the form the feature builders consume.

## Tests

```bash
python -m pytest tests/
```
