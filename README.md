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
│       ├── indicators.py        # ATR and other causal helpers
│       ├── pivots.py            # N-bar pivot detection, stamped at confirmation
│       └── levels.py            # strategy 1: horizontal S/R levels and their features
├── tests/
│   ├── test_pivots.py
│   └── test_levels.py
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

## Horizontal levels (strategy 1)

```bash
python -m src.features.levels
```

`build_level_features(df, piv)` replays the bars in order. When a pivot is confirmed its
price either joins an existing level within `merge_tol_atr` ATRs (one more touch) or
opens a new one; a later confirmation of the same swing at a larger N upgrades the
level's tier without adding a touch. Support versus resistance is decided per bar by
which side of the close the level sits on, and every close through a level is counted
as a break rather than deleting it.

Levels expire because a chart only shows a window: each swing stays visible for
`lookback[N]` bars after it occurred, with N its largest confirmed tier. The defaults
are one month for minor, six months for intermediate and two years for major swings.
Without this, nine years of swings blanket the price range and every bar sits within
half an ATR of some level.

Per-bar features, all distances in ATR units:

- `res_*` / `sup_*`: nearest level above / below the close with its distance, touch
  count, tier, number of distinct tiers (cross-timeframe confluence), age and break count
- `res_dist_atr_{N}` / `sup_dist_atr_{N}`: nearest level of tier at least N, for an
  obviousness-controlled comparison
- `n_levels_near`: levels within `near_band_atr` of the close
- `break_dir`, `break_mag_atr`, `break_vol_ratio`, `break_touches`, `break_tier`: set on
  bars where the close crossed a level since the previous close

## Tests

```bash
python -m pytest tests/
```
