"""
Assemble the per-bar feature matrix from the three strategies plus causal context.

`build_feature_matrix(df, method)` runs the pivot detector, then the horizontal-level,
trendline and Fibonacci builders, and concatenates their per-bar features with a few
context columns (volatility, recent returns). Everything is computed from bars at or
before ``t``, so the matrix can be joined to any forward-looking label without leakage.
"""

from __future__ import annotations
from src.features.fibonacci import build_fibonacci_features
from src.features.indicators import atr as _atr, rolling_mean
from src.features.levels import build_level_features
from src.features.pivots import DEFAULT_METHOD, default_ns, pivot_table
from src.features.trendlines import build_trendline_features
from typing import Sequence
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CONTEXT_COLUMNS = ["atr_pct", "ret_1_atr", "ret_4_atr", "ret_24_atr", "vol_ratio", "range_pos_24", "volume_ratio", "hour", "dow"]

def context_features(df: pd.DataFrame, atr_n: int = 14) -> pd.DataFrame:
    """
    Causal market-context columns: ATR as a fraction of price, recent returns in ATR
    units, ATR relative to its trailing week, position in the trailing day's range,
    volume relative to its trailing day, and clock features.
    """
    close = df["close"].to_numpy(dtype="float64")
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    volume = df["volume"].to_numpy(dtype="float64")
    a = _atr(high, low, close, atr_n)
    s = pd.Series(close, index=df.index)
    out = pd.DataFrame(index=df.index)
    out["atr_pct"] = a / close
    for h in (1, 4, 24):
        out[f"ret_{h}_atr"] = (s - s.shift(h)).to_numpy() / a
    out["vol_ratio"] = a / rolling_mean(a, 24 * 7)
    lo24 = pd.Series(low, index=df.index).rolling(24, min_periods=24).min()
    hi24 = pd.Series(high, index=df.index).rolling(24, min_periods=24).max()
    rng = (hi24 - lo24).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        out["range_pos_24"] = np.where(rng > 0, (close - lo24.to_numpy()) / rng, np.nan)
    out["volume_ratio"] = volume / rolling_mean(volume, 24)
    out["hour"] = df.index.hour
    out["dow"] = df.index.dayofweek
    return out

def build_feature_matrix(
    df: pd.DataFrame,
    method: str = DEFAULT_METHOD,
    ns: Sequence[int] | None = None,
    atr_n: int = 14,
    level_kw: dict | None = None,
    trendline_kw: dict | None = None,
    fib_kw: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Per-bar features from all three strategies for one pivot method.

    Returns ``(X, meta)`` where ``X`` is aligned to ``df.index`` and ``meta`` holds
    the method, its tier parameters ``ns`` and the strategy tables at the end of the
    series (levels, lines, swings) for inspection.
    """
    ns = tuple(ns) if ns is not None else default_ns(method)
    piv = pivot_table(df, ns=ns, method=method, atr_n=atr_n)
    lf, levels = build_level_features(df, piv, ns=ns, atr_n=atr_n, **(level_kw or {}))
    tf, lines = build_trendline_features(df, piv, ns=ns, atr_n=atr_n, **(trendline_kw or {}))
    ff, swings = build_fibonacci_features(df, piv, ns=ns, atr_n=atr_n, **(fib_kw or {}))
    ctx = context_features(df, atr_n)
    X = pd.concat([lf, tf, ff, ctx], axis=1)
    if X.columns.duplicated().any():
        raise RuntimeError(f"duplicate feature columns: {list(X.columns[X.columns.duplicated()])}")
    meta = {"method": method, "ns": ns, "n_pivots": int(len(piv)), "levels": levels, "lines": lines, "swings": swings}
    log.info("feature matrix %d x %d (%s, ns=%s, %d pivot rows)", *X.shape, method, ns, len(piv))
    return X, meta
