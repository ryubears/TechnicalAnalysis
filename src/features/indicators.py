"""
Small causal indicators shared by the strategy modules.

Everything here is computed from bars at or before ``t`` only, so it can be used
freely in features without introducing lookahead.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    Wilder's true range: max(high - low, |high - prev_close|, |low - prev_close|).

    The first bar uses high - low since it has no previous close.
    """
    high = np.asarray(high, dtype="float64")
    low = np.asarray(low, dtype="float64")
    close = np.asarray(close, dtype="float64")
    prev_close = np.empty_like(close)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    tr[0] = high[0] - low[0]
    return tr

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    """
    Wilder's average true range with smoothing factor 1/n, seeded from the first bar.

    Returned as a float array aligned to the inputs. Unlike a rolling mean it has no
    NaN warm-up, but the first few dozen values are still dominated by the seed.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    tr = true_range(high, low, close)
    return pd.Series(tr).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()

def rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    """
    Trailing mean over the previous ``n`` bars including the current one; NaN during warm-up.
    """
    return pd.Series(np.asarray(x, dtype="float64")).rolling(n, min_periods=n).mean().to_numpy()
