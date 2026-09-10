"""
Purged walk-forward splits with an embargo.

Adjacent hourly windows share almost all of their inputs, so a random split would let
the model memorise. Instead the series is cut into consecutive test blocks in time
order. For each block, training uses only events whose entire label horizon ends
before the block starts minus an embargo:

    train:  t + horizon < test_start - embargo
    test:   test_start <= t < test_end

The purge (dropping training events whose horizon would overlap the test block) is
therefore structural, and the embargo adds a further gap so that serial correlation
in the features cannot bleed across the boundary. ``embargo`` defaults to the
horizon and may not be smaller than it.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Fold:
    k: int
    test_start: int
    test_end: int
    train_idx: np.ndarray     # positions into the event array
    test_idx: np.ndarray

def purged_walk_forward(
    t: np.ndarray,
    n_bars: int,
    horizon: int,
    embargo: int | None = None,
    test_bars: int = 24 * 180,
    min_train_bars: int = 24 * 365,
    min_train_events: int = 200,
    min_test_events: int = 20,
) -> list[Fold]:
    """
    Build folds over event times ``t`` (bar indices) for a series of ``n_bars``.

    Test blocks are ``test_bars`` long, starting at ``min_train_bars`` and stepping
    forward without overlap. Folds with too few training or test events are skipped.
    """
    t = np.asarray(t, dtype=int)
    if embargo is None:
        embargo = horizon
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if embargo < horizon:
        raise ValueError(f"embargo ({embargo}) must be at least the horizon ({horizon})")
    if test_bars < 1 or min_train_bars < 1:
        raise ValueError("test_bars and min_train_bars must be >= 1")
    folds: list[Fold] = []
    k = 0
    for s in range(min_train_bars, n_bars, test_bars):
        e = min(s + test_bars, n_bars)
        train = np.flatnonzero(t + horizon < s - embargo)
        test = np.flatnonzero((t >= s) & (t < e))
        if train.size < min_train_events or test.size < min_test_events:
            continue
        folds.append(Fold(k, s, e, train, test))
        k += 1
    return folds

def embargoed_tail_split(t: np.ndarray, horizon: int, embargo: int, frac: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """
    Split one training set, by time, into a core and a validation tail for early
    stopping, with the same purge and embargo between them. Returns positions.
    """
    t = np.asarray(t, dtype=int)
    if t.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    cut = int(np.quantile(t, 1 - frac))
    core = np.flatnonzero(t + horizon < cut - embargo)
    tail = np.flatnonzero(t >= cut)
    return core, tail
