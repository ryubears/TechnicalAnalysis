"""
Fibonacci retracement and extension levels from confirmed swings.

This is the third of the three strategies under test. For each pivot tier ``N`` the
*active swing* runs between the latest confirmed tier-``N`` pivot high and the
latest confirmed tier-``N`` pivot low; whichever came later is the swing's end. A
tier-``N`` swing therefore changes every time a new tier-``N`` pivot confirms, so a
minor swing is redrawn every few bars while a major swing persists for weeks. That
difference is the obviousness axis: nobody draws fibs on every minor leg, everyone
draws them on the major one.

Ratio levels
------------
Levels are measured as *retracement* from the swing end back toward its start::

    level(r) = end - r * (end - start)

so ``r = 0`` is the swing end, ``r = 1`` the swing start, ``0.618`` the classic
golden retracement, and negative ``r`` an extension beyond the end (``-0.272`` and
``-0.618`` are the 1.272 and 1.618 extensions). The ``retrace`` feature is the
close expressed in the same units, so comparing it with the ratio set tells you
which level price is sitting on.

A swing smaller than ``min_range_atr`` ATRs is not drawn: an outside bar that is
both a pivot high and a pivot low would otherwise define a zero-length swing whose
retracement units blow up. The previous swing of that tier stays in force until a
proper one forms.

Touches
-------
A confirmed pivot of any tier that lands within ``touch_tol_atr`` ATRs of a ratio
level, after the swing's end bar, counts as a touch on that level. Counters reset
when the swing changes. Because a minor swing rarely lives long enough to be
touched, minor-tier touch counts are mostly zero, which is the point.

Per-bar features
----------------
For each tier ``N``, prefixed ``fib_{N}_``:

``dir``, ``retrace``, ``range_atr``, ``swing_bars``, ``age_bars``
    +1 for an up-swing (low then high), -1 for a down-swing; the close in
    retracement units; the swing's size in ATRs; bars between its endpoints; bars
    since its end.
``res_dist_atr``, ``res_ratio``, ``res_touches``
    Nearest ratio level strictly above the close, its ratio, and its touch count.
``sup_dist_atr``, ``sup_ratio``, ``sup_touches``
    Same for the nearest level at or below the close.
``break_dir``, ``break_ratio``, ``break_mag_atr``, ``break_vol_ratio``
    Set on bars where the close crossed a ratio level since the previous close:
    +1 up, -1 down, the ratio crossed (nearest to the close if several), how far
    past it the close finished, and volume over its trailing mean.

Across tiers, ``fib_n_near`` counts ratio levels within ``near_band_atr`` of the
close, the cross-timeframe confluence.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from src.features.indicators import atr as _atr, rolling_mean
from src.features.pivots import DEFAULT_NS
from typing import Sequence
import argparse
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

RATIOS: tuple[float, ...] = (-0.618, -0.272, 0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618)

@dataclass
class Swing:
    """
    The active swing for one tier and the ratio levels drawn on it.
    """
    start_idx: int
    start_price: float
    end_idx: int
    end_price: float
    formed_idx: int                                 # bar at which both endpoints were confirmed
    levels: np.ndarray = field(default_factory=lambda: np.empty(0))
    touches: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))

    def __post_init__(self) -> None:
        rng = self.end_price - self.start_price
        self.levels = self.end_price - np.asarray(RATIOS) * rng
        self.touches = np.zeros(len(RATIOS), dtype=int)

    @property
    def dir(self) -> int:
        return 1 if self.end_price >= self.start_price else -1

    @property
    def range(self) -> float:
        return abs(self.end_price - self.start_price)

    def retrace(self, price: float) -> float:
        return (self.end_price - price) / (self.end_price - self.start_price)

class _SwingBook:
    """
    Latest confirmed high and low per tier, and the swing they define.
    """

    def __init__(self, ns: tuple[int, ...], touch_tol_atr: float, min_range_atr: float) -> None:
        self.ns = ns
        self.touch_tol_atr = touch_tol_atr
        self.min_range_atr = min_range_atr
        self.last: dict[int, dict[str, tuple[int, float]]] = {n: {} for n in ns}   # n -> kind -> (idx, price)
        self.swings: dict[int, Swing | None] = {n: None for n in ns}

    def add_pivot(self, idx: int, kind: str, n: int, price: float, t: int, atr_t: float) -> None:
        # 1. Touches on every tier's current levels, for pivots after that swing's end.
        tol = self.touch_tol_atr * atr_t
        for sw in self.swings.values():
            if sw is not None and idx > sw.end_idx:
                sw.touches += (np.abs(sw.levels - price) <= tol)
        # 2. Update this tier's endpoints; ignore a stale event for an older bar.
        if n not in self.last:
            return
        cur = self.last[n].get(kind)
        if cur is not None and cur[0] >= idx:
            return
        self.last[n][kind] = (idx, price)
        hi, lo = self.last[n].get("high"), self.last[n].get("low")
        if hi is None or lo is None:
            return
        if hi[0] >= lo[0]:
            start, end = lo, hi
        else:
            start, end = hi, lo
        if abs(end[1] - start[1]) < self.min_range_atr * atr_t:
            return
        self.swings[n] = Swing(start[0], start[1], end[0], end[1], formed_idx=t)

# --------------------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------------------

TIER_FIELDS = (
    "dir", "retrace", "range_atr", "swing_bars", "age_bars",
    "res_dist_atr", "res_ratio", "res_touches",
    "sup_dist_atr", "sup_ratio", "sup_touches",
    "break_dir", "break_ratio", "break_mag_atr", "break_vol_ratio",
)

def feature_columns(ns: tuple[int, ...] = DEFAULT_NS) -> list[str]:
    cols = [f"fib_{n}_{f}" for n in ns for f in TIER_FIELDS]
    cols.append("fib_n_near")
    return cols

def build_fibonacci_features(
    df: pd.DataFrame,
    pivots: pd.DataFrame,
    atr_n: int = 14,
    touch_tol_atr: float = 0.25,
    min_range_atr: float = 1.0,
    near_band_atr: float = 1.0,
    vol_n: int = 20,
    ns: Sequence[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build per-bar Fibonacci features from a pivot table.

    ``df`` is the OHLCV frame from `src.api.binance.load_prices`; ``pivots``
    the long table from `src.features.pivots.pivot_table` (any method); ``ns``
    default to the tiers present in it. Swings smaller than ``min_range_atr``
    ATRs are ignored. Returns
    ``(features, swings)``: features aligned to ``df.index`` (NaN for a tier until
    both a high and a low of that tier have confirmed) and the active swing per tier
    at the end of the series.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError("df must be sorted by time")
    if ns is None:
        ns = tuple(sorted(int(n) for n in pivots["n"].unique())) if len(pivots) else DEFAULT_NS
    ns = tuple(ns)
    L = len(df)
    close = df["close"].to_numpy(dtype="float64")
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    volume = df["volume"].to_numpy(dtype="float64")
    atr = _atr(high, low, close, atr_n)
    vol_ma = rolling_mean(volume, vol_n)

    ev = pivots.sort_values(["confirm_idx", "n"], kind="stable")
    ev_confirm = ev["confirm_idx"].to_numpy(dtype=int)
    ev_idx = ev["idx"].to_numpy(dtype=int)
    ev_kind = ev["kind"].astype(str).to_numpy()
    ev_n = ev["n"].to_numpy(dtype=int)
    ev_price = ev["price"].to_numpy(dtype="float64")

    cols = feature_columns(ns)
    out = np.full((L, len(cols)), np.nan)
    ci = {c: i for i, c in enumerate(cols)}
    out[:, ci["fib_n_near"]] = 0.0
    for n in ns:
        out[:, ci[f"fib_{n}_break_dir"]] = 0.0
    ratios = np.asarray(RATIOS)

    book = _SwingBook(tuple(ns), touch_tol_atr, min_range_atr)
    ptr = 0
    for t in range(L):
        a = atr[t]
        while ptr < ev_confirm.size and ev_confirm[ptr] == t:
            book.add_pivot(int(ev_idx[ptr]), str(ev_kind[ptr]), int(ev_n[ptr]), float(ev_price[ptr]), t, a)
            ptr += 1
        if not np.isfinite(a) or a <= 0:
            continue
        c = close[t]
        near = 0
        for n in ns:
            sw = book.swings[n]
            if sw is None:
                continue
            p = f"fib_{n}_"
            out[t, ci[p + "dir"]] = sw.dir
            out[t, ci[p + "retrace"]] = sw.retrace(c)
            out[t, ci[p + "range_atr"]] = sw.range / a
            out[t, ci[p + "swing_bars"]] = abs(sw.end_idx - sw.start_idx)
            out[t, ci[p + "age_bars"]] = t - sw.end_idx

            diff = sw.levels - c
            near += int(np.sum(np.abs(diff) <= near_band_atr * a))
            above = diff > 0
            if above.any():
                j = int(np.argmin(np.where(above, diff, np.inf)))
                out[t, ci[p + "res_dist_atr"]] = diff[j] / a
                out[t, ci[p + "res_ratio"]] = ratios[j]
                out[t, ci[p + "res_touches"]] = sw.touches[j]
            below = ~above
            if below.any():
                j = int(np.argmin(np.where(below, -diff, np.inf)))
                out[t, ci[p + "sup_dist_atr"]] = -diff[j] / a
                out[t, ci[p + "sup_ratio"]] = ratios[j]
                out[t, ci[p + "sup_touches"]] = sw.touches[j]

            # Break: a level strictly between the previous close and this one, on a swing
            # that already existed at the previous bar.
            if t > 0 and sw.formed_idx < t:
                pc = close[t - 1]
                lo_, hi_ = (pc, c) if pc <= c else (c, pc)
                crossed = (sw.levels > lo_) & (sw.levels < hi_)
                if crossed.any():
                    direction = 1.0 if c > pc else -1.0
                    j = int(np.argmin(np.where(crossed, np.abs(diff), np.inf)))
                    out[t, ci[p + "break_dir"]] = direction
                    out[t, ci[p + "break_ratio"]] = ratios[j]
                    out[t, ci[p + "break_mag_atr"]] = abs(diff[j]) / a
                    out[t, ci[p + "break_vol_ratio"]] = volume[t] / vol_ma[t] if np.isfinite(vol_ma[t]) and vol_ma[t] > 0 else np.nan
        out[t, ci["fib_n_near"]] = near

    feats = pd.DataFrame(out, index=df.index, columns=cols)
    return feats, swing_table(book, df)

def swing_table(book: _SwingBook, df: pd.DataFrame) -> pd.DataFrame:
    """
    The active swing per tier at the end of the series.
    """
    cols = ["n", "dir", "start_idx", "start_time", "start_price", "end_idx", "end_time", "end_price",
            "range", "swing_bars", "formed_idx", "touches_total"]
    rows = []
    for n, sw in book.swings.items():
        if sw is None:
            continue
        rows.append(
            {
                "n": n, "dir": sw.dir,
                "start_idx": sw.start_idx, "start_time": df.index[sw.start_idx], "start_price": sw.start_price,
                "end_idx": sw.end_idx, "end_time": df.index[sw.end_idx], "end_price": sw.end_price,
                "range": sw.range, "swing_bars": abs(sw.end_idx - sw.start_idx),
                "formed_idx": sw.formed_idx, "touches_total": int(sw.touches.sum()),
            }
        )
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols]

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    from src.api.binance import load_prices
    from src.features.pivots import DEFAULT_METHOD, METHODS, default_ns, pivot_table
    import time

    p = argparse.ArgumentParser(description="Build Fibonacci features on the cached BTC series and summarise them.")
    p.add_argument("--refresh", action="store_true", help="update the price DB from Binance first")
    p.add_argument("--touch-tol", type=float, default=0.25, help="touch tolerance in ATRs")
    p.add_argument("--min-range", type=float, default=1.0, help="smallest swing to draw, in ATRs")
    p.add_argument("--method", choices=METHODS, default=DEFAULT_METHOD, help="pivot detector")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = load_prices(refresh=args.refresh)
    piv = pivot_table(df, method=args.method)
    t0 = time.perf_counter()
    feats, swings = build_fibonacci_features(df, piv, touch_tol_atr=args.touch_tol, min_range_atr=args.min_range)
    log.info("built %d x %d features in %.1fs", *feats.shape, time.perf_counter() - t0)

    pd.set_option("display.width", 220)
    print(f"\n[{args.method}] active swings at the end of the series:\n")
    print(swings.to_string(index=False))
    print("\nfeature summary:")
    print(feats.describe().T[["count", "mean", "50%", "min", "max"]].to_string())
    for n in default_ns(args.method):
        nb = int((feats[f"fib_{n}_break_dir"] != 0).sum())
        print(f"\ntier {n}: break bars {nb} of {len(feats)}; nearest-resistance ratio distribution:")
        print(feats[f"fib_{n}_res_ratio"].value_counts(normalize=True).sort_index().round(3).to_string())

if __name__ == "__main__":
    main()
