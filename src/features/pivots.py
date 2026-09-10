"""
Pivot (swing point) detection with confirmation-lag-aware timestamps.

A bar ``i`` is a **pivot high** at scale ``N`` if its high is strictly greater than
the highs of the ``N`` bars on either side. A **pivot low** is defined symmetrically
on the lows. Three scales are used as a proxy for how obvious the swing is on a chart:

======  =============
N       tier
======  =============
5       minor
20      intermediate
50      major
======  =============

Every major pivot is also an intermediate and a minor pivot.
The largest ``N`` a swing satisfies is its obviousness tier.

Confirmation lag
----------------
A pivot at bar ``i`` cannot be known until bar ``i + N`` has closed, because the
right-hand window must be complete. Every pivot therefore carries two timestamps:

* ``idx`` / ``time`` — where the swing *occurred* (used for the level's price and for
  drawing), and
* ``confirm_idx`` / ``confirm_time`` — the bar at whose **close** the swing became
  knowable. Any feature computed at bar ``t`` may only use pivots with
  ``confirm_idx <= t``.

Because the tiers confirm at different times (``i+5``, ``i+20``, ``i+50``), the tier
of a given swing *upgrades over time*. `known_pivots` returns the tier as it was
known at a given bar, which is the quantity that should drive the obviousness axis in
the experiments.

Detection methods
-----------------
Three detectors share the same output table and the same confirmation-lag
discipline; pick one with ``pivot_table(df, method=...)``.

``nbar`` (default)
    The N-bar rule above. ``n`` is N; confirmed at ``idx + N``.
``zigzag``
    The classic ZigZag reformulation: a candidate extreme is confirmed at the first
    bar where price has retraced ``k`` ATRs away from it. ``n`` is ``k`` (whole
    ATRs). The threshold self-adjusts: quiet regimes yield pivots from small swings,
    volatile regimes only from large ones. Confirmation time is data-driven, and
    pivots strictly alternate high / low.
``kernel``
    Nadaraya-Watson smoothing in the spirit of Lo, Mamaysky & Wang (2000), made
    causal. A Gaussian kernel of bandwidth ``n`` bars, truncated at three bandwidths,
    smooths the close; the smoothed value at ``t`` only exists once bar ``t + 3n``
    has closed, so a smoothed extremum at ``t`` is stamped ``confirm_idx =
    t + 1 + 3n``. The pivot itself is the actual high (low) within one bandwidth of
    the smoothed extremum.

Bars, not hours
---------------
Windows are counted in bars. The hourly series has a few dozen maintenance gaps, so
``N`` bars is not always ``N`` hours. Timestamps in the output are the bars' actual
open times.

Usage
-----
    from src.api.binance import load_prices
    from src.features.pivots import pivot_table, known_pivots, pivot_events

    df = load_prices(refresh=False)
    piv = pivot_table(df)                  # one row per (swing, tier)
    now = known_pivots(piv, as_of_idx=len(df) - 1)   # what a chart shows today
    ev = pivot_events(df, piv)             # wide, aligned to df.index, stamped at confirmation
"""

from __future__ import annotations
from src.features.indicators import atr as _atr
from typing import Iterable, Sequence
import argparse
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TIERS: dict[int, str] = {5: "minor", 20: "intermediate", 50: "major"}          # nbar: N bars each side
DEFAULT_NS: tuple[int, ...] = tuple(TIERS)
ZIGZAG_TIERS: dict[int, str] = {2: "minor", 4: "intermediate", 8: "major"}     # zigzag: reversal in ATRs
KERNEL_TIERS: dict[int, str] = {3: "minor", 8: "intermediate", 20: "major"}    # kernel: bandwidth in bars
METHOD_TIERS: dict[str, dict[int, str]] = {"nbar": TIERS, "zigzag": ZIGZAG_TIERS, "kernel": KERNEL_TIERS}
METHODS: tuple[str, ...] = tuple(METHOD_TIERS)
DEFAULT_METHOD = "nbar"
Record = tuple[int, str, float, int, float]   # (idx, kind, price, confirm_idx, prominence)

PIVOT_COLUMNS = [
    "idx",
    "time",
    "kind",
    "n",
    "tier",
    "price",
    "confirm_idx",
    "confirm_time",
    "prominence",
]

# --------------------------------------------------------------------------------------
# Core detection
# --------------------------------------------------------------------------------------

def _side_extrema(x: np.ndarray, n: int, how: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Rolling max/min of the ``n`` bars strictly to the left and right of each bar.

    Positions whose window runs off the edge of the series get NaN, so no pivot can
    be reported in the first or last ``n`` bars.
    """
    s = pd.Series(x, dtype="float64")
    roll = getattr(s.rolling(n, min_periods=n), how)
    left = roll().shift(1)
    right = getattr(s[::-1].rolling(n, min_periods=n), how)().shift(1)[::-1]
    return left.to_numpy(), right.to_numpy()

def detect_pivots(
    high: np.ndarray | pd.Series,
    low: np.ndarray | pd.Series,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Flag pivot highs and lows at scale ``n``.

    Returns ``(is_high, is_low, prom_high, prom_low)``. The boolean arrays are indexed
    by *occurrence* bar. Prominence is how far the pivot sticks out beyond the higher
    (lower) of its two side windows, in price units; NaN where not a pivot.

    Ties are not pivots. Pivot must be *strictly* beyond every neighbour.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    high = np.asarray(high, dtype="float64")
    low = np.asarray(low, dtype="float64")
    if high.shape != low.shape or high.ndim != 1:
        raise ValueError("high and low must be 1-D arrays of the same length")

    lmax, rmax = _side_extrema(high, n, "max")
    lmin, rmin = _side_extrema(low, n, "min")

    with np.errstate(invalid="ignore"):
        is_high = (high > lmax) & (high > rmax)
        is_low = (low < lmin) & (low < rmin)
        prom_high = np.where(is_high, high - np.maximum(lmax, rmax), np.nan)
        prom_low = np.where(is_low, np.minimum(lmin, rmin) - low, np.nan)
    return is_high, is_low, prom_high, prom_low

def detect_pivots_zigzag(
    high: np.ndarray | pd.Series,
    low: np.ndarray | pd.Series,
    atr: np.ndarray | pd.Series,
    k: float,
) -> list[Record]:
    """
    ATR-scaled ZigZag: confirm an extreme once price retraces ``k`` ATRs from it.

    While the last confirmed pivot is a low, the running maximum of ``high`` is the
    candidate high; the first bar whose low sits ``k * atr[t]`` or more below that
    candidate confirms it, and that bar's low seeds the next candidate. The mirror
    holds after a high. Before any pivot exists both candidates run until one of
    them is ``k`` ATRs away from the other. Pivots therefore alternate strictly.

    Prominence is the size of the reversal that confirmed the pivot, in price units.
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    high = np.asarray(high, dtype="float64")
    low = np.asarray(low, dtype="float64")
    atr = np.asarray(atr, dtype="float64")
    L = high.size
    out: list[Record] = []
    if L == 0:
        return out
    direction = 0                       # 0 unknown, +1 tracking a high, -1 tracking a low
    ch_i, ch_p = 0, high[0]
    cl_i, cl_p = 0, low[0]
    for t in range(1, L):
        thr = k * atr[t]
        if direction == 0:
            if high[t] > ch_p:
                ch_i, ch_p = t, high[t]
            if low[t] < cl_p:
                cl_i, cl_p = t, low[t]
            if ch_p - cl_p >= thr and ch_i != cl_i:
                if ch_i < cl_i:
                    out.append((ch_i, "high", ch_p, t, ch_p - cl_p))
                    direction = -1
                else:
                    out.append((cl_i, "low", cl_p, t, ch_p - cl_p))
                    direction = 1
        elif direction == 1:
            if high[t] > ch_p:
                ch_i, ch_p = t, high[t]
            elif ch_p - low[t] >= thr:
                out.append((ch_i, "high", ch_p, t, ch_p - low[t]))
                direction = -1
                cl_i, cl_p = t, low[t]
        else:
            if low[t] < cl_p:
                cl_i, cl_p = t, low[t]
            elif high[t] - cl_p >= thr:
                out.append((cl_i, "low", cl_p, t, high[t] - cl_p))
                direction = 1
                ch_i, ch_p = t, high[t]
    return out

def kernel_smooth(x: np.ndarray | pd.Series, bw: float) -> tuple[np.ndarray, int]:
    """
    Gaussian (Nadaraya-Watson) smoothing with bandwidth ``bw`` bars, truncated at
    three bandwidths. Returns ``(smoothed, H)`` where ``H = ceil(3 * bw)`` is the
    half-width: ``smoothed[t]`` uses bars ``t - H .. t + H`` and is only trustworthy
    for ``H <= t < len - H``.
    """
    if bw <= 0:
        raise ValueError("bw must be > 0")
    x = np.asarray(x, dtype="float64")
    H = int(np.ceil(3 * bw))
    j = np.arange(-H, H + 1)
    w = np.exp(-0.5 * (j / bw) ** 2)
    w /= w.sum()
    return np.convolve(x, w, mode="same"), H

def detect_pivots_kernel(
    high: np.ndarray | pd.Series,
    low: np.ndarray | pd.Series,
    close: np.ndarray | pd.Series,
    bw: float,
) -> list[Record]:
    """
    Pivots from local extrema of the kernel-smoothed close, stamped at confirmation.

    A smoothed maximum at ``t`` needs ``smoothed[t + 1]``, which needs bar
    ``t + 1 + H``; that is the confirmation bar. The reported pivot is the bar with
    the highest high (lowest low) within one bandwidth of ``t``, so the level sits on
    a real extreme rather than on the smooth curve. Prominence is how far that
    extreme sticks out beyond the smoothed value.
    """
    high = np.asarray(high, dtype="float64")
    low = np.asarray(low, dtype="float64")
    close = np.asarray(close, dtype="float64")
    L = close.size
    m, H = kernel_smooth(close, bw)
    w = int(np.ceil(bw))
    out: list[Record] = []
    if L < 2 * H + 3:
        return out
    inner = np.arange(H + 1, L - H - 1)            # both t-1 and t+1 trustworthy
    prev, cur, nxt = m[inner - 1], m[inner], m[inner + 1]
    seen: set[tuple[int, str]] = set()
    for t, is_max in sorted(
        [(int(t), True) for t in inner[(cur > prev) & (cur > nxt)]]
        + [(int(t), False) for t in inner[(cur < prev) & (cur < nxt)]]
    ):
        a, b = max(0, t - w), min(L, t + w + 1)
        if is_max:
            i = a + int(np.argmax(high[a:b]))
            rec: Record = (i, "high", float(high[i]), t + 1 + H, float(high[i] - m[t]))
        else:
            i = a + int(np.argmin(low[a:b]))
            rec = (i, "low", float(low[i]), t + 1 + H, float(m[t] - low[i]))
        key = (rec[0], rec[1])
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out

# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------

def _tier_name(n: int, method: str = DEFAULT_METHOD) -> str:
    return METHOD_TIERS.get(method, TIERS).get(n, f"n{n}")

def default_ns(method: str = DEFAULT_METHOD) -> tuple[int, ...]:
    """
    The three tier parameters for ``method``.
    """
    if method not in METHOD_TIERS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    return tuple(METHOD_TIERS[method])

def pivot_table(
    df: pd.DataFrame,
    ns: Sequence[int] | None = None,
    method: str = DEFAULT_METHOD,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    atr_n: int = 14,
) -> pd.DataFrame:
    """
    Long table of pivots: one row per (swing, scale).

    ``df`` must be sorted by time with a DatetimeIndex (as returned by
    `src.api.binance.load_prices`). ``method`` is ``"nbar"``, ``"zigzag"`` or
    ``"kernel"`` (see the module docstring); ``ns`` are that method's scale
    parameters and default to its three tiers. A swing that qualifies at several
    scales appears once per scale, each with its own ``confirm_idx``.

    Columns: ``idx, time, kind, n, tier, price, confirm_idx, confirm_time, prominence``.
    Sorted by ``confirm_idx`` so it can be scanned in the order information arrives.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError("df must be sorted by time")
    if method not in METHOD_TIERS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    if ns is None:
        ns = default_ns(method)
    times = df.index
    high = df[high_col].to_numpy(dtype="float64")
    low = df[low_col].to_numpy(dtype="float64")
    close = df[close_col].to_numpy(dtype="float64") if method != "nbar" else None
    atr = _atr(high, low, close, atr_n) if method == "zigzag" else None

    frames: list[pd.DataFrame] = []
    for n in ns:
        if method == "nbar":
            is_high, is_low, prom_high, prom_low = detect_pivots(high, low, n)
            for kind, mask, price, prom in (
                ("high", is_high, high, prom_high),
                ("low", is_low, low, prom_low),
            ):
                idx = np.flatnonzero(mask)
                if idx.size == 0:
                    continue
                frames.append(_frame(idx, kind, price[idx], idx + n, prom[idx], n, method, times))
            continue
        recs = detect_pivots_zigzag(high, low, atr, n) if method == "zigzag" else detect_pivots_kernel(high, low, close, n)
        if not recs:
            continue
        idx = np.array([r[0] for r in recs], dtype=int)
        kinds = np.array([r[1] for r in recs])
        price = np.array([r[2] for r in recs], dtype="float64")
        confirm = np.array([r[3] for r in recs], dtype=int)
        prom = np.array([r[4] for r in recs], dtype="float64")
        for kind in ("high", "low"):
            m = kinds == kind
            if m.any():
                frames.append(_frame(idx[m], kind, price[m], confirm[m], prom[m], n, method, times))
    if not frames:
        return pd.DataFrame(columns=PIVOT_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["kind"] = out["kind"].astype("category")
    out["tier"] = out["tier"].astype("category")
    return (
        out.sort_values(["confirm_idx", "idx", "kind"], kind="stable")
        .reset_index(drop=True)[PIVOT_COLUMNS]
    )

def _frame(idx, kind, price, confirm_idx, prom, n, method, times) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "idx": idx,
            "time": times[idx],
            "kind": kind,
            "n": n,
            "tier": _tier_name(n, method),
            "price": price,
            "confirm_idx": confirm_idx,
            "confirm_time": times[confirm_idx],
            "prominence": prom,
        }
    )

def known_pivots(pivots: pd.DataFrame, as_of_idx: int) -> pd.DataFrame:
    """
    Pivots knowable at the close of bar ``as_of_idx``, one row per swing.

    Filters to ``confirm_idx <= as_of_idx`` and collapses the scales, keeping the
    largest ``n`` confirmed so far. ``n`` and ``tier`` therefore reflect the swing's
    obviousness *as it was known at that time*, not its eventual tier.
    """
    known = pivots[pivots["confirm_idx"] <= as_of_idx]
    if known.empty:
        return pd.DataFrame(columns=["idx", "time", "kind", "price", "n", "tier", "confirm_idx", "prominence"])
    best = (
        known.sort_values("n", kind="stable")
        .groupby(["idx", "kind"], observed=True, sort=False)
        .tail(1)
    )
    return (
        best[["idx", "time", "kind", "price", "n", "tier", "confirm_idx", "prominence"]]
        .sort_values("idx", kind="stable")
        .reset_index(drop=True)
    )

def pivot_events(
    df: pd.DataFrame,
    pivots: pd.DataFrame,
    ns: Iterable[int] | None = None,
) -> pd.DataFrame:
    """
    Wide, time-aligned view stamped at **confirmation**.

    Returns a frame on ``df.index`` with one column per (kind, n), e.g. ``high_20``,
    holding the pivot *price* on the bar where that pivot became known and NaN
    elsewhere. Reading row ``t`` therefore tells you exactly which levels a
    chart-watcher could have drawn for the first time at the close of bar ``t``.
    """
    ns = sorted(set(ns) if ns is not None else set(pivots["n"].unique()))
    out = pd.DataFrame(index=df.index)
    for n in ns:
        for kind in ("high", "low"):
            col = f"{kind}_{n}"
            vals = np.full(len(df), np.nan)
            sel = pivots[(pivots["n"] == n) & (pivots["kind"] == kind)]
            vals[sel["confirm_idx"].to_numpy()] = sel["price"].to_numpy()
            out[col] = vals
    return out

def pivot_summary(pivots: pd.DataFrame, n_bars: int) -> pd.DataFrame:
    """Per (tier, kind) counts, mean spacing in bars and median prominence."""
    if pivots.empty:
        return pd.DataFrame()
    g = pivots.groupby(["n", "tier", "kind"], observed=True)
    summary = g.agg(count=("idx", "size"), median_prominence=("prominence", "median"))
    summary["bars_per_pivot"] = n_bars / summary["count"]
    return summary.reset_index()

# --------------------------------------------------------------------------------------
# Brute-force reference (for tests and --check)
# --------------------------------------------------------------------------------------

def detect_pivots_bruteforce(high: np.ndarray, low: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """O(len * n) reference implementation of `detect_pivots`."""
    high = np.asarray(high, dtype="float64")
    low = np.asarray(low, dtype="float64")
    L = len(high)
    is_high = np.zeros(L, dtype=bool)
    is_low = np.zeros(L, dtype=bool)
    for i in range(n, L - n):
        window_h = np.concatenate([high[i - n : i], high[i + 1 : i + n + 1]])
        window_l = np.concatenate([low[i - n : i], low[i + 1 : i + n + 1]])
        is_high[i] = np.all(high[i] > window_h)
        is_low[i] = np.all(low[i] < window_l)
    return is_high, is_low

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    from src.api.binance import load_prices

    p = argparse.ArgumentParser(description="Detect pivots on the cached BTC series and print a summary.")
    p.add_argument("--method", choices=METHODS, default=DEFAULT_METHOD, help="pivot detector")
    p.add_argument("--ns", type=int, nargs="+", default=None, help="scale parameters (default: the method's tiers)")
    p.add_argument("--refresh", action="store_true", help="update the price DB from Binance first")
    p.add_argument("--check", action="store_true", help="verify against the brute-force implementation")
    p.add_argument("--tail", type=int, default=10, help="print the last K pivots known today")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = load_prices(refresh=args.refresh)
    piv = pivot_table(df, args.ns, method=args.method)
    ns = tuple(args.ns) if args.ns else default_ns(args.method)

    pd.set_option("display.width", 160)
    print(f"\n{len(df)} bars, {df.index[0]} .. {df.index[-1]}  [{args.method}]\n")
    print(pivot_summary(piv, len(df)).to_string(index=False))

    if args.check and args.method == "nbar":
        high, low = df["high"].to_numpy(), df["low"].to_numpy()
        for n in ns:
            fast_h, fast_l, _, _ = detect_pivots(high, low, n)
            ref_h, ref_l = detect_pivots_bruteforce(high, low, n)
            ok = np.array_equal(fast_h, ref_h) and np.array_equal(fast_l, ref_l)
            print(f"check n={n:3d}: {'OK' if ok else 'MISMATCH'}")

    if args.tail:
        known = known_pivots(piv, len(df) - 1)
        print(f"\nlast {args.tail} swings as known at {df.index[-1]}:\n")
        print(known.tail(args.tail).to_string(index=False))

if __name__ == "__main__":
    main()
