"""
Trendlines drawn through confirmed pivots.

This is the second of the three strategies under test. A trendline is the straight
line through two pivots of the same kind: two pivot lows make a support line, two
pivot highs a resistance line. The slope is free, so unlike horizontal levels the
line's price changes every bar; everything else (touches, tier, breaks, expiry)
follows the same rules as `src.features.levels`.

How lines are built
-------------------
Bars are replayed in order. When a pivot ``P2`` is confirmed at bar ``t`` it is
compared with every alive line of its kind: if it lands within ``touch_tol_atr``
ATRs of the line's projected value at ``P2``'s bar it becomes a touch. Then, for
every alive earlier pivot ``P1`` of the same kind that is not already on a line
``P2`` just touched, a new line is drawn if the chord ``P1 -> P2`` is *clean*: no
bar low between them dips below a support chord (no bar high pokes above a
resistance chord), within tolerance. A chord that price has already closed through
between ``P2``'s bar and ``t`` is not drawn. A swing later confirmed at a larger
``N`` upgrades the line's tier without adding a touch.

A line dies the first time a close finishes beyond it by more than
``break_tol_atr`` ATRs. Chart-watchers erase a broken trendline, so unlike
horizontal levels there is no role reversal; the break is recorded in the ``break``
features on that bar and the line is gone. A line also dies quietly once it has
drifted more than ``max_dist_atr`` ATRs from the close, since a steep line running
away from price is no longer on anyone's screen and can never be closed through.
Lines and their candidate anchor pivots also expire with the same per-tier lookback
as horizontal levels.

Per-bar features
----------------
All distances in ATR units; slopes in ATRs per bar.

``tl_sup_*`` / ``tl_res_*``
    Nearest alive support line below / resistance line above the close:
    ``dist_atr``, ``slope_atr``, ``touches``, ``tier`` (smaller of the two anchor
    tiers), ``n_tiers``, ``age_bars`` (bars since the last touch), ``span_bars``
    (bars since the first anchor).
``tl_sup_dist_atr_{N}`` / ``tl_res_dist_atr_{N}``
    Same restricted to lines whose tier is at least ``N``.
``tl_n_sup``, ``tl_n_res``, ``tl_n_near``
    Alive lines on each side, and lines within ``near_band_atr`` of the close.
``tl_break_dir``, ``tl_break_mag_atr``, ``tl_break_vol_ratio``,
``tl_break_touches``, ``tl_break_tier``, ``tl_break_slope_atr``
    Set on bars where a close finished through a line. ``dir`` is +1 through a
    resistance line, -1 through a support line. If several lines broke, the one
    with the most touches is reported.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from src.features.indicators import atr as _atr, rolling_mean
from src.features.levels import DEFAULT_LOOKBACK
from src.features.pivots import DEFAULT_NS
import argparse
import heapq
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

Key = tuple[int, str]

@dataclass
class Line:
    """
    One trendline: two anchor pivots plus any later pivots that touched it.
    """
    id: int
    kind: str                       # "low" = support line, "high" = resistance line
    a_key: Key                      # earlier anchor (idx, kind)
    b_key: Key                      # later anchor
    price_a: float
    price_b: float
    created_idx: int                # bar at which the line became drawable
    members: dict[Key, int] = field(default_factory=dict)  # (idx, kind) -> largest confirmed n
    last_touch_idx: int = 0         # occurrence bar of the latest member
    expires: int = 0

    def __post_init__(self) -> None:
        self.slope = (self.price_b - self.price_a) / (self.b_key[0] - self.a_key[0])

    def value(self, t: int) -> float:
        return self.price_a + self.slope * (t - self.a_key[0])

    @property
    def touches(self) -> int:
        return len(self.members)

    @property
    def tier(self) -> int:
        return min(self.members[self.a_key], self.members[self.b_key])

    @property
    def n_tiers(self) -> int:
        return len(set(self.members.values()))

class _LineBook:
    """
    Alive pivots (candidate anchors) and alive lines, with per-kind arrays for
    vectorised evaluation at each bar.
    """

    def __init__(
        self,
        low: np.ndarray,
        high: np.ndarray,
        close: np.ndarray,
        lookback: dict[int, int] | None,
        touch_tol_atr: float,
        break_tol_atr: float,
        max_dist_atr: float,
    ) -> None:
        self.low, self.high, self.close = low, high, close
        self.lookback = lookback
        self.touch_tol_atr = touch_tol_atr
        self.break_tol_atr = break_tol_atr
        self.max_dist_atr = max_dist_atr
        self.pivots: dict[Key, tuple[int, float]] = {}      # key -> (n, price)
        self.lines: dict[int, Line] = {}
        self._kind_ids: dict[str, set[int]] = {"low": set(), "high": set()}
        self.by_key: dict[Key, set[int]] = {}               # pivot key -> line ids it belongs to
        self._pivot_expiry: list[tuple[int, Key, int]] = []
        self._line_expiry: list[tuple[int, int]] = []
        self._next_id = 0
        self._dirty = {"low": True, "high": True}
        self._arrays: dict[str, tuple[np.ndarray, ...]] = {}
        self.n_created = 0
        self.n_broken = 0
        self.n_pruned = 0

    # -- expiry -------------------------------------------------------------------

    def _pivot_life(self, idx: int, n: int) -> int:
        return idx + self.lookback[n] + 1 if self.lookback is not None else np.iinfo(np.int64).max

    def _refresh_expiry(self, line: Line) -> None:
        line.expires = max(self._pivot_life(k[0], n) for k, n in line.members.items())
        heapq.heappush(self._line_expiry, (line.expires, line.id))

    def expire(self, t: int) -> None:
        while self._pivot_expiry and self._pivot_expiry[0][0] <= t:
            _, key, n = heapq.heappop(self._pivot_expiry)
            if self.pivots.get(key, (None,))[0] == n:
                del self.pivots[key]
        while self._line_expiry and self._line_expiry[0][0] <= t:
            _, lid = heapq.heappop(self._line_expiry)
            line = self.lines.get(lid)
            if line is not None and line.expires <= t:
                self._remove(line)

    def _remove(self, line: Line) -> None:
        del self.lines[line.id]
        self._kind_ids[line.kind].discard(line.id)
        for k in line.members:
            s = self.by_key.get(k)
            if s is not None:
                s.discard(line.id)
        self._dirty[line.kind] = True

    # -- ingest -------------------------------------------------------------------

    def add_pivot(self, idx: int, kind: str, n: int, price: float, t: int, atr_t: float) -> None:
        key = (idx, kind)
        if key in self.pivots:
            # Tier upgrade: no new touch, but a bigger n and a longer life.
            if n > self.pivots[key][0]:
                self.pivots[key] = (n, price)
                heapq.heappush(self._pivot_expiry, (self._pivot_life(idx, n), key, n))
                for lid in self.by_key.get(key, ()):
                    line = self.lines[lid]
                    line.members[key] = n
                    self._refresh_expiry(line)
                    self._dirty[kind] = True
            return
        self.pivots[key] = (n, price)
        heapq.heappush(self._pivot_expiry, (self._pivot_life(idx, n), key, n))
        self.by_key.setdefault(key, set())
        tol = self.touch_tol_atr * atr_t
        btol = self.break_tol_atr * atr_t

        # 1. Touches on existing lines of this kind.
        touched_keys: set[Key] = set()
        ids, vals, _, _ = self.values(kind, idx)
        for lid in ids[np.abs(vals - price) <= tol]:
            line = self.lines[int(lid)]
            line.members[key] = n
            line.last_touch_idx = idx
            self.by_key[key].add(line.id)
            touched_keys.update(line.members)
            self._refresh_expiry(line)
            self._dirty[kind] = True

        # 2. New lines to earlier anchors not already on a line this pivot touched.
        for k1, n1, p1 in self._candidate_anchors(kind, idx, price, tol, touched_keys):
            if not self._chord_clean(kind, k1[0], p1, idx, price, tol):
                continue
            if self._already_broken(kind, k1[0], p1, idx, price, t, btol):
                continue
            line = Line(
                id=self._next_id, kind=kind, a_key=k1, b_key=key, price_a=p1, price_b=price,
                created_idx=t, members={k1: n1, key: n}, last_touch_idx=idx,
            )
            self._next_id += 1
            self.lines[line.id] = line
            self._kind_ids[kind].add(line.id)
            self.by_key[k1].add(line.id)
            self.by_key[key].add(line.id)
            self._refresh_expiry(line)
            self._dirty[kind] = True
            self.n_created += 1

    def _candidate_anchors(
        self, kind: str, i2: int, p2: float, tol: float, exclude: set[Key]
    ) -> list[tuple[Key, int, float]]:
        """
        Alive earlier pivots of ``kind`` whose chord to ``(i2, p2)`` is not already
        ruled out by an intermediate pivot of the same kind.

        This is an exact pre-filter for `_chord_clean`: a pivot low below a
        support chord is itself a bar low below it, so every rejection here is
        correct, and survivors still get the full bar-by-bar check.
        """
        keys = [k for k in self.pivots if k[1] == kind and k[0] < i2 and k not in exclude]
        if not keys:
            return []
        i1 = np.fromiter((k[0] for k in keys), dtype=int, count=len(keys))
        p1 = np.fromiter((self.pivots[k][1] for k in keys), dtype="float64", count=len(keys))
        slope = (p2 - p1) / (i2 - i1)
        # chord[c, j]: value of candidate c's chord at candidate j's bar.
        chord = p1[:, None] + slope[:, None] * (i1[None, :] - i1[:, None])
        between = i1[None, :] > i1[:, None]
        if kind == "low":
            bad = between & (p1[None, :] < chord - tol)
        else:
            bad = between & (p1[None, :] > chord + tol)
        ok = np.flatnonzero(~bad.any(axis=1))
        return [(keys[c], self.pivots[keys[c]][0], float(p1[c])) for c in ok]

    def _chord_clean(self, kind: str, i1: int, p1: float, i2: int, p2: float, tol: float) -> bool:
        if i2 - i1 < 2:
            return True
        s = (p2 - p1) / (i2 - i1)
        line = p1 + s * np.arange(1, i2 - i1)
        if kind == "low":
            return bool(np.all(self.low[i1 + 1 : i2] >= line - tol))
        return bool(np.all(self.high[i1 + 1 : i2] <= line + tol))

    def _already_broken(self, kind: str, i1: int, p1: float, i2: int, p2: float, t: int, btol: float) -> bool:
        """
        Has a close between the second anchor and now already finished through the chord?
        """
        if t <= i2:
            return False
        s = (p2 - p1) / (i2 - i1)
        line = p2 + s * np.arange(1, t - i2 + 1)
        closes = self.close[i2 + 1 : t + 1]
        if kind == "low":
            return bool(np.any(closes < line - btol))
        return bool(np.any(closes > line + btol))

    # -- evaluation ---------------------------------------------------------------

    def arrays(self, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        (ids, a_idx, price_a, slope, tier) for alive lines of ``kind``.
        """
        if self._dirty[kind]:
            ls = [self.lines[i] for i in self._kind_ids[kind]]
            self._arrays[kind] = (
                np.fromiter((l.id for l in ls), dtype=int, count=len(ls)),
                np.fromiter((l.a_key[0] for l in ls), dtype=int, count=len(ls)),
                np.fromiter((l.price_a for l in ls), dtype="float64", count=len(ls)),
                np.fromiter((l.slope for l in ls), dtype="float64", count=len(ls)),
                np.fromiter((l.tier for l in ls), dtype=int, count=len(ls)),
            )
            self._dirty[kind] = False
        return self._arrays[kind]

    def values(self, kind: str, t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ids, a_idx, price_a, slope, tier = self.arrays(kind)
        return ids, price_a + slope * (t - a_idx), slope, tier

    def check_breaks(self, t: int, atr_t: float) -> list[tuple[Line, float]]:
        """
        Retire every line the close at ``t`` finished through; return (line, value) pairs.
        """
        btol = self.break_tol_atr * atr_t
        c = self.close[t]
        broken: list[tuple[Line, float]] = []
        for kind in ("low", "high"):
            ids, vals, _, _ = self.values(kind, t)
            if ids.size == 0:
                continue
            hit = c < vals - btol if kind == "low" else c > vals + btol
            for lid, v in zip(ids[hit], vals[hit]):
                broken.append((self.lines[int(lid)], float(v)))
        for line, _ in broken:
            self._remove(line)
            self.n_broken += 1
        return broken

    def prune_far(self, t: int, atr_t: float) -> None:
        """
        Quietly drop lines that have drifted off the screen.
        """
        limit = self.max_dist_atr * atr_t
        c = self.close[t]
        for kind in ("low", "high"):
            ids, vals, _, _ = self.values(kind, t)
            if ids.size == 0:
                continue
            for lid in ids[np.abs(vals - c) > limit]:
                self._remove(self.lines[int(lid)])
                self.n_pruned += 1

# --------------------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------------------

SIDE_FIELDS = ("dist_atr", "slope_atr", "touches", "tier", "n_tiers", "age_bars", "span_bars")
BREAK_FIELDS = ("dir", "mag_atr", "vol_ratio", "touches", "tier", "slope_atr")

def feature_columns(ns: tuple[int, ...] = DEFAULT_NS) -> list[str]:
    cols = [f"tl_{side}_{f}" for side in ("sup", "res") for f in SIDE_FIELDS]
    cols += [f"tl_{side}_dist_atr_{n}" for n in ns for side in ("sup", "res")]
    cols += ["tl_n_sup", "tl_n_res", "tl_n_near"]
    cols += [f"tl_break_{f}" for f in BREAK_FIELDS]
    return cols

def build_trendline_features(
    df: pd.DataFrame,
    pivots: pd.DataFrame,
    atr_n: int = 14,
    touch_tol_atr: float = 0.25,
    break_tol_atr: float = 0.0,
    max_dist_atr: float = 30.0,
    near_band_atr: float = 1.0,
    vol_n: int = 20,
    ns: tuple[int, ...] = DEFAULT_NS,
    lookback: dict[int, int] | None = DEFAULT_LOOKBACK,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build per-bar trendline features from a pivot table.

    ``df`` is the OHLCV frame from `src.api.binance.load_prices`; ``pivots``
    the long table from `src.features.pivots.pivot_table`. ``max_dist_atr``
    is how far a line may drift from the close before it is dropped as off-screen.
    Returns ``(features, lines)``: features aligned to ``df.index`` (NaN where no
    line exists on that side) and a table of the lines still alive at the end.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError("df must be sorted by time")
    if lookback is not None:
        missing = set(ns) - set(lookback)
        if missing:
            raise ValueError(f"lookback has no entry for tiers {sorted(missing)}")
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
    for c in ("tl_n_sup", "tl_n_res", "tl_n_near", "tl_break_dir"):
        out[:, ci[c]] = 0.0

    book = _LineBook(low, high, close, lookback, touch_tol_atr, break_tol_atr, max_dist_atr)
    ptr = 0
    for t in range(L):
        a = atr[t]
        book.expire(t)
        while ptr < ev_confirm.size and ev_confirm[ptr] == t:
            book.add_pivot(int(ev_idx[ptr]), str(ev_kind[ptr]), int(ev_n[ptr]), float(ev_price[ptr]), t, a)
            ptr += 1
        if not book.lines or not np.isfinite(a) or a <= 0:
            continue
        c = close[t]

        broken = book.check_breaks(t, a)
        if broken:
            line, v = max(broken, key=lambda lv: (lv[0].touches, lv[0].tier))
            direction = -1.0 if line.kind == "low" else 1.0
            out[t, ci["tl_break_dir"]] = direction
            out[t, ci["tl_break_mag_atr"]] = direction * (c - v) / a
            out[t, ci["tl_break_vol_ratio"]] = volume[t] / vol_ma[t] if np.isfinite(vol_ma[t]) and vol_ma[t] > 0 else np.nan
            out[t, ci["tl_break_touches"]] = line.touches
            out[t, ci["tl_break_tier"]] = line.tier
            out[t, ci["tl_break_slope_atr"]] = line.slope / a
        book.prune_far(t, a)

        near = 0
        for kind, side, sign in (("low", "sup", 1.0), ("high", "res", -1.0)):
            ids, vals, slopes, tiers = book.values(kind, t)
            out[t, ci[f"tl_n_{side}"]] = ids.size
            if ids.size == 0:
                continue
            dist = sign * (c - vals) / a          # >= -break_tol for every alive line
            near += int(np.sum(np.abs(dist) <= near_band_atr))
            j = int(np.argmin(dist))
            line = book.lines[int(ids[j])]
            out[t, ci[f"tl_{side}_dist_atr"]] = dist[j]
            out[t, ci[f"tl_{side}_slope_atr"]] = slopes[j] / a
            out[t, ci[f"tl_{side}_touches"]] = line.touches
            out[t, ci[f"tl_{side}_tier"]] = line.tier
            out[t, ci[f"tl_{side}_n_tiers"]] = line.n_tiers
            out[t, ci[f"tl_{side}_age_bars"]] = t - line.last_touch_idx
            out[t, ci[f"tl_{side}_span_bars"]] = t - line.a_key[0]
            for n in ns:
                m = tiers >= n
                if m.any():
                    out[t, ci[f"tl_{side}_dist_atr_{n}"]] = dist[m].min()
        out[t, ci["tl_n_near"]] = near

    feats = pd.DataFrame(out, index=df.index, columns=cols)
    return feats, line_table(book, df, atr)

def line_table(book: _LineBook, df: pd.DataFrame, atr: np.ndarray) -> pd.DataFrame:
    """
    Every line alive at the end of the series, most-touched first.
    """
    cols = ["id", "kind", "a_idx", "a_time", "b_idx", "b_time", "price_a", "slope", "slope_atr",
            "touches", "tier", "n_tiers", "created_idx", "last_touch_idx", "value_now"]
    L = len(df) - 1
    rows = []
    for l in book.lines.values():
        rows.append(
            {
                "id": l.id, "kind": l.kind,
                "a_idx": l.a_key[0], "a_time": df.index[l.a_key[0]],
                "b_idx": l.b_key[0], "b_time": df.index[l.b_key[0]],
                "price_a": l.price_a, "slope": l.slope, "slope_atr": l.slope / atr[L],
                "touches": l.touches, "tier": l.tier, "n_tiers": l.n_tiers,
                "created_idx": l.created_idx, "last_touch_idx": l.last_touch_idx,
                "value_now": l.value(L),
            }
        )
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].sort_values(["touches", "tier"], ascending=False, kind="stable").reset_index(drop=True)

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    from src.api.binance import load_prices
    from src.features.pivots import pivot_table
    import time

    p = argparse.ArgumentParser(description="Build trendline features on the cached BTC series and summarise them.")
    p.add_argument("--refresh", action="store_true", help="update the price DB from Binance first")
    p.add_argument("--touch-tol", type=float, default=0.25, help="touch tolerance in ATRs")
    p.add_argument("--break-tol", type=float, default=0.0, help="close-through tolerance in ATRs")
    p.add_argument("--max-dist", type=float, default=30.0, help="drop lines further than this many ATRs from price")
    p.add_argument("--top", type=int, default=10, help="print the most-touched alive lines")
    p.add_argument("--no-expiry", action="store_true", help="keep anchors and lines forever")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = load_prices(refresh=args.refresh)
    piv = pivot_table(df)
    t0 = time.perf_counter()
    feats, lines = build_trendline_features(
        df, piv, touch_tol_atr=args.touch_tol, break_tol_atr=args.break_tol, max_dist_atr=args.max_dist,
        lookback=None if args.no_expiry else DEFAULT_LOOKBACK,
    )
    log.info("built %d x %d features in %.1fs", *feats.shape, time.perf_counter() - t0)

    pd.set_option("display.width", 220)
    print(f"\n{len(lines)} lines alive at the end\n")
    print("alive-line touches distribution:")
    print(lines["touches"].value_counts().sort_index().to_string())
    print("\nfeature summary:")
    print(feats.describe().T[["count", "mean", "50%", "min", "max"]].to_string())
    nb = int((feats["tl_break_dir"] != 0).sum())
    print(f"\nbreak bars: {nb} of {len(feats)}")
    print(f"\ntop {args.top} alive lines by touches:\n")
    print(lines.head(args.top).to_string(index=False))

if __name__ == "__main__":
    main()
