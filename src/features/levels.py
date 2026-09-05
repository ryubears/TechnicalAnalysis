"""
Horizontal support / resistance levels built from confirmed pivots.

This is the first of the three strategies under test. It turns the pivot stream into
a set of price levels and, for every bar, emits numeric features describing where
price sits relative to them. The thesis is that the *reaction* at a level should
scale with how obvious the level is, so the features carry the ingredients of
obviousness (touch count, tier, cross-tier confluence, age) alongside distance.

How levels are built
--------------------
Bars are processed in order. When a pivot becomes *confirmed* at bar ``t`` (see
`src.features.pivots`) its price is compared with the existing levels. If one
lies within ``merge_tol_atr`` ATRs it joins that level as an extra touch and the
level's price becomes the mean of its members; otherwise a new level is created.
A swing that is later confirmed at a larger ``N`` upgrades the level's tier rather
than adding a touch. Nothing about a level can be known before its first member is
confirmed, so features at bar ``t`` only ever reflect pivots with
``confirm_idx <= t``.

A level is never deleted when price passes through it. Whether it is support or
resistance is decided fresh at every bar by which side of the close it sits on
(classic role reversal), and each break is counted so weak levels can be told apart.

Levels do expire, though, because a chart only shows a window. Each member swing
stays visible for ``lookback[N]`` bars after it occurred, where ``N`` is its largest
confirmed tier, so a minor swing drops off after a month while a major one lingers
for two years. A level with no visible members is removed. Without this, nine years
of swings blanket the price range and every bar sits within half an ATR of a level.

Per-bar features
----------------
All distances are in ATR units and positive.

``res_*`` / ``sup_*``
    Nearest level strictly above / below the close: ``dist_atr``, ``touches``,
    ``tier`` (largest confirmed N among members), ``n_tiers`` (distinct tiers among
    members, the cross-timeframe confluence), ``age_bars`` (bars since the last
    member was confirmed), ``breaks`` (times the close has crossed it).
``res_dist_atr_{N}`` / ``sup_dist_atr_{N}``
    Same nearest-distance idea restricted to levels whose tier is at least ``N``,
    giving a direct obviousness-controlled comparison.
``n_levels_near``
    Number of levels within ``near_band_atr`` ATRs of the close.
``break_dir``, ``break_mag_atr``, ``break_vol_ratio``, ``break_touches``,
``break_tier``
    Set on bars where the close crossed a level since the previous close.
    ``break_dir`` is +1 upward, -1 downward, 0 none. Magnitude is how far beyond the
    level the close finished, in ATRs; the volume ratio is this bar's volume over
    its trailing mean. If several levels were crossed the one with the most touches
    is reported.

Usage
-----
::

    from src.api.binance import load_prices
    from src.features.pivots import pivot_table
    from src.features.levels import build_level_features

    df = load_prices(refresh=False)
    feats, levels = build_level_features(df, pivot_table(df))
"""

from __future__ import annotations
from dataclasses import dataclass, field
from src.features.indicators import atr as _atr, rolling_mean
from src.features.pivots import DEFAULT_NS
import argparse
import heapq
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# How many bars a swing of each tier stays "on the chart" after it occurred.
DEFAULT_LOOKBACK: dict[int, int] = {5: 24 * 30, 20: 24 * 180, 50: 24 * 730}

@dataclass
class Level:
    """
    One horizontal level and the confirmed pivots that make it up.
    """
    id: int
    price: float
    members: dict[tuple[int, str], int] = field(default_factory=dict)  # (idx, kind) -> largest confirmed n
    first_idx: int = 0   # occurrence bar of the earliest member
    last_idx: int = 0    # confirmation bar of the latest member (or upgrade)
    breaks: int = 0

    @property
    def touches(self) -> int:
        return len(self.members)

    @property
    def tier(self) -> int:
        return max(self.members.values())

    @property
    def n_tiers(self) -> int:
        return len(set(self.members.values()))

    @property
    def kinds(self) -> str:
        ks = {k for _, k in self.members}
        return "both" if len(ks) == 2 else next(iter(ks))

class _LevelBook:
    """
    Mutable set of levels with sorted price arrays for fast nearest lookups.

    One sorted view is kept per tier threshold so that "nearest level of tier >= N"
    is a single searchsorted call.
    """

    def __init__(self, tier_thresholds: tuple[int, ...], lookback: dict[int, int] | None = None) -> None:
        self.levels: dict[int, Level] = {}
        self.tier_thresholds = tier_thresholds
        self.lookback = lookback
        self._owner: dict[tuple[int, str], int] = {}        # (idx, kind) -> level id
        self._member_price: dict[tuple[int, str], float] = {}
        self._expiry: list[tuple[int, tuple[int, str], int]] = []  # (bar, key, n), lazily validated
        self._next_id = 0
        self._dirty = True
        self._prices = np.empty(0)
        self._ids = np.empty(0, dtype=int)
        self._by_tier: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def add_pivot(self, idx: int, kind: str, n: int, price: float, confirm_idx: int, tol: float) -> Level:
        key = (idx, kind)
        # Tier upgrade of an already-known swing: no new touch, just a bigger n.
        owner = self._owner.get(key)
        if owner is not None:
            lvl = self.levels[owner]
            if n > lvl.members[key]:
                lvl.members[key] = n
                lvl.last_idx = confirm_idx
                self._schedule_expiry(key, n)
                self._dirty = True
            return lvl
        # Otherwise merge into the closest level within tolerance, or open a new one.
        self._ensure_sorted()
        if self._prices.size:
            pos = np.searchsorted(self._prices, price)
            cands = [p for p in (pos - 1, pos) if 0 <= p < self._prices.size]
            best = min(cands, key=lambda p: abs(self._prices[p] - price))
            if abs(self._prices[best] - price) <= tol:
                lvl = self.levels[int(self._ids[best])]
                lvl.members[key] = n
                self._member_price[key] = price
                self._owner[key] = lvl.id
                lvl.price = float(np.mean([self._member_price[k] for k in lvl.members]))
                lvl.first_idx = min(lvl.first_idx, idx)
                lvl.last_idx = confirm_idx
                self._schedule_expiry(key, n)
                self._dirty = True
                return lvl
        lvl = Level(id=self._next_id, price=price, members={key: n}, first_idx=idx, last_idx=confirm_idx)
        self._member_price[key] = price
        self._owner[key] = lvl.id
        self.levels[lvl.id] = lvl
        self._next_id += 1
        self._schedule_expiry(key, n)
        self._dirty = True
        return lvl

    def _schedule_expiry(self, key: tuple[int, str], n: int) -> None:
        if self.lookback is None:
            return
        idx = key[0]
        heapq.heappush(self._expiry, (idx + self.lookback[n] + 1, key, n))

    def expire(self, t: int) -> None:
        """
        Drop members whose swing has scrolled off the chart by bar ``t``.

        Heap entries are validated lazily: an entry is stale if the member has since
        been upgraded to a larger tier (which pushed a later entry) or already removed.
        """
        while self._expiry and self._expiry[0][0] <= t:
            _, key, n = heapq.heappop(self._expiry)
            owner = self._owner.get(key)
            if owner is None:
                continue
            lvl = self.levels[owner]
            if lvl.members.get(key) != n:
                continue
            del lvl.members[key]
            del self._owner[key]
            del self._member_price[key]
            if lvl.members:
                lvl.price = float(np.mean([self._member_price[k] for k in lvl.members]))
                lvl.first_idx = min(k[0] for k in lvl.members)
            else:
                del self.levels[owner]
            self._dirty = True

    def _ensure_sorted(self) -> None:
        if not self._dirty:
            return
        if not self.levels:
            self._prices = np.empty(0)
            self._ids = np.empty(0, dtype=int)
            self._by_tier = {n: (self._prices, self._ids) for n in self.tier_thresholds}
        else:
            ids = np.fromiter(self.levels.keys(), dtype=int, count=len(self.levels))
            prices = np.fromiter((self.levels[i].price for i in ids), dtype="float64", count=ids.size)
            order = np.argsort(prices, kind="stable")
            self._prices, self._ids = prices[order], ids[order]
            tiers = np.fromiter((self.levels[i].tier for i in self._ids), dtype=int, count=ids.size)
            self._by_tier = {}
            for n in self.tier_thresholds:
                m = tiers >= n
                self._by_tier[n] = (self._prices[m], self._ids[m])
        self._dirty = False

    def nearest_above_below(self, price: float, min_tier: int | None = None) -> tuple[Level | None, Level | None]:
        """
        Nearest level strictly above and strictly below ``price``.
        """
        self._ensure_sorted()
        prices, ids = self._by_tier[min_tier] if min_tier is not None else (self._prices, self._ids)
        if prices.size == 0:
            return None, None
        hi = np.searchsorted(prices, price, side="right")
        lo = np.searchsorted(prices, price, side="left") - 1
        above = self.levels[int(ids[hi])] if hi < prices.size else None
        below = self.levels[int(ids[lo])] if lo >= 0 else None
        return above, below

    def between(self, a: float, b: float) -> list[Level]:
        """
        Levels with price in the closed interval [min(a, b), max(a, b)].
        """
        self._ensure_sorted()
        lo, hi = (a, b) if a <= b else (b, a)
        i = np.searchsorted(self._prices, lo, side="left")
        j = np.searchsorted(self._prices, hi, side="right")
        return [self.levels[int(k)] for k in self._ids[i:j]]

    def count_within(self, lo: float, hi: float) -> int:
        self._ensure_sorted()
        return int(np.searchsorted(self._prices, hi, side="right") - np.searchsorted(self._prices, lo, side="left"))

# --------------------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------------------

SIDE_FIELDS = ("dist_atr", "touches", "tier", "n_tiers", "age_bars", "breaks")

def feature_columns(ns: tuple[int, ...] = DEFAULT_NS) -> list[str]:
    cols = [f"{side}_{f}" for side in ("res", "sup") for f in SIDE_FIELDS]
    cols += [f"{side}_dist_atr_{n}" for n in ns for side in ("res", "sup")]
    cols += ["n_levels_near", "break_dir", "break_mag_atr", "break_vol_ratio", "break_touches", "break_tier"]
    return cols

def build_level_features(
    df: pd.DataFrame,
    pivots: pd.DataFrame,
    atr_n: int = 14,
    merge_tol_atr: float = 0.5,
    near_band_atr: float = 1.0,
    vol_n: int = 20,
    ns: tuple[int, ...] = DEFAULT_NS,
    lookback: dict[int, int] | None = DEFAULT_LOOKBACK,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build per-bar horizontal-level features from a pivot table.

    ``df`` is the OHLCV frame from `src.api.binance.load_prices`; ``pivots``
    the long table from `src.features.pivots.pivot_table`. ``lookback`` maps
    each tier ``N`` to how many bars a swing of that tier stays on the chart; pass
    ``None`` to keep every level forever. Returns ``(features, levels)``: the
    features aligned to ``df.index`` (NaN where no level exists on that side) and a
    table of every level still alive at the end of the series.
    """
    if lookback is not None:
        missing = set(ns) - set(lookback)
        if missing:
            raise ValueError(f"lookback has no entry for tiers {sorted(missing)}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("df must be sorted by time")
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
    out[:, ci["break_dir"]] = 0.0
    out[:, ci["n_levels_near"]] = 0.0

    book = _LevelBook(tuple(ns), lookback)
    ptr = 0
    for t in range(L):
        a = atr[t]
        book.expire(t)
        while ptr < ev_confirm.size and ev_confirm[ptr] == t:
            book.add_pivot(int(ev_idx[ptr]), str(ev_kind[ptr]), int(ev_n[ptr]), float(ev_price[ptr]), t, merge_tol_atr * a)
            ptr += 1
        if not book.levels or not np.isfinite(a) or a <= 0:
            continue
        c = close[t]

        # Breakouts: levels crossed between the previous close and this one.
        if t > 0:
            crossed = book.between(close[t - 1], c)
            crossed = [lvl for lvl in crossed if lvl.price != c]  # sitting exactly on it is not a break
            if crossed:
                best = max(crossed, key=lambda lvl: (lvl.touches, lvl.tier, -abs(lvl.price - c)))
                direction = 1.0 if c > close[t - 1] else -1.0
                for lvl in crossed:
                    lvl.breaks += 1
                out[t, ci["break_dir"]] = direction
                out[t, ci["break_mag_atr"]] = direction * (c - best.price) / a
                out[t, ci["break_vol_ratio"]] = volume[t] / vol_ma[t] if np.isfinite(vol_ma[t]) and vol_ma[t] > 0 else np.nan
                out[t, ci["break_touches"]] = best.touches
                out[t, ci["break_tier"]] = best.tier

        # Nearest resistance / support across all tiers.
        above, below = book.nearest_above_below(c)
        for side, lvl in (("res", above), ("sup", below)):
            if lvl is None:
                continue
            out[t, ci[f"{side}_dist_atr"]] = abs(lvl.price - c) / a
            out[t, ci[f"{side}_touches"]] = lvl.touches
            out[t, ci[f"{side}_tier"]] = lvl.tier
            out[t, ci[f"{side}_n_tiers"]] = lvl.n_tiers
            out[t, ci[f"{side}_age_bars"]] = t - lvl.last_idx
            out[t, ci[f"{side}_breaks"]] = lvl.breaks

        # Nearest of at least tier N.
        for n in ns:
            above_n, below_n = book.nearest_above_below(c, min_tier=n)
            if above_n is not None:
                out[t, ci[f"res_dist_atr_{n}"]] = (above_n.price - c) / a
            if below_n is not None:
                out[t, ci[f"sup_dist_atr_{n}"]] = (c - below_n.price) / a

        out[t, ci["n_levels_near"]] = book.count_within(c - near_band_atr * a, c + near_band_atr * a)

    feats = pd.DataFrame(out, index=df.index, columns=cols)
    return feats, level_table(book, df)

def level_table(book: _LevelBook, df: pd.DataFrame) -> pd.DataFrame:
    """
    Every level alive at the end of the series as a DataFrame, most-touched first.
    """
    rows = []
    for lvl in book.levels.values():
        rows.append(
            {
                "id": lvl.id,
                "price": lvl.price,
                "touches": lvl.touches,
                "tier": lvl.tier,
                "n_tiers": lvl.n_tiers,
                "kinds": lvl.kinds,
                "first_idx": lvl.first_idx,
                "first_time": df.index[lvl.first_idx],
                "last_idx": lvl.last_idx,
                "last_time": df.index[lvl.last_idx],
                "breaks": lvl.breaks,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["id", "price", "touches", "tier", "n_tiers", "kinds", "first_idx", "first_time", "last_idx", "last_time", "breaks"])
    return pd.DataFrame(rows).sort_values(["touches", "tier"], ascending=False, kind="stable").reset_index(drop=True)

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    from src.api.binance import load_prices
    from src.features.pivots import pivot_table
    import time

    p = argparse.ArgumentParser(description="Build horizontal-level features on the cached BTC series and summarise them.")
    p.add_argument("--refresh", action="store_true", help="update the price DB from Binance first")
    p.add_argument("--merge-tol", type=float, default=0.5, help="cluster tolerance in ATRs")
    p.add_argument("--near-band", type=float, default=1.0, help="confluence band in ATRs")
    p.add_argument("--top", type=int, default=10, help="print the most-touched levels")
    p.add_argument("--no-expiry", action="store_true", help="keep every level forever (no chart lookback)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = load_prices(refresh=args.refresh)
    piv = pivot_table(df)
    t0 = time.perf_counter()
    feats, levels = build_level_features(
        df, piv, merge_tol_atr=args.merge_tol, near_band_atr=args.near_band,
        lookback=None if args.no_expiry else DEFAULT_LOOKBACK,
    )
    log.info("built %d x %d features in %.1fs", *feats.shape, time.perf_counter() - t0)

    pd.set_option("display.width", 200)
    print(f"\n{len(levels)} levels alive at the end, from {len(piv)} pivot rows\n")
    print("touches distribution:")
    print(levels["touches"].value_counts().sort_index().to_string())
    print("\nfeature summary:")
    print(feats.describe().T[["count", "mean", "50%", "min", "max"]].to_string())
    print(f"\nbreak bars: {(feats['break_dir'] != 0).sum()} of {len(feats)}")
    print(f"\ntop {args.top} levels by touches:\n")
    print(levels.head(args.top).to_string(index=False))

if __name__ == "__main__":
    main()
