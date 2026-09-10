"""
Approach events and reaction labels.

The thesis is about *reaction* at a level, not direction, so the unit of analysis is
an approach: the first bar at which the close comes within ``near_atr`` ATRs of the
nearest level on one side (resistance above, support below). Each approach is then
labelled by a race over the next ``horizon`` bars:

* **bounce (1)** if the close first moves ``react_atr`` ATRs *away* from the level,
* **break (0)** if the close first moves ``break_atr`` ATRs *toward and through* it,
* **undecided (NaN)** if neither happens in time; these are dropped from modelling
  but their share is reported.

With ``barrier="close"`` (default) both distances are measured from the event bar's
close, so the two barriers are symmetric and a random walk bounces half the time;
any excess is reaction. With ``barrier="level"`` they are measured from the level's
price instead, which is the more literal chart reading but biases the base rate:
an approach that starts half an ATR short of the level then needs only half an ATR
to bounce and one and a half to break. ATR is frozen at the event bar. Events whose
horizon runs past the end of the series have no label and are dropped, which is
also what keeps the walk-forward purge simple.

One event table is built per strategy. Each strategy exposes a nearest level above
and below the close, and its side-specific feature columns are renamed to generic
ones (``dist_atr``, ``touches``, ``tier_rank`` ...) so the same model code serves
horizontal levels, trendlines and Fibonacci ratios. ``feature_groups`` says which of
those columns are *geometry* (where price is relative to the level) and which are
*obviousness* (how watched the level is); the harness ablates the latter.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from src.features.indicators import atr as _atr
from typing import Sequence
import numpy as np
import pandas as pd

STRATEGIES: tuple[str, ...] = ("levels", "trendlines", "fibonacci")
CONTEXT = ["atr_pct", "ret_1_atr", "ret_4_atr", "ret_24_atr", "vol_ratio", "range_pos_24", "volume_ratio", "hour", "dow"]
EVENT_COLUMNS = ["t", "time", "side", "level_price", "label", "hit_bar", "strategy"]

@dataclass
class EventSpec:
    """
    How to read one strategy's feature columns for one side of the close.
    """
    features: dict[str, str]              # generic name -> column in X (side-specific)
    common: dict[str, str] = field(default_factory=dict)   # generic name -> column in X (side-agnostic)
    other_dist: str = ""                  # column with the distance to the level on the other side
    geometry: tuple[str, ...] = ()        # generic names that describe position relative to the level
    obviousness: tuple[str, ...] = ()     # generic names that describe how watched the level is
    tier_value: int | None = None         # fixed tier (Fibonacci pools one spec per tier)

def _levels_spec(side: str, other: str, ns: Sequence[int]) -> EventSpec:
    feats = {k: f"{side}_{k}" for k in ("dist_atr", "touches", "tier", "n_tiers", "age_bars", "breaks")}
    feats.update({f"dist_atr_{n}": f"{side}_dist_atr_{n}" for n in ns})
    return EventSpec(
        features=feats, common={"n_near": "n_levels_near"}, other_dist=f"{other}_dist_atr",
        geometry=("dist_atr", "other_dist_atr", "side"),
        obviousness=("touches", "tier_rank", "n_tiers", "age_bars", "breaks", "n_near") + tuple(f"dist_atr_{n}" for n in ns),
    )

def _trendlines_spec(side: str, other: str, ns: Sequence[int]) -> EventSpec:
    feats = {k: f"tl_{side}_{k}" for k in ("dist_atr", "slope_atr", "touches", "tier", "n_tiers", "age_bars", "span_bars")}
    feats.update({f"dist_atr_{n}": f"tl_{side}_dist_atr_{n}" for n in ns})
    return EventSpec(
        features=feats, common={"n_near": "tl_n_near", "n_side": f"tl_n_{side}"}, other_dist=f"tl_{other}_dist_atr",
        geometry=("dist_atr", "slope_atr", "other_dist_atr", "side"),
        obviousness=("touches", "tier_rank", "n_tiers", "age_bars", "span_bars", "n_near", "n_side") + tuple(f"dist_atr_{n}" for n in ns),
    )

def _fib_spec(side: str, other: str, n: int) -> EventSpec:
    p = f"fib_{n}_"
    return EventSpec(
        features={"dist_atr": f"{p}{side}_dist_atr", "ratio": f"{p}{side}_ratio", "touches": f"{p}{side}_touches"},
        common={"dir": f"{p}dir", "retrace": f"{p}retrace", "range_atr": f"{p}range_atr", "swing_bars": f"{p}swing_bars",
                "age_bars": f"{p}age_bars", "n_near": "fib_n_near"},
        other_dist=f"{p}{other}_dist_atr",
        geometry=("dist_atr", "other_dist_atr", "side", "dir", "retrace"),
        obviousness=("tier_rank", "touches", "ratio", "range_atr", "swing_bars", "age_bars", "n_near"),
        tier_value=n,
    )

def specs_for(strategy: str, ns: Sequence[int]) -> list[tuple[int, EventSpec]]:
    """
    ``(side, spec)`` pairs for a strategy: side +1 is the level above the close
    (resistance), -1 the level below (support). Fibonacci yields one pair per tier.
    """
    if strategy == "levels":
        return [(1, _levels_spec("res", "sup", ns)), (-1, _levels_spec("sup", "res", ns))]
    if strategy == "trendlines":
        return [(1, _trendlines_spec("res", "sup", ns)), (-1, _trendlines_spec("sup", "res", ns))]
    if strategy == "fibonacci":
        return [(s, _fib_spec(a, b, n)) for n in ns for s, a, b in ((1, "res", "sup"), (-1, "sup", "res"))]
    raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")

def feature_groups(strategy: str, ns: Sequence[int]) -> tuple[list[str], list[str]]:
    """
    ``(geometry_and_context, obviousness)`` generic column names for a strategy.
    """
    spec = specs_for(strategy, ns)[0][1]
    return list(spec.geometry) + CONTEXT, list(spec.obviousness)

def label_events(
    close: np.ndarray,
    atr: np.ndarray,
    t: np.ndarray,
    level_price: np.ndarray,
    side: np.ndarray,
    horizon: int,
    react_atr: float,
    break_atr: float,
    barrier: str = "close",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Race the two barriers over ``horizon`` bars after each event.

    ``barrier`` is ``"close"`` (barriers at the event close minus / plus the ATR
    multiples, symmetric) or ``"level"`` (barriers at the level's price minus / plus
    them). Returns ``(label, hit_bar)``: label 1.0 for a bounce, 0.0 for a break, NaN
    if undecided; ``hit_bar`` is the number of bars until the deciding close (NaN if
    undecided). A decision reached before the series ends counts even if the full
    horizon is not available; `build_events` blanks those events separately.
    """
    if barrier not in ("close", "level"):
        raise ValueError("barrier must be 'close' or 'level'")
    n = t.size
    label = np.full(n, np.nan)
    hit = np.full(n, np.nan)
    a = atr[t]
    origin = close[t] if barrier == "close" else level_price
    away = origin - side * react_atr * a      # bounce barrier: away from the level
    through = origin + side * break_atr * a   # break barrier: toward / through the level
    L = close.size
    open_ = np.ones(n, dtype=bool)
    for h in range(1, horizon + 1):
        tt = t + h
        ok = open_ & (tt < L)
        if not ok.any():
            break
        c = close[np.minimum(tt, L - 1)]
        bounced = ok & (side * (away - c) >= 0)     # resistance: c <= away ; support: c >= away
        broke = ok & (side * (c - through) >= 0)    # resistance: c >= through ; support: c <= through
        label[bounced] = 1.0
        label[broke & ~bounced] = 0.0
        decided = bounced | broke
        hit[decided] = h
        open_ &= ~decided
    return label, hit

def build_events(
    df: pd.DataFrame,
    X: pd.DataFrame,
    strategy: str,
    ns: Sequence[int],
    near_atr: float = 0.5,
    horizon: int = 24,
    react_atr: float = 1.0,
    break_atr: float = 1.0,
    events_only: bool = True,
    atr_n: int = 14,
    barrier: str = "close",
) -> pd.DataFrame:
    """
    Approach events for one strategy, with generic feature columns and labels.

    ``events_only`` keeps only the first bar of each approach (a new approach starts
    when the close re-enters the ``near_atr`` band or the level itself changes);
    ``False`` keeps every bar inside the band, which is far more autocorrelated.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
    ns = tuple(sorted(int(n) for n in ns))
    rank = {n: i for i, n in enumerate(ns)}
    close = df["close"].to_numpy(dtype="float64")
    a = _atr(df["high"].to_numpy(dtype="float64"), df["low"].to_numpy(dtype="float64"), close, atr_n)
    L = len(df)
    frames: list[pd.DataFrame] = []
    for side, spec in specs_for(strategy, ns):
        dist = X[spec.features["dist_atr"]].to_numpy(dtype="float64")
        near = np.isfinite(dist) & (dist <= near_atr)
        p = close + side * dist * a
        if events_only:
            prev_near = np.concatenate([[False], near[:-1]])
            prev_p = np.concatenate([[np.nan], p[:-1]])
            with np.errstate(invalid="ignore"):
                same = np.abs(p - prev_p) <= 0.05 * a
            keep = near & ~(prev_near & same)
        else:
            keep = near
        t = np.flatnonzero(keep)
        if t.size == 0:
            continue
        ev = pd.DataFrame({"t": t, "time": df.index[t], "side": side, "level_price": p[t]})
        for name, col in spec.features.items():
            ev[name] = X[col].to_numpy()[t]
        for name, col in spec.common.items():
            ev[name] = X[col].to_numpy()[t]
        ev["other_dist_atr"] = X[spec.other_dist].to_numpy()[t]
        tier = np.full(t.size, spec.tier_value, dtype="float64") if spec.tier_value is not None else ev["tier"].to_numpy(dtype="float64")
        ev["tier"] = tier
        ev["tier_rank"] = pd.Series(tier).map(lambda v: rank.get(int(v), np.nan) if np.isfinite(v) else np.nan).to_numpy()
        for c in CONTEXT:
            ev[c] = X[c].to_numpy()[t]
        frames.append(ev)
    if not frames:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    ev = pd.concat(frames, ignore_index=True).sort_values(["t", "side"], kind="stable").reset_index(drop=True)
    label, hit = label_events(close, a, ev["t"].to_numpy(dtype=int), ev["level_price"].to_numpy(), ev["side"].to_numpy(dtype="float64"), horizon, react_atr, break_atr, barrier)
    ev["label"] = label
    ev["hit_bar"] = hit
    ev["strategy"] = strategy
    ev.loc[ev["t"] + horizon >= L, "label"] = np.nan     # no complete future
    front = EVENT_COLUMNS
    return ev[front + [c for c in ev.columns if c not in front]]
