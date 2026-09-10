"""
Tests for src.features.pivots.
"""

from src.features.indicators import atr
from src.features.pivots import (
    METHODS,
    default_ns,
    detect_pivots,
    detect_pivots_bruteforce,
    detect_pivots_kernel,
    detect_pivots_zigzag,
    kernel_smooth,
    known_pivots,
    pivot_events,
    pivot_table,
)
import numpy as np
import pandas as pd
import pytest

def _frame(high, low):
    idx = pd.date_range("2020-01-01", periods=len(high), freq="h", tz="UTC")
    return pd.DataFrame({"high": high, "low": low}, index=idx)

@pytest.mark.parametrize("n", [1, 3, 5, 20])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_bruteforce_on_random_walk(n, seed):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(size=600))
    high = close + rng.uniform(0, 1, size=close.size)
    low = close - rng.uniform(0, 1, size=close.size)
    fh, fl, _, _ = detect_pivots(high, low, n)
    rh, rl = detect_pivots_bruteforce(high, low, n)
    assert np.array_equal(fh, rh)
    assert np.array_equal(fl, rl)

def test_simple_peak_and_trough():
    high = np.array([1, 2, 3, 5, 3, 2, 1, 2, 3, 4], dtype=float)
    low = high - 1
    is_h, is_l, prom_h, prom_l = detect_pivots(high, low, n=2)
    assert np.flatnonzero(is_h).tolist() == [3]
    assert np.flatnonzero(is_l).tolist() == [6]
    assert prom_h[3] == pytest.approx(5 - 3)  # beats max(3,3) of side windows
    assert prom_l[6] == pytest.approx(1 - 0)  # low[6] = 0, neighbours' min = 1
    assert np.isnan(prom_h[0])

def test_no_pivots_within_n_of_edges():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    n = 7
    is_h, is_l, _, _ = detect_pivots(x, x, n)
    assert not is_h[:n].any() and not is_h[-n:].any()
    assert not is_l[:n].any() and not is_l[-n:].any()

def test_ties_are_not_pivots():
    high = np.array([1, 2, 5, 5, 2, 1, 1, 1], dtype=float)
    is_h, _, _, _ = detect_pivots(high, high - 1, n=2)
    assert not is_h.any()

def test_confirm_idx_is_occurrence_plus_n():
    rng = np.random.default_rng(3)
    close = np.cumsum(rng.normal(size=400))
    df = _frame(close + 0.5, close - 0.5)
    piv = pivot_table(df, ns=(5, 20))
    assert (piv["confirm_idx"] == piv["idx"] + piv["n"]).all()
    assert (piv["confirm_idx"] < len(df)).all()
    assert (piv["confirm_time"] == df.index[piv["confirm_idx"]]).all()
    assert piv["confirm_idx"].is_monotonic_increasing

def test_tiers_nest():
    """Every pivot at a larger n must also be a pivot at every smaller n."""
    rng = np.random.default_rng(4)
    close = np.cumsum(rng.normal(size=3000))
    df = _frame(close + 0.5, close - 0.5)
    piv = pivot_table(df, ns=(5, 20, 50))
    for kind in ("high", "low"):
        k = piv[piv["kind"] == kind]
        s5 = set(k[k["n"] == 5]["idx"])
        s20 = set(k[k["n"] == 20]["idx"])
        s50 = set(k[k["n"] == 50]["idx"])
        assert s50 <= s20 <= s5
        assert len(s50) < len(s20) < len(s5)

def test_known_pivots_respects_confirmation_and_upgrades_tier():
    rng = np.random.default_rng(5)
    close = np.cumsum(rng.normal(size=3000))
    df = _frame(close + 0.5, close - 0.5)
    piv = pivot_table(df, ns=(5, 20, 50))
    major = piv[piv["n"] == 50].iloc[0]
    i = int(major["idx"])

    # Before the minor confirmation nothing about this swing is known.
    k = known_pivots(piv, i + 4)
    assert not ((k["idx"] == i) & (k["kind"] == major["kind"])).any()

    # Between the minor and intermediate confirmations it is a minor pivot.
    k = known_pivots(piv, i + 5)
    row = k[(k["idx"] == i) & (k["kind"] == major["kind"])]
    assert len(row) == 1 and row["n"].item() == 5

    # After the major confirmation it reports as major.
    k = known_pivots(piv, i + 50)
    row = k[(k["idx"] == i) & (k["kind"] == major["kind"])]
    assert len(row) == 1 and row["n"].item() == 50

    # One row per swing, never duplicated across tiers.
    assert not k.duplicated(["idx", "kind"]).any()

def test_pivot_events_stamped_at_confirmation():
    rng = np.random.default_rng(6)
    close = np.cumsum(rng.normal(size=500))
    df = _frame(close + 0.5, close - 0.5)
    piv = pivot_table(df, ns=(5,))
    ev = pivot_events(df, piv)
    assert list(ev.columns) == ["high_5", "low_5"]
    assert len(ev) == len(df)
    highs = piv[piv["kind"] == "high"]
    assert ev["high_5"].notna().sum() == len(highs)
    for _, r in highs.iterrows():
        assert ev["high_5"].iloc[int(r["confirm_idx"])] == pytest.approx(r["price"])
    # Every bar that is not a confirmation bar must be NaN.
    not_confirm = np.ones(len(df), dtype=bool)
    not_confirm[highs["confirm_idx"].to_numpy()] = False
    assert ev["high_5"].to_numpy()[not_confirm].size == not_confirm.sum()
    assert np.isnan(ev["high_5"].to_numpy()[not_confirm]).all()

def test_empty_result_has_columns():
    df = _frame(np.ones(10), np.zeros(10))
    piv = pivot_table(df, ns=(3,))
    assert piv.empty
    assert "confirm_idx" in piv.columns

def _ohlc(close, spread=0.5):
    close = np.asarray(close, dtype="float64")
    idx = pd.date_range("2020-01-01", periods=close.size, freq="h", tz="UTC")
    return pd.DataFrame({"open": close, "high": close + spread, "low": close - spread, "close": close}, index=idx)

def _walk(seed, size=3000):
    rng = np.random.default_rng(seed)
    return 100 + np.cumsum(rng.normal(size=size))

def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        pivot_table(_ohlc(_walk(0, 300)), method="bogus")
    assert set(METHODS) == {"nbar", "zigzag", "kernel"}
    assert default_ns("nbar") == (5, 20, 50)

def test_zigzag_alternates_and_confirms_on_reversal():
    df = _ohlc(_walk(1))
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    a = atr(h, l, c)
    k = 2.0
    recs = detect_pivots_zigzag(h, l, a, k)
    assert len(recs) > 20
    kinds = [r[1] for r in recs]
    assert all(x != y for x, y in zip(kinds, kinds[1:]))           # strict alternation
    idxs = [r[0] for r in recs]
    assert all(x < y for x, y in zip(idxs, idxs[1:]))               # in time order
    for i, kind, price, confirm, prom in recs:
        assert confirm > i
        if kind == "high":
            assert price - l[confirm] >= k * a[confirm] - 1e-9 or prom >= k * a[confirm] - 1e-9
            assert h[i] == price and h[i] == h[: confirm + 1][i]
        else:
            assert h[confirm] - price >= k * a[confirm] - 1e-9 or prom >= k * a[confirm] - 1e-9
            assert l[i] == price

def test_zigzag_larger_threshold_fewer_pivots():
    df = _ohlc(_walk(2))
    n = {k: len(pivot_table(df, ns=(k,), method="zigzag")) for k in (2, 4, 8)}
    assert n[2] > n[4] > n[8] > 0

def test_zigzag_prefix_invariance():
    df = _ohlc(_walk(3))
    T = 2000
    full = pivot_table(df, method="zigzag")
    part = pivot_table(df.iloc[:T], method="zigzag")
    f = full[full["confirm_idx"] < T].reset_index(drop=True)
    pd.testing.assert_frame_equal(f, part.reset_index(drop=True))

def test_kernel_finds_sine_peaks_and_stamps_confirmation():
    t = np.arange(1200)
    close = 100 + 10 * np.sin(2 * np.pi * t / 100)     # peaks every 100 bars at t = 25, 125, ...
    df = _ohlc(close, spread=0.0)
    bw = 3
    recs = detect_pivots_kernel(df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(), bw)
    highs = sorted(r[0] for r in recs if r[1] == "high")
    lows = sorted(r[0] for r in recs if r[1] == "low")
    H = int(np.ceil(3 * bw))
    for i in highs:
        assert min(abs(i - p) for p in range(25, 1200, 100)) <= 1
    for i in lows:
        assert min(abs(i - p) for p in range(75, 1200, 100)) <= 1
    assert len(highs) >= 10 and len(lows) >= 10
    for i, kind, price, confirm, prom in recs:
        assert confirm > i and confirm - i <= 2 * H + 2
        assert prom >= 0

def test_kernel_smooth_shape_and_halfwidth():
    m, H = kernel_smooth(np.ones(50), 2.0)
    assert H == 6 and m.shape == (50,)
    np.testing.assert_allclose(m[H:-H], 1.0)

def test_kernel_larger_bandwidth_fewer_pivots():
    df = _ohlc(_walk(4))
    n = {bw: len(pivot_table(df, ns=(bw,), method="kernel")) for bw in (3, 8, 20)}
    assert n[3] > n[8] > n[20] > 0

def test_kernel_prefix_invariance():
    df = _ohlc(_walk(5))
    T = 2000
    full = pivot_table(df, method="kernel")
    part = pivot_table(df.iloc[:T], method="kernel")
    f = full[full["confirm_idx"] < T].reset_index(drop=True)
    pd.testing.assert_frame_equal(f, part.reset_index(drop=True))

def test_all_methods_share_table_shape():
    df = _ohlc(_walk(6))
    for method in METHODS:
        piv = pivot_table(df, method=method)
        assert list(piv.columns) == ["idx", "time", "kind", "n", "tier", "price", "confirm_idx", "confirm_time", "prominence"]
        assert (piv["confirm_idx"] > piv["idx"]).all()
        assert piv["confirm_idx"].is_monotonic_increasing
        assert set(piv["n"]) <= set(default_ns(method))
        assert set(piv["tier"].astype(str)) <= {"minor", "intermediate", "major"}
