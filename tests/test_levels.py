"""
Tests for src.features.levels.
"""

from src.features.indicators import atr, true_range
from src.features.levels import DEFAULT_LOOKBACK, build_level_features, feature_columns
from src.features.pivots import pivot_table
import numpy as np
import pandas as pd
import pytest

def _frame(close, spread=0.5, volume=None):
    close = np.asarray(close, dtype="float64")
    idx = pd.date_range("2020-01-01", periods=close.size, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": close,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": np.ones(close.size) if volume is None else volume,
        },
        index=idx,
    )

def _random_walk(seed, size=1500):
    rng = np.random.default_rng(seed)
    return 100 + np.cumsum(rng.normal(size=size))

def test_true_range_and_atr_basic():
    high = np.array([2, 3, 5, 4], dtype=float)
    low = np.array([1, 2, 3, 3], dtype=float)
    close = np.array([1.5, 2.5, 4.5, 3.5], dtype=float)
    tr = true_range(high, low, close)
    assert tr[0] == 1.0
    assert tr[1] == pytest.approx(max(1.0, abs(3 - 1.5), abs(2 - 1.5)))
    a = atr(high, low, close, n=2)
    assert a.shape == high.shape and np.all(a > 0)

def test_no_lookahead_prefix_invariance():
    """
    Features on the first T bars must not change when later bars are appended.
    """
    df = _frame(_random_walk(1))
    full, _ = build_level_features(df, pivot_table(df, ns=(5, 20)), ns=(5, 20))
    T = 900
    part, _ = build_level_features(df.iloc[:T], pivot_table(df.iloc[:T], ns=(5, 20)), ns=(5, 20))
    pd.testing.assert_frame_equal(full.iloc[:T], part)

def test_columns_and_alignment():
    df = _frame(_random_walk(2))
    feats, levels = build_level_features(df, pivot_table(df))
    assert list(feats.columns) == feature_columns()
    assert feats.index.equals(df.index)
    assert (feats["break_dir"].isin([-1, 0, 1])).all()
    dist = feats.filter(like="dist_atr")
    assert ((dist >= 0) | dist.isna()).all().all()
    assert (levels["touches"] >= 1).all()

def test_features_nan_before_first_confirmation():
    df = _frame(_random_walk(3))
    piv = pivot_table(df, ns=(5,))
    feats, _ = build_level_features(df, piv, ns=(5,))
    first = int(piv["confirm_idx"].min())
    assert feats["res_dist_atr"].iloc[:first].isna().all()
    assert feats["sup_dist_atr"].iloc[:first].isna().all()
    assert feats[["res_dist_atr", "sup_dist_atr"]].iloc[first:].notna().any(axis=1).all()

def test_repeated_touches_cluster_into_one_level():
    """
    Three separate swing highs at the same price should form one level with 3 touches.
    """
    base = np.full(400, 100.0)
    close = base.copy()
    for peak in (60, 160, 260):
        close[peak - 10 : peak] = np.linspace(100, 110, 10)
        close[peak] = 112.0
        close[peak + 1 : peak + 11] = np.linspace(110, 100, 10)
    df = _frame(close, spread=0.1)
    piv = pivot_table(df, ns=(5,))
    feats, levels = build_level_features(df, piv, ns=(5,), merge_tol_atr=1.0)
    top = levels.iloc[0]
    assert top["touches"] == 3
    assert top["price"] == pytest.approx(112.1)  # high = close + spread
    # Once the third peak is confirmed the resistance seen from the flat 100 region has 3 touches.
    t = int(piv[piv["idx"] == 260]["confirm_idx"].item()) + 5
    assert feats["res_touches"].iloc[t] == 3
    assert feats["res_dist_atr"].iloc[t] > 0

def test_break_detected_when_close_crosses_level():
    close = np.full(300, 100.0)
    close[40:50] = np.linspace(100, 105, 10)
    close[50] = 106.0
    close[51:61] = np.linspace(105, 100, 10)
    close[200:] = 120.0  # decisive close through the level at bar 200
    df = _frame(close, spread=0.1)
    piv = pivot_table(df, ns=(5,))
    feats, levels = build_level_features(df, piv, ns=(5,))
    assert feats["break_dir"].iloc[200] == 1
    assert feats["break_mag_atr"].iloc[200] > 0
    assert feats["break_touches"].iloc[200] >= 1
    assert (feats["break_dir"].iloc[201:] == 0).all()
    assert levels["breaks"].max() >= 1
    # After the break the old resistance is now support (role reversal).
    assert feats["sup_touches"].iloc[210] >= 1
    assert np.isnan(feats["res_dist_atr"].iloc[210])

def test_tier_restricted_distances_are_never_closer_than_unrestricted():
    df = _frame(_random_walk(4, size=4000))
    feats, _ = build_level_features(df, pivot_table(df))
    for side in ("res", "sup"):
        base = feats[f"{side}_dist_atr"]
        for n in (5, 20, 50):
            col = feats[f"{side}_dist_atr_{n}"]
            m = col.notna() & base.notna()
            assert (col[m] >= base[m] - 1e-12).all()
        # tier >= 5 is every level, so it must equal the unrestricted distance.
        m = feats[f"{side}_dist_atr_5"].notna()
        np.testing.assert_allclose(feats[f"{side}_dist_atr_5"][m], base[m])

def test_tier_upgrade_does_not_add_touch():
    df = _frame(_random_walk(5, size=3000))
    piv = pivot_table(df)
    _, levels = build_level_features(df, piv, lookback=None)
    # Total members across levels equals unique swings, not pivot rows.
    n_swings = piv[["idx", "kind"]].drop_duplicates().shape[0]
    assert levels["touches"].sum() == n_swings

def test_minor_level_expires_after_lookback():
    """
    A lone minor swing must stop being a level once it scrolls off the chart.
    """
    close = np.full(400, 100.0)
    close[40:50] = np.linspace(100, 105, 10)
    close[50] = 106.0
    close[51:61] = np.linspace(105, 100, 10)
    df = _frame(close, spread=0.1)
    piv = pivot_table(df, ns=(5,))
    lookback = {5: 100}
    feats, levels = build_level_features(df, piv, ns=(5,), lookback=lookback)
    confirm = int(piv[piv["idx"] == 50]["confirm_idx"].item())
    assert feats["res_touches"].iloc[confirm] == 1
    assert feats["res_touches"].iloc[50 + 100] == 1        # still on chart at the last visible bar
    assert np.isnan(feats["res_touches"].iloc[50 + 101])   # gone the bar after
    assert (levels["price"] < 106).all() or levels.empty    # the 106 level is not alive at the end

def test_tier_upgrade_extends_lifetime():
    """
    A swing confirmed at a larger N inherits that tier's longer lookback.
    """
    df = _frame(_random_walk(7, size=3000))
    piv = pivot_table(df)
    short = {5: 50, 20: 50, 50: 2000}
    _, levels = build_level_features(df, piv, lookback=short)
    # Everything alive at the end must be major (tier 50) or recent.
    L = len(df)
    assert ((levels["tier"] == 50) | (levels["last_idx"] >= L - 50 - 50)).all()

def test_default_lookback_covers_default_tiers():
    assert set(DEFAULT_LOOKBACK) == {5, 20, 50}
    with pytest.raises(ValueError):
        df = _frame(_random_walk(8, size=300))
        build_level_features(df, pivot_table(df, ns=(5,)), ns=(5,), lookback={20: 10})
