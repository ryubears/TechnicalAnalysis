"""
Tests for src.features.fibonacci.
"""

from src.features.fibonacci import RATIOS, build_fibonacci_features, feature_columns
from src.features.pivots import pivot_table
import numpy as np
import pandas as pd
import pytest

def _frame(close, spread=0.5):
    close = np.asarray(close, dtype="float64")
    idx = pd.date_range("2020-01-01", periods=close.size, freq="h", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": close + spread, "low": close - spread, "close": close, "volume": np.ones(close.size)},
        index=idx,
    )

def _random_walk(seed, size=1500):
    rng = np.random.default_rng(seed)
    return 100 + np.cumsum(rng.normal(size=size))

def _up_swing_then_pullback(low_at=50, high_at=150, low_price=100.0, high_price=200.0, pull_to=161.8, size=400):
    """
    Ramp from a low to a high, then pull back to ``pull_to`` and hold there.
    """
    close = np.full(size, low_price + 20.0)
    close[low_at - 10 : low_at] = np.linspace(low_price + 20.0, low_price + 1.0, 10)
    close[low_at] = low_price
    close[low_at + 1 : high_at] = np.linspace(low_price + 1.0, high_price - 1.0, high_at - low_at - 1)
    close[high_at] = high_price
    close[high_at + 1 : high_at + 21] = np.linspace(high_price - 1.0, pull_to, 20)
    close[high_at + 21 :] = pull_to
    return close

def test_columns_and_alignment():
    df = _frame(_random_walk(1))
    feats, swings = build_fibonacci_features(df, pivot_table(df))
    assert list(feats.columns) == feature_columns()
    assert feats.index.equals(df.index)
    for n in (5, 20, 50):
        assert feats[f"fib_{n}_dir"].dropna().isin([-1, 1]).all()
        assert feats[f"fib_{n}_break_dir"].isin([-1, 0, 1]).all()
    assert set(swings["n"]) <= {5, 20, 50}

def test_no_lookahead_prefix_invariance():
    df = _frame(_random_walk(2))
    full, _ = build_fibonacci_features(df, pivot_table(df, ns=(5, 20)), ns=(5, 20))
    T = 900
    part, _ = build_fibonacci_features(df.iloc[:T], pivot_table(df.iloc[:T], ns=(5, 20)), ns=(5, 20))
    pd.testing.assert_frame_equal(full.iloc[:T], part)

def test_up_swing_pullback_sits_on_the_0382_level():
    df = _frame(_up_swing_then_pullback(), spread=0.0)
    piv = pivot_table(df, ns=(5,))
    feats, swings = build_fibonacci_features(df, piv, ns=(5,))
    t = 300
    sw = swings[swings["n"] == 5].iloc[0]
    assert sw["dir"] == 1 and sw["start_price"] == 100.0 and sw["end_price"] == 200.0
    assert feats["fib_5_dir"].iloc[t] == 1
    assert feats["fib_5_retrace"].iloc[t] == pytest.approx(0.382, abs=1e-9)
    # 161.8 is exactly the 0.382 level: it is "at or below" the close, so it is the support.
    assert feats["fib_5_sup_ratio"].iloc[t] == pytest.approx(0.382)
    assert feats["fib_5_sup_dist_atr"].iloc[t] == pytest.approx(0.0, abs=1e-9)
    assert feats["fib_5_res_ratio"].iloc[t] == pytest.approx(0.236)
    assert feats["fib_5_range_atr"].iloc[t] > 0
    assert feats["fib_5_swing_bars"].iloc[t] == 100

def test_down_swing_is_mirrored():
    close = 300.0 - _up_swing_then_pullback()      # high at 50 (200), low at 150 (100), bounce to 138.2
    df = _frame(close, spread=0.0)
    feats, swings = build_fibonacci_features(df, pivot_table(df, ns=(5,)), ns=(5,))
    sw = swings[swings["n"] == 5].iloc[0]
    assert sw["dir"] == -1 and sw["start_price"] == 200.0 and sw["end_price"] == 100.0
    assert feats["fib_5_retrace"].iloc[300] == pytest.approx(0.382, abs=1e-9)
    # The close sits exactly on the 0.382 level (138.2); floating point decides which side it lands on,
    # so the level must show up as either the support or the resistance at zero distance.
    row = feats.iloc[300]
    on_sup = row["fib_5_sup_ratio"] == pytest.approx(0.382) and row["fib_5_sup_dist_atr"] == pytest.approx(0.0, abs=1e-9)
    on_res = row["fib_5_res_ratio"] == pytest.approx(0.382) and row["fib_5_res_dist_atr"] == pytest.approx(0.0, abs=1e-9)
    assert on_sup or on_res
    # And the neighbouring levels are 0.236 below (123.6) and 0.5 above (150) in a down-swing.
    other_sup = 0.236 if on_res else 0.382
    other_res = 0.5 if on_sup else 0.382
    assert row["fib_5_sup_ratio"] == pytest.approx(other_sup)
    assert row["fib_5_res_ratio"] == pytest.approx(other_res)

def test_break_emitted_when_close_crosses_a_level():
    close = _up_swing_then_pullback(pull_to=190.0)
    close[300:] = 140.0                       # jumps through several levels at bar 300
    df = _frame(close, spread=0.0)
    feats, _ = build_fibonacci_features(df, pivot_table(df, ns=(5,)), ns=(5,))
    assert feats["fib_5_break_dir"].iloc[300] == -1
    assert feats["fib_5_break_ratio"].iloc[300] == pytest.approx(0.5)   # 150 is the level nearest the new close of 140
    assert feats["fib_5_break_mag_atr"].iloc[300] > 0
    assert (feats["fib_5_break_dir"].iloc[301:] == 0).all()

def test_later_minor_pivot_on_a_level_counts_as_touch_on_the_major_swing():
    """
    A minor pivot high landing on the major swing's 0.236 level (176.4) is a touch on that level.

    The peak is placed at bar 197: within 50 bars of the swing high at 150 so it is not itself a
    major pivot (which would redraw the swing), and confirmed at 202, after the major swing forms
    at 200.
    """
    close = _up_swing_then_pullback(pull_to=161.8)
    close[192:197] = np.linspace(161.8, 173.0, 5)
    close[197] = 176.4
    close[198:203] = np.linspace(173.0, 161.8, 5)
    df = _frame(close, spread=0.0)
    piv = pivot_table(df, ns=(5, 50))
    assert 197 in set(piv[(piv["n"] == 5) & (piv["kind"] == "high")]["idx"])
    assert 197 not in set(piv[(piv["n"] == 50) & (piv["kind"] == "high")]["idx"])
    feats, swings = build_fibonacci_features(df, piv, ns=(5, 50), touch_tol_atr=0.5)
    sw50 = swings[swings["n"] == 50].iloc[0]
    assert sw50["start_price"] == 100.0 and sw50["end_price"] == 200.0
    assert sw50["touches_total"] == 1
    t = 210
    assert feats["fib_50_res_ratio"].iloc[t] == pytest.approx(0.236)
    assert feats["fib_50_res_touches"].iloc[t] == 1
    assert np.isnan(feats["fib_50_res_touches"].iloc[199])   # major swing only forms at 200
    assert feats["fib_50_res_touches"].iloc[201] == 0        # swing exists, touch not confirmed until 202

def test_swing_is_redrawn_when_new_tier_pivot_confirms():
    df = _frame(_random_walk(3, size=3000))
    piv = pivot_table(df, ns=(5,))
    feats, _ = build_fibonacci_features(df, piv, ns=(5,))
    age = feats["fib_5_age_bars"].dropna()
    # Age must reset to something small at every confirmation of a new tier-5 pivot.
    resets = (age.diff() < 0).sum()
    assert resets > 50
    assert age.max() < 200

def test_n_near_counts_levels_within_band():
    df = _frame(_random_walk(4, size=3000))
    feats, _ = build_fibonacci_features(df, pivot_table(df), near_band_atr=0.0)
    assert (feats["fib_n_near"] <= 3).all()          # at most one level per tier can be at exactly zero distance
    feats_wide, _ = build_fibonacci_features(df, pivot_table(df), near_band_atr=1e9)
    tiers_active = feats[[f"fib_{n}_dir" for n in (5, 20, 50)]].notna().sum(axis=1)
    assert (feats_wide["fib_n_near"] == tiers_active * len(RATIOS)).all()

def test_tiny_swing_is_not_drawn():
    df = _frame(_random_walk(5, size=2000))
    piv = pivot_table(df, ns=(5,))
    none, _ = build_fibonacci_features(df, piv, ns=(5,), min_range_atr=1e9)
    some, _ = build_fibonacci_features(df, piv, ns=(5,), min_range_atr=0.0)
    assert none["fib_5_dir"].isna().all()
    assert some["fib_5_dir"].notna().sum() > 1000
    assert np.isfinite(some["fib_5_retrace"].dropna()).all()
