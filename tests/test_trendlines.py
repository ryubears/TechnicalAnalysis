"""
Tests for src.features.trendlines.
"""

from src.features.pivots import pivot_table
from src.features.trendlines import build_trendline_features, feature_columns
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

def _zigzag_with_lows_on_line(lows_at, slope, base=100.0, size=400, amp=6.0, width=8):
    """
    Flat series with V-shaped dips whose bottoms sit exactly on the line base + slope * idx.
    """
    close = np.full(size, base + 20.0)
    for i in lows_at:
        bottom = base + slope * i
        close[i - width : i] = np.linspace(base + 20.0, bottom + 1.0, width)
        close[i] = bottom
        close[i + 1 : i + width + 1] = np.linspace(bottom + 1.0, base + 20.0, width)
    return close

def test_columns_and_alignment():
    df = _frame(_random_walk(1))
    feats, lines = build_trendline_features(df, pivot_table(df))
    assert list(feats.columns) == feature_columns()
    assert feats.index.equals(df.index)
    assert feats["tl_break_dir"].isin([-1, 0, 1]).all()
    assert (feats["tl_n_sup"] >= 0).all() and (feats["tl_n_res"] >= 0).all()

def test_no_lookahead_prefix_invariance():
    df = _frame(_random_walk(2))
    full, _ = build_trendline_features(df, pivot_table(df, ns=(5, 20)), ns=(5, 20))
    T = 900
    part, _ = build_trendline_features(df.iloc[:T], pivot_table(df.iloc[:T], ns=(5, 20)), ns=(5, 20))
    pd.testing.assert_frame_equal(full.iloc[:T], part)

def test_three_lows_on_a_line_make_one_line_with_three_touches():
    lows_at = (60, 160, 260)
    slope = 0.02
    df = _frame(_zigzag_with_lows_on_line(lows_at, slope), spread=0.1)
    piv = pivot_table(df, ns=(5,))
    feats, lines = build_trendline_features(df, piv, ns=(5,), touch_tol_atr=0.5, max_dist_atr=np.inf)
    assert len(lines) >= 1
    top = lines.iloc[0]
    assert top["kind"] == "low"
    assert top["touches"] == 3
    assert top["slope"] == pytest.approx(slope, rel=1e-6)
    # After the third touch is confirmed, the nearest support line reports 3 touches and a positive slope.
    t = int(piv[piv["idx"] == 260]["confirm_idx"].item()) + 5
    assert feats["tl_sup_touches"].iloc[t] == 3
    assert feats["tl_sup_slope_atr"].iloc[t] > 0
    assert feats["tl_sup_dist_atr"].iloc[t] > 0

def test_dirty_chord_is_not_drawn():
    """
    Two lows with a deeper low between them cannot be joined by a support line.
    """
    close = np.full(400, 120.0)
    for i, bottom in ((60, 100.0), (160, 90.0), (260, 100.0)):
        close[i - 8 : i] = np.linspace(120, bottom + 1, 8)
        close[i] = bottom
        close[i + 1 : i + 9] = np.linspace(bottom + 1, 120, 8)
    df = _frame(close, spread=0.1)
    piv = pivot_table(df, ns=(5,))
    _, lines = build_trendline_features(df, piv, ns=(5,))
    pairs = set(zip(lines["a_idx"], lines["b_idx"]))
    assert (60, 260) not in pairs

def test_break_retires_line_and_emits_features():
    lows_at = (60, 160)
    slope = 0.05
    close = _zigzag_with_lows_on_line(lows_at, slope)
    close[300:] = 50.0                      # decisive close through the rising support line
    df = _frame(close, spread=0.1)
    piv = pivot_table(df, ns=(5,))
    feats, lines = build_trendline_features(df, piv, ns=(5,), max_dist_atr=np.inf)
    assert feats["tl_break_dir"].iloc[300] == -1
    assert feats["tl_break_mag_atr"].iloc[300] > 0
    assert feats["tl_break_touches"].iloc[300] == 2
    assert (feats["tl_n_sup"].iloc[301:] == 0).all()
    assert not ((lines["a_idx"] == 60) & (lines["b_idx"] == 160)).any()

def test_resistance_line_is_above_close_and_support_below():
    df = _frame(_random_walk(3, size=3000))
    feats, _ = build_trendline_features(df, pivot_table(df))
    sup = feats["tl_sup_dist_atr"].dropna()
    res = feats["tl_res_dist_atr"].dropna()
    assert (sup >= -1e-9).all()
    assert (res >= -1e-9).all()

def test_tier_restricted_distance_is_never_closer():
    df = _frame(_random_walk(4, size=4000))
    feats, _ = build_trendline_features(df, pivot_table(df))
    for side in ("sup", "res"):
        base = feats[f"tl_{side}_dist_atr"]
        for n in (5, 20, 50):
            col = feats[f"tl_{side}_dist_atr_{n}"]
            m = col.notna() & base.notna()
            assert (col[m] >= base[m] - 1e-12).all()
        m = feats[f"tl_{side}_dist_atr_5"].notna()
        np.testing.assert_allclose(feats[f"tl_{side}_dist_atr_5"][m], base[m])

def test_tier_upgrade_does_not_add_touch():
    lows_at = (60, 160, 260)
    df = _frame(_zigzag_with_lows_on_line(lows_at, 0.02, size=600), spread=0.1)
    piv = pivot_table(df, ns=(5, 20, 50))
    _, lines = build_trendline_features(df, piv, ns=(5, 20, 50), touch_tol_atr=0.5, max_dist_atr=np.inf)
    top = lines.iloc[0]
    assert top["touches"] == 3
    assert top["tier"] == 50   # the dips are 100 bars apart so every low is a major pivot

def test_line_far_from_price_is_pruned_without_a_break():
    """
    A support line that price runs away from is dropped once it is off-screen, not counted as a break.
    """
    lows_at = (60, 160)
    close = _zigzag_with_lows_on_line(lows_at, -0.5)   # steeply falling support line
    df = _frame(close, spread=0.5)
    piv = pivot_table(df, ns=(5,))
    kept, kept_lines = build_trendline_features(df, piv, ns=(5,), max_dist_atr=np.inf)
    pruned, pruned_lines = build_trendline_features(df, piv, ns=(5,), max_dist_atr=5.0)
    assert kept["tl_n_sup"].iloc[-1] >= 1 and not kept_lines.empty     # never closed through, so it survives
    assert pruned["tl_n_sup"].iloc[-1] == 0 and pruned_lines.empty     # but it is off-screen, so it is dropped
    assert (pruned["tl_n_sup"] >= 1).sum() < (kept["tl_n_sup"] >= 1).sum()
    assert (pruned["tl_break_dir"] == 0).all()
