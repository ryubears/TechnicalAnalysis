"""
Tests for the modelling harness: feature matrix, events and labels, purged walk-forward, end-to-end run.
"""

from src.model.events import STRATEGIES, build_events, feature_groups, label_events
from src.model.features import CONTEXT_COLUMNS, build_feature_matrix, context_features
from src.model.harness import RunConfig, cochran_armitage, obviousness_table, run, wilson_interval
from src.model.walkforward import embargoed_tail_split, purged_walk_forward
import numpy as np
import pandas as pd
import pytest

def _frame(close, spread=0.5, seed=0):
    close = np.asarray(close, dtype="float64")
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=close.size, freq="h", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": close + spread, "low": close - spread, "close": close,
         "volume": rng.uniform(50, 150, size=close.size)},
        index=idx,
    )

def _walk(seed, size=6000):
    rng = np.random.default_rng(seed)
    return 100 + np.cumsum(rng.normal(size=size))

# --------------------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------------------

def test_context_features_are_causal():
    df = _frame(_walk(1, 3000))
    full = context_features(df)
    part = context_features(df.iloc[:2000])
    pd.testing.assert_frame_equal(full.iloc[:2000], part)
    assert list(full.columns) == CONTEXT_COLUMNS

def test_feature_matrix_has_all_strategies():
    df = _frame(_walk(2, 3000))
    X, meta = build_feature_matrix(df, method="nbar", ns=(5, 20))
    assert X.index.equals(df.index)
    for c in ("res_dist_atr", "tl_sup_dist_atr", "fib_5_retrace", "atr_pct"):
        assert c in X.columns
    assert not X.columns.duplicated().any()
    assert meta["ns"] == (5, 20)

# --------------------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------------------

def test_label_race_bounce_break_undecided_level_barriers():
    """
    Barriers measured from the level: resistance at 101 -> bounce below 100, break above 102.
    """
    close = np.full(100, 100.0)
    atr = np.ones(100)
    close[11:14] = [100.5, 100.2, 99.9]        # event 0 at t=10: falls through 100 at h=3 -> bounce
    close[31:33] = [101.5, 102.2]              # event 1 at t=30: closes above 102 at h=2 -> break
    close[51:60] = 99.4                        # event 2 at t=50: support at 99 -> barriers 100 / 98, sideways -> undecided
    close[96:100] = 99.5                       # event 3 at t=95: same support, only 4 bars left, nothing decided
    t = np.array([10, 30, 50, 95])
    p = np.array([101.0, 101.0, 99.0, 99.0])
    side = np.array([1.0, 1.0, -1.0, -1.0])
    label, hit = label_events(close, atr, t, p, side, horizon=8, react_atr=1.0, break_atr=1.0, barrier="level")
    assert label[0] == 1.0 and hit[0] == 3
    assert label[1] == 0.0 and hit[1] == 2
    assert np.isnan(label[2]) and np.isnan(hit[2])
    assert np.isnan(label[3])

def test_label_close_barriers_are_symmetric_around_event_close():
    """
    Default barriers sit at the event close +/- the ATR multiples, whatever the level's distance.
    """
    close = np.full(60, 100.0)
    atr = np.ones(60)
    close[11:14] = [99.6, 99.2, 98.9]          # event 0: resistance at 100.4; away barrier 99 -> bounce at h=3
    close[31:34] = [100.6, 100.9, 101.1]       # event 1: same resistance; through barrier 101 -> break at h=3
    t = np.array([10, 30]); p = np.array([100.4, 100.4]); side = np.array([1.0, 1.0])
    label, hit = label_events(close, atr, t, p, side, 8, 1.0, 1.0, barrier="close")
    assert label[0] == 1.0 and hit[0] == 3
    assert label[1] == 0.0 and hit[1] == 3
    with pytest.raises(ValueError):
        label_events(close, atr, t, p, side, 8, 1.0, 1.0, barrier="bogus")

def test_close_barriers_give_random_walk_a_half_base_rate():
    rng = np.random.default_rng(11)
    close = 100 + np.cumsum(rng.normal(size=60000))
    atr = np.ones(60000)
    t = np.arange(100, 59000, 7)
    p = close[t] + 0.3                               # a "resistance" 0.3 above every event close
    side = np.ones(t.size)
    lab_close, _ = label_events(close, atr, t, p, side, 48, 1.0, 1.0, barrier="close")
    lab_level, _ = label_events(close, atr, t, p, side, 48, 1.0, 1.0, barrier="level")
    assert abs(np.nanmean(lab_close) - 0.5) < 0.02          # symmetric: null is one half
    assert np.nanmean(lab_level) > 0.55                     # level-anchored: biased toward "bounce"

def test_label_support_is_mirrored():
    close = np.full(60, 100.0)
    atr = np.ones(60)
    close[11:14] = [99.6, 99.9, 100.6]        # support at 99: away barrier 100 -> bounce at h=3 (100.6 >= 100)
    close[31:34] = [99.2, 98.5, 97.9]         # support at 99: through barrier 98 -> break at h=3 (97.9 <= 98)
    label, hit = label_events(close, atr, np.array([10, 30]), np.array([99.0, 99.0]), np.array([-1.0, -1.0]), 8, 1.0, 1.0, barrier="level")
    assert label[0] == 1.0 and hit[0] == 3
    assert label[1] == 0.0 and hit[1] == 3

def test_events_first_bar_of_approach_only():
    df = _frame(_walk(3, 4000))
    X, meta = build_feature_matrix(df, method="nbar", ns=(5, 20))
    ev_first = build_events(df, X, "levels", meta["ns"], events_only=True)
    ev_all = build_events(df, X, "levels", meta["ns"], events_only=False)
    assert len(ev_all) > len(ev_first) > 0
    assert ev_first["label"].isin([0.0, 1.0]).sum() > 0
    # no two consecutive bars against the same level on the same side
    e = ev_first.sort_values(["side", "t"])
    same_side = e["side"].to_numpy()[1:] == e["side"].to_numpy()[:-1]
    consecutive = np.diff(e["t"].to_numpy()) == 1
    same_level = np.abs(np.diff(e["level_price"].to_numpy())) < 1e-9
    assert not (same_side & consecutive & same_level).any()
    # events near the end have no label
    assert ev_first.loc[ev_first["t"] + 24 >= len(df), "label"].isna().all()

def test_events_all_strategies_and_feature_groups():
    df = _frame(_walk(4, 4000))
    X, meta = build_feature_matrix(df, method="zigzag")
    for strategy in STRATEGIES:
        ev = build_events(df, X, strategy, meta["ns"])
        geometry, obviousness = feature_groups(strategy, meta["ns"])
        assert set(geometry) <= set(ev.columns) or ev.empty
        assert set(obviousness) <= set(ev.columns) or ev.empty
        assert not (set(geometry) & set(obviousness))
        if not ev.empty:
            assert ev["tier_rank"].dropna().isin([0, 1, 2]).all()
            assert ev["side"].isin([1, -1]).all()

# --------------------------------------------------------------------------------------
# walk-forward
# --------------------------------------------------------------------------------------

def test_purged_walk_forward_geometry():
    rng = np.random.default_rng(0)
    t = np.sort(rng.integers(0, 20000, size=3000))
    H, E = 24, 48
    folds = purged_walk_forward(t, 20000, H, E, test_bars=2000, min_train_bars=4000, min_train_events=10, min_test_events=5)
    assert len(folds) >= 5
    seen_test = np.zeros(t.size, dtype=bool)
    for f in folds:
        assert (t[f.train_idx] + H < f.test_start - E).all()          # purge + embargo
        assert (t[f.test_idx] >= f.test_start).all() and (t[f.test_idx] < f.test_end).all()
        assert not seen_test[f.test_idx].any()                           # no event tested twice
        seen_test[f.test_idx] = True
        assert not np.intersect1d(f.train_idx, f.test_idx).size
    starts = [f.test_start for f in folds]
    assert starts == sorted(starts) and all(b - a == 2000 for a, b in zip(starts, starts[1:]))
    with pytest.raises(ValueError):
        purged_walk_forward(t, 20000, H, embargo=H - 1)

def test_embargoed_tail_split_respects_gap():
    t = np.arange(0, 5000, 3)
    core, tail = embargoed_tail_split(t, horizon=24, embargo=24)
    assert core.size and tail.size
    assert t[core].max() + 24 < t[tail].min() - 24

# --------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------

def test_wilson_and_trend():
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi and hi - lo < 0.25
    z, p = cochran_armitage([100, 100, 100], [30, 50, 70])
    assert z > 3 and p < 0.01
    z0, p0 = cochran_armitage([100, 100, 100], [50, 50, 50])
    assert abs(z0) < 1e-9 and p0 == pytest.approx(1.0)

def test_obviousness_table_bins():
    e = pd.DataFrame({
        "label": np.r_[np.ones(60), np.zeros(40)],
        "touches": np.r_[np.full(50, 2.0), np.full(50, 4.0)],
        "tier_rank": np.r_[np.zeros(50), np.full(50, 2.0)],
        "n_tiers": np.r_[np.ones(50), np.full(50, 3.0)],
        "p_full": np.full(100, 0.6),
    })
    tab = obviousness_table(e)
    assert set(tab["axis"]) == {"touches", "tier", "n_tiers"}
    assert list(tab[tab["axis"] == "tier"]["bin"]) == ["minor", "major"]      # rank order, not alphabetical
    assert tab["n"].sum() == 300
    assert (tab["ci_low"] <= tab["bounce_rate"]).all() and (tab["bounce_rate"] <= tab["ci_high"]).all()

# --------------------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------------------

def test_harness_runs_end_to_end_on_synthetic_series():
    df = _frame(_walk(5, 9000))
    cfg = RunConfig(strategy="levels", method="nbar", horizon=12, test_bars=1500, min_train_bars=3000, num_rounds=40, early_stopping=10)
    X, meta = build_feature_matrix(df, method="nbar")
    res = run(df, cfg, X=X, ns=meta["ns"])
    s = res.summary
    assert s["n_folds"] >= 2 and s["n_oos"] > 0
    assert set(s["pooled"]) == {"base", "geometry", "full"}
    assert np.isfinite(s["pooled"]["full"]["logloss"])
    assert len(res.folds) == s["n_folds"]
    assert (res.events["p_full"].dropna().between(0, 1)).all()
    assert set(res.importance["feature"]) == set(s["feature_groups"]["geometry"] + s["feature_groups"]["obviousness"])
    # out-of-sample predictions exist only inside test blocks, never before the first one
    first = res.folds["test_start"].min()
    assert res.events.loc[res.events["time"] < first, "p_full"].isna().all()
