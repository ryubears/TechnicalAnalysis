"""
Modelling harness: does reaction at a level scale with how obvious the level is?

For one strategy and one pivot method the harness

1. builds the feature matrix and the approach events with reaction labels,
2. walks forward over purged, embargoed folds,
3. in each fold fits two LightGBM classifiers, one on geometry + context features
   only and one that also sees the *obviousness* features (touches, tier,
   cross-tier confluence, age, breaks ...), plus a constant base-rate baseline,
4. pools the out-of-sample predictions and reports fold metrics, the obviousness
   ablation, and a direct table of bounce rate by obviousness bin with a
   Cochran-Armitage trend test.

The ablation is the thesis test. If adding obviousness features does not improve
out-of-sample log loss, and the bounce rate is flat across bins, the levels are
noise regardless of how the aggregate statistics look.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence
from pathlib import Path
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from src.features.pivots import DEFAULT_METHOD, METHODS, default_ns
from src.model.events import STRATEGIES, build_events, feature_groups
from src.model.features import build_feature_matrix
from src.model.walkforward import Fold, embargoed_tail_split, purged_walk_forward
import argparse
import json
import lightgbm as lgb
import logging
import math
import numpy as np
import pandas as pd
import time

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

LGB_PARAMS: dict = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 7,
}

@dataclass
class RunConfig:
    strategy: str = "levels"
    method: str = DEFAULT_METHOD
    near_atr: float = 0.5
    horizon: int = 24
    react_atr: float = 1.0
    break_atr: float = 1.0
    barrier: str = "close"
    events_only: bool = True
    embargo: int | None = None
    test_bars: int = 24 * 180
    min_train_bars: int = 24 * 365
    num_rounds: int = 500
    early_stopping: int = 50
    params: dict = field(default_factory=lambda: dict(LGB_PARAMS))

@dataclass
class RunResult:
    config: RunConfig
    events: pd.DataFrame            # labelled events with out-of-sample predictions
    folds: pd.DataFrame             # per-fold metrics
    summary: dict                   # pooled metrics and ablation
    obviousness: pd.DataFrame       # bounce rate by obviousness bin
    importance: pd.DataFrame        # mean gain importance of the full model

# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------

def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    out = {"n": int(y.size), "base_rate": float(y.mean()) if y.size else np.nan, "logloss": np.nan, "brier": np.nan, "auc": np.nan}
    if y.size == 0:
        return out
    out["logloss"] = float(log_loss(y, p, labels=[0, 1]))
    out["brier"] = float(brier_score_loss(y, p))
    if 0 < y.mean() < 1:
        out["auc"] = float(roc_auc_score(y, p))
    return out

def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

def cochran_armitage(counts: Sequence[int], successes: Sequence[int]) -> tuple[float, float]:
    """
    Trend test for a success rate across ordered bins. Returns ``(z, p_two_sided)``.
    """
    n = np.asarray(counts, dtype="float64")
    r = np.asarray(successes, dtype="float64")
    keep = n > 0
    n, r = n[keep], r[keep]
    if n.size < 2 or n.sum() == 0:
        return (np.nan, np.nan)
    x = np.arange(n.size, dtype="float64")
    N, R = n.sum(), r.sum()
    pbar = R / N
    if pbar in (0.0, 1.0):
        return (np.nan, np.nan)
    T = float(np.sum(x * (r - n * pbar)))
    var = pbar * (1 - pbar) * (np.sum(n * x * x) - np.sum(n * x) ** 2 / N)
    if var <= 0:
        return (np.nan, np.nan)
    z = T / math.sqrt(var)
    return (z, math.erfc(abs(z) / math.sqrt(2)))

# --------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------

def _fit_predict(
    X_train: pd.DataFrame, y_train: np.ndarray, t_train: np.ndarray, X_test: pd.DataFrame, cfg: RunConfig,
) -> tuple[np.ndarray, lgb.Booster]:
    """
    Fit with early stopping on an embargoed tail of the training set, then predict.
    """
    embargo = cfg.embargo if cfg.embargo is not None else cfg.horizon
    core, tail = embargoed_tail_split(t_train, cfg.horizon, embargo)
    if tail.size < 50 or core.size < 100 or len(np.unique(y_train[tail])) < 2:
        core, tail = np.arange(t_train.size), np.empty(0, dtype=int)
    dtrain = lgb.Dataset(X_train.iloc[core], label=y_train[core], free_raw_data=False)
    valid = [lgb.Dataset(X_train.iloc[tail], label=y_train[tail], reference=dtrain)] if tail.size else []
    callbacks = [lgb.early_stopping(cfg.early_stopping, verbose=False)] if tail.size else []
    booster = lgb.train(cfg.params, dtrain, num_boost_round=cfg.num_rounds, valid_sets=valid, callbacks=callbacks)
    n_iter = booster.best_iteration or cfg.num_rounds
    return booster.predict(X_test, num_iteration=n_iter), booster

def run(
    df: pd.DataFrame,
    cfg: RunConfig,
    X: pd.DataFrame | None = None,
    ns: Sequence[int] | None = None,
) -> RunResult:
    """
    Run the full harness for one strategy / method. ``X`` and ``ns`` may be passed to
    reuse a feature matrix across strategies.
    """
    if cfg.strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {cfg.strategy!r}")
    if cfg.method not in METHODS:
        raise ValueError(f"unknown method {cfg.method!r}")
    if X is None:
        X, meta = build_feature_matrix(df, cfg.method)
        ns = meta["ns"]
    ns = tuple(ns) if ns is not None else default_ns(cfg.method)
    ev = build_events(df, X, cfg.strategy, ns, cfg.near_atr, cfg.horizon, cfg.react_atr, cfg.break_atr, cfg.events_only, barrier=cfg.barrier)
    n_all = len(ev)
    ev = ev[ev["label"].notna()].reset_index(drop=True)
    geometry, obviousness = feature_groups(cfg.strategy, ns)
    geometry = [c for c in geometry if c in ev.columns]
    obviousness = [c for c in obviousness if c in ev.columns]
    full_cols = geometry + obviousness
    y = ev["label"].to_numpy(dtype="float64")
    t = ev["t"].to_numpy(dtype=int)
    embargo = cfg.embargo if cfg.embargo is not None else cfg.horizon
    folds = purged_walk_forward(t, len(df), cfg.horizon, embargo, cfg.test_bars, cfg.min_train_bars)
    if not folds:
        raise RuntimeError("no usable folds: series too short for min_train_bars / test_bars, or too few events")

    p_full = np.full(len(ev), np.nan)
    p_geo = np.full(len(ev), np.nan)
    p_base = np.full(len(ev), np.nan)
    fold_rows = []
    gains: list[pd.Series] = []
    for f in folds:
        t0 = time.perf_counter()
        ytr, yte = y[f.train_idx], y[f.test_idx]
        base = float(ytr.mean())
        pf, booster = _fit_predict(ev[full_cols].iloc[f.train_idx], ytr, t[f.train_idx], ev[full_cols].iloc[f.test_idx], cfg)
        pg, _ = _fit_predict(ev[geometry].iloc[f.train_idx], ytr, t[f.train_idx], ev[geometry].iloc[f.test_idx], cfg)
        p_full[f.test_idx], p_geo[f.test_idx], p_base[f.test_idx] = pf, pg, base
        gains.append(pd.Series(booster.feature_importance("gain"), index=full_cols))
        mf, mg, mb = _metrics(yte, pf), _metrics(yte, pg), _metrics(yte, np.full(yte.size, base))
        fold_rows.append(
            {
                "fold": f.k, "test_start": df.index[f.test_start], "test_end": df.index[min(f.test_end, len(df) - 1)],
                "n_train": int(f.train_idx.size), "n_test": int(f.test_idx.size), "base_rate": mf["base_rate"],
                "logloss_base": mb["logloss"], "logloss_geometry": mg["logloss"], "logloss_full": mf["logloss"],
                "auc_geometry": mg["auc"], "auc_full": mf["auc"], "brier_full": mf["brier"],
                "seconds": round(time.perf_counter() - t0, 1),
            }
        )
        log.info("fold %d %s..%s train %d test %d | logloss base %.4f geo %.4f full %.4f | auc geo %.3f full %.3f",
                 f.k, str(df.index[f.test_start])[:10], str(df.index[min(f.test_end, len(df) - 1)])[:10], f.train_idx.size, f.test_idx.size,
                 mb["logloss"], mg["logloss"], mf["logloss"], mg["auc"] if mg["auc"] == mg["auc"] else -1, mf["auc"] if mf["auc"] == mf["auc"] else -1)

    ev["p_full"], ev["p_geometry"], ev["p_base"] = p_full, p_geo, p_base
    oos = ev["p_full"].notna()
    e = ev[oos]
    pooled = {
        "base": _metrics(e["label"].to_numpy(), e["p_base"].to_numpy()),
        "geometry": _metrics(e["label"].to_numpy(), e["p_geometry"].to_numpy()),
        "full": _metrics(e["label"].to_numpy(), e["p_full"].to_numpy()),
    }
    fold_df = pd.DataFrame(fold_rows)
    delta = (fold_df["logloss_geometry"] - fold_df["logloss_full"])
    summary = {
        "strategy": cfg.strategy, "method": cfg.method, "ns": list(ns), "barrier": cfg.barrier,
        "horizon": cfg.horizon, "near_atr": cfg.near_atr, "react_atr": cfg.react_atr, "break_atr": cfg.break_atr,
        "n_events_all": int(n_all), "n_events_labelled": int(len(ev)), "undecided_share": float(1 - len(ev) / n_all) if n_all else np.nan,
        "n_folds": int(len(folds)), "n_oos": int(oos.sum()),
        "pooled": pooled,
        "ablation": {
            "logloss_gain_mean": float(delta.mean()), "logloss_gain_std": float(delta.std(ddof=1)) if len(delta) > 1 else np.nan,
            "folds_improved": int((delta > 0).sum()), "auc_gain_mean": float((fold_df["auc_full"] - fold_df["auc_geometry"]).mean()),
        },
        "feature_groups": {"geometry": geometry, "obviousness": obviousness},
    }
    importance = pd.concat(gains, axis=1).mean(axis=1).sort_values(ascending=False).rename("gain").reset_index().rename(columns={"index": "feature"})
    return RunResult(cfg, ev, fold_df, summary, obviousness_table(e), importance)

# --------------------------------------------------------------------------------------
# Obviousness analysis
# --------------------------------------------------------------------------------------

def obviousness_table(e: pd.DataFrame) -> pd.DataFrame:
    """
    Out-of-sample bounce rate by touches bin and by tier, with Wilson intervals, the
    mean full-model probability, and a Cochran-Armitage trend z / p per axis.
    """
    rows = []
    axes: list[tuple[str, pd.Series]] = []
    if "touches" in e.columns and e["touches"].notna().any():
        tb = pd.cut(e["touches"], bins=[-np.inf, 1, 2, 3, 5, np.inf], labels=["0-1", "2", "3", "4-5", "6+"])
        axes.append(("touches", tb))
    if "tier_rank" in e.columns and e["tier_rank"].notna().any():
        names = ["minor", "intermediate", "major"]
        labels = e["tier_rank"].map(lambda v: names[int(v)] if v == v and 0 <= int(v) < 3 else np.nan)
        axes.append(("tier", pd.Categorical(labels, categories=names, ordered=True)))
    if "n_tiers" in e.columns and e["n_tiers"].notna().any():
        axes.append(("n_tiers", e["n_tiers"].round().astype("Int64")))
    for axis, key in axes:
        g = e.groupby(key, observed=True)
        counts, succ = [], []
        for name, grp in g:
            k, n = int(grp["label"].sum()), int(len(grp))
            lo, hi = wilson_interval(k, n)
            rows.append({"axis": axis, "bin": str(name), "n": n, "bounce_rate": k / n if n else np.nan, "ci_low": lo, "ci_high": hi,
                         "model_mean_p": float(grp["p_full"].mean())})
            counts.append(n); succ.append(k)
        z, pval = cochran_armitage(counts, succ)
        for r in rows:
            if r["axis"] == axis:
                r["trend_z"], r["trend_p"] = z, pval
    return pd.DataFrame(rows)

# --------------------------------------------------------------------------------------
# Reporting / CLI
# --------------------------------------------------------------------------------------

def save_result(res: RunResult, out_dir: Path = RESULTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{res.config.strategy}_{res.config.method}_h{res.config.horizon}_{res.config.barrier}"
    res.folds.to_csv(out_dir / f"{stem}_folds.csv", index=False)
    res.obviousness.to_csv(out_dir / f"{stem}_obviousness.csv", index=False)
    res.importance.to_csv(out_dir / f"{stem}_importance.csv", index=False)
    res.events.to_csv(out_dir / f"{stem}_events.csv", index=False)
    with open(out_dir / f"{stem}_summary.json", "w") as f:
        json.dump(res.summary, f, indent=2, default=str)
    return out_dir / f"{stem}_summary.json"

def print_report(res: RunResult) -> None:
    s = res.summary
    pd.set_option("display.width", 200)
    print(f"\n=== {s['strategy']} / {s['method']} (ns={s['ns']}) | horizon {s['horizon']} | barriers at {s['barrier']} "
          f"+/-{s['react_atr']}/{s['break_atr']} ATR | approach band {s['near_atr']} ATR ===")
    print(f"events: {s['n_events_all']} approaches, {s['n_events_labelled']} labelled ({s['undecided_share']:.1%} undecided dropped), "
          f"{s['n_folds']} folds, {s['n_oos']} out-of-sample")
    print("\nper fold:")
    cols = ["fold", "test_start", "n_train", "n_test", "base_rate", "logloss_base", "logloss_geometry", "logloss_full", "auc_geometry", "auc_full"]
    f = res.folds[cols].copy()
    f["test_start"] = f["test_start"].dt.strftime("%Y-%m-%d")
    print(f.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\npooled out-of-sample:")
    for k in ("base", "geometry", "full"):
        m = s["pooled"][k]
        print(f"  {k:9} logloss {m['logloss']:.4f}  brier {m['brier']:.4f}  auc {m['auc'] if m['auc'] == m['auc'] else float('nan'):.3f}")
    a = s["ablation"]
    print(f"\nobviousness ablation: logloss gain (geometry -> full) mean {a['logloss_gain_mean']:+.4f} "
          f"sd {a['logloss_gain_std']:.4f}, improved in {a['folds_improved']}/{s['n_folds']} folds; auc gain {a['auc_gain_mean']:+.3f}")
    print("\nbounce rate by obviousness bin (out-of-sample):")
    print(res.obviousness.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\ntop features by gain:")
    print(res.importance.head(12).to_string(index=False, float_format=lambda v: f"{v:.0f}"))

def main(argv: list[str] | None = None) -> None:
    from src.api.binance import load_prices

    p = argparse.ArgumentParser(description="Purged walk-forward test of reaction at TA levels vs. their obviousness.")
    p.add_argument("--strategy", choices=STRATEGIES, default="levels")
    p.add_argument("--method", choices=METHODS, default=DEFAULT_METHOD)
    p.add_argument("--horizon", type=int, default=24, help="label horizon in bars")
    p.add_argument("--embargo", type=int, default=None, help="embargo in bars (default: horizon)")
    p.add_argument("--near", type=float, default=0.5, help="approach band in ATRs")
    p.add_argument("--react", type=float, default=1.0, help="bounce barrier in ATRs from the level")
    p.add_argument("--break-atr", type=float, default=1.0, help="break barrier in ATRs toward / through the level")
    p.add_argument("--barrier", choices=("close", "level"), default="close", help="measure barriers from the event close (symmetric) or the level")
    p.add_argument("--test-days", type=int, default=180)
    p.add_argument("--min-train-days", type=int, default=365)
    p.add_argument("--all-bars", action="store_true", help="use every bar in the band, not just the first of each approach")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = load_prices(refresh=args.refresh)
    cfg = RunConfig(
        strategy=args.strategy, method=args.method, near_atr=args.near, horizon=args.horizon, react_atr=args.react,
        break_atr=args.break_atr, barrier=args.barrier, events_only=not args.all_bars, embargo=args.embargo,
        test_bars=24 * args.test_days, min_train_bars=24 * args.min_train_days,
    )
    res = run(df, cfg)
    print_report(res)
    if not args.no_save:
        print(f"\nsaved to {save_result(res)}")

if __name__ == "__main__":
    main()
