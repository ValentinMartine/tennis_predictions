"""Compare tennis model variants and selective betting thresholds.

The test split (2024+) is only reported after model/blend/betting choices are selected
on the 2023 validation split.
"""
from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import FEATURES, build_features, load_matches, market_prob  # noqa: E402

VAL_START = np.datetime64("2023-01-01")
TEST_START = np.datetime64("2024-01-01")


@dataclass(frozen=True)
class Score:
    name: str
    val_acc: float
    val_ll: float
    test_acc: float
    test_ll: float
    blend_w: float
    roi_val: float | None
    bets_val: int
    roi_test: float | None
    bets_test: int
    edge_min: float
    conf_min: float


def _metrics(p: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    pp, yy = p[mask], y[mask]
    return (
        float(((pp > 0.5) == (yy == 1)).mean()),
        float(log_loss(yy, np.clip(pp, 1e-6, 1 - 1e-6))),
    )


def _market_p1(meta: dict) -> float | None:
    q = market_prob(meta["avgw"], meta["avgl"])
    if q is None:
        return None
    return q if meta["p1"] == meta["winner"] else 1 - q


def _betting_stats(
    p: np.ndarray,
    y: np.ndarray,
    metas: list[dict],
    mask: np.ndarray,
    edge_min: float,
    conf_min: float,
) -> tuple[int, float | None]:
    profits: list[float] = []
    for i, meta in enumerate(metas):
        if not mask[i]:
            continue
        try:
            ow, ol = float(meta["avgw"]), float(meta["avgl"])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(ow) or not np.isfinite(ol) or ow <= 1 or ol <= 1:
            continue
        p1 = float(p[i])
        candidates = [
            (p1, ol if meta["p1"] == meta["loser"] else ow, y[i] == 1),
            (1 - p1, ow if meta["p1"] == meta["loser"] else ol, y[i] == 0),
        ]
        prob, odds, hit = max(candidates, key=lambda item: item[0] * item[1])
        if prob < conf_min or prob * odds - 1 < edge_min:
            continue
        profits.append(odds - 1 if hit else -1.0)
    if not profits:
        return 0, None
    return len(profits), float(np.mean(profits))


def _best_blend(y: np.ndarray, p_ml: np.ndarray, p_elo: np.ndarray, val: np.ndarray) -> tuple[float, np.ndarray]:
    best_w, best_ll = 0.0, float("inf")
    for w in np.arange(0.0, 1.001, 0.01):
        pred = w * p_ml[val] + (1 - w) * p_elo[val]
        ll = log_loss(y[val], np.clip(pred, 1e-6, 1 - 1e-6))
        if ll < best_ll:
            best_w, best_ll = float(w), ll
    return best_w, best_w * p_ml + (1 - best_w) * p_elo


def _best_betting_threshold(
    p: np.ndarray,
    y: np.ndarray,
    metas: list[dict],
    val: np.ndarray,
) -> tuple[float, float, int, float | None]:
    best = (0.05, 0.5, 0, None)
    for edge_min, conf_min in itertools.product(
        [0.00, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20],
        [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
    ):
        n, roi = _betting_stats(p, y, metas, val, edge_min, conf_min)
        if n < 30 or roi is None:
            continue
        _, _, best_n, best_roi = best
        if best_roi is None or roi > best_roi or (roi == best_roi and n > best_n):
            best = (edge_min, conf_min, n, roi)
    return best


def _score_variant(
    name: str,
    p_ml: np.ndarray,
    p_elo: np.ndarray,
    y: np.ndarray,
    metas: list[dict],
    val: np.ndarray,
    test: np.ndarray,
) -> Score:
    blend_w, p = _best_blend(y, p_ml, p_elo, val)
    val_acc, val_ll = _metrics(p, y, val)
    test_acc, test_ll = _metrics(p, y, test)
    edge_min, conf_min, bets_val, roi_val = _best_betting_threshold(p, y, metas, val)
    bets_test, roi_test = _betting_stats(p, y, metas, test, edge_min, conf_min)
    return Score(
        name=name,
        val_acc=val_acc,
        val_ll=val_ll,
        test_acc=test_acc,
        test_ll=test_ll,
        blend_w=blend_w,
        roi_val=roi_val,
        bets_val=bets_val,
        roi_test=roi_test,
        bets_test=bets_test,
        edge_min=edge_min,
        conf_min=conf_min,
    )


def main() -> int:
    matches = load_matches()
    X, y, metas, _ = build_features(matches)
    dates = np.array([meta["date"] for meta in metas], dtype="datetime64[ns]")
    train = dates < VAL_START
    val = (dates >= VAL_START) & (dates < TEST_START)
    test = dates >= TEST_START
    p_elo = X["p_elo"].to_numpy()

    variants: list[tuple[str, object]] = [
        ("lgbm_current", LGBMClassifier(
            n_estimators=500, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
            random_state=42, verbose=-1,
        )),
        ("lgbm_regularized", LGBMClassifier(
            n_estimators=900, learning_rate=0.02, num_leaves=15,
            subsample=0.75, colsample_bytree=0.75, min_child_samples=80,
            reg_alpha=0.15, reg_lambda=1.0, random_state=42, verbose=-1,
        )),
        ("lgbm_accuracy", LGBMClassifier(
            n_estimators=700, learning_rate=0.025, num_leaves=63,
            subsample=0.9, colsample_bytree=0.8, min_child_samples=30,
            reg_lambda=0.3, random_state=42, verbose=-1,
        )),
        ("logistic", LogisticRegression(max_iter=2000, C=0.5)),
    ]

    print("walk_forward_year,variant,val_acc,val_ll,w")
    for year in (2021, 2022, 2023):
        fold_val_start = np.datetime64(f"{year}-01-01")
        fold_val_end = np.datetime64(f"{year + 1}-01-01")
        fold_train = dates < fold_val_start
        fold_val = (dates >= fold_val_start) & (dates < fold_val_end)
        for name, model in variants:
            model.fit(X.loc[fold_train, FEATURES], y[fold_train])
            p_ml = model.predict_proba(X[FEATURES])[:, 1]
            blend_w, p = _best_blend(y, p_ml, p_elo, fold_val)
            acc, ll = _metrics(p, y, fold_val)
            print(f"{year},{name},{acc:.4f},{ll:.4f},{blend_w:.2f}")

    print("\nheld_out_2024_plus")
    scores: list[Score] = []
    for name, model in variants:
        model.fit(X.loc[train, FEATURES], y[train])
        p_ml = model.predict_proba(X[FEATURES])[:, 1]
        scores.append(_score_variant(name, p_ml, p_elo, y, metas, val, test))

    scores.sort(key=lambda s: (s.test_acc, -s.test_ll), reverse=True)
    print("variant,val_acc,val_ll,test_acc,test_ll,w,edge,conf,val_bets,val_roi,test_bets,test_roi")
    for score in scores:
        val_roi = "" if score.roi_val is None else f"{score.roi_val:.4f}"
        test_roi = "" if score.roi_test is None else f"{score.roi_test:.4f}"
        print(
            f"{score.name},{score.val_acc:.4f},{score.val_ll:.4f},"
            f"{score.test_acc:.4f},{score.test_ll:.4f},{score.blend_w:.2f},"
            f"{score.edge_min:.2f},{score.conf_min:.2f},{score.bets_val},"
            f"{val_roi},{score.bets_test},{test_roi}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
