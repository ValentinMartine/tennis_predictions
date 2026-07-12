"""Entraîne le modèle tennis (LightGBM + blend Elo) et mesure sur un test tenu à l'écart.

Découpage temporel : entraînement < 2023, validation 2023 (poids du blend), test >= 2024.
Compare : baseline Elo, ML seul, blend ML+Elo, et le marché (cotes moyennes, marge retirée).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import FEATURES, build_features, load_matches, market_prob  # noqa: E402

VAL_START, TEST_START = "2023-01-01", "2024-01-01"


def evaluate(name, p, y, mask):
    p, y = p[mask], y[mask]
    acc = ((p > 0.5) == (y == 1)).mean()
    ll = log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
    print(f"{name:22s} acc={acc:.3f}  log-loss={ll:.4f}  (n={mask.sum()})")
    return acc, ll


def main() -> int:
    allm = load_matches()
    print(f"{len(allm)} matchs chargés ({allm['Date'].min().date()} -> {allm['Date'].max().date()})")
    X, y, metas, _ = build_features(allm)
    dates = np.array([m["date"] for m in metas], dtype="datetime64[ns]")

    tr = dates < np.datetime64(VAL_START)
    va = (dates >= np.datetime64(VAL_START)) & (dates < np.datetime64(TEST_START))
    te = dates >= np.datetime64(TEST_START)

    model = LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31,
                           subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
                           random_state=42, verbose=-1)
    model.fit(X[tr], y[tr])
    p_ml = model.predict_proba(X)[:, 1]
    p_elo = X["p_elo"].to_numpy()

    # poids du blend réglé sur la validation
    best_w, best_ll = 0.5, 1e9
    for w in np.arange(0.0, 1.001, 0.01):
        ll = log_loss(y[va], np.clip(w * p_ml[va] + (1 - w) * p_elo[va], 1e-6, 1 - 1e-6))
        if ll < best_ll:
            best_w, best_ll = float(w), ll
    p_blend = best_w * p_ml + (1 - best_w) * p_elo
    print(f"poids blend ML w*={best_w:.2f} (val 2023)")

    print("\n== TEST (>= 2024) ==")
    evaluate("Baseline Elo", p_elo, y, te)
    evaluate("ML (LightGBM)", p_ml, y, te)
    evaluate("Blend ML+Elo", p_blend, y, te)

    # marché : probabilité implicite du vainqueur -> reconstituer p(p1 gagne)
    pm, ym = [], []
    for i, m in enumerate(metas):
        if not te[i]:
            continue
        q = market_prob(m["avgw"], m["avgl"])
        if q is None:
            continue
        pm.append(q if m["p1"] == m["winner"] else 1 - q)
        ym.append(y[i])
    if pm:
        pm, ym = np.array(pm), np.array(ym)
        acc = ((pm > 0.5) == (ym == 1)).mean()
        print(f"{'Marché (cotes moy.)':22s} acc={acc:.3f}  log-loss={log_loss(ym, np.clip(pm, 1e-6, 1-1e-6)):.4f}  (n={len(pm)})")

    # importance des features
    imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda t: -t[1])
    print("\nimportance:", ", ".join(f"{k}={v}" for k, v in imp[:6]))

    # persiste le modèle + méta pour l'export
    import joblib
    out = Path(__file__).resolve().parents[1] / "data" / "model.joblib"
    joblib.dump({"model": model, "blend_w": best_w}, out)
    print(f"\nmodèle -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
