"""Entraîne le modèle tennis pur et mesure sur un test tenu à l'écart.

Découpage temporel :
- entraînement < 2023 ;
- validation 2023 pour choisir le poids ML/Elo et calibrer les probabilités ;
- test >= 2024, jamais utilisé pour régler le modèle.

Les cotes ne sont pas utilisées par le modèle : elles restent un benchmark externe.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.modeling import (  # noqa: E402
    best_blend_weight,
    fit_probability_calibrator,
    make_lightgbm,
    probability_metrics,
    yearly_metrics,
)
from src.pipeline import FEATURES, build_features, load_matches, market_prob  # noqa: E402

VAL_START, TEST_START = "2023-01-01", "2024-01-01"


def evaluate(name: str, p, y, mask) -> dict:
    metrics = probability_metrics(y, p, mask)
    print(
        f"{name:24s} acc={metrics['accuracy']:.3f}  "
        f"log-loss={metrics['log_loss']:.4f}  brier={metrics['brier']:.4f}  "
        f"ece={metrics['ece']:.4f}  (n={metrics['n']})"
    )
    return metrics


def model_path(tour: str) -> Path:
    name = "model.joblib" if tour == "atp" else f"model_{tour}.joblib"
    return Path(__file__).resolve().parents[1] / "data" / name


def main(argv: list[str] | None = None) -> int:
    tour = (argv or sys.argv[1:] or ["atp"])[0].lower()
    allm = load_matches(tour)
    print(f"[{tour}] {len(allm)} matchs chargés ({allm['Date'].min().date()} -> {allm['Date'].max().date()})")
    X, y, metas, _ = build_features(allm)
    dates = np.array([m["date"] for m in metas], dtype="datetime64[ns]")

    tr = dates < np.datetime64(VAL_START)
    va = (dates >= np.datetime64(VAL_START)) & (dates < np.datetime64(TEST_START))
    te = dates >= np.datetime64(TEST_START)

    model = make_lightgbm()
    model.fit(X[tr], y[tr])
    p_ml = model.predict_proba(X[FEATURES])[:, 1]
    p_elo = X["p_elo"].to_numpy()

    # Poids ML/Elo réglé sur la validation temporelle.
    best_w = best_blend_weight(y, p_ml, p_elo, va)
    p_blend = best_w * p_ml + (1 - best_w) * p_elo

    # Calibration du modèle pur sur la validation temporelle, sans aucune cote.
    calibrator = fit_probability_calibrator(y[va], p_blend[va])
    p_calibrated = calibrator.predict(p_blend)

    print(f"poids blend ML w*={best_w:.2f} (val 2023)")
    print(f"calibration modèle pur={calibrator.method} (val 2023)")

    print("\n== TEST (>= 2024) ==")
    evaluate("Baseline Elo", p_elo, y, te)
    evaluate("ML (LightGBM)", p_ml, y, te)
    evaluate("Blend ML+Elo brut", p_blend, y, te)
    evaluate("Modèle pur calibré", p_calibrated, y, te)

    # Marché : probabilité implicite du vainqueur -> reconstituer p(p1 gagne).
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
        pm_arr, ym_arr = np.array(pm), np.array(ym)
        acc = ((pm_arr > 0.5) == (ym_arr == 1)).mean()
        print(
            f"{'Marché (cotes moy.)':24s} acc={acc:.3f}  "
            f"log-loss={log_loss(ym_arr, np.clip(pm_arr, 1e-6, 1 - 1e-6)):.4f}  "
            f"(n={len(pm_arr)})"
        )

    print("\n== ROBUSTESSE TEMPORELLE DU MODÈLE PUR CALIBRÉ ==")
    for year, stats in yearly_metrics(y, p_calibrated, dates).items():
        print(
            f"{year}: acc={stats['accuracy']:.3f} log-loss={stats['log_loss']:.4f} "
            f"brier={stats['brier']:.4f} ece={stats['ece']:.4f} (n={stats['n']})"
        )

    imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda t: -t[1])
    print("\nimportance:", ", ".join(f"{k}={v}" for k, v in imp[:10]))

    import joblib
    out = model_path(tour)
    joblib.dump({
        "model": model,
        "calibrator": calibrator,
        "blend_w": best_w,
        "tour": tour,
        "features": FEATURES,
        "model_version": "independent-calibrated-v2",
    }, out)
    print(f"\nmodèle -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
