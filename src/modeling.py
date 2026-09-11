"""Modèle tennis pur, calibration temporelle et métriques probabilistes.

Le marché ne sert ici qu'à comparer le modèle : aucune cote n'entre dans les
features ni dans la calibration indépendante.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


EPS = 1e-6


def clip_prob(probabilities) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), EPS, 1 - EPS)


def logit(probabilities) -> np.ndarray:
    p = clip_prob(probabilities)
    return np.log(p / (1 - p)).reshape(-1, 1)


def probability_metrics(y_true, probabilities, mask=None) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = clip_prob(probabilities)
    if mask is not None:
        idx = np.asarray(mask, dtype=bool)
        y, p = y[idx], p[idx]
    if len(y) == 0:
        return {"n": 0, "accuracy": None, "log_loss": None, "brier": None, "ece": None}
    return {
        "n": int(len(y)),
        "accuracy": round(float(((p > 0.5) == (y == 1)).mean()), 4),
        "log_loss": round(float(log_loss(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "ece": round(float(expected_calibration_error(y, p)), 4),
    }


def expected_calibration_error(y_true, probabilities, n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=int)
    p = clip_prob(probabilities)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            idx = (p >= lo) & (p <= hi)
        else:
            idx = (p >= lo) & (p < hi)
        if not idx.any():
            continue
        ece += float(idx.mean()) * abs(float(p[idx].mean()) - float(y[idx].mean()))
    return ece


def calibration_bins(y_true, probabilities, n_bins: int = 10) -> list[dict]:
    y = np.asarray(y_true, dtype=int)
    p = clip_prob(probabilities)
    bins: list[dict] = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            idx = (p >= lo) & (p <= hi)
        else:
            idx = (p >= lo) & (p < hi)
        if not idx.any():
            continue
        bins.append({
            "range": f"{lo:.1f}-{hi:.1f}",
            "n": int(idx.sum()),
            "confidence": round(float(p[idx].mean()), 4),
            "observed": round(float(y[idx].mean()), 4),
        })
    return bins


@dataclass
class ProbabilityCalibrator:
    method: str = "identity"
    sigmoid: LogisticRegression | None = None

    def predict(self, probabilities) -> np.ndarray:
        p = clip_prob(probabilities)
        if self.method == "sigmoid" and self.sigmoid is not None:
            return clip_prob(self.sigmoid.predict_proba(logit(p))[:, 1])
        return p


def fit_probability_calibrator(y_true, probabilities) -> ProbabilityCalibrator:
    y = np.asarray(y_true, dtype=int)
    p = clip_prob(probabilities)
    if len(np.unique(y)) < 2 or len(y) < 50:
        return ProbabilityCalibrator()

    # La calibration est elle-même un modèle. On évite donc de choisir sigmoid/isotonic
    # sur les mêmes points que ceux utilisés pour les ajuster : la première moitié de la
    # validation temporelle sert à fitter, la seconde à sélectionner contre l'identité.
    split = max(25, len(y) // 2)
    y_fit, p_fit = y[:split], p[:split]
    y_sel, p_sel = y[split:], p[split:]
    if len(np.unique(y_fit)) < 2 or len(np.unique(y_sel)) < 2:
        return ProbabilityCalibrator()

    identity_loss = float(log_loss(y_sel, p_sel))

    sigmoid = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    sigmoid.fit(logit(p_fit), y_fit)
    sigmoid_loss = float(log_loss(y_sel, clip_prob(sigmoid.predict_proba(logit(p_sel))[:, 1])))

    # Une calibration non-parametrique (isotonic) est trop flexible pour une seule saison
    # de validation et a un vrai risque d'overfitting. On ne retient donc que le sigmoid
    # Platt scaling, et seulement s'il bat clairement l'identite sur la sous-validation.
    if sigmoid_loss <= identity_loss - 0.001:
        return ProbabilityCalibrator(method="sigmoid", sigmoid=sigmoid)
    return ProbabilityCalibrator()


def best_blend_weight(y_true, left_prob, right_prob, mask) -> float:
    idx = np.asarray(mask, dtype=bool)
    y = np.asarray(y_true, dtype=int)
    left = clip_prob(left_prob)
    right = clip_prob(right_prob)
    if not idx.any():
        return 0.5
    best_w, best_ll = 0.5, float("inf")
    for w in np.arange(0.0, 1.001, 0.01):
        p = w * left[idx] + (1 - w) * right[idx]
        ll = log_loss(y[idx], clip_prob(p))
        if ll < best_ll:
            best_w, best_ll = float(w), float(ll)
    return best_w


def make_lightgbm() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.015,
        num_leaves=17,
        subsample=0.78,
        colsample_bytree=0.78,
        min_child_samples=70,
        reg_alpha=0.20,
        reg_lambda=1.30,
        random_state=42,
        verbose=-1,
    )


def yearly_metrics(y_true, probabilities, dates, start_year: int = 2024) -> dict[str, dict]:
    years = np.array([int(str(d)[:4]) for d in dates])
    out: dict[str, dict] = {}
    for year in sorted(set(years)):
        if year >= start_year:
            out[str(year)] = probability_metrics(y_true, probabilities, years == year)
    return out
