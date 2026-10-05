"""Calibration des signaux d'actualité sur les titres étiquetés de data/actualite/a_etiqueter.csv.

L'échantillon est stratifié : tous les titres à p >= 0,2, et un tirage aléatoire en dessous.
Chaque strate est repondérée par l'inverse de son taux d'échantillonnage.

    python scripts/calibration_actualite.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOSSIER = ROOT / "data" / "actualite"
SEUIL_STRATE = 0.2


def calibrer(lignes: list[dict], poids: dict[str, float], seuil: float = 0.5) -> dict:
    etiquetees = [lig for lig in lignes if lig["etiquette"] in ("oui", "non")]
    alertes = [lig for lig in etiquetees if lig["p"] >= seuil]
    vrais = [lig for lig in alertes if lig["etiquette"] == "oui"]
    total_poids = sum(poids[lig["strate"]] for lig in etiquetees)
    brier = sum(poids[lig["strate"]] * (lig["p"] - (lig["etiquette"] == "oui")) ** 2 for lig in etiquetees) / total_poids
    return {
        "n_etiquetes": len(etiquetees),
        "precision": round(len(vrais) / len(alertes), 3) if alertes else None,
        "faux_negatifs_observes": sum(1 for lig in etiquetees if lig["p"] < seuil and lig["etiquette"] == "oui"),
        "brier_pondere": round(brier, 4),
    }


def main() -> int:
    with (DOSSIER / "a_etiqueter.csv").open(encoding="utf-8", newline="") as f:
        lignes = [{**r, "p": float(r["p"])} for r in csv.DictReader(f)]
    for lig in lignes:
        lig["strate"] = "haute" if lig["p"] >= SEUIL_STRATE else "basse"
    poids = {}
    for strate in ("haute", "basse"):
        population = sum(1 for lig in lignes if lig["strate"] == strate)
        etiquetes = sum(1 for lig in lignes if lig["strate"] == strate and lig["etiquette"] in ("oui", "non"))
        poids[strate] = population / etiquetes if etiquetes else 0.0
    bilan = calibrer(lignes, poids)
    print(json.dumps(bilan, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
