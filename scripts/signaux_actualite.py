"""Juge l'actualité des joueurs des matchs à venir avec Nimble (Ollama local).

Local uniquement : Ollama n'existe pas en CI. Écrit data/actualite/signaux.json (lu par
src.export), complète l'archive datée titres.jsonl et le fichier d'étiquetage a_etiqueter.csv.

    python scripts/signaux_actualite.py [--jours 7] [--titres 5] [--modele nimble:9b-q4_K_M]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.download_data import known_surnames, short_name_from_full  # noqa: E402
from src.actualite import MODELE, archiver, juger_ollama, signal_joueur, telecharger_rss  # noqa: E402

DOSSIER = ROOT / "data" / "actualite"
COLONNES = ["date", "joueur", "titre", "source", "lien", "p", "etiquette", "etiqueteur"]


def joueurs_a_suivre() -> list[str]:
    """Noms complets des joueurs présents dans les matchs à venir exportés (ATP et WTA)."""
    brut = json.loads((ROOT / "data" / "upcoming_raw.json").read_text(encoding="utf-8"))["matches"]
    noms: set[str] = set()
    for tour, suffixe in (("atp", ""), ("wta", "_wta")):
        export = ROOT / "data" / f"tennis_predictions{suffixe}.json"
        if not export.exists():
            continue
        paires = {(m["player1"], m["player2"]) for m in json.loads(export.read_text(encoding="utf-8"))["upcoming"]}
        surnames = known_surnames(tour=tour)
        for m in brut:
            if (short_name_from_full(m["player1"], surnames), short_name_from_full(m["player2"], surnames)) in paires:
                noms.update((m["player1"], m["player2"]))
    return sorted(noms)


def mettre_a_jour_etiquetage(archive: Path, csv_path: Path) -> None:
    """Réécrit le fichier d'étiquetage en conservant les étiquettes déjà saisies."""
    etiquettes = {}
    if csv_path.exists():
        with csv_path.open(encoding="utf-8", newline="") as f:
            etiquettes = {(r["joueur"], r["lien"]): (r["etiquette"], r.get("etiqueteur", "")) for r in csv.DictReader(f)}
    lignes = [json.loads(x) for x in archive.read_text(encoding="utf-8").splitlines()]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLONNES, extrasaction="ignore")
        w.writeheader()
        for lig in lignes:
            etiquette, etiqueteur = etiquettes.get((lig["joueur"], lig["lien"]), ("", ""))
            w.writerow({**lig, "etiquette": etiquette, "etiqueteur": etiqueteur})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jours", type=int, default=7)
    parser.add_argument("--titres", type=int, default=5, help="titres les plus récents jugés par joueur")
    parser.add_argument("--modele", default=MODELE)
    args = parser.parse_args()

    maintenant = datetime.now(timezone.utc)
    joueurs = joueurs_a_suivre()
    print(f"{len(joueurs)} joueurs à suivre")
    signaux, a_archiver = {}, []
    for joueur in joueurs:
        try:
            titres = telecharger_rss(joueur, args.jours)
        except OSError as erreur:
            print(f"  {joueur} : flux indisponible ({erreur})")
            continue
        titres = sorted((t for t in titres if t["date"]), key=lambda t: t["date"], reverse=True)[: args.titres]
        evalues = [{**t, "p": juger_ollama(joueur, t["titre"], args.modele)} for t in titres]
        a_archiver += [{**e, "joueur": joueur, "modele": args.modele, "juge_le": maintenant.isoformat()} for e in evalues]
        signal = signal_joueur(evalues, maintenant, args.jours)
        if signal:
            signaux[joueur] = signal
            if signal["p"] >= 0.5:
                print(f"  {joueur} : {signal['p']:.2f} | {signal['titre'][:90]}")

    DOSSIER.mkdir(parents=True, exist_ok=True)
    sortie = {"generated_at": maintenant.isoformat(), "modele": args.modele, "fenetre_jours": args.jours, "joueurs": signaux}
    (DOSSIER / "signaux.json").write_text(json.dumps(sortie, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    n = archiver(DOSSIER / "titres.jsonl", a_archiver)
    mettre_a_jour_etiquetage(DOSSIER / "titres.jsonl", DOSSIER / "a_etiqueter.csv")
    print(f"{len(signaux)} signaux écrits, {n} titres ajoutés à l'archive")
    return 0


if __name__ == "__main__":
    sys.exit(main())
