"""Signaux d'actualité avant match : blessure, maladie, forfait, retour de blessure.

Chaque titre Google News daté est jugé par un modèle de décision local (Nimble via l'API
System One d'Ollama). Tant que l'archive n'est pas assez longue pour un test hors
échantillon, le signal est affiché à côté de la prédiction et n'entre pas dans le modèle.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

OLLAMA_SYSTEMONE = "http://localhost:11434/v1/systemone"
MODELE = "nimble:9b-q4_K_M"


def url_rss(joueur: str, jours: int = 7) -> str:
    requete = {"q": f'"{joueur}" tennis when:{jours}d', "hl": "en-US", "gl": "US", "ceid": "US:en"}
    return "https://news.google.com/rss/search?" + urllib.parse.urlencode(requete)


def lire_rss(contenu: bytes) -> list[dict]:
    titres = []
    for item in ET.fromstring(contenu).iter("item"):
        date = item.findtext("pubDate")
        titres.append({
            "titre": (item.findtext("title") or "").strip(),
            "date": parsedate_to_datetime(date).isoformat() if date else None,
            "source": (item.findtext("source") or "").strip(),
            "lien": (item.findtext("link") or "").strip(),
        })
    return titres


def telecharger_rss(joueur: str, jours: int = 7, delai: int = 20) -> list[dict]:
    requete = urllib.request.Request(url_rss(joueur, jours), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(requete, timeout=delai) as reponse:
        return lire_rss(reponse.read())


def question_forme(joueur: str) -> str:
    # En anglais : les titres le sont, et Nimble répond mieux dans la langue du texte.
    return (f"Does this headline report that {joueur} is injured, ill, physically limited, "
            f"returning from injury, or doubtful or withdrawn for an upcoming match?")


def juger_ollama(joueur: str, titre: str, modele: str = MODELE, delai: int = 300) -> float:
    corps = {
        "model": modele,
        "state": {"joueur": joueur, "titre": titre},
        "questions": {"forme": {"type": "noul", "instructions": question_forme(joueur)}},
    }
    requete = urllib.request.Request(OLLAMA_SYSTEMONE, data=json.dumps(corps).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(requete, timeout=delai) as reponse:
        return float(json.loads(reponse.read())["answers"]["forme"]["noul"])


def signal_joueur(evalues: list[dict], maintenant: datetime, jours: int = 7) -> dict | None:
    debut = maintenant - timedelta(days=jours)
    fenetre = [e for e in evalues if e.get("date") and debut <= datetime.fromisoformat(e["date"]) <= maintenant]
    if not fenetre:
        return None
    pire = max(fenetre, key=lambda e: e["p"])
    return {"p": round(pire["p"], 3), "titre": pire["titre"], "date": pire["date"],
            "source": pire.get("source"), "lien": pire.get("lien"), "n_titres": len(fenetre)}


def archiver(chemin: Path, lignes: list[dict]) -> int:
    """Ajoute les titres jamais vus ; l'archive datée servira au test hors échantillon."""
    vus = set()
    if chemin.exists():
        for ligne in chemin.read_text(encoding="utf-8").splitlines():
            d = json.loads(ligne)
            vus.add((d["joueur"], d["lien"]))
    nouvelles = [lig for lig in lignes if (lig["joueur"], lig["lien"]) not in vus]
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("a", encoding="utf-8", newline="\n") as f:
        for ligne in nouvelles:
            f.write(json.dumps(ligne, ensure_ascii=False) + "\n")
    return len(nouvelles)
