"""Récupère les matchs à venir depuis Valuebetennis (ATP et WTA confondus).

Le fichier de saison ne contient que des matchs joués : la page « matchs à venir » est
la seule source publique du calendrier avec horaires, classements et cotes. Le tri par
circuit et par niveau se fait plus tard, à l'export, en recoupant l'identifiant de
tournoi avec la saison en cours.
"""
from __future__ import annotations

import html
import json
import re
import sys
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "upcoming_raw.json"
URL = "https://www.valuebetennis.com/matchs-a-venir.htm"
REFERER = "https://www.valuebetennis.com/donnees.htm"

MONTHS = {
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
    "juillet": 7, "août": 8, "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11,
    "décembre": 12, "decembre": 12,
}

SURFACES = {
    "dur": "Hard", "dur intérieur": "Hard", "terre battue": "Clay",
    "gazon": "Grass", "moquette": "Carpet",
}

ROUNDS = {
    "qualifications": "Qualifying 1", "1er tour": "1st Round", "2e tour": "2nd Round",
    "3e tour": "3rd Round", "4e tour": "4th Round", "8e de finale": "4th Round",
    "quart de finale": "Quarterfinals", "quarts de finale": "Quarterfinals",
    "demi-finale": "Semifinals", "demi-finales": "Semifinals", "finale": "The Final",
    "poule": "Round Robin",
}


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def parse_french_date(label: str) -> str | None:
    match = re.search(r"(\d{1,2})\s+([a-zéûôA-Z]+)\s+(\d{4})", label)
    if not match:
        return None
    month = MONTHS.get(match.group(2).lower())
    if not month:
        return None
    return date(int(match.group(3)), month, int(match.group(1))).isoformat()


def _number(raw: str) -> float | None:
    try:
        return float(raw.replace(",", ".").strip())
    except (TypeError, ValueError):
        return None


def parse(page: str) -> list[dict]:
    matches: list[dict] = []
    # La page alterne <h2> jour, puis un <h3> et un tableau par tournoi.
    for day_block in re.split(r"<h2[^>]*>", page)[1:]:
        day = parse_french_date(_text(day_block.split("</h2>", 1)[0]))
        for block in re.split(r"<h3[^>]*>", day_block)[1:]:
            head, _, body = block.partition("</h3>")
            link = re.search(r'href="[^"]*?/tournaments/\d{4}/[a-z0-9-]+-(\d+)\.htm"[^>]*>(.*?)</a>', head, re.S)
            if not link:
                continue
            tournament_id, tournament = int(link.group(1)), _text(link.group(2))
            surface_label = _text(re.search(r"<small[^>]*>(.*?)</small>", head, re.S).group(1)) if "<small" in head else ""
            surface = SURFACES.get(surface_label.lstrip("· ").strip().lower(), "")
            for row in re.findall(r"<tr[^>]*>(.*?)</tr>", body.split("</table>", 1)[0], re.S):
                cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
                if len(cells) < 4:
                    continue
                players = re.search(r">\s*([^<>]+?)\s*<span[^>]*>vs</span>\s*([^<>]+?)\s*</a>", cells[1], re.S)
                if not players:
                    continue
                ranks = re.search(r"n°\s*(\d+)\s*/\s*n°\s*(\d+)", _text(cells[1]))
                odds = re.search(r"([\d,]+)\s*/\s*([\d,]+)", _text(cells[3]))
                round_label = _text(cells[2]).lower()
                matches.append({
                    "date": day,
                    "time": _text(cells[0]) or None,
                    "tournament_id": tournament_id,
                    "tournament": tournament,
                    "surface": surface,
                    "round": ROUNDS.get(round_label, round_label),
                    "player1": html.unescape(players.group(1)).strip(),
                    "player2": html.unescape(players.group(2)).strip(),
                    "rank1": int(ranks.group(1)) if ranks else None,
                    "rank2": int(ranks.group(2)) if ranks else None,
                    "odds1": _number(odds.group(1)) if odds else None,
                    "odds2": _number(odds.group(2)) if odds else None,
                })
    return matches


def main() -> int:
    request = urllib.request.Request(URL, headers={
        "User-Agent": "Mozilla/5.0 valentinmartine-tennis-predictions/1.0",
        "Referer": REFERER,
    })
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            page = response.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        # Le calendrier est un plus : son absence ne doit pas faire échouer la régénération.
        print(f"matchs à venir indisponibles : {exc}", file=sys.stderr)
        OUT.write_text(json.dumps({
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": URL, "error": str(exc), "matches": [],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0

    matches = parse(page)
    OUT.write_text(json.dumps({
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": URL, "error": None, "matches": matches,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"écrit {OUT.name} : {len(matches)} matchs à venir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
