"""Complète les résultats ATP récents que tennis-data.co.uk n'a pas encore publiés, via
Sofascore. Ciblé sur les Grands Chelems (les tournois où le retard de tennis-data fait le
plus mal : 2 semaines de matchs), en suivant leur uniqueTournament Sofascore.

Écrit data/results_patch.csv (schéma tennis-data) que src.pipeline.load_matches fusionne.
Local uniquement (navigateur headless) : Sofascore bloque les IP des runners CI.
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "results_patch.csv"
API = "https://api.sofascore.com/api/v1"

# uniqueTournament Sofascore des Grands Chelems (simple messieurs)
SLAMS = {
    "Australian Open": 2363,
    "Roland Garros": 2480,
    "Wimbledon": 2361,
    "US Open": 2449,
}


def _surface(ground: str, slam: str) -> str:
    g = str(ground or "").lower()
    if "clay" in g or slam == "Roland Garros":
        return "Clay"
    if "grass" in g or slam == "Wimbledon":
        return "Grass"
    return "Hard"


def _to_tennisdata_name(short: str) -> str:
    """Sofascore « J. Sinner » -> tennis-data « Sinner J. » (initiales en fin, sans accents)."""
    import re
    import unicodedata
    s = unicodedata.normalize("NFKD", str(short or "").strip()).encode("ascii", "ignore").decode()
    m = re.match(r"^((?:[A-Z]\.\s*)+)(.+)$", s)
    if not m:
        return s
    initials = m.group(1).replace(" ", "")
    return f"{m.group(2).strip()} {initials}"


def _round(name: str) -> str:
    n = str(name or "")
    return {"Final": "The Final", "Semifinals": "Semifinals",
            "Quarterfinals": "Quarterfinals"}.get(n, n or "1st Round")


def _last_known_ranks() -> dict[str, float]:
    ranks: dict[str, float] = {}
    for f in sorted(DATA.glob("atp_*.xlsx"))[-2:]:
        df = pd.read_excel(f)
        for _, r in df.iterrows():
            if pd.notna(r.get("WRank")):
                ranks[str(r["Winner"]).strip()] = float(r["WRank"])
            if pd.notna(r.get("LRank")):
                ranks[str(r["Loser"]).strip()] = float(r["LRank"])
    return ranks


def main() -> int:
    try:
        from ScraperFC.utils import botasaurus_browser_get_json as get
    except Exception as exc:  # noqa: BLE001
        print(f"ScraperFC indisponible : {exc}", file=sys.stderr)
        return 0

    newest = sorted(DATA.glob("atp_*.xlsx"))[-1]
    have = pd.read_excel(newest)
    last_covered = pd.to_datetime(have["Date"]).max().date()
    ranks = _last_known_ranks()
    year = str(date.today().year)

    rows = []
    for slam, uid in SLAMS.items():
        try:
            seasons = (get(f"{API}/unique-tournament/{uid}/seasons") or {}).get("seasons", [])
            sid = next((s["id"] for s in seasons if str(s.get("year")) == year), None)
            if not sid:
                continue
            events, i = [], 0
            while True:
                d = get(f"{API}/unique-tournament/{uid}/season/{sid}/events/last/{i}")
                if not isinstance(d, dict) or "events" not in d:
                    break
                events += d["events"]
                i += 1
            n0 = len(rows)
            for ev in events:
                try:
                    if ev.get("status", {}).get("type") != "finished":
                        continue
                    ts = ev.get("startTimestamp")
                    day = datetime.utcfromtimestamp(ts).date() if ts else None
                    if not day or day <= last_covered:
                        continue  # déjà couvert par tennis-data
                    home, away = ev.get("homeTeam", {}), ev.get("awayTeam", {})
                    hn = home.get("shortName") or home.get("name")
                    an = away.get("shortName") or away.get("name")
                    if not hn or not an or "/" in hn or "/" in an:
                        continue
                    hn, an = _to_tennisdata_name(hn), _to_tennisdata_name(an)
                    wc = ev.get("winnerCode")
                    if wc not in (1, 2):
                        continue
                    win, lose = (hn, an) if wc == 1 else (an, hn)
                    hs, as_ = ev.get("homeScore", {}), ev.get("awayScore", {})
                    row = {
                        "ATP": None, "Location": None, "Tournament": slam,
                        "Date": pd.Timestamp(day), "Series": "Grand Slam",
                        "Court": "Outdoor", "Surface": _surface(ev.get("groundType"), slam),
                        "Round": _round((ev.get("roundInfo") or {}).get("name")),
                        "Best of": 5, "Winner": win, "Loser": lose,
                        "WRank": ranks.get(win), "LRank": ranks.get(lose),
                        "WPts": None, "LPts": None,
                        "Wsets": hs.get("current") if wc == 1 else as_.get("current"),
                        "Lsets": as_.get("current") if wc == 1 else hs.get("current"),
                        "Comment": "Completed",
                    }
                    for k in range(1, 6):
                        hp, ap = hs.get(f"period{k}"), as_.get(f"period{k}")
                        row[f"W{k}"] = hp if wc == 1 else ap
                        row[f"L{k}"] = ap if wc == 1 else hp
                    rows.append(row)
                except Exception:  # noqa: BLE001
                    continue
            print(f"  {slam}: +{len(rows) - n0} matchs après le {last_covered}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  {slam}: échec ({exc})", file=sys.stderr)

    if rows:
        pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"patch : {len(rows)} match(s) -> {OUT.name} (tennis-data couvre jusqu'au {last_covered})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
