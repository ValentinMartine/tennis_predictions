"""Télécharge les saisons ATP depuis tennis-data.co.uk (résultats + cotes, 2010-2026).

Même source de confiance que football-data.co.uk (même éditeur). Chaque saison est un
fichier Excel ; on ne retélécharge que la saison courante (les saisons passées sont figées).
"""
from __future__ import annotations

import sys
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIRST_SEASON = 2010
URL = "http://www.tennis-data.co.uk/{y}/{y}.xlsx"


def main() -> int:
    DATA.mkdir(exist_ok=True)
    current = date.today().year
    for y in range(FIRST_SEASON, current + 1):
        out = DATA / f"atp_{y}.xlsx"
        if out.exists() and y < current:
            continue  # saison passée figée
        try:
            urllib.request.urlretrieve(URL.format(y=y), out)
            print(f"  atp_{y}.xlsx OK")
        except Exception as exc:  # noqa: BLE001
            print(f"  atp_{y}.xlsx ÉCHEC : {exc}", file=sys.stderr)
            if y < current:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
