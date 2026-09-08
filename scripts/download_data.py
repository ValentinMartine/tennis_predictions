"""Télécharge les saisons ATP depuis tennis-data.co.uk (résultats + cotes, 2010-2026).

Même source de confiance que football-data.co.uk (même éditeur). Chaque saison est un
fichier Excel ; on ne retélécharge que la saison courante (les saisons passées sont figées).
"""
from __future__ import annotations

import shutil
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIRST_SEASON = 2000
URL = "http://www.tennis-data.co.uk/{y}/{y}.xlsx"
MAX_ATTEMPTS = 4
RETRY_DELAYS_SECONDS = (2, 5, 10)


def download_with_retries(year: int, destination: Path) -> None:
    """Download one season with retries, then atomically replace the target file."""
    tmp = destination.with_suffix(".xlsx.tmp")
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(URL.format(y=year), timeout=30) as response:
                with tmp.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            tmp.replace(destination)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if tmp.exists():
                tmp.unlink()
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_DELAYS_SECONDS[attempt - 1])

    if last_error is not None:
        raise last_error


def main() -> int:
    DATA.mkdir(exist_ok=True)
    current = date.today().year
    for y in range(FIRST_SEASON, current + 1):
        out = DATA / f"atp_{y}.xlsx"
        if out.exists() and y < current:
            continue  # saison passée figée
        try:
            download_with_retries(y, out)
            print(f"  atp_{y}.xlsx OK")
        except Exception as exc:  # noqa: BLE001
            print(f"  atp_{y}.xlsx ÉCHEC : {exc}", file=sys.stderr)
            if y < current or not out.exists():
                return 1
            print(f"  atp_{y}.xlsx conserve depuis le fichier existant", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
