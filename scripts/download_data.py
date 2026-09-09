"""Download ATP seasons from tennis-data.co.uk with GitHub CSV fallbacks.

The primary source keeps the historical betting odds. When it is unavailable, the
pipeline falls back to Sackmann-compatible public CSV mirrors so the model can
still refresh results and Elo features without pretending that market odds were
available.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCE_STATUS = DATA / "source_status.json"
FIRST_SEASON = 2000
URL = "http://www.tennis-data.co.uk/{y}/{y}.xlsx"
SACKMANN_URLS = (
    "https://raw.githubusercontent.com/azhou5849/tennis-atp-sackmann-data/master/atp_matches_{y}.csv",
    "https://raw.githubusercontent.com/beta2k/tennis_atp/master/atp_matches_{y}.csv",
    "https://raw.githubusercontent.com/FerrisPhilippi/tennis_atp_data/master/atp_matches_{y}.csv",
    "https://raw.githubusercontent.com/aadvikchaturvedi/tennis_atp_dataset/master/atp_matches_{y}.csv",
    "https://raw.githubusercontent.com/migumax/tennis_atp/master/{y}.csv",
)
MAX_ATTEMPTS = 4
RETRY_DELAYS_SECONDS = (2, 5, 10)


def download_with_retries(year: int, destination: Path) -> None:
    """Download one tennis-data season with retries and atomic replacement."""
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


def series_from_sackmann_level(level: object) -> str:
    """Map Sackmann tournament levels to tennis-data style labels."""
    mapping = {
        "G": "Grand Slam",
        "M": "Masters 1000",
        "F": "Tour Finals",
        "A": "ATP",
        "D": "Davis Cup",
    }
    return mapping.get(str(level or "").strip(), "ATP")


def score_columns(score: object) -> dict[str, int | None]:
    """Extract tennis-data compatible set columns from a Sackmann score string."""
    values: dict[str, int | None] = {}
    winner_sets = 0
    loser_sets = 0
    for index in range(1, 6):
        values[f"W{index}"] = None
        values[f"L{index}"] = None

    for index, token in enumerate(str(score or "").split()[:5], start=1):
        clean = token.split("(", 1)[0]
        if "-" not in clean:
            continue
        winner_games, loser_games = clean.split("-", 1)
        try:
            w_games = int(winner_games)
            l_games = int(loser_games)
        except ValueError:
            continue
        values[f"W{index}"] = w_games
        values[f"L{index}"] = l_games
        if w_games > l_games:
            winner_sets += 1
        elif l_games > w_games:
            loser_sets += 1

    values["Wsets"] = winner_sets or None
    values["Lsets"] = loser_sets or None
    return values


def download_sackmann_fallback(year: int, destination: Path) -> str:
    """Build a tennis-data compatible Excel file from a public ATP CSV mirror."""
    last_error: Exception | None = None
    for url_template in SACKMANN_URLS:
        url = url_template.format(y=year)
        try:
            raw = pd.read_csv(url)
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    else:
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"No Sackmann fallback URL configured for {year}")

    converted = pd.DataFrame(
        {
            "ATP": raw.get("tourney_id", ""),
            "Location": raw.get("tourney_name", ""),
            "Tournament": raw.get("tourney_name", ""),
            "Date": pd.to_datetime(raw["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"),
            "Series": raw.get("tourney_level", "").map(series_from_sackmann_level),
            "Court": "Outdoor",
            "Surface": raw.get("surface", ""),
            "Round": raw.get("round", ""),
            "Best of": raw.get("best_of", 3),
            "Winner": raw.get("winner_name", ""),
            "Loser": raw.get("loser_name", ""),
            "WRank": raw.get("winner_rank", pd.NA),
            "LRank": raw.get("loser_rank", pd.NA),
            "WPts": raw.get("winner_rank_points", pd.NA),
            "LPts": raw.get("loser_rank_points", pd.NA),
            "AvgW": pd.NA,
            "AvgL": pd.NA,
            "Comment": raw.get("score", "").astype(str).str.contains("W/O", regex=False).map(
                {True: "Walkover", False: "Completed"}
            ),
        }
    )
    score_frame = pd.DataFrame([score_columns(score) for score in raw.get("score", "")])
    converted = pd.concat([converted, score_frame], axis=1)
    converted = converted.dropna(subset=["Date", "Winner", "Loser"])
    converted.to_excel(destination, index=False)
    return url


def write_source_status(records: list[dict], healthy: bool, current: int) -> None:
    """Persist source health so the site can disclose stale or fallback data."""
    SOURCE_STATUS.write_text(json.dumps({
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "primary_source": "tennis-data.co.uk",
        "fallback_sources": list(SACKMANN_URLS),
        "records": records,
        "healthy": healthy,
        "current_year_source": next((record["source"] for record in records if record["year"] == current), None),
        "current_year_url": next((record["url"] for record in records if record["year"] == current), None),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    DATA.mkdir(exist_ok=True)
    current = date.today().year
    use_primary_source = True
    records = []
    for year in range(FIRST_SEASON, current + 1):
        out = DATA / f"atp_{year}.xlsx"
        if out.exists() and year < current:
            records.append({"year": year, "source": "cache", "ok": True, "url": None, "error": None})
            continue
        try:
            if use_primary_source:
                download_with_retries(year, out)
                print(f"  atp_{year}.xlsx OK")
                records.append({
                    "year": year,
                    "source": "tennis-data.co.uk",
                    "ok": True,
                    "url": URL.format(y=year),
                    "error": None,
                })
            else:
                fallback_url = download_sackmann_fallback(year, out)
                print(f"  atp_{year}.xlsx OK via fallback")
                records.append({
                    "year": year,
                    "source": "sackmann-compatible-fallback",
                    "ok": True,
                    "url": fallback_url,
                    "error": None,
                })
        except Exception as exc:  # noqa: BLE001
            use_primary_source = False
            print(f"  atp_{year}.xlsx ECHEC : {exc}", file=sys.stderr)
            try:
                fallback_url = download_sackmann_fallback(year, out)
                print(f"  atp_{year}.xlsx OK via Sackmann-compatible fallback")
                records.append({
                    "year": year,
                    "source": "sackmann-compatible-fallback",
                    "ok": True,
                    "url": fallback_url,
                    "primary_error": str(exc),
                    "error": None,
                })
            except Exception as fallback_exc:  # noqa: BLE001
                print(f"  atp_{year}.xlsx fallback ECHEC : {fallback_exc}", file=sys.stderr)
                if year < current or not out.exists():
                    records.append({
                        "year": year,
                        "source": "none",
                        "ok": False,
                        "url": None,
                        "primary_error": str(exc),
                        "error": str(fallback_exc),
                    })
                    write_source_status(records, healthy=False, current=current)
                    return 1
                print(f"  atp_{year}.xlsx conserve depuis le fichier existant", file=sys.stderr)
                records.append({
                    "year": year,
                    "source": "cache",
                    "ok": True,
                    "url": None,
                    "primary_error": str(exc),
                    "fallback_error": str(fallback_exc),
                    "error": None,
                })

    write_source_status(records, healthy=any(record["ok"] for record in records), current=current)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
