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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCE_STATUS = DATA / "source_status.json"
FIRST_SEASON = 2000
URL = "http://www.tennis-data.co.uk/{y}/{y}.xlsx"
VALUEBETENNIS_URL = "https://www.valuebetennis.com/datasets/valuebetennis-matchs-{y}.csv"
VALUEBETENNIS_REFERER = "https://www.valuebetennis.com/donnees.htm"
SACKMANN_URLS = (
    "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{y}.csv",
    "https://raw.githubusercontent.com/azhou5849/tennis-atp-sackmann-data/master/atp_matches_{y}.csv",
    "https://raw.githubusercontent.com/beta2k/tennis_atp/master/atp_matches_{y}.csv",
    "https://raw.githubusercontent.com/FerrisPhilippi/tennis_atp_data/master/atp_matches_{y}.csv",
    "https://raw.githubusercontent.com/aadvikchaturvedi/tennis_atp_dataset/master/atp_matches_{y}.csv",
    "https://raw.githubusercontent.com/migumax/tennis_atp/master/{y}.csv",
)
MAX_ATTEMPTS = 4
RETRY_DELAYS_SECONDS = (2, 5, 10)
CURRENT_YEAR_MAX_STALENESS_DAYS = 3


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


def valuebetennis_surface(surface: object) -> str:
    """Map Valuebetennis French surface labels to tennis-data labels."""
    mapping = {
        "dur": "Hard",
        "terre battue": "Clay",
        "gazon": "Grass",
        "moquette": "Carpet",
    }
    return mapping.get(str(surface or "").strip().lower(), str(surface or ""))


def valuebetennis_series(category: object) -> str:
    """Map Valuebetennis categories to tennis-data style series labels."""
    mapping = {
        "grand chelem": "Grand Slam",
        "masters": "Masters 1000",
        "main tour": "ATP",
        "challenger": "Challenger",
        "itf": "ITF",
    }
    return mapping.get(str(category or "").strip().lower(), "ATP")


def valuebetennis_round(round_code: object) -> str:
    """Map Valuebetennis numeric rounds to labels used by the existing UI/export."""
    try:
        value = int(float(round_code))
    except (TypeError, ValueError):
        return str(round_code or "1st Round")
    return {
        12: "The Final",
        11: "Semifinals",
        10: "Quarterfinals",
        9: "4th Round",
        8: "3rd Round",
        7: "2nd Round",
        6: "1st Round",
    }.get(value, f"Round {value}")


def score_columns_from_valuebetennis(score: object, winner_is_player1: bool) -> dict[str, int | None]:
    """Extract set scores when Valuebetennis scores are expressed player1-player2."""
    values = {f"{side}{index}": None for index in range(1, 6) for side in ("W", "L")}
    winner_sets = 0
    loser_sets = 0
    complete_sets = 0
    for index, token in enumerate(str(score or "").split()[:5], start=1):
        clean = token.split("(", 1)[0]
        if "-" not in clean:
            continue
        player1_games, player2_games = clean.split("-", 1)
        try:
            p1_games = int(player1_games)
            p2_games = int(player2_games)
        except ValueError:
            continue
        if p1_games == p2_games:
            continue
        w_games, l_games = (p1_games, p2_games) if winner_is_player1 else (p2_games, p1_games)
        values[f"W{index}"] = w_games
        values[f"L{index}"] = l_games
        complete_sets += 1
        if w_games > l_games:
            winner_sets += 1
        else:
            loser_sets += 1
    values["Wsets"] = winner_sets or None
    values["Lsets"] = loser_sets or None
    values["_complete_sets"] = complete_sets
    return values


def download_valuebetennis_fallback(year: int, destination: Path) -> str:
    """Build a tennis-data compatible Excel file from the fresh Valuebetennis CSV."""
    url = VALUEBETENNIS_URL.format(y=year)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 valentinmartine-tennis-predictions/1.0",
            "Referer": VALUEBETENNIS_REFERER,
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = pd.read_csv(response, sep=";")

    raw = raw[raw["genre"].astype(str).str.lower().eq("atp")].copy()
    raw["Date"] = pd.to_datetime(raw["date"], errors="coerce", utc=True).dt.tz_localize(None)
    raw = raw.dropna(subset=["Date", "joueur1", "joueur2", "vainqueur_id", "score"])

    rows = []
    for _, match in raw.iterrows():
        winner_is_player1 = str(match["vainqueur_id"]) == str(match["joueur1_id"])
        winner = match["joueur1"] if winner_is_player1 else match["joueur2"]
        loser = match["joueur2"] if winner_is_player1 else match["joueur1"]
        score_frame = score_columns_from_valuebetennis(match["score"], winner_is_player1)
        if score_frame.pop("_complete_sets", 0) < 2:
            continue
        avg_w = match["cote1_cloture"] if winner_is_player1 else match["cote2_cloture"]
        avg_l = match["cote2_cloture"] if winner_is_player1 else match["cote1_cloture"]
        rows.append({
            "ATP": match["tournoi_id"],
            "Location": match["tournoi"],
            "Tournament": match["tournoi"],
            "Date": match["Date"],
            "Series": valuebetennis_series(match["categorie"]),
            "Court": "Outdoor",
            "Surface": valuebetennis_surface(match["surface"]),
            "Round": valuebetennis_round(match["tour"]),
            "Best of": 5 if valuebetennis_series(match["categorie"]) == "Grand Slam" else 3,
            "Winner": winner,
            "Loser": loser,
            "WRank": pd.NA,
            "LRank": pd.NA,
            "WPts": pd.NA,
            "LPts": pd.NA,
            "AvgW": avg_w,
            "AvgL": avg_l,
            "Comment": "Completed",
            **score_frame,
        })

    if not rows:
        raise RuntimeError(f"No completed ATP rows in Valuebetennis file for {year}")
    pd.DataFrame(rows).to_excel(destination, index=False)
    return url


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
        "fresh_csv_fallback": "Valuebetennis",
        "fresh_csv_fallback_url": VALUEBETENNIS_REFERER,
        "current_year_max_staleness_days": CURRENT_YEAR_MAX_STALENESS_DAYS,
        "fallback_sources": list(SACKMANN_URLS),
        "records": records,
        "healthy": healthy,
        "current_year_source": next((record["source"] for record in records if record["year"] == current), None),
        "current_year_url": next((record["url"] for record in records if record["year"] == current), None),
        "current_year_latest_match_date": next(
            (record.get("latest_match_date") for record in records if record["year"] == current),
            None,
        ),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def download_fallback(year: int, destination: Path) -> tuple[str, str, str | None, int | None]:
    """Download the freshest available public fallback and return source metadata."""
    try:
        url = download_valuebetennis_fallback(year, destination)
        converted = pd.read_excel(destination)
        latest = pd.to_datetime(converted["Date"]).max()
        return "valuebetennis", url, latest.date().isoformat() if pd.notna(latest) else None, len(converted)
    except Exception:  # noqa: BLE001
        sackmann_url = download_sackmann_fallback(year, destination)
        converted = pd.read_excel(destination)
        latest = pd.to_datetime(converted["Date"]).max()
        return (
            "sackmann-compatible-fallback",
            sackmann_url,
            latest.date().isoformat() if pd.notna(latest) else None,
            len(converted),
        )


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
                primary = pd.read_excel(out)
                latest = pd.to_datetime(primary["Date"]).max()
                if (
                    year == current
                    and pd.notna(latest)
                    and latest.date() < date.today() - timedelta(days=CURRENT_YEAR_MAX_STALENESS_DAYS)
                ):
                    raise RuntimeError(
                        f"tennis-data.co.uk current-year file stale: latest match {latest.date().isoformat()}"
                    )
                print(f"  atp_{year}.xlsx OK")
                records.append({
                    "year": year,
                    "source": "tennis-data.co.uk",
                    "ok": True,
                    "url": URL.format(y=year),
                    "latest_match_date": latest.date().isoformat() if pd.notna(latest) else None,
                    "rows": int(len(primary)),
                    "error": None,
                })
            else:
                fallback_source, fallback_url, latest_date, rows = download_fallback(year, out)
                print(f"  atp_{year}.xlsx OK via fallback")
                records.append({
                    "year": year,
                    "source": fallback_source,
                    "ok": True,
                    "url": fallback_url,
                    "latest_match_date": latest_date,
                    "rows": rows,
                    "error": None,
                })
        except Exception as exc:  # noqa: BLE001
            use_primary_source = False
            print(f"  atp_{year}.xlsx ECHEC : {exc}", file=sys.stderr)
            try:
                fallback_source, fallback_url, latest_date, rows = download_fallback(year, out)
                print(f"  atp_{year}.xlsx OK via {fallback_source}")
                records.append({
                    "year": year,
                    "source": fallback_source,
                    "ok": True,
                    "url": fallback_url,
                    "latest_match_date": latest_date,
                    "rows": rows,
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
