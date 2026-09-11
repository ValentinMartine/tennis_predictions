"""Download ATP seasons from tennis-data.co.uk with GitHub CSV fallbacks.

The primary source keeps the historical betting odds. When it is unavailable, the
pipeline falls back to Sackmann-compatible public CSV mirrors so the model can
still refresh results and Elo features without pretending that market odds were
available.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import time
import unicodedata
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIRST_SEASON = 2000
VALUEBETENNIS_URL = "https://www.valuebetennis.com/datasets/valuebetennis-matchs-{y}.csv"
VALUEBETENNIS_REFERER = "https://www.valuebetennis.com/donnees.htm"

# Un circuit = un préfixe de fichier, une source primaire et une liste de miroirs au
# format Sackmann. Le reste du pipeline est identique pour l'ATP et la WTA.
TOURS = {
    "atp": {
        "primary": "http://www.tennis-data.co.uk/{y}/{y}.xlsx",
        "genre": "atp",
        "mirrors": (
            "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{y}.csv",
            "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main/atp/atp_matches_{y}.csv",
            "https://raw.githubusercontent.com/azhou5849/tennis-atp-sackmann-data/master/atp_matches_{y}.csv",
            "https://raw.githubusercontent.com/beta2k/tennis_atp/master/atp_matches_{y}.csv",
            "https://raw.githubusercontent.com/FerrisPhilippi/tennis_atp_data/master/atp_matches_{y}.csv",
            "https://raw.githubusercontent.com/aadvikchaturvedi/tennis_atp_dataset/master/atp_matches_{y}.csv",
            "https://raw.githubusercontent.com/migumax/tennis_atp/master/{y}.csv",
        ),
    },
    "wta": {
        "primary": "http://www.tennis-data.co.uk/{y}w/{y}.xlsx",
        "genre": "wta",
        "mirrors": (
            "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{y}.csv",
            "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main/wta/wta_matches_{y}.csv",
            "https://raw.githubusercontent.com/azhou5849/tennis-wta-sackmann-data/master/wta_matches_{y}.csv",
        ),
    },
}
SHORT_NAME_RE = re.compile(r"^(.+?)\s+((?:[A-Z]\.){1,3})$")
# Corpus consolidé, versionné : c'est lui qui rend le modèle reproductible. Les .xlsx
# restent des fichiers de travail ignorés par git, la CI n'ayant plus à les retélécharger.
NUMERIC_COLUMNS = (
    ["WRank", "LRank", "WPts", "LPts", "Best of", "Wsets", "Lsets"]
    + [f"{side}{index}" for index in range(1, 6) for side in ("W", "L")]
)
# Périmètre de tennis-data.co.uk : tableau final des épreuves du circuit principal.
# Ni Challenger, ni ITF, ni Coupe Davis, ni qualifications.
MAIN_TOUR_SERIES = frozenset({
    "Grand Slam", "Masters 1000", "ATP", "Tour Finals",
    "WTA 1000", "WTA 500", "WTA 250",
})
QUALIFYING_ROUNDS = frozenset({"Qualifying 1", "Qualifying 2", "Qualifying 3"})
SACKMANN_LEVELS = {
    "atp": {
        "G": "Grand Slam",
        "M": "Masters 1000",
        "A": "ATP",
        "F": "Tour Finals",
        "D": "Davis Cup",
    },
    "wta": {
        "G": "Grand Slam",
        "PM": "WTA 1000",
        "P": "WTA 1000",
        "T1": "WTA 1000",
        "T2": "WTA 500",
        "T3": "WTA 250",
        "I": "WTA 250",
        "D": "Fed Cup",
    },
}
MAX_ATTEMPTS = 4
RETRY_DELAYS_SECONDS = (2, 5, 10)
CURRENT_YEAR_MAX_STALENESS_DAYS = 3


def download_with_retries(year: int, destination: Path, tour: str = "atp", attempts: int = MAX_ATTEMPTS) -> None:
    """Download one tennis-data season with retries and atomic replacement."""
    tmp = destination.with_suffix(".xlsx.tmp")
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(TOURS[tour]["primary"].format(y=year), timeout=30) as response:
                with tmp.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            tmp.replace(destination)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if tmp.exists():
                tmp.unlink()
            if attempt < attempts:
                time.sleep(RETRY_DELAYS_SECONDS[attempt - 1])

    if last_error is not None:
        raise last_error


def _name_key(text: object) -> str:
    """Clé de comparaison : sans accent, sans ponctuation, en minuscules."""
    folded = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(c for c in folded.lower() if c.isalnum())


def _tournament_key(text: object) -> str:
    """Loose tournament key shared by Valuebetennis and Sackmann labels."""
    head = str(text or "").split("-", 1)[0]
    return _name_key(head)


_SEASON_NAMES_CACHE: dict[tuple[str, int | None], dict[str, dict[str, str]]] = {}


def _initials(tokens: list[str]) -> str:
    return "".join(f"{token[0].upper()}." for token in tokens if token)


def known_surnames(exclude_year: int | None = None, tour: str = "atp") -> dict[str, dict[str, str]]:
    """Indexe la convention tennis-data : nom de famille normalisé -> initiales -> nom court."""
    key = (tour, exclude_year)
    if key in _SEASON_NAMES_CACHE:
        return _SEASON_NAMES_CACHE[key]
    index: dict[str, dict[str, str]] = {}
    for path in sorted(DATA.glob(f"{tour}_*.xlsx")):
        try:
            year = int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            year = None
        if exclude_year is not None and year == exclude_year:
            continue
        try:
            frame = pd.read_excel(path, usecols=lambda column: column in {"Winner", "Loser"})
        except Exception:  # noqa: BLE001
            continue
        for value in pd.concat([frame.get("Winner", pd.Series(dtype="string")),
                                frame.get("Loser", pd.Series(dtype="string"))]).dropna():
            match = SHORT_NAME_RE.match(str(value).strip())
            if not match:
                continue
            surname, initials = match.groups()
            index.setdefault(_name_key(surname), {})[initials] = str(value).strip()
    _SEASON_NAMES_CACHE[key] = index
    return index


def short_name_from_full(name: object, surnames: dict[str, dict[str, str]]) -> str:
    """Convertit un nom complet Valuebetennis/Sackmann vers la convention tennis-data."""
    tokens = [token for token in re.split(r"\s+", str(name or "").strip()) if token]
    if len(tokens) <= 1:
        return str(name or "").strip()

    for split in range(1, len(tokens)):
        surname = " ".join(tokens[split:])
        surname_key = _name_key(surname)
        choices = surnames.get(surname_key)
        if not choices:
            continue
        initials = _initials(tokens[:split])
        if initials in choices:
            return choices[initials]
        same_first_initial = [
            short for known_initials, short in choices.items()
            if known_initials[:1] == initials[:1]
        ]
        if len(same_first_initial) == 1:
            return same_first_initial[0]

    # Certains flux écrivent plusieurs noms de famille ("Jaume Antoni Munar Clar")
    # alors que tennis-data/Sackmann n'en gardent qu'un ("Munar J.").
    for surname_index in range(len(tokens) - 2, 0, -1):
        surname_key = _name_key(tokens[surname_index])
        choices = surnames.get(surname_key)
        if not choices:
            continue
        initials = _initials(tokens[:surname_index])
        if initials in choices:
            return choices[initials]
        same_first_initial = [
            short for known_initials, short in choices.items()
            if known_initials[:1] == initials[:1]
        ]
        if len(same_first_initial) == 1:
            return same_first_initial[0]

    surname = tokens[-1]
    initials = _initials(tokens[:-1])
    return f"{surname} {initials}".strip()


def seasons_path(tour: str) -> Path:
    return DATA / f"seasons_{tour}.parquet"


def load_seasons(tour: str) -> pd.DataFrame | None:
    """Corpus consolidé du circuit, ou None s'il n'a pas encore été construit."""
    path = seasons_path(tour)
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    return frame


def normalise_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Types stables pour Parquet : « NR » et cases vides deviennent des NaN."""
    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    for column in frame.columns:
        if column == "Date":
            continue
        if column in NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            frame[column] = frame[column].astype("string")
    return frame


def consolidate_seasons(tour: str) -> tuple[Path, int]:
    """Réécrit le corpus : tout .xlsx présent fait autorité, le Parquet fournit le reste.

    Cette règle évite la perte d'une saison présente sur le disque mais absente du corpus,
    et rend l'opération idempotente quel que soit ce qui a été retéléchargé.
    """
    on_disk = {int(path.stem.rsplit("_", 1)[1]): path for path in DATA.glob(f"{tour}_*.xlsx")}
    frames = []
    previous = load_seasons(tour)
    if previous is not None:
        kept = previous[~previous["Date"].dt.year.isin(on_disk)]
        if len(kept):
            frames.append(kept)
    for year in sorted(on_disk):
        frames.append(pd.read_excel(on_disk[year]))
    if not frames:
        raise RuntimeError(f"Aucune saison à consolider pour {tour}")
    merged = normalise_types(pd.concat(frames, ignore_index=True))
    merged = merged.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    target = seasons_path(tour)
    merged.to_parquet(target, index=False, compression="zstd")
    return target, len(merged)


def series_from_sackmann_level(level: object, tour: str = "atp") -> str:
    """Map Sackmann tournament levels to tennis-data style labels."""
    mapping = SACKMANN_LEVELS[tour]
    return mapping.get(str(level or "").strip(), "ATP" if tour == "atp" else "WTA 250")


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


def complete_set_count(values: dict[str, int | None]) -> int:
    """Count fully parsed sets in tennis-data compatible score columns."""
    return sum(
        values.get(f"W{index}") is not None and values.get(f"L{index}") is not None
        for index in range(1, 6)
    )


def score_is_decisive(values: dict[str, int | None], best_of: object) -> bool:
    """Return True when the score gives the announced winner enough sets."""
    try:
        best = int(float(best_of))
    except (TypeError, ValueError):
        best = 3
    target_sets = 3 if best >= 5 else 2
    winner_sets = int(values.get("Wsets") or 0)
    loser_sets = int(values.get("Lsets") or 0)
    return winner_sets >= target_sets and winner_sets > loser_sets


def read_sackmann_raw(year: int, tour: str) -> tuple[pd.DataFrame, str]:
    """Read the first available Sackmann-format mirror for a season."""
    required = {"tourney_date", "winner_name", "loser_name", "score"}
    last_error: Exception | None = None
    for url_template in TOURS[tour]["mirrors"]:
        url = url_template.format(y=year)
        try:
            raw = pd.read_csv(url)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
        if required.issubset(raw.columns):
            return raw, url
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"No Sackmann fallback URL configured for {tour} {year}")


def sackmann_score_repairs(
    year: int,
    tour: str,
    surnames: dict[str, dict[str, str]],
) -> dict[tuple[str, str, str], dict[str, int | None]]:
    """Build conservative score repairs for Valuebetennis partial scores."""
    try:
        raw, _ = read_sackmann_raw(year, tour)
    except Exception:  # noqa: BLE001
        return {}

    repairs: dict[tuple[str, str, str], dict[str, int | None]] = {}
    pair_repairs: dict[tuple[str, str], list[dict[str, int | None]]] = {}
    for _, match in raw.iterrows():
        match_date = pd.to_datetime(str(match.get("tourney_date")), format="%Y%m%d", errors="coerce")
        if pd.isna(match_date):
            continue
        values = score_columns(match.get("score"))
        if not score_is_decisive(values, match.get("best_of", 3)):
            continue
        winner = short_name_from_full(match.get("winner_name"), surnames)
        loser = short_name_from_full(match.get("loser_name"), surnames)
        if winner and loser:
            repairs[(f"date:{match_date.date().isoformat()}", winner, loser)] = values
            tournament_key = _tournament_key(match.get("tourney_name"))
            if tournament_key:
                repairs[(f"tournament:{tournament_key}", winner, loser)] = values
            pair_repairs.setdefault((winner, loser), []).append(values)
    for (winner, loser), candidates in pair_repairs.items():
        if len(candidates) == 1:
            repairs[("pair", winner, loser)] = candidates[0]
    return repairs


def maybe_repair_score_from_sackmann(
    score_frame: dict[str, int | None],
    match_date: object,
    tournament: object,
    winner: str,
    loser: str,
    repairs: dict[tuple[str, str, str], dict[str, int | None]],
) -> tuple[dict[str, int | None], bool, int]:
    """Replace a partial Valuebetennis score with a complete Sackmann score when possible."""
    partial_score = bool(score_frame.pop("_partial", False))
    complete_sets = int(score_frame.pop("_complete_sets", 0) or 0)
    if partial_score:
        parsed_date = pd.to_datetime(match_date, errors="coerce")
        keys: list[tuple[str, str, str]] = []
        if not pd.isna(parsed_date):
            keys.append((f"date:{parsed_date.date().isoformat()}", winner, loser))
        tournament_key = _tournament_key(tournament)
        if tournament_key:
            keys.append((f"tournament:{tournament_key}", winner, loser))
        keys.append(("pair", winner, loser))
        for key in keys:
            repaired = repairs.get(key)
            if repaired:
                return dict(repaired), False, complete_set_count(repaired)
    return score_frame, partial_score, complete_sets


def valuebetennis_surface(surface: object) -> str:
    """Map Valuebetennis French surface labels to tennis-data labels."""
    mapping = {
        "dur": "Hard",
        "terre battue": "Clay",
        "gazon": "Grass",
        "moquette": "Carpet",
    }
    return mapping.get(str(surface or "").strip().lower(), str(surface or ""))


def valuebetennis_series(category: object, tour: str = "atp") -> str:
    """Map Valuebetennis categories to tennis-data style series labels."""
    if tour == "wta":
        return {
            "grand chelem": "Grand Slam", "masters": "WTA 1000", "main tour": "WTA 250",
            "challenger": "Challenger", "itf": "ITF", "coupe davis / bjk cup": "Fed Cup",
        }.get(str(category or "").strip().lower(), "WTA 250")
    mapping = {
        "grand chelem": "Grand Slam",
        "masters": "Masters 1000",
        "main tour": "ATP",
        "challenger": "Challenger",
        "itf": "ITF",
        "coupe davis / bjk cup": "Davis Cup",
    }
    return mapping.get(str(category or "").strip().lower(), "ATP")


# Codes `tour` de Valuebetennis : 1-3 = qualifications, 4-7 = tours du tableau final
# comptés depuis le début (un 250 saute 6 et 7), 8 = poules, puis 9/10/12 absolus.
# Vérifié sur Australian Open (4:64, 5:32, 6:16, 7:8, 9:4, 10:2, 12:1),
# Indian Wells et un ATP 250 (4:12, 5:8, 9:4, 10:2, 12:1).
VALUEBETENNIS_ROUNDS = {
    1: "Qualifying 1",
    2: "Qualifying 2",
    3: "Qualifying 3",
    4: "1st Round",
    5: "2nd Round",
    6: "3rd Round",
    7: "4th Round",
    8: "Round Robin",
    9: "Quarterfinals",
    10: "Semifinals",
    12: "The Final",
}


def valuebetennis_round(round_code: object) -> str:
    """Map Valuebetennis numeric rounds to labels used by the existing UI/export."""
    try:
        value = int(float(round_code))
    except (TypeError, ValueError):
        return str(round_code or "1st Round")
    return VALUEBETENNIS_ROUNDS.get(value, f"Round {value}")


def is_finished_set(a: int, b: int) -> bool:
    """Un set terminé : 6-0 à 6-4, 7-5, 7-6, ou un super tie-break à 10 et plus."""
    high, low = max(a, b), min(a, b)
    return high >= 10 or (high == 6 and low <= 4) or (high == 7 and low in (5, 6))


def score_columns_from_valuebetennis(score: object, winner_is_player1: bool) -> dict[str, int | None]:
    """Extract set scores when Valuebetennis scores are expressed player1-player2.

    La source tronque le dernier set d'un match sur huit : « 5-7 3-6 7-5 6-3 5-5 » pour un
    match gagné en cinq. Les sets inachevés sont écartés du décompte et le match est marqué
    partiel, sinon le score affiché contredit le vainqueur annoncé.
    """
    values = {f"{side}{index}": None for index in range(1, 6) for side in ("W", "L")}
    winner_sets = 0
    loser_sets = 0
    complete_sets = 0
    partial = False
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
        if not is_finished_set(p1_games, p2_games):
            partial = True
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
    values["_partial"] = partial
    return values


def download_valuebetennis_fallback(year: int, destination: Path, tour: str = "atp") -> str:
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

    raw = raw[raw["genre"].astype(str).str.lower().eq(TOURS[tour]["genre"])].copy()
    # Le fichier Valuebetennis descend jusqu'aux ITF. La source primaire couvre le seul
    # main tour : garder le même périmètre, sinon les métriques ne sont plus comparables
    # et le classement Elo mélange des circuits sans rapport.
    raw = raw[raw["categorie"].map(lambda c: valuebetennis_series(c, tour)).isin(MAIN_TOUR_SERIES)].copy()
    raw["Date"] = pd.to_datetime(raw["date"], errors="coerce", utc=True).dt.tz_localize(None)
    raw = raw.dropna(subset=["Date", "joueur1", "joueur2", "vainqueur_id", "score"])

    surnames = known_surnames(exclude_year=year, tour=tour)
    score_repairs = sackmann_score_repairs(year, tour, surnames)
    rows = []
    for _, match in raw.iterrows():
        winner_is_player1 = str(match["vainqueur_id"]) == str(match["joueur1_id"])
        winner = short_name_from_full(match["joueur1"] if winner_is_player1 else match["joueur2"], surnames)
        loser = short_name_from_full(match["joueur2"] if winner_is_player1 else match["joueur1"], surnames)
        round_label = valuebetennis_round(match["tour"])
        if round_label in QUALIFYING_ROUNDS:
            continue
        score_frame = score_columns_from_valuebetennis(match["score"], winner_is_player1)
        score_frame, partial_score, complete_sets = maybe_repair_score_from_sackmann(
            score_frame,
            match["Date"],
            match["tournoi"],
            winner,
            loser,
            score_repairs,
        )
        # Un abandon au premier set reste un résultat : il suffit d'un set terminé pour
        # que le match compte. Sans set exploitable, la ligne n'apporte rien.
        if complete_sets < 1:
            continue
        avg_w = match["cote1_cloture"] if winner_is_player1 else match["cote2_cloture"]
        avg_l = match["cote2_cloture"] if winner_is_player1 else match["cote1_cloture"]
        rows.append({
            "ATP": match["tournoi_id"],
            "Location": match["tournoi"],
            "Tournament": match["tournoi"],
            "Date": match["Date"],
            "Series": valuebetennis_series(match["categorie"], tour),
            "Court": "Outdoor",
            "Surface": valuebetennis_surface(match["surface"]),
            "Round": round_label,
            # La WTA joue au meilleur des 3 sets partout, Grands Chelems compris.
            "Best of": 5 if (tour == "atp" and valuebetennis_series(match["categorie"], tour) == "Grand Slam") else 3,
            "Winner": winner,
            "Loser": loser,
            "WRank": pd.NA,
            "LRank": pd.NA,
            "WPts": pd.NA,
            "LPts": pd.NA,
            "AvgW": avg_w,
            "AvgL": avg_l,
            "Comment": "Completed",
            "ScoreStatus": "partial" if partial_score else "complete",
            **score_frame,
        })

    if not rows:
        raise RuntimeError(f"No completed {tour.upper()} main-tour rows in Valuebetennis file for {year}")
    pd.DataFrame(rows).to_excel(destination, index=False)
    return url


def download_sackmann_fallback(year: int, destination: Path, tour: str = "atp") -> str:
    """Build a tennis-data compatible Excel file from a public CSV mirror."""
    raw, url = read_sackmann_raw(year, tour)

    surnames = known_surnames(exclude_year=year, tour=tour)
    converted = pd.DataFrame(
        {
            "ATP": raw.get("tourney_id", ""),
            "Location": raw.get("tourney_name", ""),
            "Tournament": raw.get("tourney_name", ""),
            "Date": pd.to_datetime(raw["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"),
            "Series": raw.get("tourney_level", "").map(lambda lv: series_from_sackmann_level(lv, tour)),
            "Court": "Outdoor",
            "Surface": raw.get("surface", ""),
            "Round": raw.get("round", ""),
            "Best of": raw.get("best_of", 3),
            "Winner": raw.get("winner_name", "").map(lambda n: short_name_from_full(n, surnames)),
            "Loser": raw.get("loser_name", "").map(lambda n: short_name_from_full(n, surnames)),
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
    # Même périmètre que le secours Valuebetennis : ni Coupe Davis, ni BJK Cup.
    converted = converted[converted["Series"].isin(MAIN_TOUR_SERIES)]
    converted.to_excel(destination, index=False)
    return url


def status_path(tour: str) -> Path:
    return DATA / ("source_status.json" if tour == "atp" else f"source_status_{tour}.json")


def write_source_status(records: list[dict], healthy: bool, current: int, tour: str = "atp") -> None:
    """Persist source health so the site can disclose stale or fallback data."""
    status_path(tour).write_text(json.dumps({
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tour": tour,
        "primary_source": "tennis-data.co.uk",
        "fresh_csv_fallback": "Valuebetennis",
        "fresh_csv_fallback_url": VALUEBETENNIS_REFERER,
        "current_year_max_staleness_days": CURRENT_YEAR_MAX_STALENESS_DAYS,
        "fallback_sources": list(TOURS[tour]["mirrors"]),
        "records": records,
        "healthy": healthy,
        "current_year_source": next((record["source"] for record in records if record["year"] == current), None),
        "current_year_url": next((record["url"] for record in records if record["year"] == current), None),
        "current_year_latest_match_date": next(
            (record.get("latest_match_date") for record in records if record["year"] == current),
            None,
        ),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def download_fallback(year: int, destination: Path, tour: str = "atp") -> tuple[str, str, str | None, int | None]:
    """Download the freshest available public fallback and return source metadata."""
    try:
        url = download_valuebetennis_fallback(year, destination, tour)
        converted = pd.read_excel(destination)
        latest = pd.to_datetime(converted["Date"]).max()
        return "valuebetennis", url, latest.date().isoformat() if pd.notna(latest) else None, len(converted)
    except Exception:  # noqa: BLE001
        sackmann_url = download_sackmann_fallback(year, destination, tour)
        converted = pd.read_excel(destination)
        latest = pd.to_datetime(converted["Date"]).max()
        return (
            "sackmann-compatible-fallback",
            sackmann_url,
            latest.date().isoformat() if pd.notna(latest) else None,
            len(converted),
        )


def main(argv: list[str] | None = None) -> int:
    tour = (argv or sys.argv[1:] or ["atp"])[0].lower()
    if tour not in TOURS:
        print(f"circuit inconnu : {tour} (attendu : {', '.join(TOURS)})", file=sys.stderr)
        return 2
    DATA.mkdir(exist_ok=True)
    current = date.today().year
    records = []
    primary_failures = 0
    seasons = load_seasons(tour)
    covered = set(seasons["Date"].dt.year.dropna().astype(int)) if seasons is not None else set()
    for year in range(FIRST_SEASON, current + 1):
        out = DATA / f"{tour}_{year}.xlsx"
        if year < current and (year in covered or out.exists()):
            records.append({"year": year, "source": "corpus" if year in covered else "cache",
                            "ok": True, "url": None, "error": None})
            continue
        try:
            # Chaque saison retente la source primaire, mais sans repayer quatre essais
            # avec pauses quand elle est visiblement hors service.
            download_with_retries(year, out, tour, 1 if primary_failures >= 3 else MAX_ATTEMPTS)
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
            print(f"  {tour}_{year}.xlsx OK")
            records.append({
                "year": year,
                "source": "tennis-data.co.uk",
                "ok": True,
                "url": TOURS[tour]["primary"].format(y=year),
                "latest_match_date": latest.date().isoformat() if pd.notna(latest) else None,
                "rows": int(len(primary)),
                "error": None,
            })
        except Exception as exc:  # noqa: BLE001
            # Un 503 ponctuel sur 2000 ne doit pas basculer les 26 saisons suivantes
            # sur un fallback dépourvu de classements et de cotes.
            primary_failures += 1
            print(f"  {tour}_{year}.xlsx ECHEC : {exc}", file=sys.stderr)
            try:
                fallback_source, fallback_url, latest_date, rows = download_fallback(year, out, tour)
                print(f"  {tour}_{year}.xlsx OK via {fallback_source}")
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
                print(f"  {tour}_{year}.xlsx fallback ECHEC : {fallback_exc}", file=sys.stderr)
                if year < current or not out.exists():
                    records.append({
                        "year": year,
                        "source": "none",
                        "ok": False,
                        "url": None,
                        "primary_error": str(exc),
                        "error": str(fallback_exc),
                    })
                    write_source_status(records, healthy=False, current=current, tour=tour)
                    return 1
                print(f"  {tour}_{year}.xlsx conserve depuis le fichier existant", file=sys.stderr)
                records.append({
                    "year": year,
                    "source": "cache",
                    "ok": True,
                    "url": None,
                    "primary_error": str(exc),
                    "fallback_error": str(fallback_exc),
                    "error": None,
                })

    target, rows = consolidate_seasons(tour)
    print(f"  {target.name} : {rows} matchs consolidés")
    write_source_status(records, healthy=all(record["ok"] for record in records), current=current, tour=tour)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
