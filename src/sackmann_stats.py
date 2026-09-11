"""Jeff Sackmann ATP/WTA match statistics.

The tennis-data.co.uk files used by the main pipeline contain results, ranks and odds,
but not serve/return details. Jeff Sackmann's match CSVs add tour-level match
stats such as aces, double faults, service points and break points.
"""
from __future__ import annotations

import math
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE_DIR = DATA / "sackmann"
URLS = {
    "atp": [
        "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/refs/heads/master/atp_matches_{year}.csv",
        "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main/atp/atp_matches_{year}.csv",
    ],
    "wta": [
        "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/refs/heads/master/wta_matches_{year}.csv",
        "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main/wta/wta_matches_{year}.csv",
    ],
}

STAT_FEATURES = [
    "d_srv_pts_won",
    "d_ret_pts_won",
    "d_ace_rate",
    "d_df_rate",
    "d_bp_saved",
    "d_bp_converted",
]


def _norm_key(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return "".join(c.lower() for c in text if c.isalnum())


def _format_initials(names: list[str]) -> str:
    initials = "".join(f"{part[0].upper()}." for part in names if part)
    return initials or "?"


def _name_candidates(full_name: str) -> list[str]:
    parts = [p for p in str(full_name or "").replace("-", " ").split() if p]
    if len(parts) < 2:
        return [full_name]
    candidates: list[str] = []
    particles = {"da", "de", "del", "der", "di", "du", "la", "le", "van", "von"}

    def add(surname_parts: list[str], first_parts: list[str]) -> None:
        if not surname_parts or not first_parts:
            return
        surname = " ".join(part.capitalize() for part in surname_parts)
        initials = _format_initials(first_parts)
        for sep_name in {surname, surname.replace(" ", "-")}:
            candidate = f"{sep_name} {initials}"
            if candidate not in candidates:
                candidates.append(candidate)

    add([parts[-1]], parts[:-1])
    if len(parts) >= 3:
        add(parts[-2:], parts[:-2])
    if len(parts) >= 3 and parts[-2].lower() in particles:
        add(parts[-2:], parts[:-2])
    if len(parts) >= 4 and parts[-3].lower() in particles:
        add(parts[-3:], parts[:-3])
    return candidates


def make_player_mapper(existing_players: set[str] | None = None):
    """Return a mapper from Sackmann full names to tennis-data display names."""
    lookup = {_norm_key(player): player for player in (existing_players or set())}

    def mapper(full_name: str) -> str:
        for candidate in _name_candidates(full_name):
            found = lookup.get(_norm_key(candidate))
            if found:
                return found
        return _name_candidates(full_name)[0]

    return mapper


def download_match_stats(
    start_year: int = 2000,
    end_year: int | None = None,
    force: bool = False,
    tour: str = "atp",
) -> list[Path]:
    """Download Sackmann yearly match-stat CSVs into data/sackmann."""
    tour = tour.lower()
    if tour not in URLS:
        raise ValueError(f"Unsupported tour: {tour}")
    end_year = end_year or datetime.now().year
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for year in range(start_year, end_year + 1):
        out = CACHE_DIR / f"{tour}_matches_{year}.csv"
        if out.exists() and not force:
            paths.append(out)
            continue
        for template in URLS[tour]:
            response = requests.get(template.format(year=year), timeout=30)
            if response.status_code == 404:
                continue
            response.raise_for_status()
            out.write_bytes(response.content)
            paths.append(out)
            break
    return paths


def load_match_stats(
    existing_players: set[str] | None = None,
    start_year: int = 2000,
    end_year: int | None = None,
    tour: str = "atp",
) -> pd.DataFrame:
    """Load cached Sackmann match-stat CSVs and normalize names to tennis-data format."""
    tour = tour.lower()
    end_year = end_year or datetime.now().year
    frames = []
    for year in range(start_year, end_year + 1):
        path = CACHE_DIR / f"{tour}_matches_{year}.csv"
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    if "tourney_date" not in data:
        return pd.DataFrame()
    data["Date"] = pd.to_datetime(data["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    mapper = make_player_mapper(existing_players)
    data["Winner"] = data["winner_name"].map(mapper)
    data["Loser"] = data["loser_name"].map(mapper)
    return data.dropna(subset=["Date", "Winner", "Loser"]).sort_values("Date").reset_index(drop=True)


def _num(row, key: str) -> float:
    try:
        value = float(row.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


@dataclass
class StatAccumulator:
    matches: int = 0
    service_points: float = 0.0
    service_points_won: float = 0.0
    aces: float = 0.0
    double_faults: float = 0.0
    first_in: float = 0.0
    first_won: float = 0.0
    second_won: float = 0.0
    service_games: float = 0.0
    bp_saved: float = 0.0
    bp_faced: float = 0.0
    return_points: float = 0.0
    return_points_won: float = 0.0
    bp_chances: float = 0.0
    bp_converted: float = 0.0

    def update_service(self, row, prefix: str) -> None:
        svpt = _num(row, f"{prefix}_svpt")
        first_won = _num(row, f"{prefix}_1stWon")
        second_won = _num(row, f"{prefix}_2ndWon")
        self.matches += 1
        self.service_points += svpt
        self.service_points_won += first_won + second_won
        self.aces += _num(row, f"{prefix}_ace")
        self.double_faults += _num(row, f"{prefix}_df")
        self.first_in += _num(row, f"{prefix}_1stIn")
        self.first_won += first_won
        self.second_won += second_won
        self.service_games += _num(row, f"{prefix}_SvGms")
        self.bp_saved += _num(row, f"{prefix}_bpSaved")
        self.bp_faced += _num(row, f"{prefix}_bpFaced")

    def update_return(self, opponent, prefix: str) -> None:
        svpt = _num(opponent, f"{prefix}_svpt")
        first_won = _num(opponent, f"{prefix}_1stWon")
        second_won = _num(opponent, f"{prefix}_2ndWon")
        bp_faced = _num(opponent, f"{prefix}_bpFaced")
        bp_saved = _num(opponent, f"{prefix}_bpSaved")
        self.return_points += svpt
        self.return_points_won += max(0.0, svpt - first_won - second_won)
        self.bp_chances += bp_faced
        self.bp_converted += max(0.0, bp_faced - bp_saved)

    def rate(self, num: float, den: float, default: float = 0.0) -> float:
        return num / den if den else default

    def rates(self) -> dict[str, float | int]:
        return {
            "sackmann_matches": self.matches,
            "service_points": int(self.service_points),
            "ace_rate": self.rate(self.aces, self.service_points),
            "double_fault_rate": self.rate(self.double_faults, self.service_points),
            "first_in_rate": self.rate(self.first_in, self.service_points),
            "first_serve_won_rate": self.rate(self.first_won, self.first_in),
            "second_serve_won_rate": self.rate(self.second_won, self.service_points - self.first_in),
            "service_points_won_rate": self.rate(self.service_points_won, self.service_points),
            "break_points_saved_rate": self.rate(self.bp_saved, self.bp_faced),
            "return_points_won_rate": self.rate(self.return_points_won, self.return_points),
            "break_points_converted_rate": self.rate(self.bp_converted, self.bp_chances),
        }


def _z(value: float, values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    sd = math.sqrt(var)
    return 0.0 if sd == 0 else (value - mean) / sd


def aggregate_player_stats(matches: pd.DataFrame, min_service_points: int = 300) -> dict[str, dict]:
    """Aggregate serve/return statistics and derive custom statistical ratings."""
    if matches.empty:
        return {}
    stats: dict[str, StatAccumulator] = defaultdict(StatAccumulator)
    for _, row in matches.iterrows():
        winner, loser = row["Winner"], row["Loser"]
        stats[winner].update_service(row, "w")
        stats[winner].update_return(row, "l")
        stats[loser].update_service(row, "l")
        stats[loser].update_return(row, "w")

    rows = {player: acc.rates() for player, acc in stats.items() if acc.service_points >= min_service_points}
    service_values = [r["service_points_won_rate"] for r in rows.values()]
    return_values = [r["return_points_won_rate"] for r in rows.values()]
    ace_values = [r["ace_rate"] for r in rows.values()]
    df_values = [r["double_fault_rate"] for r in rows.values()]
    bp_saved_values = [r["break_points_saved_rate"] for r in rows.values()]
    bp_conv_values = [r["break_points_converted_rate"] for r in rows.values()]

    out: dict[str, dict] = {}
    for player, row in rows.items():
        serve_score = (
            0.55 * _z(row["service_points_won_rate"], service_values)
            + 0.20 * _z(row["ace_rate"], ace_values)
            - 0.15 * _z(row["double_fault_rate"], df_values)
            + 0.10 * _z(row["break_points_saved_rate"], bp_saved_values)
        )
        return_score = (
            0.75 * _z(row["return_points_won_rate"], return_values)
            + 0.25 * _z(row["break_points_converted_rate"], bp_conv_values)
        )
        pressure_score = (
            0.50 * _z(row["break_points_saved_rate"], bp_saved_values)
            + 0.50 * _z(row["break_points_converted_rate"], bp_conv_values)
        )
        stat_score = 0.42 * serve_score + 0.38 * return_score + 0.20 * pressure_score
        out[player] = {
            **row,
            "serve_elo": round(1500 + 100 * serve_score, 1),
            "return_elo": round(1500 + 100 * return_score, 1),
            "pressure_elo": round(1500 + 100 * pressure_score, 1),
            "stat_elo": round(1500 + 100 * stat_score, 1),
        }
    return out


@dataclass
class RollingSackmannStats:
    matches: pd.DataFrame
    index: int = 0
    stats: dict[str, StatAccumulator] = field(default_factory=lambda: defaultdict(StatAccumulator))

    @classmethod
    def from_cache(cls, existing_players: set[str] | None = None, tour: str = "atp") -> "RollingSackmannStats":
        return cls(load_match_stats(existing_players=existing_players, tour=tour))

    def advance_before(self, date) -> None:
        if self.matches.empty:
            return
        while self.index < len(self.matches) and self.matches.iloc[self.index]["Date"] < date:
            row = self.matches.iloc[self.index]
            winner, loser = row["Winner"], row["Loser"]
            self.stats[winner].update_service(row, "w")
            self.stats[winner].update_return(row, "l")
            self.stats[loser].update_service(row, "l")
            self.stats[loser].update_return(row, "w")
            self.index += 1

    def diff_features(self, a: str, b: str) -> dict[str, float]:
        ar = self.stats[a].rates()
        br = self.stats[b].rates()
        return {
            "d_srv_pts_won": ar["service_points_won_rate"] - br["service_points_won_rate"],
            "d_ret_pts_won": ar["return_points_won_rate"] - br["return_points_won_rate"],
            "d_ace_rate": ar["ace_rate"] - br["ace_rate"],
            "d_df_rate": br["double_fault_rate"] - ar["double_fault_rate"],
            "d_bp_saved": ar["break_points_saved_rate"] - br["break_points_saved_rate"],
            "d_bp_converted": ar["break_points_converted_rate"] - br["break_points_converted_rate"],
        }
