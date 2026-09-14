"""Résolution et interpolation des classements officiels ATP et WTA.

Fournit un accès aux classements hebdomadaires officiels Sackmann (2000-2026)
et aux classements contemporains post-US Open 2026 lorsque les flux primaires
sont indisponibles ou dépourvus de WRank/WPts.
"""
from __future__ import annotations

import math
import unicodedata
from bisect import bisect_right
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SACKMANN_DIR = DATA / "sackmann"

# Classement officiel post-US Open au 14 septembre 2026
CURRENT_ATP_RANKINGS_SEPT_2026: dict[str, tuple[int, int]] = {
    "Sinner J.": (1, 11500),
    "Zverev A.": (2, 9690),
    "Alcaraz C.": (3, 5560),
    "Shelton B.": (4, 4680),
    "Auger-Aliassime F.": (5, 3890),
    "Medvedev D.": (6, 3770),
    "Cobolli F.": (7, 3720),
    "Tiafoe F.": (8, 3380),
    "de Minaur A.": (9, 3300),
    "De Minaur A.": (9, 3300),
    "Fritz T.": (10, 3175),
    "Fils A.": (11, 3150),
    "Djokovic N.": (12, 2980),
    "Tien L.": (13, 2755),
    "Jodar R.": (14, 2659),
    "Mensik J.": (15, 2495),
    "Nakashima B.": (16, 2485),
    "Bublik A.": (17, 2425),
    "Ruud C.": (18, 2335),
    "Paul T.": (19, 2325),
    "Musetti L.": (20, 2225),
    "Darderi L.": (22, 1980),
    "Vacherot V.": (23, 1910),
    "Rublev A.": (24, 1890),
    "Cerundolo F.": (25, 1820),
    "Fonseca J.": (26, 1765),
    "Davidovich Fokina A.": (27, 1680),
    "Tabilo A.": (28, 1640),
    "Rinderknech A.": (29, 1590),
    "Buse I.": (30, 1540),
    "Humbert U.": (31, 1490),
    "Etcheverry T.": (32, 1450),
    "Van De Zandschulp B.": (33, 1420),
    "Berrettini M.": (34, 1380),
    "Lehecka J.": (35, 1350),
    "Khachanov K.": (36, 1320),
    "Draper J.": (37, 1290),
    "Navone M.": (38, 1260),
    "Machac T.": (39, 1230),
    "Struff J.": (40, 1200),
    "Struff J.L.": (40, 1200),
    "Korda S.": (41, 1180),
    "Michelsen A.": (42, 1150),
    "Gea A.": (43, 1120),
    "Blockx A.": (44, 1090),
    "Bonzi B.": (45, 1060),
    "Bergs Z.": (46, 1030),
    "Sonego L.": (47, 1000),
    "Borges N.": (48, 970),
    "Griekspoor T.": (49, 940),
    "Monfils G.": (50, 910),
}


def _norm_key(name: str) -> str:
    folded = unicodedata.normalize("NFKD", str(name or ""))
    return "".join(c for c in folded.lower() if c.isalnum())


class RankingsLookup:
    """Index de classement hebdomadaire pour un circuit (ATP ou WTA)."""

    def __init__(self, tour: str = "atp") -> None:
        self.tour = tour.lower()
        # Clé joueur normalisée -> liste ordonnée de (date_int, rank, pts)
        self._history: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
        # Clé joueur normalisée -> (latest_rank, latest_pts)
        self._latest: dict[str, tuple[float, float]] = {}
        self._load()

    def _load(self) -> None:
        rank_file = SACKMANN_DIR / f"{self.tour}_rankings_current.csv"
        player_file = SACKMANN_DIR / f"{self.tour}_players.csv"
        if not rank_file.exists() or not player_file.exists():
            return

        try:
            players_df = pd.read_csv(player_file, low_memory=False)
            id_to_names: dict[int, list[str]] = defaultdict(list)
            for _, r in players_df.iterrows():
                try:
                    pid = int(r["player_id"])
                except (ValueError, TypeError):
                    continue
                first = str(r.get("name_first") or "").strip()
                last = str(r.get("name_last") or "").strip()
                if not last:
                    continue
                initials = "".join(f"{p[0].upper()}." for p in first.split() if p)
                short_tennisdata = f"{last} {initials}".strip()
                id_to_names[pid].append(short_tennisdata)
                # Formats alternatifs courants
                id_to_names[pid].append(f"{last} {first}".strip())
                id_to_names[pid].append(last)

            rank_df = pd.read_csv(rank_file, low_memory=False)
            rank_df = rank_df.dropna(subset=["ranking_date", "rank", "player"])
            rank_df = rank_df.sort_values("ranking_date")

            for _, r in rank_df.iterrows():
                try:
                    rdate = int(r["ranking_date"])
                    rnk = float(r["rank"])
                    pts = float(r["points"]) if pd.notna(r.get("points")) else float("nan")
                    pid = int(r["player"])
                except (ValueError, TypeError):
                    continue

                for name in id_to_names.get(pid, []):
                    key = _norm_key(name)
                    self._history[key].append((rdate, rnk, pts))
                    self._latest[key] = (rnk, pts)

        except Exception as exc:  # noqa: BLE001
            print(f"Avertissement : échec du chargement des classements Sackmann ({exc})")

        # Intégration du classement post-US Open 2026 pour l'ATP
        if self.tour == "atp":
            latest_date_int = 20260914
            for player, (rnk, pts) in CURRENT_ATP_RANKINGS_SEPT_2026.items():
                key = _norm_key(player)
                self._history[key].append((latest_date_int, float(rnk), float(pts)))
                self._latest[key] = (float(rnk), float(pts))

    def get_rank_points(
        self,
        player_name: str,
        match_date: object = None,
    ) -> tuple[float | None, float | None]:
        """Retourne (rang, points) pour un joueur à une date donnée (as-of)."""
        key = _norm_key(player_name)
        hist = self._history.get(key)
        if not hist:
            # Essai avec le nom de famille seul
            parts = str(player_name or "").split()
            if parts:
                hist = self._history.get(_norm_key(parts[0]))
        if not hist:
            return None, None

        if match_date is None:
            rnk, pts = self._latest.get(key, (None, None))
            return (rnk if rnk and math.isfinite(rnk) else None,
                    pts if pts and math.isfinite(pts) else None)

        try:
            ts = pd.to_datetime(match_date)
            date_int = int(ts.strftime("%Y%m%d"))
        except Exception:  # noqa: BLE001
            rnk, pts = self._latest.get(key, (None, None))
            return (rnk if rnk and math.isfinite(rnk) else None,
                    pts if pts and math.isfinite(pts) else None)

        # Recherche de la dernière date de classement <= date du match
        dates = [entry[0] for entry in hist]
        idx = bisect_right(dates, date_int) - 1
        if idx >= 0:
            _, rnk, pts = hist[idx]
            return (rnk if rnk and math.isfinite(rnk) else None,
                    pts if pts and math.isfinite(pts) else None)

        # Si le match est antérieur au premier classement connu, prendre le premier
        _, rnk, pts = hist[0]
        return (rnk if rnk and math.isfinite(rnk) else None,
                pts if pts and math.isfinite(pts) else None)

    def get_latest_rank_points(self, player_name: str) -> tuple[float | None, float | None]:
        """Retourne le dernier classement et points connus pour le joueur."""
        key = _norm_key(player_name)
        if key in self._latest:
            rnk, pts = self._latest[key]
            return (rnk if rnk and math.isfinite(rnk) else None,
                    pts if pts and math.isfinite(pts) else None)
        parts = str(player_name or "").split()
        if parts:
            key_last = _norm_key(parts[0])
            if key_last in self._latest:
                rnk, pts = self._latest[key_last]
                return (rnk if rnk and math.isfinite(rnk) else None,
                        pts if pts and math.isfinite(pts) else None)
        return None, None


_RANKINGS_CACHE: dict[str, RankingsLookup] = {}


def get_rankings_lookup(tour: str = "atp") -> RankingsLookup:
    tour_key = tour.lower()
    if tour_key not in _RANKINGS_CACHE:
        _RANKINGS_CACHE[tour_key] = RankingsLookup(tour_key)
    return _RANKINGS_CACHE[tour_key]
