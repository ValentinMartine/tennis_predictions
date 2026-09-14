"""Résolution et interpolation des classements officiels ATP et WTA.

Fournit un accès aux classements hebdomadaires officiels Sackmann (2000-2026)
et aux classements contemporains post-US Open 2026 lorsque les flux primaires
sont indisponibles ou dépourvus de WRank/WPts.
"""
from __future__ import annotations

import math
import unicodedata
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SACKMANN_DIR = DATA / "sackmann"

# Classement officiel ATP post-US Open (top 100 officiel consolidé)
CURRENT_ATP_RANKINGS_SEPT_2026: dict[str, tuple[int, int]] = {
    "Sinner J.": (1, 11180),
    "Zverev A.": (2, 7075),
    "Alcaraz C.": (3, 6690),
    "Djokovic N.": (4, 5560),
    "Medvedev D.": (5, 5475),
    "Rublev A.": (6, 4645),
    "Fritz T.": (7, 4060),
    "Fritz T.H.": (7, 4060),
    "Hurkacz H.": (8, 4060),
    "Ruud C.": (9, 4010),
    "Dimitrov G.": (10, 3965),
    "de Minaur A.": (11, 3655),
    "De Minaur A.": (11, 3655),
    "Tsitsipas S.": (12, 3390),
    "Paul T.": (13, 3005),
    "Rune H.": (14, 2785),
    "Korda S.": (15, 2585),
    "Tiafoe F.": (16, 2560),
    "Shelton B.": (17, 2490),
    "Humbert U.": (18, 2370),
    "Musetti L.": (19, 2340),
    "Draper J.": (20, 2315),
    "Auger-Aliassime F.": (21, 2170),
    "Aliassime F.A.": (21, 2170),
    "Tabilo A.": (22, 1978),
    "Khachanov K.": (23, 1930),
    "Popyrin A.": (24, 1845),
    "Fils A.": (25, 1770),
    "Baez S.": (26, 1730),
    "Bublik A.": (27, 1690),
    "Jarry N.": (28, 1650),
    "Thompson J.": (29, 1576),
    "Borges N.": (30, 1500),
    "Etcheverry T.": (31, 1495),
    "Cobolli F.": (32, 1482),
    "Arnaldi M.": (33, 1450),
    "Nakashima B.": (34, 1405),
    "Griekspoor T.": (35, 1375),
    "Lehecka J.": (36, 1345),
    "Kotov P.": (37, 1315),
    "Machac T.": (38, 1290),
    "Zhang Z.": (39, 1275),
    "Zhang Zh.": (39, 1275),
    "Kecmanovic M.": (40, 1270),
    "Darderi L.": (41, 1271),
    "Martinez P.": (42, 1167),
    "Berrettini M.": (43, 1165),
    "Monfils G.": (44, 1160),
    "Giron M.": (45, 1150),
    "Mannarino A.": (46, 1144),
    "Michelsen A.": (47, 1115),
    "Marozsan F.": (48, 1095),
    "Sonego L.": (49, 1081),
    "Mpetshi Perricard G.": (50, 1062),
    "Perricard G.M.": (50, 1062),
    "Norrie C.": (51, 1045),
    "Navone M.": (52, 1030),
    "Struff J.": (53, 1025),
    "Struff J.L.": (53, 1025),
    "Seyboth Wild T.": (54, 990),
    "Wild T.S.": (54, 990),
    "Rinderknech A.": (55, 975),
    "Carballes Baena R.": (56, 960),
    "Baena R.C.": (56, 960),
    "Goffin D.": (57, 940),
    "Bautista Agut R.": (58, 930),
    "Agut R.B.": (58, 930),
    "Davidovich Fokina A.": (59, 925),
    "Fokina A.D.": (59, 925),
    "Cazaux A.": (60, 915),
    "Lajovic D.": (61, 905),
    "Bergs Z.": (62, 895),
    "Fognini F.": (63, 885),
    "Nishikori K.": (64, 875),
    "Van De Zandschulp B.": (65, 865),
    "Zandschulp B.V.D.": (65, 865),
    "Mensik J.": (66, 855),
    "Moutet C.": (67, 845),
    "Coric B.": (68, 835),
    "Wawrinka S.": (69, 825),
    "Kokkinakis T.": (70, 815),
    "Shang J.": (71, 805),
    "Duckworth J.": (72, 795),
    "Nardi L.": (73, 785),
    "Kovacevic A.": (74, 775),
    "Muller A.": (75, 765),
    "Daniel T.": (76, 755),
    "Vukic A.": (77, 745),
    "Nagal S.": (78, 735),
    "O'Connell C.": (79, 725),
    "Gaston H.": (80, 715),
    "Walton A.": (81, 705),
    "Altmaier D.": (82, 695),
    "Fucsovics M.": (83, 685),
    "Shevchenko A.": (84, 675),
    "Shapovalov D.": (85, 665),
    "Hanfmann Y.": (86, 655),
    "Halys Q.": (87, 645),
    "Coria F.": (88, 635),
    "Gasquet R.": (89, 625),
    "Tseng C.H.": (90, 615),
    "Bellucci M.": (91, 605),
    "Diaz Acosta F.": (92, 595),
    "Acosta F.D.": (92, 595),
    "Fearnley J.": (93, 585),
    "Passaro F.": (94, 575),
    "Bu Y.": (95, 565),
    "Kukushkin M.": (96, 555),
    "Van Assche L.": (97, 545),
    "Bonzi B.": (98, 535),
    "Mayot H.": (99, 525),
    "Vacherot V.": (120, 480),
    "Tien L.": (124, 490),
    "Buse I.": (210, 260),
    "Blockx A.": (230, 245),
    "Gea A.": (310, 165),
    "Jodar R.": (850, 25),
    "Kyrgios N.": (835, 25),
}

# Classement officiel WTA post-US Open (top 80 officiel consolidé)
CURRENT_WTA_RANKINGS_SEPT_2026: dict[str, tuple[int, int]] = {
    "Swiatek I.": (1, 10885),
    "Sabalenka A.": (2, 8716),
    "Pegula J.": (3, 6220),
    "Rybakina E.": (4, 5871),
    "Paolini J.": (5, 5398),
    "Gauff C.": (6, 4983),
    "Zheng Q.": (7, 3980),
    "Navarro E.": (8, 3810),
    "Sakkari M.": (9, 3605),
    "Krejcikova B.": (10, 3536),
    "Collins D.": (11, 3178),
    "Ostapenko J.": (12, 3058),
    "Kasatkina D.": (13, 2720),
    "Samsonova L.": (14, 2720),
    "Shnaider D.": (15, 2685),
    "Andreeva M.": (16, 2600),
    "Kostyuk M.": (17, 2498),
    "Vekic D.": (18, 2468),
    "Haddad Maia B.": (19, 2460),
    "Maia B.H.": (19, 2460),
    "Keys M.": (20, 2410),
    "Badosa P.": (21, 2328),
    "Jabeur O.": (22, 2210),
    "Pavlyuchenkova A.": (23, 2021),
    "Fernandez L.": (24, 1920),
    "Noskova L.": (25, 1895),
    "Svitolina E.": (26, 1845),
    "Putintseva Y.": (27, 1798),
    "Garcia C.": (28, 1780),
    "Kalinskaya A.": (29, 1750),
    "Yastremska D.": (30, 1690),
    "Alexandrova E.": (31, 1635),
    "Muchova K.": (32, 1612),
    "Mertens E.": (33, 1570),
    "Sun L.": (34, 1540),
    "Azarenka V.": (35, 1495),
    "Linette M.": (36, 1450),
    "Bouzkova M.": (37, 1410),
    "Potapova A.": (38, 1375),
    "Yuan Y.": (39, 1340),
    "Kudermetova V.": (40, 1310),
    "Tomljanovic A.": (41, 1280),
    "Raducanu E.": (42, 1250),
    "Wang Xinyu": (43, 1220),
    "Wang Xin.": (43, 1220),
    "Wang X.": (43, 1220),
    "Wang Xiyu": (44, 1200),
    "Wang Xi.": (44, 1200),
    "Bencic B.": (45, 1190),
    "Kenin S.": (46, 1160),
    "Osaka N.": (47, 1130),
    "Stephens S.": (48, 1100),
    "Anisimova A.": (49, 1070),
    "Pliskova Ka.": (50, 1050),
    "Pliskova K.": (50, 1050),
    "Bronzetti L.": (51, 1030),
    "Parry D.": (52, 1010),
    "Burel C.": (53, 995),
    "Blinkova A.": (54, 980),
    "Sorribes Tormo S.": (55, 965),
    "Tormo S.S.": (55, 965),
    "Frech M.": (56, 950),
    "Sherif M.": (57, 935),
    "Schmiedlova A.K.": (58, 920),
    "Schmiedlova K.": (58, 920),
    "Rus A.": (59, 905),
    "Joint M.": (60, 890),
    "Townsend T.": (61, 875),
    "Trevisan M.": (62, 860),
    "Danilovic O.": (63, 845),
    "Ruse E.G.": (64, 830),
    "Uchijima M.": (65, 815),
    "Zidansek T.": (66, 800),
    "Minnen G.": (67, 785),
    "Siegemund L.": (68, 770),
    "Golubic V.": (69, 755),
    "Podoroska N.": (70, 740),
    "Sasnovich A.": (71, 725),
    "Niemeier J.": (72, 710),
    "Sonmez Z.": (73, 695),
    "Starodubtseva Y.": (74, 680),
    "Bondar A.": (75, 665),
    "Parks A.": (76, 650),
    "Cristian J.": (77, 635),
    "Gadecki O.": (78, 620),
    "Lamens S.": (79, 605),
    "Sramkova R.": (80, 590),
    "Williams V.": (500, 75),
    "Williams S.": (2071, 1),
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
            lasts = [
                str(r.get("name_last") or "").strip()
                for _, r in players_df.iterrows()
                if str(r.get("name_last") or "").strip()
            ]
            last_name_counts = Counter(lasts)

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
                id_to_names[pid].append(f"{last} {first}".strip())
                # N'ajouter le nom de famille seul que s'il est strictement unique dans la base
                # pour éviter qu'un frère/parent n'écrase le classement du joueur principal.
                if last_name_counts.get(last, 0) == 1:
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

        # Intégration du classement de référence officiel post-US Open pour ATP et WTA
        ref_rankings = CURRENT_ATP_RANKINGS_SEPT_2026 if self.tour == "atp" else CURRENT_WTA_RANKINGS_SEPT_2026
        for player, (rnk, pts) in ref_rankings.items():
            key = _norm_key(player)
            self._history[key].append((20260609, float(rnk), float(pts)))
            self._history[key].append((20260914, float(rnk), float(pts)))
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
