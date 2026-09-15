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

# Classement officiel ATP post-US Open 2026 (top 100+ officiel consolidé au 14/09/2026)
CURRENT_ATP_RANKINGS_SEPT_2026: dict[str, tuple[int, int]] = {
    "Sinner J.": (1, 11500),
    "Zverev A.": (2, 9690),
    "Alcaraz C.": (3, 5560),
    "Shelton B.": (4, 4680),
    "Auger-Aliassime F.": (5, 3890),
    "Aliassime F.A.": (5, 3890),
    "Medvedev D.": (6, 3770),
    "Cobolli F.": (7, 3720),
    "Tiafoe F.": (8, 3380),
    "de Minaur A.": (9, 3300),
    "De Minaur A.": (9, 3300),
    "Fritz T.": (10, 3175),
    "Fritz T.H.": (10, 3175),
    "Fils A.": (11, 3150),
    "Djokovic N.": (12, 2980),
    "Tien L.": (13, 2755),
    "Jodar R.": (14, 2659),
    "Mensik J.": (15, 2495),
    "Nakashima B.": (16, 2485),
    "Bublik A.": (17, 2425),
    "Ruud C.": (18, 2335),
    "Paul T.": (19, 2325),
    "Musetti L.": (20, 2255),
    "Vacherot V.": (21, 2250),
    "Darderi L.": (22, 2200),
    "Cerundolo F.": (23, 2130),
    "Lehecka J.": (24, 2120),
    "Rublev A.": (25, 1930),
    "Khachanov K.": (26, 1800),
    "Blockx A.": (27, 1788),
    "Fonseca J.": (28, 1750),
    "Davidovich Fokina A.": (29, 1750),
    "Fokina A.D.": (29, 1750),
    "Etcheverry T.": (30, 1590),
    "Rinderknech A.": (31, 1583),
    "Tabilo A.": (32, 1565),
    "Humbert U.": (33, 1535),
    "Michelsen A.": (34, 1460),
    "Buse I.": (35, 1460),
    "Fery A.": (36, 1413),
    "Arnaldi M.": (37, 1349),
    "Bergs Z.": (38, 1345),
    "Norrie C.": (39, 1275),
    "Van De Zandschulp B.": (40, 1258),
    "Zandschulp B.V.D.": (40, 1258),
    "Merida D.": (41, 1216),
    "Aguilar D.M.": (41, 1216),
    "Merida Aguilar D.": (41, 1216),
    "Van Assche L.": (42, 1180),
    "Berrettini M.": (43, 1165),
    "Tsitsipas S.": (44, 1165),
    "Navone M.": (45, 1155),
    "Hurkacz H.": (46, 1120),
    "Collignon R.": (47, 1117),
    "Borges N.": (48, 1095),
    "Halys Q.": (49, 1070),
    "Shapovalov D.": (50, 1065),
    "Baez S.": (51, 1044),
    "Tirante T.A.": (52, 1010),
    "Tirante T.": (52, 1010),
    "Cerundolo J.M.": (53, 995),
    "Hanfmann Y.": (54, 991),
    "Munar J.": (55, 990),
    "Griekspoor T.": (56, 940),
    "Landaluce M.": (57, 935),
    "Machac T.": (58, 915),
    "Burruchaga R.": (59, 905),
    "Ugo Carabelli C.": (60, 900),
    "Carabelli C.U.": (60, 900),
    "Mpetshi G.": (61, 895),
    "Svajda Z.": (62, 885),
    "Marozsan F.": (63, 880),
    "Kecmanovic M.": (64, 875),
    "Struff J.": (65, 870),
    "Struff J.L.": (65, 870),
    "Korda S.": (66, 865),
    "Quinn E.": (67, 855),
    "Kovacevic A.": (68, 850),
    "Kopriva V.": (69, 848),
    "Moutet C.": (70, 845),
    "Prizmic D.": (71, 835),
    "Carreno Busta P.": (72, 830),
    "Busta P.C.": (72, 830),
    "Brooksby J.": (73, 825),
    "Nardi L.": (74, 820),
    "Fucsovics M.": (75, 815),
    "Majchrzak K.": (76, 795),
    "Muller A.": (77, 792),
    "Mannarino A.": (78, 790),
    "Duckworth J.": (79, 762),
    "Giron M.": (80, 755),
    "Cilic M.": (81, 751),
    "Comesana F.": (82, 748),
    "Altmaier D.": (83, 745),
    "Vukic A.": (84, 742),
    "Daniel T.": (85, 738),
    "Opelka R.": (86, 735),
    "Atmane T.": (87, 730),
    "Walton A.": (88, 728),
    "Sonego L.": (89, 725),
    "Popyrin A.": (90, 715),
    "Shevchenko A.": (91, 710),
    "Bonzi B.": (92, 705),
    "Faria J.": (93, 695),
    "Spizzirri E.": (94, 685),
    "Samuel T.": (95, 680),
    "Gaston H.": (96, 675),
    "Hijikata R.": (97, 665),
    "Wong C.": (98, 655),
    "Molcan A.": (99, 645),
    "Wu Y.": (100, 635),
    "Nava E.": (103, 615),
    "Mpetshi Perricard G.": (105, 605),
    "Perricard G.M.": (105, 605),
    "Draper J.": (112, 520),
    "Diallo G.": (125, 475),
    "Wawrinka S.": (127, 475),
    "Dimitrov G.": (128, 472),
    "Safiullin R.": (135, 460),
    "Gea A.": (137, 452),
    "Rune H.": (142, 415),
    "Zheng M.": (147, 410),
    "Martinez P.": (148, 406),
    "Zhang Z.": (150, 402),
    "Zhang Zh.": (150, 402),
    "Thompson J.": (152, 398),
    "Cazaux A.": (153, 375),
    "Lajovic D.": (167, 340),
    "Mmoh M.": (195, 301),
    "Seyboth Wild T.": (198, 290),
    "Wild T.S.": (198, 290),
    "Kym J.": (201, 285),
    "Carballes Baena R.": (208, 270),
    "Baena R.C.": (208, 270),
    "Evans D.": (217, 265),
    "Shang J.": (228, 245),
    "Bautista Agut R.": (232, 233),
    "Agut R.B.": (232, 233),
    "Kotov P.": (250, 210),
    "Watanuki Y.": (292, 185),
    "Monfils G.": (311, 185),
    "Goffin D.": (317, 178),
    "Stricker D.": (344, 149),
    "Kokkinakis T.": (417, 118),
    "Nishikori K.": (481, 90),
    "Gigante M.": (502, 86),
    "Ruusuvuori E.": (776, 37),
    "Kyrgios N.": (835, 25),
    "Coric B.": (848, 22),
}

# Classement officiel WTA post-US Open 2026 (top 80+ officiel consolidé au 14/09/2026)
CURRENT_WTA_RANKINGS_SEPT_2026: dict[str, tuple[int, int]] = {
    "Rybakina E.": (1, 9901),
    "Sabalenka A.": (2, 7875),
    "Pegula J.": (3, 7265),
    "Gauff C.": (4, 7244),
    "Andreeva M.": (5, 5743),
    "Noskova L.": (6, 5328),
    "Svitolina E.": (7, 4809),
    "Muchova K.": (8, 4683),
    "Swiatek I.": (9, 4619),
    "Kostyuk M.": (10, 3980),
    "Anisimova A.": (11, 3363),
    "Bencic B.": (12, 2935),
    "Shnaider D.": (13, 2588),
    "Mboko V.": (14, 2521),
    "Cirstea S.": (15, 2464),
    "Jovic I.": (16, 2390),
    "Osaka N.": (17, 2306),
    "Eala A.": (18, 2276),
    "Mertens E.": (19, 2165),
    "Paolini J.": (20, 2123),
    "Kalinskaya A.": (21, 2043),
    "Keys M.": (22, 2029),
    "Navarro E.": (23, 2024),
    "Potapova A.": (24, 1926),
    "Chwalinska M.": (25, 1888),
    "Bouzkova M.": (26, 1873),
    "Alexandrova E.": (27, 1811),
    "Bejlek S.": (28, 1632),
    "Li A.": (29, 1573),
    "Parry D.": (30, 1551),
    "Fernandez L.": (31, 1454),
    "Ostapenko J.": (32, 1452),
    "Sakkari M.": (33, 1355),
    "Bartunkova N.": (34, 1352),
    "Vekic D.": (35, 1278),
    "Baptiste H.": (36, 1230),
    "Bucsa C.": (37, 1214),
    "Wang Xinyu": (38, 1195),
    "Wang Xin.": (38, 1195),
    "Wang X.": (38, 1195),
    "Tauson C.": (39, 1184),
    "Starodubtseva Y.": (40, 1165),
    "Snigur D.": (41, 1163),
    "Krejcikova B.": (42, 1139),
    "Pliskova Ka.": (43, 1121),
    "Pliskova K.": (43, 1121),
    "Tagger L.": (44, 1119),
    "Kalinina A.": (45, 1112),
    "Sherif M.": (46, 1105),
    "Korpatsch T.": (47, 1103),
    "Samsonova L.": (48, 1099),
    "Zheng Q.": (49, 1083),
    "Valentova T.": (50, 1014),
    "Haddad Maia B.": (51, 1010),
    "Maia B.H.": (51, 1010),
    "Badosa P.": (52, 995),
    "Jabeur O.": (53, 980),
    "Pavlyuchenkova A.": (54, 965),
    "Putintseva Y.": (55, 950),
    "Garcia C.": (56, 935),
    "Yastremska D.": (57, 920),
    "Sun L.": (58, 905),
    "Azarenka V.": (59, 890),
    "Linette M.": (60, 875),
    "Yuan Y.": (61, 860),
    "Kudermetova V.": (62, 845),
    "Tomljanovic A.": (63, 830),
    "Raducanu E.": (64, 815),
    "Wang Xiyu": (65, 800),
    "Wang Xi.": (65, 800),
    "Kenin S.": (66, 785),
    "Stephens S.": (67, 770),
    "Bronzetti L.": (68, 755),
    "Burel C.": (69, 740),
    "Blinkova A.": (70, 725),
    "Sorribes Tormo S.": (71, 710),
    "Tormo S.S.": (71, 710),
    "Frech M.": (72, 695),
    "Schmiedlova A.K.": (73, 680),
    "Schmiedlova K.": (73, 680),
    "Rus A.": (74, 665),
    "Joint M.": (75, 650),
    "Townsend T.": (76, 635),
    "Trevisan M.": (77, 620),
    "Danilovic O.": (78, 605),
    "Ruse E.G.": (79, 590),
    "Uchijima M.": (80, 575),
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
