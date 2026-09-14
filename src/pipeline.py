"""Chargement des saisons tennis-data.co.uk et construction chronologique des features
(sans fuite : chaque match n'utilise que l'information disponible avant lui).

Features par match (perspective joueur 1, ordre aléatoire déterministe) :
- p_elo : probabilité Elo pré-match (global + surface)
- d_selo : écart d'Elo de surface
- d_lrank / d_lpts : écarts de log(classement ATP) et log(points)
- d_form10 / d_form25 : écarts de taux de victoire récents
- d_activity : écarts de matchs joués sur 60 jours (rythme/rouille)
- h2h : bilan des confrontations directes (part de victoires de p1)
- best_of, surface (one-hot)
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from .elo import TennisElo, norm_surface
from .rankings import get_rankings_lookup
from .sackmann_stats import STAT_FEATURES, RollingSackmannStats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

NA = float("nan")
# Hors périmètre : circuits secondaires et compétitions par équipes, absentes de la
# source primaire. Les miroirs Sackmann, eux, les fournissent (niveau « D »).
OUT_OF_SCOPE_SERIES = frozenset({"Challenger", "ITF", "Davis Cup", "Fed Cup"})
YEAR_END_ACTIVE_DAYS = 365


def _finite(value: float) -> bool:
    return value == value and value > 0


def _log_diff(a: float, b: float) -> float:
    """log(a) - log(b), écart neutre si l'un des deux classements est inconnu.

    Mesuré sur le test 2024+ : l'écart neutre donne log-loss 0,6241 contre 0,6401 en
    laissant un NaN, que LightGBM n'a jamais rencontré à l'entraînement. Le classement
    reste exporté à None : neutraliser la variable ne justifie pas d'afficher un faux rang.
    """
    return math.log(a) - math.log(b) if _finite(a) and _finite(b) else 0.0


def _log1p_diff(a: float, b: float) -> float:
    return math.log1p(a) - math.log1p(b) if _finite(a) and _finite(b) else 0.0


def _resolve_market_odds(m) -> tuple[float | None, float | None]:
    """Extrait ou reconstruit la cote de marché moyenne / consensus du match."""
    aw = m.get("AvgW")
    al = m.get("AvgL")
    try:
        aw_f, al_f = float(aw), float(al)
        if math.isfinite(aw_f) and math.isfinite(al_f) and aw_f > 1.0 and al_f > 1.0:
            return aw_f, al_f
    except (TypeError, ValueError):
        pass

    bookmaker_pairs = [
        ("PSW", "PSL"),      # Pinnacle
        ("B365W", "B365L"),  # Bet365
        ("CBW", "CBL"),      # Centrebet
        ("GBW", "GBL"),      # Gamebookers
        ("IWW", "IWL"),      # Interwetten
        ("SBW", "SBL"),      # Sportingbet
        ("UBW", "UBL"),      # Unibet
        ("LBW", "LBL"),      # Ladbrokes
        ("MaxW", "MaxL"),
    ]
    w_odds = []
    l_odds = []
    for w_col, l_col in bookmaker_pairs:
        w_val = m.get(w_col)
        l_val = m.get(l_col)
        try:
            w_f, l_f = float(w_val), float(l_val)
            if math.isfinite(w_f) and math.isfinite(l_f) and w_f > 1.0 and l_f > 1.0:
                w_odds.append(w_f)
                l_odds.append(l_f)
        except (TypeError, ValueError):
            continue
    if w_odds and l_odds:
        return round(float(np.mean(w_odds)), 3), round(float(np.mean(l_odds)), 3)
    return None, None

FEATURES = [
    "p_elo", "d_selo", "d_lrank", "d_lpts", "d_form10", "d_form25",
    "d_surface_form10", "d_surface_form25", "d_season_winrate", "d_season_matches",
    "d_win_streak", "d_loss_streak", "d_best_of5_winrate", "d_slam_winrate",
    "d_activity", "h2h", "h2h_count", "h2h_surface", "h2h_slam",
    "h2h_recent", *STAT_FEATURES,
    "d_tourn_sets", "d_tourn_games", "d_recent_sets", "d_recent_games",
    "d_retirement_365", "d_recent_retirement",
    "d_rest_days", "rank_both_known", "pts_both_known",
    "level_grand_slam", "level_masters", "level_500", "level_250",
    "round_final", "round_semifinal", "round_quarterfinal",
    "best_of", "s_hard", "s_clay", "s_grass",
]


def _rest_days(player: str, last_dates: dict[str, deque], date) -> float:
    if not last_dates[player]:
        return 21.0
    return float(min(21, max(0, (date - last_dates[player][-1]).days)))


def _series_flags(series) -> dict[str, float]:
    text = str(series or "").lower()
    return {
        "level_grand_slam": 1.0 if "grand slam" in text else 0.0,
        "level_masters": 1.0 if "masters" in text or "1000" in text else 0.0,
        "level_500": 1.0 if "500" in text else 0.0,
        "level_250": 1.0 if "250" in text or text == "atp" or text == "wta" else 0.0,
    }


def _round_flags(round_name) -> dict[str, float]:
    text = str(round_name or "").lower().replace("-", " ").replace("_", " ")
    is_semi = "semi" in text
    is_quarter = "quarter" in text or "quarters" in text
    is_final = (
        text in {"the final", "final", "finals"}
        or text.endswith(" final")
    ) and not is_semi and not is_quarter
    return {
        "round_final": 1.0 if is_final else 0.0,
        "round_semifinal": 1.0 if is_semi else 0.0,
        "round_quarterfinal": 1.0 if is_quarter else 0.0,
    }


def _elo_rank_map(elo: TennisElo, last_dates: dict[str, deque], date) -> dict[str, float]:
    """Classement ordinal derive d'Elo pre-match, utilise seulement en feature de secours."""
    active = [
        (player, rating)
        for player, rating in elo.rating.items()
        if (
            elo.count[player] >= 15
            and last_dates[player]
            and (date - last_dates[player][-1]).days <= YEAR_END_ACTIVE_DAYS
        )
    ]
    active.sort(key=lambda item: -item[1])
    return {player: float(index + 1) for index, (player, _) in enumerate(active)}


def _rank_feature_value(
    player: str,
    raw_rank: float,
    last_rank: float,
    fallback_elo_ranks: dict[str, float],
) -> float:
    if _finite(raw_rank):
        return raw_rank
    if _finite(last_rank):
        return last_rank
    return _finite_or_nan(fallback_elo_ranks.get(player))


def _normalise_patch_tournament(value) -> str:
    name = str(value or "").strip().lower()
    aliases = {
        "wimbledon - london": "wimbledon",
        "wimbledon": "wimbledon",
        "u.s. open - new york": "us open",
        "us open - new york": "us open",
        "us open": "us open",
        "french open - paris": "french open",
        "roland garros": "french open",
        "australian open - melbourne": "australian open",
    }
    return aliases.get(name, name)


def _normalise_patch_round(value) -> str:
    r = str(value or "").strip().lower()
    mapping = {
        "round of 128": "1st round",
        "1st round": "1st round",
        "round of 64": "2nd round",
        "2nd round": "2nd round",
        "round of 32": "3rd round",
        "3rd round": "3rd round",
        "round of 16": "4th round",
        "4th round": "4th round",
        "the final": "the final",
        "final": "the final",
        "semifinals": "semifinals",
        "semifinal": "semifinals",
        "quarterfinals": "quarterfinals",
        "quarterfinal": "quarterfinals",
    }
    return mapping.get(r, r)


def _normalise_patch_player(value) -> str:
    name = str(value or "").strip()
    key = name.lower()
    aliases = {
        "aliassime f.a.": "auger-aliassime f.",
        "auger-aliassime f.": "auger-aliassime f.",
        "fritz t.h.": "fritz t.",
        "fritz t.": "fritz t.",
        "zandschulp b.v.d.": "van de zandschulp b.",
        "van de zandschulp b.": "van de zandschulp b.",
        "struff j.l.": "struff j.",
        "struff j.": "struff j.",
        "damm jr m.": "damm m.",
        "damm m.": "damm m.",
        "merida d.": "aguilar d.m.",
        "aguilar d.m.": "aguilar d.m.",
        "de minaur a.": "de minaur a.",
    }
    return aliases.get(key, key)


def _patch_key(row) -> tuple:
    return (
        pd.Timestamp(row["Date"]).year,
        _normalise_patch_tournament(row.get("Tournament")),
        _normalise_patch_round(row.get("Round")),
        _normalise_patch_player(row.get("Winner")),
        _normalise_patch_player(row.get("Loser")),
    )


def _patch_pair_key(row) -> tuple:
    players = tuple(sorted((
        _normalise_patch_player(row.get("Winner")),
        _normalise_patch_player(row.get("Loser")),
    )))
    return (
        pd.Timestamp(row["Date"]).year,
        _normalise_patch_tournament(row.get("Tournament")),
        _normalise_patch_round(row.get("Round")),
        players,
    )


def _patch_tournament_pair_key(row) -> tuple:
    players = tuple(sorted((
        _normalise_patch_player(row.get("Winner")),
        _normalise_patch_player(row.get("Loser")),
    )))
    return (
        pd.Timestamp(row["Date"]).year,
        _normalise_patch_tournament(row.get("Tournament")),
        players,
    )


def load_matches(tour: str = "atp") -> pd.DataFrame:
    # Le corpus consolidé est versionné : local et CI partent du même fichier. Les .xlsx
    # ne servent que de fichiers de travail entre deux exécutions de download_data.py.
    corpus = DATA / f"seasons_{tour}.parquet"
    if corpus.exists():
        allm = pd.read_parquet(corpus)
    else:
        frames = [pd.read_excel(f) for f in sorted(DATA.glob(f"{tour}_*.xlsx"))]
        if not frames:
            raise FileNotFoundError(f"Aucune saison {tour} : lancer scripts/download_data.py {tour}")
        allm = pd.concat(frames, ignore_index=True)
    allm["Date"] = pd.to_datetime(allm["Date"])
    allm = allm[allm.get("Comment", "Completed").fillna("Completed") != "Walkover"]
    # Filet de sécurité : les fichiers déjà en cache peuvent contenir du Challenger/ITF
    # importé par un ancien fallback. Le périmètre du projet reste le main tour ATP.
    if "Series" in allm.columns:
        allm = allm[~allm["Series"].astype(str).isin(OUT_OF_SCOPE_SERIES)]
    if "Tournament" in allm.columns:
        allm = allm[~allm["Tournament"].astype(str).str.contains("juniors", case=False, na=False)]
    allm = allm.dropna(subset=["Winner", "Loser"])
    allm = allm.sort_values(["Date", "ATP", "Round"]).reset_index(drop=True)
    allm.attrs["tour"] = tour
    # patch résultats récents (Sofascore) éventuel
    patch = DATA / ("results_patch.csv" if tour == "atp" else f"results_patch_{tour}.csv")
    if patch.exists():
        extra = pd.read_csv(patch, parse_dates=["Date"])
        if not extra.empty:
            if tour == "atp":
                last_covered = allm["Date"].max()
                extra = extra[
                    (extra["Date"].dt.year == last_covered.year)
                ]
                if not extra.empty:
                    extra_keys = {_patch_key(r) for _, r in extra.iterrows()}
                    extra_pair_keys = {_patch_pair_key(r) for _, r in extra.iterrows()}
                    extra_tourn_pair_keys = {_patch_tournament_pair_key(r) for _, r in extra.iterrows()}
                    keep = [
                        _patch_key(r) not in extra_keys
                        and _patch_pair_key(r) not in extra_pair_keys
                        and _patch_tournament_pair_key(r) not in extra_tourn_pair_keys
                        for _, r in allm.iterrows()
                    ]
                    removed = len(allm) - sum(keep)
                    allm = pd.concat([allm.loc[keep], extra], ignore_index=True)
                    allm = allm.sort_values("Date").reset_index(drop=True)
                    print(f"  patch résultats ATP : {len(extra)} ligne(s), {removed} remplacée(s)")
            else:
                extra_keys = {_patch_key(r) for _, r in extra.iterrows()}
                extra_pair_keys = {_patch_pair_key(r) for _, r in extra.iterrows()}
                keep = [
                    _patch_key(r) not in extra_keys
                    and _patch_pair_key(r) not in extra_pair_keys
                    for _, r in allm.iterrows()
                ]
                removed = len(allm) - sum(keep)
                allm = pd.concat([allm.loc[keep], extra], ignore_index=True)
                allm = allm.sort_values("Date").reset_index(drop=True)
                print(f"  patch résultats : {len(extra)} ligne(s), {removed} remplacée(s)")
    return allm


def _p1_is_winner(date, w, l) -> bool:
    """Ordre p1/p2 aléatoire mais déterministe (évite le biais 'p1 = vainqueur')."""
    h = hashlib.md5(f"{date}|{w}|{l}".encode()).digest()
    return h[0] % 2 == 0


def _pair(p1: str, p2: str) -> tuple[str, str]:
    return (p1, p2) if p1 < p2 else (p2, p1)


def _share_for_player(counts: list[int], pair: tuple[str, str], player: str) -> float:
    total = counts[0] + counts[1]
    if total == 0:
        return 0.0
    wins = counts[0] if player == pair[0] else counts[1]
    return wins / total - 0.5


def _recent_share_for_player(outcomes: deque, pair: tuple[str, str], player: str) -> float:
    if not outcomes:
        return 0.0
    side = 0 if player == pair[0] else 1
    return sum(1 for outcome in outcomes if outcome == side) / len(outcomes) - 0.5


def _win_rate(outcomes, default: float = 0.5) -> float:
    values = list(outcomes)
    return sum(values) / len(values) if values else default


def _count_since(dates, date, days: int) -> int:
    return sum(1 for d in dates if 0 <= (date - d).days <= days)


def upcoming_features(
    state: dict,
    p1: str,
    p2: str,
    surface: str,
    best_of: int,
    date,
    series=None,
    round_name=None,
    tournament_id=None,
    tournament=None,
    rank1=None,
    rank2=None,
) -> dict:
    """Features d'un match non joué, à partir de l'état laissé par `build_features`.

    Mêmes expressions que la boucle d'entraînement, avec l'état arrêté au dernier match
    connu : rien de postérieur n'entre ici, la règle de non-fuite reste vérifiée.
    """
    elo = state["elo"]
    recent, last_dates = state["recent"], state["last_dates"]
    surface_recent = state.get("surface_recent", {})
    season_recent = state.get("season_recent", {})
    best_of5_recent = state.get("best_of5_recent", {})
    slam_recent = state.get("slam_recent", {})
    win_streak = state.get("win_streak", {})
    loss_streak = state.get("loss_streak", {})
    retirements = state.get("retirements", {})
    recent_load = state.get("recent_load", {})
    tournament_load = state.get("tournament_load", {})
    rank, pts = state["rank"], state["pts"]
    surface = norm_surface(surface)
    fallback_elo_ranks = _elo_rank_map(elo, last_dates, date)

    def form(player, n):
        window = list(recent[player])[-n:]
        return sum(window) / len(window) if window else 0.5

    def activity(player):
        return sum(1 for d in last_dates[player] if (date - d).days <= 60)

    def load_since(player, days):
        sets = games = 0
        for d, s, g in recent_load.get(player, []):
            if 0 <= (date - d).days <= days:
                sets += s
                games += g
        return sets, games

    key = _pair(p1, p2)
    counts = state["h2h"][key]
    total = counts[0] + counts[1]
    sackmann = state["sackmann"]
    event_key = (date.year, str(tournament_id or ""), str(tournament or ""))
    p1_t_sets, p1_t_games = tournament_load.get(event_key, {}).get(p1, [0, 0])
    p2_t_sets, p2_t_games = tournament_load.get(event_key, {}).get(p2, [0, 0])
    p1_r_sets, p1_r_games = load_since(p1, 14)
    p2_r_sets, p2_r_games = load_since(p2, 14)
    r1 = _rank_feature_value(p1, _finite_or_nan(rank1), _finite_or_nan(rank.get(p1)), fallback_elo_ranks)
    r2 = _rank_feature_value(p2, _finite_or_nan(rank2), _finite_or_nan(rank.get(p2)), fallback_elo_ranks)
    q1 = _finite_or_nan(pts.get(p1))
    q2 = _finite_or_nan(pts.get(p2))
    return {
        "p_elo": elo.prob(p1, p2, surface),
        "d_selo": (elo.surf[surface][p1] - elo.surf[surface][p2]) / 100.0,
        "d_lrank": _log_diff(r2, r1),
        "d_lpts": _log1p_diff(q1, q2),
        "d_form10": form(p1, 10) - form(p2, 10),
        "d_form25": form(p1, 25) - form(p2, 25),
        "d_surface_form10": _win_rate(list(surface_recent.get((p1, surface), []))[-10:]) - _win_rate(list(surface_recent.get((p2, surface), []))[-10:]),
        "d_surface_form25": _win_rate(list(surface_recent.get((p1, surface), []))[-25:]) - _win_rate(list(surface_recent.get((p2, surface), []))[-25:]),
        "d_season_winrate": _win_rate(season_recent.get((p1, date.year), [])) - _win_rate(season_recent.get((p2, date.year), [])),
        "d_season_matches": (len(season_recent.get((p1, date.year), [])) - len(season_recent.get((p2, date.year), []))) / 30.0,
        "d_win_streak": (win_streak.get(p1, 0) - win_streak.get(p2, 0)) / 10.0,
        "d_loss_streak": (loss_streak.get(p2, 0) - loss_streak.get(p1, 0)) / 10.0,
        "d_best_of5_winrate": _win_rate(best_of5_recent.get(p1, [])) - _win_rate(best_of5_recent.get(p2, [])),
        "d_slam_winrate": _win_rate(slam_recent.get(p1, [])) - _win_rate(slam_recent.get(p2, [])),
        "d_activity": (activity(p1) - activity(p2)) / 10.0,
        "h2h": _share_for_player(counts, key, p1),
        "h2h_count": math.log1p(total),
        "h2h_surface": _share_for_player(state["h2h_surface"][(key[0], key[1], surface)], key, p1),
        "h2h_slam": _share_for_player(state["h2h_slam"][key], key, p1),
        "h2h_recent": _recent_share_for_player(state["h2h_recent"][key], key, p1),
        **sackmann.diff_features(p1, p2),
        "d_tourn_sets": (p1_t_sets - p2_t_sets) / 10.0,
        "d_tourn_games": (p1_t_games - p2_t_games) / 100.0,
        "d_recent_sets": (p1_r_sets - p2_r_sets) / 20.0,
        "d_recent_games": (p1_r_games - p2_r_games) / 200.0,
        "d_retirement_365": (_count_since(retirements.get(p1, []), date, 365) - _count_since(retirements.get(p2, []), date, 365)) / 3.0,
        "d_recent_retirement": (_count_since(retirements.get(p1, []), date, 45) - _count_since(retirements.get(p2, []), date, 45)),
        "d_rest_days": (_rest_days(p1, last_dates, date) - _rest_days(p2, last_dates, date)) / 21.0,
        "rank_both_known": 1.0 if _finite(r1) and _finite(r2) else 0.0,
        "pts_both_known": 1.0 if _finite(q1) and _finite(q2) else 0.0,
        **_series_flags(series),
        **_round_flags(round_name),
        "best_of": 1.0 if float(best_of or 3) >= 5 else 0.0,
        "s_hard": 1.0 if surface == "Hard" else 0.0,
        "s_clay": 1.0 if surface == "Clay" else 0.0,
        "s_grass": 1.0 if surface == "Grass" else 0.0,
    }


def _finite_or_nan(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return NA
    return number if _finite(number) else NA


def _score_workload(m) -> tuple[int, int]:
    """Approxime la charge d'un match par nombre de sets et jeux disputés."""
    sets = 0
    games = 0
    for i in range(1, 6):
        try:
            w_games = int(float(m.get(f"W{i}")))
            l_games = int(float(m.get(f"L{i}")))
        except (TypeError, ValueError):
            continue
        sets += 1
        games += max(0, w_games) + max(0, l_games)
    if sets == 0:
        try:
            sets = int(float(m.get("Wsets", 0) or 0)) + int(float(m.get("Lsets", 0) or 0))
        except (TypeError, ValueError):
            sets = 0
    return sets, games


def build_features(allm: pd.DataFrame):
    """Parcourt les matchs en ordre chronologique et produit X, y, meta (+ état final)."""
    elo = TennisElo()
    all_players = set(allm["Winner"].astype(str).str.strip()) | set(allm["Loser"].astype(str).str.strip())
    tour = str(allm.attrs.get("tour", "atp")).lower()
    rankings_lookup = get_rankings_lookup(tour)
    sackmann = RollingSackmannStats.from_cache(existing_players=all_players, tour=tour)
    recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=25))
    surface_recent: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=25))
    season_recent: dict[tuple[str, int], deque] = defaultdict(lambda: deque(maxlen=80))
    best_of5_recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=25))
    slam_recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=40))
    win_streak: dict[str, int] = defaultdict(int)
    loss_streak: dict[str, int] = defaultdict(int)
    retirements: dict[str, deque] = defaultdict(lambda: deque(maxlen=8))
    last_dates: dict[str, deque] = defaultdict(lambda: deque(maxlen=40))
    h2h: dict[tuple, list] = defaultdict(lambda: [0, 0])
    h2h_surface: dict[tuple, list] = defaultdict(lambda: [0, 0])
    h2h_slam: dict[tuple, list] = defaultdict(lambda: [0, 0])
    h2h_recent: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=5))
    tournament_load: dict[tuple, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    recent_load: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
    rank: dict[str, float] = {}
    pts: dict[str, float] = {}
    # Historique Elo « en direct » : trajectoire (date, elo) par joueur, top de fin d'année,
    # finales de Grands Chelems (palmarès).
    history: dict[str, list] = defaultdict(list)
    year_end: dict[int, list] = {}
    slam_finals: list[dict] = []
    cur_year = None

    def _snapshot_top(snapshot_date, n=5):
        act = [
            (p, r)
            for p, r in elo.rating.items()
            if (
                elo.count[p] >= 15
                and last_dates[p]
                and (snapshot_date - last_dates[p][-1]).days <= YEAR_END_ACTIVE_DAYS
            )
        ]
        act.sort(key=lambda t: -t[1])
        return [{"player": p, "elo": round(r, 1)} for p, r in act[:n]]

    rows, ys, metas = [], [], []
    date = None
    for _, m in allm.iterrows():
        w, l = str(m["Winner"]).strip(), str(m["Loser"]).strip()
        surface = norm_surface(m.get("Surface"))
        date = m["Date"]

        def _num(v, default):
            try:
                f = float(v)
                return f if f == f and f > 0 else default  # NaN / 0 -> défaut
            except (TypeError, ValueError):  # 'NR' (non classé) dans les vieilles saisons
                return default

        # Classement et points absents : rester à NaN plutôt que d'inventer un 250 commun.
        # Un défaut constant annule d_lrank/d_lpts sur toute la période concernée et crée
        # un décalage entraînement/test invisible ; LightGBM gère nativement les NaN.
        wr_raw = _num(m.get("WRank"), NA)
        lr_raw = _num(m.get("LRank"), NA)
        wp_raw = _num(m.get("WPts"), NA)
        lp_raw = _num(m.get("LPts"), NA)

        # Si le classement officiel est manquant, consulter la table hebdomadaire Sackmann
        if not _finite(wr_raw):
            r_val, p_val = rankings_lookup.get_rank_points(w, date)
            if r_val is not None:
                wr_raw = float(r_val)
            if not _finite(wp_raw) and p_val is not None:
                wp_raw = float(p_val)
        if not _finite(lr_raw):
            r_val, p_val = rankings_lookup.get_rank_points(l, date)
            if r_val is not None:
                lr_raw = float(r_val)
            if not _finite(lp_raw) and p_val is not None:
                lp_raw = float(p_val)

        needs_elo_rank_fallback = (
            not _finite(wr_raw)
            and not _finite(_finite_or_nan(rank.get(w)))
        ) or (
            not _finite(lr_raw)
            and not _finite(_finite_or_nan(rank.get(l)))
        )
        fallback_elo_ranks = _elo_rank_map(elo, last_dates, date) if needs_elo_rank_fallback else {}
        wr = _rank_feature_value(w, wr_raw, _finite_or_nan(rank.get(w)), fallback_elo_ranks)
        lr = _rank_feature_value(l, lr_raw, _finite_or_nan(rank.get(l)), fallback_elo_ranks)
        wp = wp_raw if _finite(wp_raw) else _finite_or_nan(pts.get(w))
        lp = lp_raw if _finite(lp_raw) else _finite_or_nan(pts.get(l))

        p1_win = _p1_is_winner(date, w, l)
        p1, p2 = (w, l) if p1_win else (l, w)
        r1, r2 = (wr, lr) if p1_win else (lr, wr)
        q1, q2 = (wp, lp) if p1_win else (lp, wp)

        def form(p, n):
            d = list(recent[p])[-n:]
            return sum(d) / len(d) if d else 0.5

        def activity(p):
            return sum(1 for d in last_dates[p] if (date - d).days <= 60)

        def load_since(p, days):
            sets = games = 0
            for d, s, g in recent_load[p]:
                if (date - d).days <= days:
                    sets += s
                    games += g
            return sets, games

        k = _pair(p1, p2)
        hh = h2h[k]
        tot = hh[0] + hh[1]
        h2h_p1 = _share_for_player(hh, k, p1)
        h2h_surface_p1 = _share_for_player(h2h_surface[(k[0], k[1], surface)], k, p1)
        h2h_slam_p1 = _share_for_player(h2h_slam[k], k, p1)
        h2h_recent_p1 = _recent_share_for_player(h2h_recent[k], k, p1)
        event_key = (date.year, str(m.get("ATP")), str(m.get("Tournament")))
        p1_t_sets, p1_t_games = tournament_load[event_key][p1]
        p2_t_sets, p2_t_games = tournament_load[event_key][p2]
        p1_r_sets, p1_r_games = load_since(p1, 14)
        p2_r_sets, p2_r_games = load_since(p2, 14)
        sackmann.advance_before(date)

        feats = {
            "p_elo": elo.prob(p1, p2, surface),
            "d_selo": (elo.surf[surface][p1] - elo.surf[surface][p2]) / 100.0,
            "d_lrank": _log_diff(r2, r1),  # >0 si p1 mieux classé
            "d_lpts": _log1p_diff(q1, q2),
            "d_form10": form(p1, 10) - form(p2, 10),
            "d_form25": form(p1, 25) - form(p2, 25),
            "d_surface_form10": _win_rate(list(surface_recent[(p1, surface)])[-10:]) - _win_rate(list(surface_recent[(p2, surface)])[-10:]),
            "d_surface_form25": _win_rate(surface_recent[(p1, surface)]) - _win_rate(surface_recent[(p2, surface)]),
            "d_season_winrate": _win_rate(season_recent[(p1, date.year)]) - _win_rate(season_recent[(p2, date.year)]),
            "d_season_matches": (len(season_recent[(p1, date.year)]) - len(season_recent[(p2, date.year)])) / 30.0,
            "d_win_streak": (win_streak[p1] - win_streak[p2]) / 10.0,
            "d_loss_streak": (loss_streak[p2] - loss_streak[p1]) / 10.0,
            "d_best_of5_winrate": _win_rate(best_of5_recent[p1]) - _win_rate(best_of5_recent[p2]),
            "d_slam_winrate": _win_rate(slam_recent[p1]) - _win_rate(slam_recent[p2]),
            "d_activity": (activity(p1) - activity(p2)) / 10.0,
            "h2h": h2h_p1,
            "h2h_count": math.log1p(tot),
            "h2h_surface": h2h_surface_p1,
            "h2h_slam": h2h_slam_p1,
            "h2h_recent": h2h_recent_p1,
            **sackmann.diff_features(p1, p2),
            "d_tourn_sets": (p1_t_sets - p2_t_sets) / 10.0,
            "d_tourn_games": (p1_t_games - p2_t_games) / 100.0,
            "d_recent_sets": (p1_r_sets - p2_r_sets) / 20.0,
            "d_recent_games": (p1_r_games - p2_r_games) / 200.0,
            "d_retirement_365": (_count_since(retirements[p1], date, 365) - _count_since(retirements[p2], date, 365)) / 3.0,
            "d_recent_retirement": (_count_since(retirements[p1], date, 45) - _count_since(retirements[p2], date, 45)),
            "d_rest_days": (_rest_days(p1, last_dates, date) - _rest_days(p2, last_dates, date)) / 21.0,
            "rank_both_known": 1.0 if _finite(r1) and _finite(r2) else 0.0,
            "pts_both_known": 1.0 if _finite(q1) and _finite(q2) else 0.0,
            **_series_flags(m.get("Series")),
            **_round_flags(m.get("Round")),
            "best_of": 1.0 if float(m.get("Best of", 3) or 3) >= 5 else 0.0,
            "s_hard": 1.0 if surface == "Hard" else 0.0,
            "s_clay": 1.0 if surface == "Clay" else 0.0,
            "s_grass": 1.0 if surface == "Grass" else 0.0,
        }
        rows.append(feats)
        ys.append(1 if p1_win else 0)
        m_avgw, m_avgl = _resolve_market_odds(m)
        metas.append({
            "date": date, "tournament": m.get("Tournament"), "series": m.get("Series"),
            "surface": surface, "round": m.get("Round"), "best_of": m.get("Best of"),
            "winner": w, "loser": l, "wrank": wr_raw, "lrank": lr_raw,
            "wrank_feature": wr, "lrank_feature": lr,
            "p1": p1, "p2": p2,
            "avgw": m_avgw, "avgl": m_avgl,
            "score": _score_str(m),
            "comment": m.get("Comment"),
            "score_status": m.get("ScoreStatus"),
        })

        # mises à jour post-match
        if cur_year is None:
            cur_year = date.year
        elif date.year > cur_year:
            year_end[cur_year] = _snapshot_top(date - pd.Timedelta(days=1))
            cur_year = date.year
        elo.update(w, l, surface, m.get("Series"))
        history[w].append((date, round(elo.rating[w], 1)))
        history[l].append((date, round(elo.rating[l], 1)))
        if str(m.get("Series")) == "Grand Slam" and str(m.get("Round")) == "The Final":
            slam_finals.append({"year": date.year, "tournament": str(m.get("Tournament")),
                                "winner": w, "loser": l, "score": _score_str(m)})
        recent[w].append(1)
        recent[l].append(0)
        surface_recent[(w, surface)].append(1)
        surface_recent[(l, surface)].append(0)
        season_recent[(w, date.year)].append(1)
        season_recent[(l, date.year)].append(0)
        if float(m.get("Best of", 3) or 3) >= 5:
            best_of5_recent[w].append(1)
            best_of5_recent[l].append(0)
        if str(m.get("Series")) == "Grand Slam":
            slam_recent[w].append(1)
            slam_recent[l].append(0)
        win_streak[w] += 1
        loss_streak[w] = 0
        loss_streak[l] += 1
        win_streak[l] = 0
        if str(m.get("Comment") or "") == "Retired":
            retirements[l].append(date)
        last_dates[w].append(date)
        last_dates[l].append(date)
        winner_side = 0 if w == k[0] else 1
        hh[winner_side] += 1
        h2h_surface[(k[0], k[1], surface)][winner_side] += 1
        if str(m.get("Series")) == "Grand Slam":
            h2h_slam[k][winner_side] += 1
        h2h_recent[k].append(winner_side)
        sets_played, games_played = _score_workload(m)
        tournament_load[event_key][w][0] += sets_played
        tournament_load[event_key][w][1] += games_played
        tournament_load[event_key][l][0] += sets_played
        tournament_load[event_key][l][1] += games_played
        recent_load[w].append((date, sets_played, games_played))
        recent_load[l].append((date, sets_played, games_played))
        if _finite(wr_raw):
            rank[w] = wr_raw
        if _finite(lr_raw):
            rank[l] = lr_raw
        if _finite(wp_raw):
            pts[w] = wp_raw
        if _finite(lp_raw):
            pts[l] = lp_raw

    if cur_year is not None:
        year_end[cur_year] = _snapshot_top(date)  # année en cours = état actuel

    # Injection du classement contemporain officiel pour les joueurs actifs
    for p in set(list(rank.keys()) + list(elo.rating.keys())):
        lat_r, lat_p = rankings_lookup.get_latest_rank_points(p)
        if lat_r is not None:
            rank[p] = lat_r
        if lat_p is not None:
            pts[p] = lat_p

    X = pd.DataFrame(rows)[FEATURES]
    y = np.array(ys)
    state = {"elo": elo, "recent": recent, "surface_recent": surface_recent,
             "season_recent": season_recent, "best_of5_recent": best_of5_recent,
             "slam_recent": slam_recent, "win_streak": win_streak, "loss_streak": loss_streak,
             "retirements": retirements, "recent_load": recent_load, "tournament_load": tournament_load,
             "rank": rank, "pts": pts, "last_dates": last_dates,
             "history": history, "year_end": year_end, "slam_finals": slam_finals,
             # Compteurs de confrontations directes : rejoués tels quels pour un match
             # à venir, sinon ses features h2h seraient neutres alors qu'on les connaît.
             "h2h": h2h, "h2h_surface": h2h_surface, "h2h_slam": h2h_slam,
             "h2h_recent": h2h_recent, "sackmann": sackmann, "last_date": date}
    return X, y, metas, state


def _score_str(m) -> str | None:
    parts = []
    for i in range(1, 6):
        w, l = m.get(f"W{i}"), m.get(f"L{i}")
        try:
            parts.append(f"{int(float(w))}-{int(float(l))}")
        except (TypeError, ValueError):  # NaN / ' ' dans les vieilles saisons
            break
    return " ".join(parts) or None


def market_prob(avgw, avgl) -> float | None:
    """Probabilité implicite du marché pour le vainqueur (marge retirée)."""
    try:
        aw, al = float(avgw), float(avgl)
        if not (math.isfinite(aw) and math.isfinite(al)) or aw <= 1 or al <= 1:
            return None
        iw, il = 1.0 / aw, 1.0 / al
        return iw / (iw + il)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
