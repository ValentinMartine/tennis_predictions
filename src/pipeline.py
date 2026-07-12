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

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

FEATURES = [
    "p_elo", "d_selo", "d_lrank", "d_lpts", "d_form10", "d_form25",
    "d_activity", "h2h", "h2h_count", "h2h_surface", "h2h_slam",
    "h2h_recent", "best_of", "s_hard", "s_clay", "s_grass",
]


def load_matches() -> pd.DataFrame:
    frames = []
    for f in sorted(DATA.glob("atp_*.xlsx")):
        df = pd.read_excel(f)
        frames.append(df)
    allm = pd.concat(frames, ignore_index=True)
    allm["Date"] = pd.to_datetime(allm["Date"])
    allm = allm[allm.get("Comment", "Completed").fillna("Completed") != "Walkover"]
    allm = allm.dropna(subset=["Winner", "Loser"])
    allm = allm.sort_values(["Date", "ATP", "Round"]).reset_index(drop=True)
    # patch résultats récents (Sofascore) éventuel
    patch = DATA / "results_patch.csv"
    if patch.exists():
        extra = pd.read_csv(patch, parse_dates=["Date"])
        key = set(zip(allm["Date"].dt.date.astype(str), allm["Winner"], allm["Loser"]))
        rows = [r for _, r in extra.iterrows()
                if (str(r["Date"].date()), r["Winner"], r["Loser"]) not in key]
        if rows:
            allm = pd.concat([allm, pd.DataFrame(rows)], ignore_index=True)
            allm = allm.sort_values("Date").reset_index(drop=True)
            print(f"  +{len(rows)} match(s) du patch live")
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
    recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=25))
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

    def _snapshot_top(n=5):
        act = [(p, r) for p, r in elo.rating.items() if elo.count[p] >= 15]
        act.sort(key=lambda t: -t[1])
        return [{"player": p, "elo": round(r, 1)} for p, r in act[:n]]

    rows, ys, metas = [], [], []
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

        wr = _num(m.get("WRank"), 250.0)
        lr = _num(m.get("LRank"), 250.0)
        wp = _num(m.get("WPts"), 10.0)
        lp = _num(m.get("LPts"), 10.0)

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

        feats = {
            "p_elo": elo.prob(p1, p2, surface),
            "d_selo": (elo.surf[surface][p1] - elo.surf[surface][p2]) / 100.0,
            "d_lrank": math.log(r2) - math.log(r1),  # >0 si p1 mieux classé
            "d_lpts": math.log1p(q1) - math.log1p(q2),
            "d_form10": form(p1, 10) - form(p2, 10),
            "d_form25": form(p1, 25) - form(p2, 25),
            "d_activity": (activity(p1) - activity(p2)) / 10.0,
            "h2h": h2h_p1,
            "h2h_count": math.log1p(tot),
            "h2h_surface": h2h_surface_p1,
            "h2h_slam": h2h_slam_p1,
            "h2h_recent": h2h_recent_p1,
            "d_tourn_sets": (p1_t_sets - p2_t_sets) / 10.0,
            "d_tourn_games": (p1_t_games - p2_t_games) / 100.0,
            "d_recent_sets": (p1_r_sets - p2_r_sets) / 20.0,
            "d_recent_games": (p1_r_games - p2_r_games) / 200.0,
            "best_of": 1.0 if float(m.get("Best of", 3) or 3) >= 5 else 0.0,
            "s_hard": 1.0 if surface == "Hard" else 0.0,
            "s_clay": 1.0 if surface == "Clay" else 0.0,
            "s_grass": 1.0 if surface == "Grass" else 0.0,
        }
        rows.append(feats)
        ys.append(1 if p1_win else 0)
        metas.append({
            "date": date, "tournament": m.get("Tournament"), "series": m.get("Series"),
            "surface": surface, "round": m.get("Round"), "best_of": m.get("Best of"),
            "winner": w, "loser": l, "wrank": wr, "lrank": lr,
            "p1": p1, "p2": p2,
            "avgw": m.get("AvgW"), "avgl": m.get("AvgL"),
            "score": _score_str(m),
            "comment": m.get("Comment"),
        })

        # mises à jour post-match
        if cur_year is None:
            cur_year = date.year
        elif date.year > cur_year:
            year_end[cur_year] = _snapshot_top()
            cur_year = date.year
        elo.update(w, l, surface, m.get("Series"))
        history[w].append((date, round(elo.rating[w], 1)))
        history[l].append((date, round(elo.rating[l], 1)))
        if str(m.get("Series")) == "Grand Slam" and str(m.get("Round")) == "The Final":
            slam_finals.append({"year": date.year, "tournament": str(m.get("Tournament")),
                                "winner": w, "loser": l, "score": _score_str(m)})
        recent[w].append(1)
        recent[l].append(0)
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
        rank[w], rank[l] = wr, lr
        pts[w], pts[l] = wp, lp

    if cur_year is not None:
        year_end[cur_year] = _snapshot_top()  # année en cours = état actuel

    X = pd.DataFrame(rows)[FEATURES]
    y = np.array(ys)
    state = {"elo": elo, "recent": recent, "rank": rank, "pts": pts, "last_dates": last_dates,
             "history": history, "year_end": year_end, "slam_finals": slam_finals}
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
