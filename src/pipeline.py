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

FEATURES = ["p_elo", "d_selo", "d_lrank", "d_lpts", "d_form10", "d_form25",
            "d_activity", "h2h", "best_of", "s_hard", "s_clay", "s_grass"]


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


def build_features(allm: pd.DataFrame):
    """Parcourt les matchs en ordre chronologique et produit X, y, meta (+ état final)."""
    elo = TennisElo()
    recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=25))
    last_dates: dict[str, deque] = defaultdict(lambda: deque(maxlen=40))
    h2h: dict[tuple, list] = defaultdict(lambda: [0, 0])
    rank: dict[str, float] = {}
    pts: dict[str, float] = {}

    rows, ys, metas = [], [], []
    for _, m in allm.iterrows():
        w, l = str(m["Winner"]).strip(), str(m["Loser"]).strip()
        surface = norm_surface(m.get("Surface"))
        date = m["Date"]
        wr = float(m["WRank"]) if pd.notna(m.get("WRank")) else 250.0
        lr = float(m["LRank"]) if pd.notna(m.get("LRank")) else 250.0
        wp = float(m["WPts"]) if pd.notna(m.get("WPts")) else 10.0
        lp = float(m["LPts"]) if pd.notna(m.get("LPts")) else 10.0

        p1_win = _p1_is_winner(date, w, l)
        p1, p2 = (w, l) if p1_win else (l, w)
        r1, r2 = (wr, lr) if p1_win else (lr, wr)
        q1, q2 = (wp, lp) if p1_win else (lp, wp)

        def form(p, n):
            d = list(recent[p])[-n:]
            return sum(d) / len(d) if d else 0.5

        def activity(p):
            return sum(1 for d in last_dates[p] if (date - d).days <= 60)

        k = (p1, p2) if p1 < p2 else (p2, p1)
        hh = h2h[k]
        tot = hh[0] + hh[1]
        h2h_p1 = ((hh[0] if p1 < p2 else hh[1]) / tot - 0.5) if tot else 0.0

        feats = {
            "p_elo": elo.prob(p1, p2, surface),
            "d_selo": (elo.surf[surface][p1] - elo.surf[surface][p2]) / 100.0,
            "d_lrank": math.log(r2) - math.log(r1),  # >0 si p1 mieux classé
            "d_lpts": math.log1p(q1) - math.log1p(q2),
            "d_form10": form(p1, 10) - form(p2, 10),
            "d_form25": form(p1, 25) - form(p2, 25),
            "d_activity": (activity(p1) - activity(p2)) / 10.0,
            "h2h": h2h_p1,
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
        elo.update(w, l, surface, m.get("Series"))
        recent[w].append(1)
        recent[l].append(0)
        last_dates[w].append(date)
        last_dates[l].append(date)
        hh[0 if w == k[0] else 1] += 1
        rank[w], rank[l] = wr, lr
        pts[w], pts[l] = wp, lp

    X = pd.DataFrame(rows)[FEATURES]
    y = np.array(ys)
    state = {"elo": elo, "recent": recent, "rank": rank, "pts": pts, "last_dates": last_dates}
    return X, y, metas, state


def _score_str(m) -> str | None:
    parts = []
    for i in range(1, 6):
        w, l = m.get(f"W{i}"), m.get(f"L{i}")
        if pd.isna(w) or pd.isna(l):
            break
        parts.append(f"{int(w)}-{int(l)}")
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
