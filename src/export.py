"""Exporte data/tennis_predictions.json pour le site : classement Elo, matchs récents
prédits (avant match) vs réels, tournois récents, métriques du backtest.

Usage : python -m src.export
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.elo import SURFACES  # noqa: E402
from src.pipeline import FEATURES, build_features, load_matches, market_prob  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "tennis_predictions.json"
SITE = Path("C:/Users/valen/Documents/Projets/Projets_Informatiques/Perso/valentinmartine/public/data/tennis_predictions.json")
TEST_START = np.datetime64("2024-01-01")
N_RECENT = 150
N_TOURNAMENTS = 4


def main() -> int:
    allm = load_matches()
    X, y, metas, state = build_features(allm)
    bundle = joblib.load(ROOT / "data" / "model.joblib")
    model, w = bundle["model"], bundle["blend_w"]
    p_ml = model.predict_proba(X[FEATURES])[:, 1]
    p_blend = w * p_ml + (1 - w) * X["p_elo"].to_numpy()

    dates = np.array([m["date"] for m in metas], dtype="datetime64[ns]")
    te = dates >= TEST_START

    def block(p):
        pp, yy = p[te], y[te]
        return {"accuracy": round(float(((pp > 0.5) == (yy == 1)).mean()), 4),
                "log_loss": round(float(log_loss(yy, np.clip(pp, 1e-6, 1 - 1e-6))), 4)}

    pm, ym = [], []
    for i, m in enumerate(metas):
        if not te[i]:
            continue
        q = market_prob(m["avgw"], m["avgl"])
        if q is not None:
            pm.append(q if m["p1"] == m["winner"] else 1 - q)
            ym.append(y[i])
    pm, ym = np.array(pm), np.array(ym)
    metrics = {
        "n_test": int(te.sum()),
        "elo": block(X["p_elo"].to_numpy()),
        "ml": block(p_ml),
        "blend": block(p_blend),
        "market": {"accuracy": round(float(((pm > 0.5) == (ym == 1)).mean()), 4),
                   "log_loss": round(float(log_loss(ym, np.clip(pm, 1e-6, 1 - 1e-6))), 4),
                   "n": int(len(pm))},
        "blend_w": w,
    }

    # précision par surface (test)
    by_surface = {}
    for s in SURFACES:
        idx = [i for i in range(len(metas)) if te[i] and metas[i]["surface"] == s]
        if idx:
            pp = p_blend[idx]
            yy = y[idx]
            by_surface[s] = {"n": len(idx), "accuracy": round(float(((pp > 0.5) == (yy == 1)).mean()), 4)}
    metrics["by_surface"] = by_surface

    # classement Elo des joueurs actifs (>= 1 match sur 6 mois, >= 30 matchs au total)
    elo, recent = state["elo"], state["recent"]
    last_dates, rank, pts = state["last_dates"], state["rank"], state["pts"]
    now = allm["Date"].max()
    players = []
    for p, r in elo.rating.items():
        if elo.count[p] < 30 or not last_dates[p] or (now - last_dates[p][-1]).days > 180:
            continue
        d10 = list(recent[p])[-10:]
        streak = 0
        for v in reversed(list(recent[p])):
            if streak == 0:
                streak = 1 if v else -1
            elif (v and streak > 0) or (not v and streak < 0):
                streak += 1 if v else -1
            else:
                break
        players.append({
            "player": p, "elo": round(r, 1),
            "hard": round(elo.surf["Hard"][p], 1),
            "clay": round(elo.surf["Clay"][p], 1),
            "grass": round(elo.surf["Grass"][p], 1),
            "rank": int(rank.get(p, 0)) or None,
            "pts": int(pts.get(p, 0)) or None,
            "form10": f"{sum(d10)}-{len(d10) - sum(d10)}",
            "streak": streak,
            "matches": elo.count[p],
        })
    players.sort(key=lambda x: -x["elo"])
    players = players[:60]

    # matchs récents (du plus récent au plus ancien)
    recent_matches = []
    for i in range(len(metas) - 1, -1, -1):
        m = metas[i]
        pw = float(p_blend[i] if m["p1"] == m["winner"] else 1 - p_blend[i])
        q = market_prob(m["avgw"], m["avgl"])
        recent_matches.append({
            "date": str(np.datetime_as_string(dates[i], unit="D")),
            "tournament": m["tournament"], "series": m["series"],
            "surface": m["surface"], "round": m["round"],
            "winner": m["winner"], "loser": m["loser"],
            "wrank": int(m["wrank"]) if m["wrank"] else None,
            "lrank": int(m["lrank"]) if m["lrank"] else None,
            "p_win": round(pw, 3),
            "market_p_win": round(float(q), 3) if q is not None else None,
            "score": m["score"],
            "retired": str(m.get("comment") or "") == "Retired",
        })
        if len(recent_matches) >= N_RECENT:
            break

    # tournois récents groupés
    seen, tournaments = [], []
    for rm in recent_matches:
        key = rm["tournament"]
        if key not in seen:
            seen.append(key)
    for key in seen[:N_TOURNAMENTS]:
        ms = [rm for rm in recent_matches if rm["tournament"] == key]
        tournaments.append({"tournament": key, "surface": ms[0]["surface"],
                            "series": ms[0]["series"], "n": len(ms),
                            "last_date": ms[0]["date"]})

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "Elo global+surface (K type FiveThirtyEight) + LightGBM (classement, forme, "
                  "activité, H2H), blend réglé sur validation 2023, test 2024+",
        "source": "tennis-data.co.uk (ATP, saisons 2010-2026)",
        "n_matches": int(len(allm)),
        "last_match": str(allm["Date"].max().date()),
        "metrics": metrics,
        "elo_rankings": players,
        "recent_matches": recent_matches,
        "recent_tournaments": tournaments,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    if SITE.parent.exists():
        SITE.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"écrit {OUT.name} : {len(players)} joueurs, {len(recent_matches)} matchs récents")
    print("top 5 Elo :", ", ".join(f"{p['player']} ({p['elo']})" for p in players[:5]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
