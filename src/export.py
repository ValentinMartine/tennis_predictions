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
import pandas as pd
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.elo import SURFACES  # noqa: E402
from scripts.download_data import known_surnames, short_name_from_full  # noqa: E402
from src.pipeline import (  # noqa: E402
    FEATURES, build_features, load_matches, market_prob, upcoming_features,
)
from src.sackmann_stats import aggregate_player_stats, load_match_stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
UPCOMING_RAW = ROOT / "data" / "upcoming_raw.json"


def _suffix(tour: str) -> str:
    return "" if tour == "atp" else f"_{tour}"


def out_paths(tour: str) -> tuple[Path, Path, Path, Path]:
    """Prédictions, historique Elo, statut source et modèle du circuit demandé."""
    data = ROOT / "data"
    return (
        data / f"tennis_predictions{_suffix(tour)}.json",
        data / f"tennis_elo_history{_suffix(tour)}.json",
        data / f"source_status{_suffix(tour)}.json",
        data / f"model{_suffix(tour)}.joblib",
    )
SITE_DIR = Path("C:/Users/valen/Documents/Projets/Projets_Informatiques/Perso/valentinmartine/public/data")
VAL_START = np.datetime64("2023-01-01")
TEST_START = np.datetime64("2024-01-01")
N_TOURNAMENTS = 4


def _opt_int(value) -> int | None:
    """Entier exportable, ou None quand le classement est inconnu (NaN, 0, vide)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number == number and number > 0 else None


def _score_parts(match) -> list[tuple[int, int]]:
    parts = []
    for i in range(1, 6):
        try:
            w_games = int(float(match.get(f"W{i}")))
            l_games = int(float(match.get(f"L{i}")))
        except (TypeError, ValueError):
            continue
        parts.append((w_games, l_games))
    return parts


def _player_match_stats(allm):
    stats: dict[str, dict] = {}

    def entry(player: str) -> dict:
        if player not in stats:
            stats[player] = {
                "player": player, "matches": 0, "wins": 0, "losses": 0,
                "sets_for": 0, "sets_against": 0, "games_for": 0, "games_against": 0,
                "tiebreak_wins": 0, "tiebreak_losses": 0,
                "deciding_wins": 0, "deciding_losses": 0,
                "straight_wins": 0, "grand_slam_matches": 0, "grand_slam_wins": 0,
                "hard_matches": 0, "hard_wins": 0, "clay_matches": 0, "clay_wins": 0,
                "grass_matches": 0, "grass_wins": 0,
            }
        return stats[player]

    for _, match in allm.iterrows():
        winner, loser = str(match["Winner"]).strip(), str(match["Loser"]).strip()
        surface = str(match.get("Surface") or "Hard").strip().title()
        series = str(match.get("Series") or "")
        parts = _score_parts(match)
        if not parts:
            continue
        try:
            w_sets = int(float(match.get("Wsets", sum(1 for w, l in parts if w > l)) or 0))
            l_sets = int(float(match.get("Lsets", sum(1 for w, l in parts if l > w)) or 0))
            best_of = int(float(match.get("Best of", 3) or 3))
        except (TypeError, ValueError):
            w_sets = sum(1 for w, l in parts if w > l)
            l_sets = sum(1 for w, l in parts if l > w)
            best_of = 3

        for player, won in ((winner, True), (loser, False)):
            row = entry(player)
            row["matches"] += 1
            row["wins" if won else "losses"] += 1
            row["sets_for"] += w_sets if won else l_sets
            row["sets_against"] += l_sets if won else w_sets
            row["games_for"] += sum(w for w, _ in parts) if won else sum(l for _, l in parts)
            row["games_against"] += sum(l for _, l in parts) if won else sum(w for w, _ in parts)
            surface_key = {"Hard": "hard", "Clay": "clay", "Grass": "grass"}.get(surface, "hard")
            row[f"{surface_key}_matches"] += 1
            if won:
                row[f"{surface_key}_wins"] += 1
            if series == "Grand Slam":
                row["grand_slam_matches"] += 1
                if won:
                    row["grand_slam_wins"] += 1

        total_sets = w_sets + l_sets
        if total_sets == best_of:
            entry(winner)["deciding_wins"] += 1
            entry(loser)["deciding_losses"] += 1
        if l_sets == 0:
            entry(winner)["straight_wins"] += 1
        for w_games, l_games in parts:
            if (w_games, l_games) == (7, 6):
                entry(winner)["tiebreak_wins"] += 1
                entry(loser)["tiebreak_losses"] += 1
            elif (w_games, l_games) == (6, 7):
                entry(winner)["tiebreak_losses"] += 1
                entry(loser)["tiebreak_wins"] += 1

    def pct(num, den):
        return round(num / den, 4) if den else None

    out = []
    for row in stats.values():
        matches = row["matches"]
        if matches < 30:
            continue
        tiebreaks = row["tiebreak_wins"] + row["tiebreak_losses"]
        deciders = row["deciding_wins"] + row["deciding_losses"]
        grand_slams = row["grand_slam_matches"]
        sets = row["sets_for"] + row["sets_against"]
        games = row["games_for"] + row["games_against"]
        row.update({
            "win_rate": pct(row["wins"], matches),
            "set_win_rate": pct(row["sets_for"], sets),
            "game_win_rate": pct(row["games_for"], games),
            "tiebreak_win_rate": pct(row["tiebreak_wins"], tiebreaks),
            "deciding_win_rate": pct(row["deciding_wins"], deciders),
            "straight_win_rate": pct(row["straight_wins"], row["wins"]),
            "grand_slam_win_rate": pct(row["grand_slam_wins"], grand_slams),
        })
        for surface in ("hard", "clay", "grass"):
            row[f"{surface}_win_rate"] = pct(row[f"{surface}_wins"], row[f"{surface}_matches"])
        out.append(row)
    out.sort(key=lambda x: (-x["matches"], x["player"]))
    return out


def _enrich_with_sackmann_stats(base_stats: list[dict], allm) -> list[dict]:
    all_players = set(allm["Winner"].astype(str).str.strip()) | set(allm["Loser"].astype(str).str.strip())
    sackmann_matches = load_match_stats(existing_players=all_players)
    sackmann_stats = aggregate_player_stats(sackmann_matches)
    if not sackmann_stats:
        return base_stats
    enriched = []
    for row in base_stats:
        extra = sackmann_stats.get(row["player"])
        enriched.append({**row, **extra} if extra else row)
    return enriched


def build_upcoming(tour: str, allm, state, model, blend_w) -> list[dict]:
    """Matchs à venir du circuit, prédits avec l'état arrêté au dernier match connu."""
    if not UPCOMING_RAW.exists():
        return []
    raw = json.loads(UPCOMING_RAW.read_text(encoding="utf-8"))
    season = allm[allm["Date"].dt.year == allm["Date"].max().year]
    # Le calendrier mêle ATP, WTA, Challenger et ITF sans indiquer le niveau. Seul
    # l'identifiant de tournoi tranche : les tableaux masculin et féminin d'un même
    # Grand Chelem portent le même nom mais deux identifiants (16866 et 16865 pour
    # l'US Open 2026). Conséquence assumée : un tournoi qui n'a pas encore produit de
    # match terminé cette saison n'est pas encore reconnu.
    known_ids = {_id_key(x) for x in season["ATP"].dropna().unique()}
    surnames = known_surnames(tour=tour)
    today = datetime.now(timezone.utc).date().isoformat()

    rows, metas = [], []
    for match in raw.get("matches", []):
        if (match.get("date") or "") < today:
            continue
        if _id_key(match.get("tournament_id")) not in known_ids:
            continue
        p1 = short_name_from_full(match["player1"], surnames)
        p2 = short_name_from_full(match["player2"], surnames)
        if p1 == p2:
            continue
        best_of = 5 if (tour == "atp" and _is_slam(season, match)) else 3
        series = _series_for_tournament(season, match)
        rows.append(upcoming_features(state, p1, p2, match.get("surface") or "Hard",
                                      best_of, state["last_date"]))
        metas.append((match, p1, p2, series))

    if not rows:
        return []
    frame = pd.DataFrame(rows)[FEATURES]
    p_ml = model.predict_proba(frame)[:, 1]
    p_blend = blend_w * p_ml + (1 - blend_w) * frame["p_elo"].to_numpy()

    upcoming = []
    for (match, p1, p2, series), probability in zip(metas, p_blend):
        probability = float(probability)
        market = market_prob(match.get("odds1"), match.get("odds2"))
        upcoming.append({
            "date": match["date"], "time": match.get("time"),
            "tournament": match["tournament"], "series": series, "surface": match.get("surface"),
            "round": match.get("round"),
            "player1": p1, "player2": p2,
            "rank1": match.get("rank1"), "rank2": match.get("rank2"),
            "odds1": match.get("odds1"), "odds2": match.get("odds2"),
            "p1_win": round(probability, 3),
            "predicted_player": p1 if probability >= 0.5 else p2,
            "predicted_prob": round(probability if probability >= 0.5 else 1 - probability, 3),
            "market_p1_win": round(market, 3) if market is not None else None,
        })
    upcoming.sort(key=lambda m: (m["date"], m["time"] or ""))
    return upcoming


def _id_key(value) -> str:
    """Identifiant de tournoi comparable : pandas rend 16866.0 là où le JSON dit 16866."""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def _is_slam(season, match) -> bool:
    same = season[season["ATP"].map(_id_key) == _id_key(match.get("tournament_id"))]
    return bool(len(same)) and str(same["Series"].iloc[0]) == "Grand Slam"


def _series_for_tournament(season, match) -> str | None:
    same = season[season["ATP"].map(_id_key) == _id_key(match.get("tournament_id"))]
    if not len(same):
        return None
    return str(same["Series"].iloc[0])


def main(argv: list[str] | None = None) -> int:
    tour = (argv or sys.argv[1:] or ["atp"])[0].lower()
    OUT, OUT_HIST, SOURCE_STATUS, MODEL = out_paths(tour)
    allm = load_matches(tour)
    X, y, metas, state = build_features(allm)
    bundle = joblib.load(MODEL)
    model, w = bundle["model"], bundle["blend_w"]
    p_ml = model.predict_proba(X[FEATURES])[:, 1]
    p_blend = w * p_ml + (1 - w) * X["p_elo"].to_numpy()

    dates = np.array([m["date"] for m in metas], dtype="datetime64[ns]")
    va = (dates >= VAL_START) & (dates < TEST_START)
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
    market = {"accuracy": None, "log_loss": None, "n": 0}
    if len(pm):
        market = {
            "accuracy": round(float(((pm > 0.5) == (ym == 1)).mean()), 4),
            "log_loss": round(float(log_loss(ym, np.clip(pm, 1e-6, 1 - 1e-6))), 4),
            "n": int(len(pm)),
        }
    metrics = {
        "n_test": int(te.sum()),
        "elo": block(X["p_elo"].to_numpy()),
        "ml": block(p_ml),
        "blend": block(p_blend),
        "market": market,
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

    # précision par niveau de tournoi et par tour (test)
    def _acc_group(key_fn):
        groups: dict[str, list] = {}
        for i in range(len(metas)):
            if te[i]:
                groups.setdefault(key_fn(metas[i]), []).append(i)
        out = {}
        for g, idx in groups.items():
            if len(idx) >= 50:
                pp, yy = p_blend[idx], y[idx]
                out[g] = {"n": len(idx), "accuracy": round(float(((pp > 0.5) == (yy == 1)).mean()), 4)}
        return out

    metrics["by_series"] = _acc_group(lambda m: str(m["series"]))
    round_map = {"The Final": "Finale", "Semifinals": "Demi-finales", "Quarterfinals": "Quarts"}
    metrics["by_round"] = _acc_group(lambda m: round_map.get(str(m["round"]), "Tours précédents"))

    # backtest de mise sur le test : mise à valeur espérée positive (cotes moyennes)
    def _backtest(mask, edge_min=0.05, conf_min=0.5):
        stakes = returns = kelly_bank = 0.0
        kelly_hist = 1.0
        n_bets = wins = 0
        for i in range(len(metas)):
            if not mask[i]:
                continue
            mm = metas[i]
            try:
                ow, ol = float(mm["avgw"]), float(mm["avgl"])
            except (TypeError, ValueError):
                continue
            if not (ow > 1 and ol > 1) or ow != ow or ol != ol:
                continue
            pw = float(p_blend[i] if mm["p1"] == mm["winner"] else 1 - p_blend[i])
            # deux côtés possibles : vainqueur réel (cote ow, proba modèle pw) ou perdant
            cands = [(pw, ow, True), (1 - pw, ol, False)]
            best = max(cands, key=lambda c: c[0] * c[1])
            ev = best[0] * best[1] - 1
            if ev < edge_min or best[0] < conf_min:
                continue
            n_bets += 1
            stakes += 1.0
            hit = best[2]
            if hit:
                wins += 1
                returns += best[1]
            # quart de Kelly (borné à 10 % de la banque)
            b = best[1] - 1
            f = max(0.0, min(0.1, 0.25 * (best[0] * b - (1 - best[0])) / b))
            kelly_hist *= (1 + f * b) if hit else (1 - f)
        roi = (returns - stakes) / stakes if stakes else 0.0
        return {"n_bets": n_bets, "win_rate": round(wins / n_bets, 3) if n_bets else None,
                "roi_flat": round(roi, 4), "kelly_final_bank": round(kelly_hist, 3),
                "edge_min": edge_min, "conf_min": conf_min}

    def _select_betting_threshold():
        best = None
        for edge_min in (0.00, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20):
            for conf_min in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
                stats = _backtest(va, edge_min, conf_min)
                if stats["n_bets"] < 30:
                    continue
                if best is None or stats["roi_flat"] > best["roi_flat"]:
                    best = stats
        return best or _backtest(va)

    validation_betting = _select_betting_threshold()
    test_betting = _backtest(te, validation_betting["edge_min"], validation_betting["conf_min"])
    test_betting["validation_n_bets"] = validation_betting["n_bets"]
    test_betting["validation_roi_flat"] = validation_betting["roi_flat"]
    metrics["betting"] = test_betting

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
            "rank": _opt_int(rank.get(p)),
            "pts": _opt_int(pts.get(p)),
            "form10": f"{sum(d10)}-{len(d10) - sum(d10)}",
            "streak": streak,
            "matches": elo.count[p],
        })
    players.sort(key=lambda x: -x["elo"])
    players = players[:60]

    # TOUS les matchs de la saison en cours (du plus récent au plus ancien)
    season_start = np.datetime64(f"{allm['Date'].max().year}-01-01")
    recent_matches = []
    for i in range(len(metas) - 1, -1, -1):
        if dates[i] < season_start:
            break
        m = metas[i]
        p1_prob = float(p_blend[i])
        predicted_player = m["p1"] if p1_prob >= 0.5 else m["p2"]
        predicted_prob = p1_prob if predicted_player == m["p1"] else 1 - p1_prob
        pw = float(p1_prob if m["p1"] == m["winner"] else 1 - p1_prob)
        q = market_prob(m["avgw"], m["avgl"])
        recent_matches.append({
            "date": str(np.datetime_as_string(dates[i], unit="D")),
            "tournament": m["tournament"], "series": m["series"],
            "surface": m["surface"], "round": m["round"],
            "winner": m["winner"], "loser": m["loser"],
            "wrank": _opt_int(m["wrank"]),
            "lrank": _opt_int(m["lrank"]),
            "p_win": round(pw, 3),
            "predicted_player": predicted_player,
            "predicted_prob": round(float(predicted_prob), 3),
            "model_correct": predicted_player == m["winner"],
            "market_p_win": round(float(q), 3) if q is not None else None,
            "score": m["score"],
            "retired": str(m.get("comment") or "") == "Retired",
            # La source de secours tronque le dernier set d'un match sur huit : le dire
            # plutôt que d'afficher un score qui contredit le vainqueur.
            "score_partial": str(m.get("score_status") or "") == "partial",
        })

    # ===== Historique Elo 2000-2026 =====
    history = state["history"]
    year_end = state["year_end"]
    slam_finals = state["slam_finals"]

    # pics Elo de tous les temps (carrière >= 150 matchs pour éviter les artefacts)
    peaks = []
    for p, hist in history.items():
        if len(hist) < 150:
            continue
        d, e = max(hist, key=lambda t: t[1])
        peaks.append({"player": p, "elo": e, "date": str(d.date())})
    peaks.sort(key=lambda x: -x["elo"])
    peaks = peaks[:20]

    # trajectoires mensuelles (fichier dédié) : pics all-time + top 60 actuel + podiums
    # de fin d'année -> l'utilisateur choisit librement les joueurs affichés.
    timeline_players = [p["player"] for p in peaks]
    for p in players:
        if p["player"] not in timeline_players:
            timeline_players.append(p["player"])
    for tops in year_end.values():
        for entry in tops[:3]:
            if entry["player"] not in timeline_players:
                timeline_players.append(entry["player"])
    elo_timeline = []
    ranking_timeline = []
    months = set()
    for p in timeline_players:
        monthly: dict[str, float] = {}
        for d, e in history.get(p, []):
            monthly[f"{d.year}-{d.month:02d}"] = e  # dernière valeur du mois
        pts_series = sorted(monthly.items())
        if len(pts_series) >= 12:
            elo_timeline.append({"player": p, "points": pts_series})
    for p, hist in history.items():
        if len(hist) < 30:
            continue
        # Troisième valeur : matchs disputés dans le mois. Le site s'en sert pour dire
        # combien un joueur a joué sur la période affichée, au lieu de laisser croire
        # qu'un Elo repris du dernier match connu vient d'une activité récente.
        monthly: dict[str, list] = {}
        for d, e in hist:
            ym = f"{d.year}-{d.month:02d}"
            entry = monthly.setdefault(ym, [ym, e, 0])
            entry[1] = e
            entry[2] += 1
            months.add(ym)
        pts_series = [monthly[ym] for ym in sorted(monthly)]
        if len(pts_series) >= 2:
            ranking_timeline.append({"player": p, "points": pts_series, "matches": len(hist)})
    # Par défaut, le graphe doit répondre à "qui est fort maintenant ?".
    # Les légendes historiques restent visibles dans la section des pics all-time,
    # mais elles ne doivent pas masquer le classement courant. On ne retient que
    # des joueurs présents dans `elo_timeline`, sinon le front recevrait une
    # sélection impossible à afficher.
    timeline_names = {serie["player"] for serie in elo_timeline}
    default_players = [p["player"] for p in players if p["player"] in timeline_names][:8]
    if not default_players:
        default_players = [p["player"] for p in peaks if p["player"] in timeline_names][:8]

    year_end_list = [{"year": yy, "top": tops} for yy, tops in sorted(year_end.items())]

    # palmarès des Grands Chelems par année
    slam_order = ["Australian Open", "French Open", "Roland Garros", "Wimbledon", "US Open"]
    slams_by_year: dict[int, dict] = {}
    for f in slam_finals:
        name = f["tournament"]
        key = "Roland Garros" if name in ("French Open", "Roland Garros") else name
        slams_by_year.setdefault(f["year"], {})[key] = {"winner": f["winner"], "loser": f["loser"], "score": f["score"]}
    slam_winners = [{"year": yy, **{k: v for k, v in d.items()}} for yy, d in sorted(slams_by_year.items())]
    _ = slam_order  # ordre géré côté site

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

    player_stats = _enrich_with_sackmann_stats(_player_match_stats(allm), allm)
    has_sackmann_stats = any("stat_elo" in row for row in player_stats)
    upcoming = build_upcoming(tour, allm, state, model, w)
    source_status = json.loads(SOURCE_STATUS.read_text(encoding="utf-8")) if SOURCE_STATUS.exists() else None
    source_label = f"tennis-data.co.uk ({tour.upper()}, seasons 2000-{allm['Date'].max().year})"
    current_year_source = source_status.get("current_year_source") if source_status else None
    current_year_url = source_status.get("current_year_url") if source_status else None
    if current_year_source and current_year_source != "tennis-data.co.uk":
        source_label += f" + fallback {current_year_source}"
    if has_sackmann_stats:
        source_label += " + Jeff Sackmann tennis_atp (serve/return statistics)"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "Elo global+surface (K type FiveThirtyEight) + LightGBM (classement, forme, "
                  "activité, H2H), blend réglé sur validation 2023, test 2024+",
        "tour": tour,
        "source": source_label,
        "source_status": source_status,
        "refresh_status": "ok" if not current_year_source or current_year_source == "tennis-data.co.uk" else "primary_source_unavailable_fallback_used",
        "current_year_source_url": current_year_url,
        "sackmann_stats_active": has_sackmann_stats,
        "n_matches": int(len(allm)),
        "last_match": str(allm["Date"].max().date()),
        "metrics": metrics,
        "elo_rankings": players,
        "player_stats": player_stats,
        "recent_matches": recent_matches,
        "recent_tournaments": tournaments,
        "elo_peaks": peaks,
        "year_end": year_end_list,
        "slam_winners": slam_winners,
        "upcoming": upcoming,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    hist_payload = {
        "default_players": default_players,
        "players": elo_timeline,
        "ranking_players": ranking_timeline,
        "months": sorted(months),
    }
    OUT_HIST.write_text(json.dumps(hist_payload, ensure_ascii=False), encoding="utf-8")
    if SITE_DIR.exists():
        (SITE_DIR / OUT.name).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        (SITE_DIR / OUT_HIST.name).write_text(json.dumps(hist_payload, ensure_ascii=False), encoding="utf-8")
    print(f"écrit {OUT.name} : {len(players)} joueurs, {len(recent_matches)} matchs saison")
    print(f"écrit {OUT_HIST.name} : {len(elo_timeline)} trajectoires, {len(upcoming)} matchs à venir")
    print("top 5 Elo :", ", ".join(f"{p['player']} ({p['elo']})" for p in players[:5]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
