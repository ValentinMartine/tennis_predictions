"""Exporte data/tennis_predictions.json pour le site : classement Elo, matchs récents
prédits (avant match) vs réels, tournois récents, métriques du backtest.

Usage : python -m src.export
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.elo import SURFACES  # noqa: E402
from src.modeling import calibration_bins, probability_metrics, yearly_metrics  # noqa: E402
from scripts.download_data import known_surnames, short_name_from_full  # noqa: E402
from src.rankings import get_rankings_lookup  # noqa: E402
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
MARKET_MODEL_GAP_ALERT = 0.15

ROUND_ORDER = {
    "Qualifying 1": 10,
    "Qualifying 2": 20,
    "Qualifying 3": 30,
    "1st Round": 100,
    "Round of 128": 100,
    "2nd Round": 200,
    "Round of 64": 200,
    "3rd Round": 300,
    "Round of 32": 300,
    "4th Round": 400,
    "Round of 16": 400,
    "Quarterfinals": 500,
    "Semifinals": 600,
    "The Final": 700,
    "Final": 700,
    "Round Robin": 50,
}


def round_order(round_name) -> int:
    """Ordre sportif du tableau, indépendant de l'ordre alphabétique ou de la date source."""
    return ROUND_ORDER.get(str(round_name or "").strip(), 0)


def _opt_int(value) -> int | None:
    """Entier exportable, ou None quand le classement est inconnu (NaN, 0, vide)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number == number and number > 0 else None


def _text_or_empty(value) -> str:
    """Chaîne sûre pour les champs optionnels pandas (NaN/pd.NA -> '')."""
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _valid_prob(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and 0.0 <= number <= 1.0)


def _best_market_blend_weight(y, model_p1, market_p1, mask) -> float:
    """Learn how much pre-match market probability to use when it is available."""
    idx = mask & np.isfinite(market_p1)
    if not idx.any():
        return 0.0
    best_w, best_ll = 0.0, float("inf")
    for w in np.arange(0.0, 1.001, 0.01):
        p = w * market_p1[idx] + (1 - w) * model_p1[idx]
        ll = log_loss(y[idx], np.clip(p, 1e-6, 1 - 1e-6))
        if ll < best_ll:
            best_w, best_ll = float(w), float(ll)
    return best_w


def _market_adjusted_probability(model_p1: float, market_p1: float | None, market_weight: float) -> float:
    """Primary probability: independent model unless market calibration is available."""
    if not _valid_prob(market_p1):
        return float(model_p1)
    return float(market_weight * float(market_p1) + (1 - market_weight) * model_p1)


def _prediction_fields(
    p1: str,
    p2: str,
    model_p1: float,
    market_p1: float | None,
    market_weight: float,
) -> dict:
    """Return independent and market-benchmark predictions from p1 perspective."""
    primary_p1 = float(model_p1)
    market_adjusted_p1 = (
        _market_adjusted_probability(model_p1, market_p1, market_weight)
        if _valid_prob(market_p1) else None
    )
    primary_player = p1 if primary_p1 >= 0.5 else p2
    model_player = p1 if model_p1 >= 0.5 else p2
    market_player = None
    if _valid_prob(market_p1):
        market_player = p1 if float(market_p1) >= 0.5 else p2
    return {
        "p1_win": round(float(primary_p1), 3),
        "predicted_player": primary_player,
        "predicted_prob": round(float(primary_p1 if primary_player == p1 else 1 - primary_p1), 3),
        "prediction_source": "independent_calibrated",
        "model_p1_win": round(float(model_p1), 3),
        "model_predicted_player": model_player,
        "model_predicted_prob": round(float(model_p1 if model_player == p1 else 1 - model_p1), 3),
        "market_predicted_player": market_player,
        "market_calibrated_p1_win": round(float(market_adjusted_p1), 3) if market_adjusted_p1 is not None else None,
        "market_model_gap": round(abs(float(market_p1) - float(model_p1)), 3) if _valid_prob(market_p1) else None,
    }


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
    sackmann_matches = load_match_stats(
        existing_players=all_players,
        tour=str(allm.attrs.get("tour", "atp")).lower(),
    )
    sackmann_stats = aggregate_player_stats(sackmann_matches)
    if not sackmann_stats:
        return base_stats
    enriched = []
    for row in base_stats:
        extra = sackmann_stats.get(row["player"])
        enriched.append({**row, **extra} if extra else row)
    return enriched


def build_upcoming(tour: str, allm, state, model, blend_w, calibrator=None, market_weight: float = 0.0) -> list[dict]:
    """Matchs à venir du circuit, prédits avec l'état arrêté au dernier match connu."""
    if not UPCOMING_RAW.exists():
        return []
    raw = json.loads(UPCOMING_RAW.read_text(encoding="utf-8"))
    season = allm[allm["Date"].dt.year == allm["Date"].max().year]
    # Le calendrier mêle ATP, WTA, Challenger et ITF sans indiquer le niveau. Seul
    # l'identifiant de tournoi tranche : les tableaux masculin et féminin d'un même
    # Grand Chelem portent le même nom mais deux identifiants (16866 et 16865 pour
    # l'US Open 2026). Conséquence assumée : un tournoi qui n'a pas encore produit de
    # match terminé cette saison n'est pas encore reconnu : c'est le cas d'un tournoi en cours
    # (Pékin et Tokyo le 30/09/2026). Il est alors reconnu par son nom, s'il a été joué par ce
    # circuit ces deux dernières saisons, et par ses deux joueurs, connus du circuit : le
    # tableau féminin du même nom et les Challenger restent écartés.
    known_ids = {_id_key(x) for x in season["ATP"].dropna().unique()}
    editions = _editions_recentes(allm)
    joueurs_connus = _joueurs_recents(allm)
    surnames = known_surnames(tour=tour)
    today = datetime.now(timezone.utc).date().isoformat()

    rows, metas = [], []
    for match in raw.get("matches", []):
        if (match.get("date") or "") < today:
            continue
        p1 = short_name_from_full(match["player1"], surnames)
        p2 = short_name_from_full(match["player2"], surnames)
        if p1 == p2:
            continue
        if _id_key(match.get("tournament_id")) in known_ids:
            series = _series_for_tournament(season, match)
        else:
            series = editions.get(_nom_tournoi(match.get("tournament")))
            if series is None or p1 not in joueurs_connus or p2 not in joueurs_connus:
                continue
        best_of = 5 if (tour == "atp" and series == "Grand Slam") else 3
        match_date = pd.to_datetime(match["date"], errors="coerce")
        if pd.isna(match_date):
            match_date = state["last_date"]
        rows.append(upcoming_features(state, p1, p2, match.get("surface") or "Hard",
                                      best_of, match_date, series, match.get("round"),
                                      tournament_id=match.get("tournament_id"),
                                      tournament=match.get("tournament"),
                                      rank1=match.get("rank1"),
                                      rank2=match.get("rank2")))
        metas.append((match, p1, p2, series))

    if not rows:
        return []
    frame = pd.DataFrame(rows)[FEATURES]
    p_ml = model.predict_proba(frame)[:, 1]
    p_blend = blend_w * p_ml + (1 - blend_w) * frame["p_elo"].to_numpy()
    p_model = calibrator.predict(p_blend) if calibrator is not None else p_blend

    upcoming = []
    for (match, p1, p2, series), probability, (_, feature_row) in zip(metas, p_model, frame.iterrows()):
        probability = float(probability)
        market = market_prob(match.get("odds1"), match.get("odds2"))
        prediction = _prediction_fields(p1, p2, probability, market, market_weight)
        upcoming.append({
            "date": match["date"], "time": match.get("time"),
            "tournament": match["tournament"], "series": series, "surface": match.get("surface"),
            "round": match.get("round"),
            "round_order": round_order(match.get("round")),
            "player1": p1, "player2": p2,
            "rank1": match.get("rank1"), "rank2": match.get("rank2"),
            "odds1": match.get("odds1"), "odds2": match.get("odds2"),
            **prediction,
            "market_p1_win": round(market, 3) if market is not None else None,
            "rank_features_available": bool(feature_row["rank_both_known"] >= 1.0),
            "probability_warning": (
                "missing_rank_features" if feature_row["rank_both_known"] < 1.0
                else "model_market_gap" if prediction["market_model_gap"] is not None
                and prediction["market_model_gap"] >= MARKET_MODEL_GAP_ALERT else None
            ),
        })
    upcoming.sort(key=lambda m: (m["date"], m["time"] or "", m.get("round_order", 0)))
    return upcoming


def _id_key(value) -> str:
    """Identifiant de tournoi comparable : pandas rend 16866.0 là où le JSON dit 16866."""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def _nom_tournoi(nom) -> str:
    """« China Open - Beijing » et « China Open » désignent le même tournoi selon la source."""
    return str(nom or "").split(" - ")[0].strip().lower()


def _editions_recentes(allm, saisons: int = 2) -> dict[str, str]:
    """Série de la dernière édition de chaque tournoi joué par le circuit ces dernières saisons."""
    if "Tournament" not in allm or allm.empty:
        return {}
    recent = allm[allm["Date"].dt.year >= allm["Date"].max().year - saisons].sort_values("Date")
    return {_nom_tournoi(t): str(s) for t, s in zip(recent["Tournament"], recent["Series"])}


def _joueurs_recents(allm, saisons: int = 2) -> set[str]:
    if "Winner" not in allm or allm.empty:
        return set()
    recent = allm[allm["Date"].dt.year >= allm["Date"].max().year - saisons]
    return set(recent["Winner"].astype(str).str.strip()) | set(recent["Loser"].astype(str).str.strip())


def _series_for_tournament(season, match) -> str | None:
    same = season[season["ATP"].map(_id_key) == _id_key(match.get("tournament_id"))]
    if not len(same):
        return None
    return str(same["Series"].iloc[0])


def main(argv: list[str] | None = None) -> int:
    tour = (argv or sys.argv[1:] or ["atp"])[0].lower()
    OUT, OUT_HIST, SOURCE_STATUS, MODEL = out_paths(tour)
    allm = load_matches(tour)
    X, y, metas, state = build_features(allm, tour=tour)
    bundle = joblib.load(MODEL)
    model, w = bundle["model"], bundle["blend_w"]
    model_features = bundle.get("features", FEATURES)
    calibrator = bundle.get("calibrator")
    p_ml = model.predict_proba(X[model_features])[:, 1]
    p_blend = w * p_ml + (1 - w) * X["p_elo"].to_numpy()
    p_model = calibrator.predict(p_blend) if calibrator is not None else p_blend

    dates = np.array([m["date"] for m in metas], dtype="datetime64[ns]")
    va = (dates >= VAL_START) & (dates < TEST_START)
    te = dates >= TEST_START
    market_p1 = np.full(len(metas), np.nan)
    for i, m in enumerate(metas):
        q = market_prob(m["avgw"], m["avgl"])
        if q is not None:
            market_p1[i] = q if m["p1"] == m["winner"] else 1 - q
    market_weight = _best_market_blend_weight(y, p_model, market_p1, va)
    p_market_calibrated = np.where(
        np.isfinite(market_p1),
        market_weight * market_p1 + (1 - market_weight) * p_model,
        p_model,
    )

    def block(p):
        out = probability_metrics(y, p, te)
        out.pop("n", None)
        return out

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
        "independent_calibrated": block(p_model),
        "market_calibrated": block(p_market_calibrated),
        "market": market,
        "blend_w": w,
        "market_blend_weight": market_weight,
        "calibration_method": getattr(calibrator, "method", "identity"),
        "model_version": bundle.get("model_version", "legacy"),
        "rolling_years": yearly_metrics(y, p_model, dates),
        "calibration_bins": calibration_bins(y[te], p_model[te]),
    }

    # précision par surface (test)
    by_surface = {}
    for s in SURFACES:
        idx = [i for i in range(len(metas)) if te[i] and metas[i]["surface"] == s]
        if idx:
            pp = p_model[idx]
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
                pp, yy = p_model[idx], y[idx]
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
        returns_list: list[float] = []
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
            pw = float(p_model[i] if mm["p1"] == mm["winner"] else 1 - p_model[i])
            # deux côtés possibles : vainqueur réel (cote ow, proba modèle pw) ou perdant
            cands = [(pw, ow, True), (1 - pw, ol, False)]
            best = max(cands, key=lambda c: c[0] * c[1])
            ev = best[0] * best[1] - 1
            if ev < edge_min or best[0] < conf_min:
                continue
            n_bets += 1
            stakes += 1.0
            hit = best[2]
            ret_item = (best[1] - 1.0) if hit else -1.0
            returns_list.append(ret_item)
            if hit:
                wins += 1
                returns += best[1]
            # quart de Kelly (borné à 10 % de la banque)
            b = best[1] - 1
            f = max(0.0, min(0.1, 0.25 * (best[0] * b - (1 - best[0])) / b))
            kelly_hist *= (1 + f * b) if hit else (1 - f)
        roi = (returns - stakes) / stakes if stakes else 0.0

        # Intervalles de confiance statistiques (Wilson pour le taux de gain, normal pour le ROI)
        win_rate = round(wins / n_bets, 3) if n_bets else None
        win_rate_ci95 = None
        roi_ci95 = None
        roi_se = 0.0
        t_stat = 0.0
        p_val = 1.0
        if n_bets > 0:
            p_hat = wins / n_bets
            z = 1.95996
            denom_w = 1.0 + (z**2) / n_bets
            ctr_w = (p_hat + (z**2) / (2.0 * n_bets)) / denom_w
            hw_w = (z * math.sqrt((p_hat * (1.0 - p_hat) / n_bets) + ((z**2) / (4.0 * (n_bets**2))))) / denom_w
            win_rate_ci95 = [round(max(0.0, ctr_w - hw_w), 3), round(min(1.0, ctr_w + hw_w), 3)]

            r_arr = np.array(returns_list, dtype=float)
            s = float(r_arr.std(ddof=1)) if n_bets > 1 else 0.0
            roi_se = round(s / math.sqrt(n_bets), 4) if n_bets > 0 else 0.0
            roi_ci95 = [round(roi - z * roi_se, 4), round(roi + z * roi_se, 4)]
            t_stat = round(roi / roi_se, 2) if roi_se > 0 else 0.0
            p_val = round(float(1.0 - 0.5 * (1.0 + math.erf(t_stat / math.sqrt(2.0)))), 4) if roi_se > 0 else 1.0

        return {
            "n_bets": n_bets,
            "win_rate": win_rate,
            "win_rate_ci95": win_rate_ci95,
            "roi_flat": round(roi, 4),
            "roi_flat_ci95": roi_ci95,
            "roi_se": roi_se,
            "roi_t_stat": t_stat,
            "roi_p_value": p_val,
            "kelly_final_bank": round(kelly_hist, 3),
            "edge_min": edge_min,
            "conf_min": conf_min,
        }

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

    # classement Elo des joueurs actifs (>= 1 match sur 18 mois pour couvrir le top 300)
    elo, recent = state["elo"], state["recent"]
    last_dates, rank, pts = state["last_dates"], state["rank"], state["pts"]
    rl = get_rankings_lookup(tour)
    now = allm["Date"].max()
    players = []
    for p, r in elo.rating.items():
        if elo.count[p] < 1 or not last_dates[p] or (now - last_dates[p][-1]).days > 540:
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
        lrnk, lpts = rl.get_latest_rank_points(p)
        final_rank = lrnk if (lrnk is not None and not math.isnan(lrnk)) else rank.get(p)
        final_pts = lpts if (lpts is not None and not math.isnan(lpts)) else pts.get(p)
        players.append({
            "player": p, "elo": round(r, 1),
            "hard": round(elo.surf["Hard"][p], 1),
            "clay": round(elo.surf["Clay"][p], 1),
            "grass": round(elo.surf["Grass"][p], 1),
            "rank": _opt_int(final_rank),
            "pts": _opt_int(final_pts),
            "form10": f"{sum(d10)}-{len(d10) - sum(d10)}",
            "streak": streak,
            "matches": elo.count[p],
        })
    seen_canonical: dict[str, dict] = {}
    for pl in players:
        key = pl["player"].strip().lower()
        if key not in seen_canonical or pl["matches"] > seen_canonical[key]["matches"]:
            seen_canonical[key] = pl
    players = list(seen_canonical.values())
    players.sort(key=lambda x: -x["elo"])
    players = players[:300]

    # TOUS les matchs de la saison en cours (du plus récent au plus ancien)
    season_start = np.datetime64(f"{allm['Date'].max().year}-01-01")
    recent_matches = []
    for i in range(len(metas) - 1, -1, -1):
        if dates[i] < season_start:
            break
        m = metas[i]
        q = market_prob(m["avgw"], m["avgl"])
        q_p1 = q if (q is not None and m["p1"] == m["winner"]) else (1 - q if q is not None else None)
        prediction = _prediction_fields(m["p1"], m["p2"], float(p_model[i]), q_p1, market_weight)
        primary_p1 = float(prediction["p1_win"])
        pw = float(primary_p1 if m["p1"] == m["winner"] else 1 - primary_p1)
        model_predicted_player = prediction["model_predicted_player"]
        recent_matches.append({
            "date": str(np.datetime_as_string(dates[i], unit="D")),
            "tournament": m["tournament"], "series": m["series"],
            "surface": m["surface"], "round": m["round"],
            "round_order": round_order(m["round"]),
            "winner": m["winner"], "loser": m["loser"],
            "wrank": _opt_int(m["wrank"]),
            "lrank": _opt_int(m["lrank"]),
            "p_win": round(pw, 3),
            **prediction,
            "model_correct": prediction["predicted_player"] == m["winner"],
            "independent_model_correct": model_predicted_player == m["winner"],
            "market_p_win": round(float(q), 3) if q is not None else None,
            "rank_features_available": _opt_int(m.get("wrank_feature")) is not None and _opt_int(m.get("lrank_feature")) is not None,
            "probability_warning": (
                "missing_rank_features" if _opt_int(m.get("wrank_feature")) is None or _opt_int(m.get("lrank_feature")) is None
                else "model_market_gap" if prediction["market_model_gap"] is not None
                and prediction["market_model_gap"] >= MARKET_MODEL_GAP_ALERT else None
            ),
            "score": m["score"],
            "retired": _text_or_empty(m.get("comment")) == "Retired",
            # Les abandons sont déjà explicités par le statut 'retired' ; 'score_partial'
            # n'est activé que si le flux a été tronqué hors abandon.
            "score_partial": _text_or_empty(m.get("score_status")) == "partial" and _text_or_empty(m.get("comment")) != "Retired",
        })

    tournament_last_date = {}
    for match in recent_matches:
        tournament_last_date[match["tournament"]] = max(
            tournament_last_date.get(match["tournament"], match["date"]),
            match["date"],
        )
    recent_matches.sort(
        key=lambda match: (
            tournament_last_date.get(match["tournament"], match["date"]),
            match.get("round_order", 0),
            match["date"],
        ),
        reverse=True,
    )

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
    for p in players[:60]:
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
    upcoming = build_upcoming(tour, allm, state, model, w, calibrator, market_weight)
    source_status = json.loads(SOURCE_STATUS.read_text(encoding="utf-8")) if SOURCE_STATUS.exists() else None
    source_label = f"tennis-data.co.uk ({tour.upper()}, seasons 2000-{allm['Date'].max().year})"
    current_year_source = source_status.get("current_year_source") if source_status else None
    current_year_url = source_status.get("current_year_url") if source_status else None
    if current_year_source and current_year_source != "tennis-data.co.uk":
        source_label += f" + fallback {current_year_source}"
    if has_sackmann_stats:
        source_label += f" + Jeff Sackmann tennis_{tour} (serve/return statistics)"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "Modèle indépendant sans cotes : Elo global+surface + LightGBM sur features "
                  "sans fuite (classement/points, forme globale/surface/saison, streaks, charge, repos, "
                  "abandons récents, H2H, niveau, tour, surface, stats service/retour si cache Sackmann disponible), "
                  "poids ML/Elo réglé sur 2023, calibration prudente sur 2023 et test tenu à l'écart depuis 2024. "
                  "Le marché reste un benchmark affiché, pas une entrée de la probabilité primaire.",
        "model_policy": "independent_no_odds_primary",
        "model_version": bundle.get("model_version", "legacy"),
        "calibration_method": getattr(calibrator, "method", "identity"),
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
