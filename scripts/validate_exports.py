"""Validate the JSON exports consumed by valentinmartine.com/projets/tennis."""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def export_paths(tour: str) -> tuple[Path, Path]:
    suffix = "" if tour == "atp" else f"_{tour}"
    return (ROOT / "data" / f"tennis_predictions{suffix}.json",
            ROOT / "data" / f"tennis_elo_history{suffix}.json")


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# Périmètre du projet : main tour ATP. Un fallback qui rapatrie du Challenger ou de
# l'ITF fausse les métriques et le classement sans lever d'erreur.
# Deux vocabulaires coexistent : tennis-data.co.uk (ATP250/ATP500/Masters Cup) et
# les sources de secours (ATP/Tour Finals).
ALLOWED_SERIES = {
    "Grand Slam", "Masters 1000", "Masters Cup", "Tour Finals", "ATP", "ATP250", "ATP500",
    "WTA 1000", "WTA 500", "WTA 250",
}

ROUND_ORDER = {
    "Qualifying 1": 10, "Qualifying 2": 20, "Qualifying 3": 30,
    "1st Round": 100, "Round of 128": 100,
    "2nd Round": 200, "Round of 64": 200,
    "3rd Round": 300, "Round of 32": 300,
    "4th Round": 400, "Round of 16": 400,
    "Quarterfinals": 500,
    "Semifinals": 600,
    "The Final": 700, "Final": 700,
    "Round Robin": 50, "poules": 50,
}
_ROUND_ORDER_CASSE = {nom.lower(): ordre for nom, ordre in ROUND_ORDER.items()}


def _round_order(match: dict) -> int:
    value = match.get("round_order")
    if value is not None:
        return int(value)
    return _ROUND_ORDER_CASSE.get(str(match.get("round") or "").strip().lower(), 0)


def validate_predictions(data: dict) -> None:
    required = {
        "generated_at", "source", "sackmann_stats_active", "n_matches", "last_match", "metrics",
        "elo_rankings", "player_stats", "recent_matches", "elo_peaks", "year_end", "slam_winners",
    }
    missing = required - set(data)
    _assert(not missing, f"Missing prediction keys: {sorted(missing)}")
    date.fromisoformat(data["last_match"])
    _assert(data["n_matches"] >= 50_000, "Unexpectedly small match history")
    _assert(len(data["elo_rankings"]) >= 30, "Too few active Elo players")
    _assert(len(data["player_stats"]) >= 300, "Too few player statistic profiles")
    _assert(len(data["recent_matches"]) >= 100, "Too few displayed season matches")

    metrics = data["metrics"]
    for key in ("elo", "ml", "blend"):
        block = metrics.get(key) or {}
        _assert(0.0 <= block.get("accuracy", -1) <= 1.0, f"Invalid {key} accuracy")
        _assert(block.get("log_loss", 99) > 0, f"Invalid {key} log-loss")
    betting = metrics.get("betting") or {}
    for key in ("n_bets", "roi_flat", "edge_min", "conf_min", "validation_n_bets", "validation_roi_flat"):
        _assert(key in betting, f"Missing betting key: {key}")
    _assert(betting["n_bets"] >= 0, "Invalid betting count")
    _assert(0.0 <= betting["edge_min"] <= 1.0, "Invalid betting edge threshold")
    _assert(0.5 <= betting["conf_min"] <= 1.0, "Invalid betting confidence threshold")

    players = {p["player"] for p in data["elo_rankings"]}
    _assert(len(players) == len(data["elo_rankings"]), "Duplicate players in Elo ranking")
    stat_players = {p["player"] for p in data["player_stats"]}
    qualified_players = {p["player"] for p in data["elo_rankings"] if p.get("matches", 0) >= 30}
    _assert(qualified_players.issubset(stat_players), "Active Elo players with >= 30 matches missing from player_stats")
    for player in data["player_stats"]:
        _assert(player["matches"] >= 30, f"Too few matches in player_stats for {player['player']}")
        for key in ("win_rate", "set_win_rate", "game_win_rate"):
            value = player.get(key)
            _assert(value is None or 0.0 <= value <= 1.0, f"Invalid {key} for {player['player']}")
        for key in (
            "ace_rate", "double_fault_rate", "first_in_rate", "first_serve_won_rate",
            "second_serve_won_rate", "service_points_won_rate", "break_points_saved_rate",
            "return_points_won_rate", "break_points_converted_rate",
        ):
            value = player.get(key)
            _assert(value is None or 0.0 <= value <= 1.0, f"Invalid {key} for {player['player']}")
        for key in ("serve_elo", "return_elo", "pressure_elo", "stat_elo"):
            value = player.get(key)
            _assert(value is None or 1000 <= value <= 2200, f"Invalid {key} for {player['player']}")
    out_of_scope = {m.get("series") for m in data["recent_matches"]} - ALLOWED_SERIES
    _assert(not out_of_scope, f"Series outside the ATP main tour: {sorted(out_of_scope)}")
    for match in data["recent_matches"]:
        pair = {match["winner"], match["loser"]}
        _assert(match.get("predicted_player") in pair, f"Prediction outside match pair: {match}")
        _assert(match.get("model_correct") == (match.get("predicted_player") == match["winner"]),
                f"Inconsistent model_correct flag: {match}")
        prob = match.get("predicted_prob")
        _assert(prob is None or 0.5 <= prob <= 1.0, f"Invalid predicted probability: {match}")
        _assert(_round_order(match) > 0, f"Missing round_order for match: {match}")

    by_tournament: dict[str, list[dict]] = {}
    for match in data["recent_matches"]:
        by_tournament.setdefault(match["tournament"], []).append(match)
    for tournament, matches in by_tournament.items():
        orders = [_round_order(match) for match in matches]
        _assert(orders == sorted(orders, reverse=True), f"Rounds are not sorted for {tournament}")
        is_grand_slam = any(match.get("series") == "Grand Slam" for match in matches)
        has_final = any(match.get("round") in {"The Final", "Final"} for match in matches)
        if is_grand_slam and has_final:
            qf = sum(match.get("round") == "Quarterfinals" for match in matches)
            sf = sum(match.get("round") == "Semifinals" for match in matches)
            _assert(qf >= 4, f"Missing quarterfinals before final for {tournament}")
            required_semifinals = 2 if "U.S. Open" in tournament else 1
            _assert(sf >= required_semifinals, f"Missing semifinals before final for {tournament}")
            for match in matches:
                if match.get("round") in {"The Final", "Final"}:
                    _assert(not match.get("score_partial"), f"Partial Grand Slam final score for {tournament}")


def validate_history(data: dict) -> None:
    required = {"default_players", "players", "ranking_players", "months"}
    missing = required - set(data)
    _assert(not missing, f"Missing history keys: {sorted(missing)}")
    months = data["months"]
    _assert(months == sorted(months), "History months must be sorted")
    _assert(len(months) >= 200, "Too few monthly history points")
    _assert(len(data["players"]) >= 20, "Too few default history trajectories")
    _assert(len(data["ranking_players"]) >= 300, "Too few ranking history players")
    default_names = {p["player"] for p in data["players"]}
    _assert(set(data["default_players"]).issubset(default_names), "Default players missing from history")
    for player in data["ranking_players"]:
        points = player.get("points") or []
        _assert(len(points) >= 2, f"Too few points for {player.get('player')}")
        _assert(points == sorted(points, key=lambda p: p[0]), f"Unsorted points for {player.get('player')}")


def validate_alignment(data: dict, history: dict) -> None:
    """Le classement Elo doit couvrir le mois du dernier match exporté."""
    last_month = str(data["last_match"])[:7]
    _assert(last_month in history["months"],
            f"History stops at {history['months'][-1]} while last match is {last_month}")
    ranked = {p["player"] for p in history["ranking_players"]}
    qualified_active = {p["player"] for p in data["elo_rankings"] if p.get("matches", 0) >= 30}
    _assert(qualified_active.issubset(ranked), "Active Elo players with >= 30 matches missing from the ranking history")


def validate_upcoming(data: dict) -> None:
    """Un match à venir n'a pas de résultat : seule sa probabilité doit être publiée."""
    today = date.today().isoformat()
    for match in data.get("upcoming") or []:
        pair = {match["player1"], match["player2"]}
        _assert(match.get("predicted_player") in pair, f"Prediction outside match pair: {match}")
        probability = match.get("predicted_prob")
        _assert(probability is None or 0.5 <= probability <= 1.0, f"Invalid upcoming probability: {match}")
        _assert((match.get("date") or "") >= today, f"Past match still listed as upcoming: {match}")
        _assert(_round_order(match) > 0, f"Missing round_order for upcoming match: {match}")
        _assert("winner" not in match and "model_correct" not in match,
                f"Upcoming match must not carry a result: {match}")


def main(argv: list[str] | None = None) -> int:
    tour = (argv or sys.argv[1:] or ["atp"])[0].lower()
    predictions_path, history_path = export_paths(tour)
    predictions = _load(predictions_path)
    history = _load(history_path)
    validate_predictions(predictions)
    validate_history(history)
    validate_alignment(predictions, history)
    validate_upcoming(predictions)
    print(f"exports {tour} OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
