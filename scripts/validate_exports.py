"""Validate the JSON exports consumed by valentinmartine.com/projets/tennis."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "data" / "tennis_predictions.json"
HISTORY = ROOT / "data" / "tennis_elo_history.json"


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_predictions(data: dict) -> None:
    required = {
        "generated_at", "source", "n_matches", "last_match", "metrics",
        "elo_rankings", "recent_matches", "elo_peaks", "year_end", "slam_winners",
    }
    missing = required - set(data)
    _assert(not missing, f"Missing prediction keys: {sorted(missing)}")
    date.fromisoformat(data["last_match"])
    _assert(data["n_matches"] >= 50_000, "Unexpectedly small match history")
    _assert(len(data["elo_rankings"]) >= 30, "Too few active Elo players")
    _assert(len(data["recent_matches"]) >= 100, "Too few displayed season matches")

    metrics = data["metrics"]
    for key in ("elo", "ml", "blend"):
        block = metrics.get(key) or {}
        _assert(0.0 <= block.get("accuracy", -1) <= 1.0, f"Invalid {key} accuracy")
        _assert(block.get("log_loss", 99) > 0, f"Invalid {key} log-loss")

    players = {p["player"] for p in data["elo_rankings"]}
    _assert(len(players) == len(data["elo_rankings"]), "Duplicate players in Elo ranking")
    for match in data["recent_matches"]:
        pair = {match["winner"], match["loser"]}
        _assert(match.get("predicted_player") in pair, f"Prediction outside match pair: {match}")
        _assert(match.get("model_correct") == (match.get("predicted_player") == match["winner"]),
                f"Inconsistent model_correct flag: {match}")
        prob = match.get("predicted_prob")
        _assert(prob is None or 0.5 <= prob <= 1.0, f"Invalid predicted probability: {match}")


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


def main() -> int:
    validate_predictions(_load(PREDICTIONS))
    validate_history(_load(HISTORY))
    print("exports OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
