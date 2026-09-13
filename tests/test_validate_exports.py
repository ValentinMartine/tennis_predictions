"""Garde-fous ajoutés après l'audit : périmètre et alignement classement/matchs."""
import pytest

from scripts.validate_exports import validate_alignment, validate_predictions


def _predictions(series="ATP", last_match="2026-09-09"):
    return {
        "generated_at": "2026-09-09T13:38:10+00:00", "source": "test",
        "sackmann_stats_active": False, "n_matches": 60_000, "last_match": last_match,
        "metrics": {
            "elo": {"accuracy": 0.65, "log_loss": 0.62},
            "ml": {"accuracy": 0.66, "log_loss": 0.61},
            "blend": {"accuracy": 0.66, "log_loss": 0.61},
            "betting": {"n_bets": 10, "roi_flat": -0.02, "edge_min": 0.2, "conf_min": 0.7,
                        "validation_n_bets": 5, "validation_roi_flat": -0.01},
        },
        "elo_rankings": [{"player": f"J{i}", "elo": 1800} for i in range(30)],
        "player_stats": [{"player": f"J{i}", "matches": 40} for i in range(300)],
        "recent_matches": [{
            "date": "2026-09-09", "tournament": "Test Open", "series": series,
            "round": "Quarterfinals", "round_order": 500, "winner": "J0", "loser": "J1",
            "predicted_player": "J0", "model_correct": True, "predicted_prob": 0.7,
        }] * 100,
        "elo_peaks": [], "year_end": [], "slam_winners": [],
    }


def test_challenger_matches_are_rejected() -> None:
    with pytest.raises(AssertionError, match="outside the ATP main tour"):
        validate_predictions(_predictions(series="Challenger"))


def test_main_tour_matches_pass() -> None:
    validate_predictions(_predictions(series="Grand Slam"))


def test_history_lagging_behind_the_last_match_is_rejected() -> None:
    history = {"months": ["2026-06", "2026-07"], "ranking_players": [{"player": "J0"}]}
    with pytest.raises(AssertionError, match="History stops at"):
        validate_alignment({"last_match": "2026-09-09", "elo_rankings": [{"player": "J0"}]}, history)
