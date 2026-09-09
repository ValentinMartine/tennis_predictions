"""Périmètre des données et classements manquants."""
import pandas as pd

from src.export import _opt_int
from src.pipeline import _log1p_diff, _log_diff, load_matches


def _match(series: str, winner: str, loser: str) -> dict:
    return {
        "ATP": 1, "Location": "Nulle part", "Tournament": f"Tournoi {series}",
        "Date": pd.Timestamp("2026-01-05"), "Series": series, "Court": "Outdoor",
        "Surface": "Hard", "Round": "1st Round", "Best of": 3,
        "Winner": winner, "Loser": loser, "Comment": "Completed",
        "W1": 6, "L1": 4, "W2": 6, "L2": 3,
    }


def test_load_matches_drops_challenger_and_itf(tmp_path, monkeypatch) -> None:
    frame = pd.DataFrame([
        _match("ATP", "Alpha", "Beta"),
        _match("Grand Slam", "Gamma", "Delta"),
        _match("Challenger", "Epsilon", "Zeta"),
        _match("ITF", "Eta", "Theta"),
    ])
    frame.to_excel(tmp_path / "atp_2026.xlsx", index=False)
    monkeypatch.setattr("src.pipeline.DATA", tmp_path)

    kept = load_matches()

    assert sorted(kept["Series"].unique()) == ["ATP", "Grand Slam"]
    assert "Epsilon" not in set(kept["Winner"])


def test_unknown_rank_gives_a_neutral_gap_not_a_fake_ranking() -> None:
    assert _log_diff(120.0, 30.0) > 0
    assert _log_diff(float("nan"), 30.0) == 0.0
    assert _log1p_diff(2000.0, float("nan")) == 0.0


def test_optional_rank_export_rejects_nan_and_zero() -> None:
    assert _opt_int(42.0) == 42
    assert _opt_int(float("nan")) is None
    assert _opt_int(0) is None
    assert _opt_int(None) is None
