from pathlib import Path

from scripts.download_data import (
    score_columns,
    score_columns_from_valuebetennis,
    valuebetennis_round,
    valuebetennis_series,
    valuebetennis_surface,
    write_source_status,
)


def test_score_columns_extracts_sets_and_games() -> None:
    values = score_columns("6-4 3-6 7-6(5)")

    assert values["W1"] == 6
    assert values["L1"] == 4
    assert values["W2"] == 3
    assert values["L2"] == 6
    assert values["W3"] == 7
    assert values["L3"] == 6
    assert values["Wsets"] == 2
    assert values["Lsets"] == 1


def test_write_source_status_records_current_year(tmp_path, monkeypatch) -> None:
    target = tmp_path / "source_status.json"
    monkeypatch.setattr("scripts.download_data.SOURCE_STATUS", target)

    write_source_status(
        [{"year": 2026, "source": "sackmann-compatible-fallback", "ok": True, "url": "https://example.test"}],
        healthy=True,
        current=2026,
    )

    content = Path(target).read_text(encoding="utf-8")
    assert '"current_year_source": "sackmann-compatible-fallback"' in content
    assert '"healthy": true' in content


def test_valuebetennis_mappings_use_existing_labels() -> None:
    assert valuebetennis_surface("dur") == "Hard"
    assert valuebetennis_surface("terre battue") == "Clay"
    assert valuebetennis_series("Grand Chelem") == "Grand Slam"
    assert valuebetennis_round(12) == "The Final"


def test_valuebetennis_score_is_converted_to_winner_perspective() -> None:
    values = score_columns_from_valuebetennis("4-6 7-5 3-6", winner_is_player1=False)

    assert values["W1"] == 6
    assert values["L1"] == 4
    assert values["W2"] == 5
    assert values["L2"] == 7
    assert values["W3"] == 6
    assert values["L3"] == 3
    assert values["Wsets"] == 2
    assert values["Lsets"] == 1
    assert values["_complete_sets"] == 3
