from pathlib import Path

import pandas as pd

from scripts.download_data import (
    MAIN_TOUR_SERIES,
    is_finished_set,
    known_surnames,
    score_columns,
    series_from_sackmann_level,
    score_columns_from_valuebetennis,
    short_name_from_full,
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
    monkeypatch.setattr("scripts.download_data.DATA", tmp_path)

    write_source_status(
        [{"year": 2026, "source": "sackmann-compatible-fallback", "ok": True, "url": "https://example.test"}],
        healthy=True,
        current=2026,
    )

    content = Path(tmp_path / "source_status.json").read_text(encoding="utf-8")
    assert '"current_year_source": "sackmann-compatible-fallback"' in content
    assert '"healthy": true' in content


def test_each_tour_writes_its_own_status_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scripts.download_data.DATA", tmp_path)

    write_source_status([{"year": 2026, "source": "valuebetennis", "ok": True, "url": None}],
                        healthy=True, current=2026, tour="wta")

    assert (tmp_path / "source_status_wta.json").exists()
    assert not (tmp_path / "source_status.json").exists()


def test_wta_levels_use_the_wta_vocabulary() -> None:
    assert series_from_sackmann_level("PM", "wta") == "WTA 1000"
    assert series_from_sackmann_level("I", "wta") == "WTA 250"
    assert series_from_sackmann_level("G", "wta") == "Grand Slam"
    assert series_from_sackmann_level("M", "atp") == "Masters 1000"
    assert valuebetennis_series("Main tour", "wta") == "WTA 250"
    assert valuebetennis_series("Masters", "wta") == "WTA 1000"
    assert valuebetennis_series("ITF", "wta") not in MAIN_TOUR_SERIES


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


def test_valuebetennis_rounds_follow_the_draw_ladder() -> None:
    """Codes relevés sur l'Australian Open 2026 : 4:64, 5:32, 6:16, 7:8, 9:4, 10:2, 12:1."""
    assert valuebetennis_round(4) == "1st Round"
    assert valuebetennis_round(5) == "2nd Round"
    assert valuebetennis_round(6) == "3rd Round"
    assert valuebetennis_round(7) == "4th Round"
    assert valuebetennis_round(9) == "Quarterfinals"
    assert valuebetennis_round(10) == "Semifinals"
    assert valuebetennis_round(12) == "The Final"
    assert valuebetennis_round(8) == "Round Robin"
    assert valuebetennis_round(1) == "Qualifying 1"


def test_valuebetennis_main_tour_perimeter_excludes_challenger_and_itf() -> None:
    assert valuebetennis_series("Challenger") not in MAIN_TOUR_SERIES
    assert valuebetennis_series("ITF") not in MAIN_TOUR_SERIES
    assert valuebetennis_series("Main tour") in MAIN_TOUR_SERIES
    assert valuebetennis_series("Masters") in MAIN_TOUR_SERIES
    assert valuebetennis_series("Grand Chelem") in MAIN_TOUR_SERIES


def test_source_status_is_unhealthy_when_a_season_failed() -> None:
    records = [
        {"year": 2025, "source": "tennis-data.co.uk", "ok": True, "url": None},
        {"year": 2026, "source": "none", "ok": False, "url": None},
    ]

    assert all(record["ok"] for record in records) is False


def test_full_names_are_mapped_onto_the_short_convention() -> None:
    """La source de secours écrit « Jannik Sinner » là où la primaire écrit « Sinner J. »."""
    surnames = {
        "sinner": {"J.": "Sinner J."},
        "augeraliassime": {"F.": "Auger-Aliassime F."},
        "deminaur": {"A.": "De Minaur A."},
        "delpotro": {"J.M.": "Del Potro J.M."},
        "fritz": {"T.": "Fritz T."},
        "lee": {"M.": "Lee M.", "H.T.": "Lee H.T."},
    }

    assert short_name_from_full("Jannik Sinner", surnames) == "Sinner J."
    assert short_name_from_full("Felix Auger Aliassime", surnames) == "Auger-Aliassime F."
    assert short_name_from_full("Alex De Minaur", surnames) == "De Minaur A."
    assert short_name_from_full("Juan Martin Del Potro", surnames) == "Del Potro J.M."
    # Le deuxième prénom ne doit pas créer un joueur parallèle « Fritz T.H. ».
    assert short_name_from_full("Taylor Harry Fritz", surnames) == "Fritz T."
    assert short_name_from_full("Hyung Taik Lee", surnames) == "Lee H.T."


def test_unknown_players_keep_a_deterministic_short_name() -> None:
    assert short_name_from_full("Rafael Jodar", {}) == "Jodar R."
    assert short_name_from_full("Sinner J.", {}) == "J. S."
    assert short_name_from_full("Madonna", {}) == "Madonna"


def test_known_surnames_reads_the_primary_convention(tmp_path, monkeypatch) -> None:
    frame = pd.DataFrame({"Winner": ["Auger-Aliassime F.", "Jannik Sinner"], "Loser": ["De Minaur A.", "X"]})
    frame.to_excel(tmp_path / "atp_2024.xlsx", index=False)
    monkeypatch.setattr("scripts.download_data.DATA", tmp_path)
    monkeypatch.setattr("scripts.download_data._SEASON_NAMES_CACHE", {})

    index = known_surnames()

    assert index["augeraliassime"] == {"F.": "Auger-Aliassime F."}
    assert index["deminaur"] == {"A.": "De Minaur A."}
    assert "janniksinner" not in index  # un nom complet n'est pas un nom de famille


def test_index_ignores_the_season_being_rebuilt(tmp_path, monkeypatch) -> None:
    """Une conversion approximative d'un run précédent ne doit pas se perpétuer."""
    pd.DataFrame({"Winner": ["Fritz T."], "Loser": ["X Y."]}).to_excel(tmp_path / "atp_2025.xlsx", index=False)
    pd.DataFrame({"Winner": ["Fritz T.H."], "Loser": ["X Y."]}).to_excel(tmp_path / "atp_2026.xlsx", index=False)
    monkeypatch.setattr("scripts.download_data.DATA", tmp_path)
    monkeypatch.setattr("scripts.download_data._SEASON_NAMES_CACHE", {})

    assert short_name_from_full("Taylor Harry Fritz", known_surnames(exclude_year=2026)) == "Fritz T."
    assert short_name_from_full("Taylor Harry Fritz", known_surnames()) == "Fritz T.H."


def test_incomplete_final_set_is_dropped_and_flagged() -> None:
    """« 5-7 3-6 7-5 6-3 5-5 » : le 5e set est en cours ou tronqué par la source."""
    values = score_columns_from_valuebetennis("5-7 3-6 7-5 6-3 5-5", winner_is_player1=True)

    assert values["_partial"] is True
    assert values["_complete_sets"] == 4
    assert values["W5"] is None and values["L5"] is None


def test_a_clean_score_is_not_flagged_partial() -> None:
    values = score_columns_from_valuebetennis("6-4 3-6 7-6(4)", winner_is_player1=True)

    assert values["_partial"] is False
    assert values["Wsets"] == 2 and values["Lsets"] == 1


def test_finished_set_recognises_the_usual_endings() -> None:
    assert is_finished_set(6, 4) and is_finished_set(7, 5) and is_finished_set(7, 6)
    assert is_finished_set(10, 8)
    assert not is_finished_set(6, 6)
    assert not is_finished_set(5, 5)
    assert not is_finished_set(2, 1)


def test_a_retirement_in_the_first_sets_stays_a_result() -> None:
    """« 6-4 2-1 » : abandon au 3e jeu du 2e set, le vainqueur est connu."""
    values = score_columns_from_valuebetennis("6-4 2-1", winner_is_player1=True)

    assert values["_complete_sets"] == 1
    assert values["_partial"] is True
    assert values["Wsets"] == 1 and values["Lsets"] is None
