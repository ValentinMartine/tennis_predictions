"""Périmètre des données et classements manquants."""
import numpy as np
import pandas as pd

from src.export import _opt_int
from src.pipeline import _log1p_diff, _log_diff, build_features, load_matches


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


def test_upcoming_keeps_only_the_tour_own_tournament_ids(tmp_path, monkeypatch) -> None:
    """Les tableaux masculin et féminin d'un Grand Chelem partagent le nom, pas l'identifiant."""
    import json

    from src import export

    saison = pd.DataFrame({
        "ATP": [16866], "Tournament": ["U.S. Open - New York"], "Series": ["Grand Slam"],
        "Date": [pd.Timestamp("2026-09-01")],
    })
    raw = {"matches": [
        {"date": "2999-01-01", "time": "20:00", "tournament_id": 16865,
         "tournament": "U.S. Open - New York", "surface": "Hard", "round": "Quarterfinals",
         "player1": "Aryna Sabalenka", "player2": "Jessica Pegula"},
        {"date": "2999-01-01", "time": "21:30", "tournament_id": 16866,
         "tournament": "U.S. Open - New York", "surface": "Hard", "round": "Quarterfinals",
         "player1": "Karen Khachanov", "player2": "Alexander Blockx"},
    ]}
    target = tmp_path / "upcoming_raw.json"
    target.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(export, "UPCOMING_RAW", target)

    gardes = []
    monkeypatch.setattr(export, "upcoming_features",
                        lambda state, p1, p2, *a, **k: gardes.append((p1, p2)) or {f: 0.0 for f in export.FEATURES})

    class _Modele:
        def predict_proba(self, frame):
            return np.column_stack([np.full(len(frame), 0.4), np.full(len(frame), 0.6)])

    upcoming = export.build_upcoming("atp", saison, {"last_date": pd.Timestamp("2026-09-08")}, _Modele(), 1.0)

    assert gardes == [("Khachanov K.", "Blockx A.")]
    assert upcoming[0]["series"] == "Grand Slam"


def test_tournament_ids_compare_across_pandas_float_coercion() -> None:
    from src.export import _id_key

    assert _id_key(16866.0) == _id_key(16866) == "16866"
    assert _id_key("16866") == "16866"


def test_year_end_elo_does_not_freeze_inactive_players() -> None:
    rows = []
    for i in range(20):
        match = _match("Grand Slam", "Ancienne Star", f"Ancien Opposant {i}")
        match["Date"] = pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)
        rows.append(match)
    for i in range(20):
        match = _match("Grand Slam", "Joueur Actif", f"Opposant Actif {i}")
        match["Date"] = pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)
        rows.append(match)

    _, _, _, state = build_features(pd.DataFrame(rows))

    current_top = {entry["player"] for entry in state["year_end"][2026]}
    assert "Ancienne Star" not in current_top
    assert "Joueur Actif" in current_top
