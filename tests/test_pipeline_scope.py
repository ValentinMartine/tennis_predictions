"""Périmètre des données et classements manquants."""
import numpy as np
import pandas as pd

from src.export import _best_market_blend_weight, _opt_int, _prediction_fields, _text_or_empty
from src.modeling import fit_probability_calibrator, probability_metrics
from src.pipeline import FEATURES, _log1p_diff, _log_diff, _round_flags, build_features, load_matches


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


def test_wta_results_patch_replaces_stale_rows_and_adds_missing_matches(tmp_path, monkeypatch) -> None:
    stale = _match("Grand Slam", "Sabalenka A.", "Noskova L.")
    stale.update({
        "ATP": 16865,
        "Tournament": "U.S. Open - New York",
        "Date": pd.Timestamp("2026-09-08"),
        "Round": "Quarterfinals",
        "W1": 6,
        "L1": 3,
        "W2": 6,
        "L2": 4,
        "W3": np.nan,
        "L3": np.nan,
    })
    current = _match("Grand Slam", "Gauff C.", "Andreeva M.")
    current.update({
        "ATP": 16865,
        "Tournament": "U.S. Open - New York",
        "Date": pd.Timestamp("2026-09-09"),
        "Round": "Quarterfinals",
    })
    pd.DataFrame([stale, current]).to_parquet(tmp_path / "seasons_wta.parquet")

    fixed = _match("Grand Slam", "Sabalenka A.", "Noskova L.")
    fixed.update({
        "ATP": 16865,
        "Tournament": "U.S. Open - New York",
        "Date": pd.Timestamp("2026-09-08"),
        "Round": "Quarterfinals",
        "W1": 7,
        "L1": 6,
        "W2": 3,
        "L2": 6,
        "W3": 7,
        "L3": 6,
    })
    missing = _match("Grand Slam", "Pegula J.", "Navarro E.")
    missing.update({
        "ATP": 16865,
        "Tournament": "U.S. Open - New York",
        "Date": pd.Timestamp("2026-09-08"),
        "Round": "Quarterfinals",
        "W1": 3,
        "L1": 6,
        "W2": 6,
        "L2": 4,
        "W3": 6,
        "L3": 3,
    })
    pd.DataFrame([fixed, missing]).to_csv(tmp_path / "results_patch_wta.csv", index=False)
    monkeypatch.setattr("src.pipeline.DATA", tmp_path)

    kept = load_matches("wta")
    pairs = set(zip(kept["Winner"], kept["Loser"]))
    sabalenka = kept[(kept["Winner"] == "Sabalenka A.") & (kept["Loser"] == "Noskova L.")]

    assert len(sabalenka) == 1
    assert sabalenka.iloc[0]["W1"] == 7
    assert ("Pegula J.", "Navarro E.") in pairs


def test_wta_results_patch_is_loaded_after_cached_source_lag(tmp_path, monkeypatch) -> None:
    base = _match("Grand Slam", "Gauff C.", "Andreeva M.")
    base.update({
        "ATP": 16865,
        "Tournament": "U.S. Open - New York",
        "Date": pd.Timestamp("2026-09-09"),
        "Round": "Quarterfinals",
    })
    pd.DataFrame([base]).to_parquet(tmp_path / "seasons_wta.parquet")

    patch = _match("Grand Slam", "Sabalenka A.", "Pegula J.")
    patch.update({
        "ATP": 16865,
        "Tournament": "U.S. Open - New York",
        "Date": pd.Timestamp("2026-09-10"),
        "Round": "Semifinals",
        "WRank": 1,
        "LRank": 3,
    })
    pd.DataFrame([patch]).to_csv(tmp_path / "results_patch_wta.csv", index=False)
    monkeypatch.setattr("src.pipeline.DATA", tmp_path)

    kept = load_matches("wta")

    assert kept["Date"].max().date().isoformat() == "2026-09-10"
    assert ("Sabalenka A.", "Pegula J.") in set(zip(kept["Winner"], kept["Loser"]))


def test_unknown_rank_gives_a_neutral_gap_not_a_fake_ranking() -> None:
    assert _log_diff(120.0, 30.0) > 0
    assert _log_diff(float("nan"), 30.0) == 0.0
    assert _log1p_diff(2000.0, float("nan")) == 0.0


def test_optional_rank_export_rejects_nan_and_zero() -> None:
    assert _opt_int(42.0) == 42
    assert _opt_int(float("nan")) is None
    assert _opt_int(0) is None
    assert _opt_int(None) is None


def test_optional_text_export_handles_pandas_missing_values() -> None:
    assert _text_or_empty(pd.NA) == ""
    assert _text_or_empty(np.nan) == ""
    assert _text_or_empty("partial") == "partial"


def test_market_blend_weight_prefers_better_calibrated_market() -> None:
    y = np.array([1, 0, 1, 0])
    weak_model = np.array([0.55, 0.45, 0.56, 0.44])
    strong_market = np.array([0.9, 0.1, 0.85, 0.15])
    mask = np.array([True, True, True, True])

    assert _best_market_blend_weight(y, weak_model, strong_market, mask) > 0.5


def test_prediction_fields_keep_independent_model_probability_visible() -> None:
    fields = _prediction_fields("Zverev A.", "Zandschulp B.V.D.", 0.56, 0.86, 1.0)

    assert fields["predicted_player"] == "Zverev A."
    assert fields["predicted_prob"] == 0.56
    assert fields["prediction_source"] == "independent_calibrated"
    assert fields["model_predicted_prob"] == 0.56
    assert fields["market_calibrated_p1_win"] == 0.86
    assert fields["market_model_gap"] == 0.3


def test_feature_list_includes_fatigue_and_context_features() -> None:
    assert "d_tourn_sets" in FEATURES
    assert "d_recent_games" in FEATURES
    assert "d_rest_days" in FEATURES
    assert "d_surface_form10" in FEATURES
    assert "d_season_winrate" in FEATURES
    assert "d_win_streak" in FEATURES
    assert "d_retirement_365" in FEATURES
    assert "rank_both_known" in FEATURES
    assert "level_grand_slam" in FEATURES
    assert "round_semifinal" in FEATURES


def test_quarterfinals_are_not_encoded_as_finals() -> None:
    flags = _round_flags("Quarterfinals")

    assert flags["round_quarterfinal"] == 1.0
    assert flags["round_semifinal"] == 0.0
    assert flags["round_final"] == 0.0


def test_features_reuse_last_known_rank_when_current_row_missing() -> None:
    first = _match("ATP", "Alpha", "Beta")
    first.update({"Date": pd.Timestamp("2026-01-01"), "WRank": 10, "LRank": 50, "WPts": 3000, "LPts": 900})
    second = _match("ATP", "Alpha", "Beta")
    second.update({"Date": pd.Timestamp("2026-01-03"), "WRank": np.nan, "LRank": np.nan, "WPts": np.nan, "LPts": np.nan})

    X, _, metas, _ = build_features(pd.DataFrame([first, second]))

    assert X.iloc[1]["rank_both_known"] == 1.0
    assert metas[1]["wrank"] != metas[1]["wrank"]
    assert metas[1]["wrank_feature"] == 10.0


def test_features_fallback_to_internal_elo_rank_when_official_rank_is_absent() -> None:
    rows = []
    for i in range(20):
        match = _match("ATP", "Alpha", f"Alpha Opponent {i}")
        match.update({"Date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i), "WRank": np.nan, "LRank": np.nan})
        rows.append(match)
    for i in range(20):
        match = _match("ATP", "Beta", f"Beta Opponent {i}")
        match.update({"Date": pd.Timestamp("2025-02-01") + pd.Timedelta(days=i), "WRank": np.nan, "LRank": np.nan})
        rows.append(match)
    target = _match("ATP", "Alpha", "Beta")
    target.update({"Date": pd.Timestamp("2025-03-15"), "WRank": np.nan, "LRank": np.nan})
    rows.append(target)

    X, _, metas, _ = build_features(pd.DataFrame(rows))

    assert X.iloc[-1]["rank_both_known"] == 1.0
    assert metas[-1]["wrank_feature"] == 1.0
    assert metas[-1]["lrank_feature"] == 2.0


def test_recent_retirement_proxy_is_pre_match_only() -> None:
    retired_loss = _match("ATP", "Beta", "Alpha")
    retired_loss.update({"Date": pd.Timestamp("2026-01-01"), "Comment": "Retired"})
    next_match = _match("ATP", "Beta", "Alpha")
    next_match.update({"Date": pd.Timestamp("2026-01-20"), "Comment": "Completed"})

    X, _, metas, _ = build_features(pd.DataFrame([retired_loss, next_match]))
    idx = 1 if metas[1]["p1"] == "Alpha" else -1

    assert X.iloc[0]["d_recent_retirement"] == 0.0
    assert idx * X.iloc[1]["d_recent_retirement"] > 0


def test_probability_calibrator_improves_miscalibrated_scale() -> None:
    y = np.array([0] * 50 + [1] * 50)
    raw = np.array([0.30] * 25 + [0.40] * 25 + [0.60] * 25 + [0.70] * 25)

    calibrator = fit_probability_calibrator(y, raw)
    calibrated = calibrator.predict(raw)

    assert probability_metrics(y, calibrated)["log_loss"] <= probability_metrics(y, raw)["log_loss"]


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
