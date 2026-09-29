import importlib.util
from pathlib import Path

from src.export import round_order

_spec = importlib.util.spec_from_file_location(
    "validate_exports", Path(__file__).resolve().parents[1] / "scripts" / "validate_exports.py")
validate_exports = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validate_exports)


def test_phase_de_poules_ordonnee_comme_le_round_robin():
    # Le 26 et le 27/09/2026, la Laver Cup (« poules » chez Valuebetennis) a fait échouer la régénération.
    assert round_order("poules") == round_order("Round Robin") == 50
    assert validate_exports._round_order({"round": "Poules"}) == 50


def test_casse_indifferente():
    assert round_order("the final") == 700
