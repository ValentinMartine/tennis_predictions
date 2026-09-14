"""Tests unitaires pour la modélisation hiérarchique du tennis (markov.py)."""
import pytest
from src.markov import match_prob_from_point_advantage, prob_hold_game, prob_match, prob_set, prob_tiebreak


class TestMarkovTennis:
    def test_prob_hold_game_symmetry(self):
        # À 50 % de points gagnés au service, la probabilité de gagner le jeu doit être 0.5
        assert round(prob_hold_game(0.5), 6) == 0.5

    def test_prob_hold_game_monotonicity(self):
        probs = [prob_hold_game(p) for p in (0.4, 0.5, 0.6, 0.65, 0.7)]
        assert probs == sorted(probs)
        assert probs[0] < 0.5
        assert probs[-1] > 0.8

    def test_prob_tiebreak_symmetry(self):
        # Deux joueurs de niveau de service identique ont 50 % de chance au tie-break
        assert round(prob_tiebreak(0.65, 0.65, target_points=7), 6) == 0.5

    def test_prob_tiebreak_advantage(self):
        # Si A est supérieur au service, sa probabilité au tie-break augmente
        assert prob_tiebreak(0.70, 0.60, target_points=7) > 0.6

    def test_prob_set_symmetry(self):
        # Deux serveurs identiques ont 50 % de chance de gagner le set
        assert round(prob_set(0.64, 0.64), 5) == 0.5

    def test_prob_match_dilation_best_of_five(self):
        # Propriété fondamentale : si un joueur est favori (p_set > 0.5),
        # sa probabilité de victoire est plus élevée en Best of 5 qu'en Best of 3
        p_bo3 = prob_match(0.68, 0.62, best_of=3)
        p_bo5 = prob_match(0.68, 0.62, best_of=5)
        assert p_bo3 > 0.5
        assert p_bo5 > p_bo3

    def test_match_prob_from_point_advantage(self):
        assert round(match_prob_from_point_advantage(0.0, best_of=3), 5) == 0.5
        assert match_prob_from_point_advantage(0.05, best_of=3) > 0.65
        assert match_prob_from_point_advantage(-0.05, best_of=3) < 0.35
