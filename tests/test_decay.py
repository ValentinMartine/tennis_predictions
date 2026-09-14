"""Tests unitaires pour le rating decay d'inactivité dans TennisElo."""
from datetime import datetime, timedelta
import pytest
from src.elo import INITIAL, TennisElo


class TestInactivityDecay:
    def test_aucun_decay_sans_date(self):
        elo = TennisElo()
        elo.rating["a"] = 1800.0
        assert elo.get_effective_rating("a") == 1800.0

    def test_aucun_decay_sous_quatre_vingt_dix_jours(self):
        elo = TennisElo()
        elo.rating["a"] = 1800.0
        d0 = datetime(2026, 1, 1)
        elo.update("a", "b", "Hard", "ATP250", date=d0)
        post_match_rating = elo.rating["a"]
        
        # 60 jours après : aucun decay par rapport au rating post-match
        d1 = d0 + timedelta(days=60)
        assert elo.get_effective_rating("a", current_date=d1) == post_match_rating

    def test_decay_progressif_apres_quatre_vingt_dix_jours(self):
        elo = TennisElo()
        elo.rating["a"] = 1800.0
        d0 = datetime(2026, 1, 1)
        elo.update("a", "b", "Hard", "ATP250", date=d0)
        post_match_rating = elo.rating["a"]
        
        # 180 jours après (3 mois d'excès) : l'Elo effectif doit régresser vers 1500
        d2 = d0 + timedelta(days=180)
        eff = elo.get_effective_rating("a", current_date=d2)
        assert eff < post_match_rating
        assert eff > 1500.0
        
        # Le rating de base stocké n'est pas écrasé
        assert elo.rating["a"] == post_match_rating

    def test_decay_joueur_sous_la_moyenne_remonte_vers_mille_cinq_cents(self):
        elo = TennisElo()
        elo.rating["faible"] = 1300.0
        d0 = datetime(2026, 1, 1)
        elo.update("b", "faible", "Hard", "ATP250", date=d0)
        post_match_rating = elo.rating["faible"]
        
        d2 = d0 + timedelta(days=180)
        eff = elo.get_effective_rating("faible", current_date=d2)
        # Un joueur mal classé inactif régresse aussi vers 1500 (remonte par rapport à son creux)
        assert eff > post_match_rating
        assert eff <= INITIAL

    def test_prob_integre_le_decay(self):
        elo = TennisElo()
        elo.rating["a"] = 1800.0
        elo.rating["b"] = 1500.0
        d0 = datetime(2026, 1, 1)
        elo.update("a", "b", "Hard", "ATP250", date=d0)
        
        p_fraiche = elo.prob("a", "b", "Hard", date=d0 + timedelta(days=10))
        p_rouille = elo.prob("a", "b", "Hard", date=d0 + timedelta(days=200))
        assert p_rouille < p_fraiche
