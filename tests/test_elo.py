"""Invariants de l'Elo tennis, global et par surface.

Protège la normalisation des surfaces, la cohérence des probabilités, la
décroissance du K avec l'expérience et l'absence de fuite temporelle.
"""

import pytest

from src.elo import INITIAL, SURFACE_WEIGHT, SURFACES, TennisElo, norm_surface


class TestNormSurface:
    @pytest.mark.parametrize("surface", SURFACES)
    def test_les_surfaces_connues_sont_conservees(self, surface):
        assert norm_surface(surface) == surface

    def test_la_casse_est_normalisee(self):
        assert norm_surface("clay") == "Clay"
        assert norm_surface("  GRASS ") == "Grass"

    def test_les_surfaces_inconnues_retombent_sur_dur(self):
        # Carpet est rare et n'a pas de classement dédié.
        assert norm_surface("Carpet") == "Hard"
        assert norm_surface("") == "Hard"
        assert norm_surface(None) == "Hard"


class TestProbabilites:
    def test_deux_joueurs_neufs_sont_a_egalite(self):
        assert TennisElo().prob("a", "b", "Hard") == 0.5

    def test_les_probabilites_des_deux_joueurs_somment_a_un(self):
        elo = TennisElo()
        elo.rating["a"] = 1800.0
        elo.surf["Clay"]["a"] = 1900.0
        assert round(elo.prob("a", "b", "Clay") + elo.prob("b", "a", "Clay"), 9) == 1.0

    def test_la_surface_pese_selon_son_poids(self):
        # Un spécialiste de terre doit être mieux noté sur terre que sur dur.
        elo = TennisElo()
        elo.surf["Clay"]["a"] = INITIAL + 400.0
        assert elo.prob("a", "b", "Clay") > elo.prob("a", "b", "Hard")
        assert 0.0 < SURFACE_WEIGHT < 1.0

    def test_une_surface_inconnue_est_traitee_comme_du_dur(self):
        elo = TennisElo()
        elo.surf["Hard"]["a"] = INITIAL + 200.0
        assert elo.prob("a", "b", "Carpet") == elo.prob("a", "b", "Hard")


class TestMiseAJour:
    def test_le_vainqueur_monte_et_le_perdant_descend(self):
        elo = TennisElo()
        elo.update("a", "b", "Hard", "ATP250")
        assert elo.rating["a"] > INITIAL
        assert elo.rating["b"] < INITIAL

    def test_seule_la_surface_jouee_est_mise_a_jour(self):
        elo = TennisElo()
        elo.update("a", "b", "Clay", "ATP250")
        assert elo.surf["Clay"]["a"] > INITIAL
        assert elo.surf["Hard"]["a"] == INITIAL
        assert elo.surf["Grass"]["a"] == INITIAL

    def test_un_grand_chelem_pese_plus_qu_un_tournoi_ordinaire(self):
        chelem, ordinaire = TennisElo(), TennisElo()
        chelem.update("a", "b", "Hard", "Grand Slam")
        ordinaire.update("a", "b", "Hard", "ATP250")
        assert chelem.rating["a"] > ordinaire.rating["a"]

    def test_le_k_decroit_avec_l_experience(self):
        valeurs = [TennisElo._k(n) for n in (0, 10, 50, 200)]
        assert valeurs == sorted(valeurs, reverse=True)
        assert all(v > 0 for v in valeurs)

    def test_battre_un_favori_rapporte_plus_que_battre_un_outsider(self):
        surprise, attendu = TennisElo(), TennisElo()
        surprise.rating["b"] = INITIAL + 400.0
        attendu.rating["b"] = INITIAL - 400.0
        surprise.update("a", "b", "Hard", "ATP250")
        attendu.update("a", "b", "Hard", "ATP250")
        assert surprise.rating["a"] > attendu.rating["a"]

    def test_le_compteur_de_matchs_suit_les_deux_joueurs(self):
        elo = TennisElo()
        elo.update("a", "b", "Hard", "ATP250")
        assert elo.count["a"] == elo.count["b"] == 1
        assert elo.surf_count["Hard"]["a"] == 1
        assert elo.surf_count["Clay"]["a"] == 0


class TestAbsenceDeFuite:
    def test_la_probabilite_pre_match_ignore_les_matchs_posterieurs(self):
        elo = TennisElo()
        elo.update("a", "b", "Hard", "ATP250")
        avant = elo.prob("a", "b", "Hard")
        elo.update("a", "b", "Hard", "Grand Slam")

        rejeu = TennisElo()
        rejeu.update("a", "b", "Hard", "ATP250")
        assert rejeu.prob("a", "b", "Hard") == avant
