"""Elo tennis : classement global + par surface (Hard/Clay/Grass), mis à jour match par match.

- K décroissant avec l'expérience (formule FiveThirtyEight : K = 250/(m+5)^0.4), ce qui
  fait converger vite les nouveaux joueurs sans déstabiliser les vétérans.
- Multiplicateur par niveau de tournoi (Grand Chelem > Masters > ATP 250).
- La probabilité pré-match combine Elo global (70 %) et Elo de surface (30 %).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

INITIAL = 1500.0
SURFACES = ("Hard", "Clay", "Grass")
LEVEL_K = {"Grand Slam": 1.1, "Masters 1000": 1.0, "Masters Cup": 1.0}  # défaut 0.9
SURFACE_WEIGHT = 0.3


def norm_surface(s: str) -> str:
    s = str(s or "Hard").strip().title()
    return s if s in SURFACES else "Hard"  # Carpet (rare) -> Hard


class TennisElo:
    def __init__(self) -> None:
        self.rating: dict[str, float] = defaultdict(lambda: INITIAL)
        self.surf: dict[str, dict[str, float]] = {s: defaultdict(lambda: INITIAL) for s in SURFACES}
        self.count: dict[str, int] = defaultdict(int)
        self.surf_count: dict[str, dict[str, int]] = {s: defaultdict(int) for s in SURFACES}
        self.last_date: dict[str, Any] = {}

    @staticmethod
    def _expect(ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def get_effective_rating(self, player: str, current_date=None) -> float:
        """Calcule l'Elo effectif en appliquant un léger amortissement après inactivité prolongée (> 90 jours)."""
        base = self.rating[player]
        if current_date is None or player not in self.last_date or self.last_date[player] is None:
            return base
        try:
            delta_days = (current_date - self.last_date[player]).days
        except AttributeError:
            return base
        if delta_days > 90:
            excess_months = (delta_days - 90) / 30.0
            # Régression vers 1500 de 3 % par mois d'inactivité excédentaire, plafonnée à 100 points
            decay = max(-100.0, min(100.0, 0.03 * (base - INITIAL) * excess_months))
            return base - decay
        return base

    def prob(self, a: str, b: str, surface: str, date=None) -> float:
        """Probabilité que `a` batte `b` (blend global + surface), avec decay d'inactivité optionnel."""
        s = norm_surface(surface)
        ra = self.get_effective_rating(a, date)
        rb = self.get_effective_rating(b, date)
        pg = self._expect(ra, rb)
        ps = self._expect(self.surf[s][a], self.surf[s][b])
        return (1 - SURFACE_WEIGHT) * pg + SURFACE_WEIGHT * ps

    @staticmethod
    def _k(matches: int) -> float:
        return 250.0 / ((matches + 5) ** 0.4)

    def update(self, winner: str, loser: str, surface: str, series: str, date=None) -> None:
        s = norm_surface(surface)
        lvl = LEVEL_K.get(str(series), 0.9)
        # global
        e = self._expect(self.rating[winner], self.rating[loser])
        self.rating[winner] += lvl * self._k(self.count[winner]) * (1 - e)
        self.rating[loser] -= lvl * self._k(self.count[loser]) * (1 - e)
        self.count[winner] += 1
        self.count[loser] += 1
        # surface
        es = self._expect(self.surf[s][winner], self.surf[s][loser])
        self.surf[s][winner] += lvl * self._k(self.surf_count[s][winner]) * (1 - es)
        self.surf[s][loser] -= lvl * self._k(self.surf_count[s][loser]) * (1 - es)
        self.surf_count[s][winner] += 1
        self.surf_count[s][loser] += 1
        # suivi temporel
        if date is not None:
            self.last_date[winner] = date
            self.last_date[loser] = date

