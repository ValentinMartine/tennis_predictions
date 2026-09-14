"""Modélisation hiérarchique du score au tennis au niveau du point.

Basé sur les travaux fondateurs de Barnett & Clarke (2005) et Kovalchik (2016) :
- Projection de la probabilité de gain d'un point au service vers le jeu de service.
- Modélisation du tie-break (7 ou 10 points).
- Modélisation du set (avec tie-break).
- Modélisation du match en fonction du format (Best of 3 vs Best of 5).
"""
from __future__ import annotations

import math


def prob_hold_game(p: float) -> float:
    """Probabilité de remporter un jeu de service standard avec règle de l'avantage (deuce).
    
    Formule fermée exacte pour une chaîne de Markov absorbante :
    g(p) = p^4 * (15 - 34p + 28p^2 - 8p^3) / (1 - 2p(1 - p))
    """
    p = max(1e-6, min(1.0 - 1e-6, float(p)))
    q = 1.0 - p
    # P(gagner à 40-0) = p^4
    # P(gagner à 40-15) = 4 * p^4 * q
    # P(gagner à 40-30) = 10 * p^4 * q^2
    win_before_deuce = p**4 * (1.0 + 4.0 * q + 10.0 * q**2)
    
    # P(deuce) = 20 * p^3 * q^3
    p_deuce = 20.0 * (p**3) * (q**3)
    
    # Depuis le deuce : p^2 / (p^2 + q^2)
    win_from_deuce = (p**2) / (p**2 + q**2)
    
    return float(win_before_deuce + p_deuce * win_from_deuce)


def prob_tiebreak(p_serve_a: float, p_serve_b: float, target_points: int = 7) -> float:
    """Probabilité que le joueur A gagne un tie-break.
    
    Chaque joueur servant en alternance, la probabilité moyenne que A gagne un point est :
    p = 0.5 * p_serve_a + 0.5 * (1 - p_serve_b).
    """
    p = 0.5 * float(p_serve_a) + 0.5 * (1.0 - float(p_serve_b))
    p = max(1e-6, min(1.0 - 1e-6, p))
    q = 1.0 - p
    
    n = target_points
    win_direct = 0.0
    for k in range(n - 1):
        comb = math.comb(n - 1 + k, k)
        win_direct += comb * (p**n) * (q**k)
        
    comb_deuce = math.comb(2 * (n - 1), n - 1)
    p_deuce = comb_deuce * (p**(n - 1)) * (q**(n - 1))
    win_from_deuce = (p**2) / (p**2 + q**2)
    
    return float(win_direct + p_deuce * win_from_deuce)


def prob_set(p_serve_a: float, p_serve_b: float, tiebreak: bool = True) -> float:
    """Probabilité que le joueur A remporte un set.
    
    g_a: probabilité que A tienne son service.
    g_b: probabilité que B tienne son service.
    """
    g_a = prob_hold_game(p_serve_a)
    g_b = prob_hold_game(p_serve_b)
    
    def _simulate_set(a_serves_first: bool) -> float:
        dp: dict[tuple[int, int], float] = {(0, 0): 1.0}
        win_a = 0.0
        
        for total_games in range(13):
            current_scores = [(i, total_games - i) for i in range(total_games + 1)]
            for i, j in current_scores:
                prob_state = dp.get((i, j), 0.0)
                if prob_state <= 0.0:
                    continue
                
                if i == 6 and j <= 4:
                    win_a += prob_state
                    continue
                if j == 6 and i <= 4:
                    continue
                if i == 7 and j == 5:
                    win_a += prob_state
                    continue
                if j == 7 and i == 5:
                    continue
                if i == 6 and j == 6:
                    if tiebreak:
                        p_tb = prob_tiebreak(p_serve_a, p_serve_b, 7)
                        win_a += prob_state * p_tb
                    continue
                
                server_is_a = (total_games % 2 == 0) if a_serves_first else (total_games % 2 == 1)
                p_win_game = g_a if server_is_a else (1.0 - g_b)
                
                dp[(i + 1, j)] = dp.get((i + 1, j), 0.0) + prob_state * p_win_game
                dp[(i, j + 1)] = dp.get((i, j + 1), 0.0) + prob_state * (1.0 - p_win_game)
                
        return win_a

    return 0.5 * _simulate_set(True) + 0.5 * _simulate_set(False)


def prob_match(p_serve_a: float, p_serve_b: float, best_of: int = 3) -> float:
    """Probabilité que le joueur A remporte le match selon le format (Bo3 vs Bo5)."""
    p_s = prob_set(p_serve_a, p_serve_b, tiebreak=True)
    q_s = 1.0 - p_s
    
    if best_of >= 5:
        return float(p_s**3 * (1.0 + 3.0 * q_s + 6.0 * (q_s**2)))
    
    return float(p_s**2 * (1.0 + 2.0 * q_s))


def match_prob_from_point_advantage(point_advantage: float, base_serve: float = 0.64, best_of: int = 3) -> float:
    """Calcule la probabilité de victoire à partir de l'avantage différentiel sur les points de service/retour."""
    adv = max(-0.25, min(0.25, float(point_advantage)))
    p_a = base_serve + adv
    p_b = base_serve - adv
    return prob_match(p_a, p_b, best_of=best_of)
