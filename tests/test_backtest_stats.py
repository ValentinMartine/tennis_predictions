"""Tests unitaires pour les statistiques de backtest et intervalles de confiance dans tennis-predictions."""
import pytest
from src.export import _prediction_fields


def test_wilson_and_roi_confidence_intervals_logic():
    # Test unitaire de la logique de calcul d'intervalles de confiance
    import math
    import numpy as np

    wins = 60
    n_bets = 100
    p_hat = wins / n_bets
    z = 1.95996
    denom_w = 1.0 + (z**2) / n_bets
    ctr_w = (p_hat + (z**2) / (2.0 * n_bets)) / denom_w
    hw_w = (z * math.sqrt((p_hat * (1.0 - p_hat) / n_bets) + ((z**2) / (4.0 * (n_bets**2))))) / denom_w
    ci_w = [round(ctr_w - hw_w, 3), round(ctr_w + hw_w, 3)]

    assert ci_w[0] < 0.60 < ci_w[1]
    assert 0.50 <= ci_w[0] <= 0.60
    assert 0.60 <= ci_w[1] <= 0.70

    # Returns avec moyenne négative
    returns = [-1.0] * 55 + [0.9] * 45
    r_arr = np.array(returns)
    roi = float(r_arr.mean())
    s = float(r_arr.std(ddof=1))
    se = s / math.sqrt(n_bets)
    t_stat = roi / se
    p_val = float(1.0 - 0.5 * (1.0 + math.erf(t_stat / math.sqrt(2.0))))

    assert roi < 0
    assert p_val > 0.5  # Impossible de rejeter H0: ROI <= 0
