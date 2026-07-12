# Prédictions au tennis (ATP)

Prédiction des matchs du circuit ATP : Elo global + par surface (K décroissant type
FiveThirtyEight), LightGBM sur variables sans fuite (classement, forme, activité, H2H,
format, surface), blend réglé sur validation temporelle (2023), test tenu à l'écart (2024+).

## Features H2H
Les confrontations directes sont enrichies : volume H2H, H2H par surface, H2H en Grand Chelem et H2H récent. Les charges en sets/jeux dans le tournoi et sur 14 jours ont été testées mais laissées hors `FEATURES`, car elles dégradaient la log-loss hors échantillon.

Dernier backtest après régularisation LightGBM : blend ML+Elo `accuracy=0,6474`, `log-loss=0,6202` sur 6 905 matchs test (2024+). Cela améliore à la fois l'ancien export (`accuracy=0,6468`, `log-loss=0,6210`) et la baseline Elo (`accuracy=0,6404`, `log-loss=0,6316`).

La stratégie de mise reste négative hors échantillon. Les seuils choisis sur 2023 (`edge >= 20 %`, `confiance >= 70 %`) donnent `ROI=-1,88 %` sur 173 paris en test 2024+. Le modèle progresse, mais ne bat pas encore les cotes moyennes du marché.

## Données
- tennis-data.co.uk : saisons ATP 2000-2026, ~70 000 matchs (résultats, classements, cotes).
- Sofascore : résultats live des Grands Chelems en cours (patch local, IP CI bloquées).

## Historique Elo
La passe chronologique capture les trajectoires Elo mensuelles par joueur, le top de
chaque fin d'année, les pics de tous les temps et le palmarès des Grands Chelems
(`data/tennis_elo_history.json` + sections dédiées de `tennis_predictions.json`).

## Usage
```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/patch_results_from_sofascore.py   # local uniquement
python scripts/train_and_backtest.py
python -m src.export
python scripts/validate_exports.py
```

Sorties : `data/tennis_predictions.json` (classement, saison complète de matchs, métriques,
pics, n°1 par année, palmarès) et `data/tennis_elo_history.json` (trajectoires mensuelles),
consommées par valentinmartine.com/projets/tennis.

Le workflow GitHub Actions exécute aussi `validate_exports.py` et commit les deux exports JSON.
