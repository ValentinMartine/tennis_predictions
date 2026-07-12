# Prédictions au tennis (ATP)

Prédiction des matchs du circuit ATP : Elo global + par surface (K décroissant type
FiveThirtyEight), LightGBM sur variables sans fuite (classement, forme, activité, H2H,
format, surface), blend réglé sur validation temporelle (2023), test tenu à l'écart (2024+).

## Features H2H
Les confrontations directes sont enrichies : volume H2H, H2H par surface, H2H en Grand Chelem et H2H récent. Les charges en sets/jeux dans le tournoi et sur 14 jours ont été testées mais laissées hors `FEATURES`, car elles dégradaient la log-loss hors échantillon.

Dernier backtest après ajout H2H fin : blend ML+Elo `accuracy=0,6468`, `log-loss=0,6210` sur 6 905 matchs test (2024+). Cela améliore légèrement le choix du vainqueur par rapport à l'ancien export (`0,6459`) mais dégrade très légèrement la calibration (`0,6205` -> `0,6210`).

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
```

Sorties : `data/tennis_predictions.json` (classement, saison complète de matchs, métriques,
pics, n°1 par année, palmarès) et `data/tennis_elo_history.json` (trajectoires mensuelles),
consommées par valentinmartine.com/projets/tennis.
