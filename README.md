# Prédictions au tennis (ATP)

Prédiction des matchs du circuit ATP : Elo global + par surface (K décroissant type
FiveThirtyEight), LightGBM sur variables sans fuite (classement, forme, activité, H2H,
format, surface), blend réglé sur validation temporelle (2023), test tenu à l'écart (2024+).

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
