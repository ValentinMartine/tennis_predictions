# Prédictions au tennis (ATP)

Prédiction des matchs du circuit ATP : Elo global + par surface (K décroissant type
FiveThirtyEight), LightGBM sur variables sans fuite (classement, forme, activité, H2H,
format, surface), blend réglé sur validation temporelle (2023), test tenu à l'écart (2024+).

## Données
- tennis-data.co.uk : saisons ATP 2010-2026 (résultats, classements, cotes).
- Sofascore : résultats live des Grands Chelems en cours (patch local, IP CI bloquées).

## Usage
```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/patch_results_from_sofascore.py   # local uniquement
python scripts/train_and_backtest.py
python -m src.export
```

Sortie : `data/tennis_predictions.json`, consommé par valentinmartine.com/projets/tennis.
