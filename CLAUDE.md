# tennis_predictions : repères projet

Prédiction des matchs ATP. Règles générales dans `~/.claude/CLAUDE.md`.

## Chemins clés
- `src/elo.py` : Elo global et **par surface**, K décroissant type FiveThirtyEight.
- `src/pipeline.py` : features sans fuite : classement, forme, activité, H2H (volume, par surface,
  en Grand Chelem, récent), format, surface.
- `src/sackmann_stats.py` : stats service/retour depuis `data/sackmann/`, chronologiques et sans fuite.
  Si le cache est absent, ces features restent **neutres** et `sackmann_stats_active=false` dans l'export.
- `src/export.py` : `data/tennis_predictions.json` et `data/tennis_elo_history.json`.
- `scripts/patch_results_from_sofascore.py` : **local uniquement**, les IP de CI sont bloquées.

## Décisions déjà prises, ne pas les refaire
- Les charges en sets/jeux (tournoi et 14 jours) **ont été testées et écartées** : elles dégradaient
  la log-loss hors échantillon. Ne pas les réintroduire sans preuve sur le test 2024+.
- Validation sur 2023, test tenu à l'écart sur 2024+ (6 905 matchs).

## État
Blend ML+Elo : accuracy 0,6474, log-loss 0,6202 sur le test 2024+, mieux que la baseline Elo
(0,6404 / 0,6316). **La stratégie de mise reste négative hors échantillon** : seuils choisis sur 2023
(`edge >= 20 %`, `confiance >= 70 %`) donnent ROI -1,88 % sur 173 paris. Le modèle progresse mais ne bat
pas les cotes moyennes du marché : ne pas prétendre le contraire.

## Usage
`pip install -r requirements.txt`, puis `scripts/download_data.py`, `scripts/train_and_backtest.py`,
`python -m src.export`, `scripts/validate_exports.py`.
