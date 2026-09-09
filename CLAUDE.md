# tennis_predictions : repères projet

Prédiction des matchs ATP. Règles générales dans `~/.claude/CLAUDE.md`.

## Chemins clés
- `scripts/download_data.py` : source primaire tennis-data.co.uk, secours Valuebetennis puis
  miroirs Sackmann. Contient la **passerelle de noms** et le **filtre de périmètre**.
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
- Validation sur 2023, test tenu à l'écart sur 2024+ (7 189 matchs).
- **Périmètre : main tour ATP, tableau final.** Challenger, ITF, Coupe Davis et qualifications sont
  écartés au téléchargement (`MAIN_TOUR_SERIES`) et au chargement (`OUT_OF_SCOPE_SERIES`, filet pour
  les fichiers déjà en cache). Sans ce filtre, le secours Valuebetennis versait 246 000 matchs dont
  60 % d'ITF et l'accuracy affichée montait artificiellement à 0,68.
- **Toute source de secours passe par `short_name_from_full`.** Valuebetennis et Sackmann écrivent
  « Jannik Sinner » là où tennis-data écrit « Sinner J. » : sans conversion, chaque joueur repart à
  1500 et le classement Elo paraît figé. L'index exclut la saison en cours de reconstruction, sinon
  une conversion approximative se réinjecte et se perpétue (« Fritz T. » contre « Fritz T.H. »).
- **Classement et points ATP absents (cas du secours) : écart neutre, pas de faux rang.** Mesuré sur
  le test 2024+ : écart neutre log-loss 0,6233, NaN 0,6401, valeur commune 250 invisible dans l'export.
  Le champ `rank` est exporté à `None` et la colonne est masquée côté site.
- `data/*.xlsx` est **gitignoré** : la CI retélécharge les 27 saisons à chaque exécution. Le corpus
  local (tennis-data en cache) et celui de la CI peuvent donc différer ; comparer `data/source_status.json`
  avant de conclure à une régression de modèle.

## État
Blend ML+Elo : accuracy 0,6433, log-loss 0,6233 sur le test 2024+ (7 189 matchs), mieux que la baseline
Elo (0,6413 / 0,6341). **La stratégie de mise reste négative hors échantillon** : seuils choisis sur 2023
(`edge >= 3 %`, `confiance >= 75 %`) donnent ROI -1,12 % sur 439 paris, capital ¼ Kelly ×0,26. Le modèle
progresse mais ne bat pas les cotes moyennes du marché (0,679 / 0,5935) : ne pas prétendre le contraire.

Toute saison servie par le secours arrive **sans classement ni points ATP** : en local seule 2026 est
concernée (0 rang sur 1 884 matchs), en CI toutes les saisons depuis 2021. Les deux variables les plus
importantes du modèle sont donc neutralisées sur une part variable du test, et les deux environnements
ne donnent pas les mêmes chiffres. Rétablir tennis-data.co.uk (HTTP 503 persistant depuis juillet 2026)
est le principal levier restant.

## Usage
`pip install -r requirements.txt`, puis `scripts/download_data.py`, `scripts/train_and_backtest.py`,
`python -m src.export`, `scripts/validate_exports.py`.
