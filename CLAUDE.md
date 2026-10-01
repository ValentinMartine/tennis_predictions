# tennis_predictions : repères projet

Prédiction des matchs **ATP et WTA**. Règles générales dans `~/.claude/CLAUDE.md`.

Tout est paramétré par circuit : `python scripts/download_data.py wta`,
`python scripts/train_and_backtest.py wta`, `python -m src.export wta`,
`python scripts/validate_exports.py wta`. Sans argument, le circuit est `atp`.
Les fichiers WTA portent le suffixe `_wta` ; ceux de l'ATP gardent leur nom d'origine.

## Chemins clés
- `scripts/download_data.py` : source primaire tennis-data.co.uk, secours Valuebetennis puis
  miroirs Sackmann. Contient la **passerelle de noms**, le **filtre de périmètre** et le
  dictionnaire `TOURS` (une source primaire et des miroirs par circuit).
- `scripts/fetch_upcoming.py` : calendrier des matchs à venir, tous circuits confondus,
  dans `data/upcoming_raw.json`. Le tri par circuit se fait à l'export.
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
- **Matchs à venir : tri par identifiant de tournoi, jamais par nom.** Les tableaux masculin
  et féminin d'un Grand Chelem portent le même nom et deux identifiants (16866 et 16865 pour
  l'US Open 2026). Le filtrage par nom faisait entrer les quarts de finale WTA dans l'export
  ATP, où les joueuses inconnues du modèle donnaient des probabilités à 50 %. Contrepartie
  assumée : un tournoi sans match terminé cette saison n'est pas encore reconnu.
- **Scores tronqués : réparer d'abord, signaler seulement si besoin.** Valuebetennis coupe parfois
  le dernier set. Le pipeline cherche alors le même match dans les miroirs Sackmann ; si le score y
  est complet, il remplace la version Valuebetennis. Sinon le match garde `ScoreStatus = partial`.
- **Le classement du site liste les joueurs actifs sur 12 mois glissants**, pas sur la fenêtre
  comparée. Exiger un match dans la fenêtre sortait le numéro un mondial dès qu'il ne jouait
  pas ce mois-là (Alcaraz premier en août 2026 parce que Sinner n'avait pas rejoué depuis
  Wimbledon). L'historique Elo exporte un compte de matchs par mois pour que l'inactivité
  se lise dans le tableau.
- `data/*.xlsx` est **gitignoré** : la CI retélécharge les 27 saisons à chaque exécution. Le corpus
  local (tennis-data en cache) et celui de la CI peuvent donc différer ; comparer `data/source_status.json`
  avant de conclure à une régression de modèle.

## Signaux d'actualité (Nimble, depuis le 01/10/2026)
- `scripts/signaux_actualite.py`, **local uniquement** (Ollama absent de la CI) : titres Google News
  des 7 derniers jours pour chaque joueur des matchs à venir, jugés par `nimble:9b-q4_K_M` via
  l'API System One (« blessé, malade, diminué, de retour de blessure ou incertain ? »).
- Sorties dans `data/actualite/` : `signaux.json` (lu par l'export, ignoré au-delà de 3 jours),
  `titres.jsonl` (archive datée) et `a_etiqueter.csv` (colonne `etiquette` à remplir oui/non).
- Le signal est **affiché** (`news1`, `news2`, badge sur le site dès p >= 0,5), **jamais une variable
  du modèle** : il n'y entrera qu'après calibration sur les étiquettes et gain de log-loss sur un test
  postérieur à la collecte. Pas d'historique antérieur au 01/10/2026 : pas de backtest possible avant.

## État
| Circuit | Matchs | Test 2024+ | Blend | Elo seul | Marché |
| --- | --- | --- | --- | --- | --- |
| ATP | 70 945 | 7 278 | 0,642 / 0,6240 | 0,641 / 0,6345 | 0,678 / 0,5937 |
| WTA | 65 397 | 6 412 | 0,650 / 0,6198 | 0,650 / 0,6232 | 0,679 / 0,5862 |

Le blend ne dépasse la baseline Elo que d'un cheveu, surtout côté WTA (poids ML 0,24), et **aucun
des deux circuits ne bat les cotes moyennes du marché** : ne pas prétendre le contraire. La
stratégie de mise reste négative hors échantillon.

Toute saison servie par le secours arrive **sans classement ni points ATP** : en local seule 2026 est
concernée (0 rang sur 1 884 matchs), en CI toutes les saisons depuis 2021. Les deux variables les plus
importantes du modèle sont donc neutralisées sur une part variable du test, et les deux environnements
ne donnent pas les mêmes chiffres. Rétablir tennis-data.co.uk (HTTP 503 persistant depuis juillet 2026)
est le principal levier restant.

## Usage
`pip install -r requirements.txt`, puis `scripts/download_data.py`, `scripts/train_and_backtest.py`,
`python -m src.export`, `scripts/validate_exports.py`.
