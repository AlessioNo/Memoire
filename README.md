# gkx_project — pipeline de prédiction de rendements (Gu, Kelly & Xiu, 2020)

## Structure du projet

```
gkx_project/
│
├── data/
│   ├── raw/            # 3 fichiers originaux (datashare.parquet, StockReturn.parquet, MacroData.parquet)
│   ├── interim/         # fichiers nettoyés (un par source), produits par le notebook 02
│   └── processed/        # panel fusionné puis prêt pour la modélisation, notebook 03
│
├── notebooks/
│   ├── 00_lancer_pipeline.ipynb            # orchestration automatique -- "le bouton", voir plus bas
│   ├── 01_exploration.ipynb
│   ├── 02_nettoyage_donnees.ipynb
│   ├── 03_construction_panel.ipynb
│   ├── 04_modele_lineaire.ipynb
│   ├── 05_modele_elastic_net.ipynb
│   ├── 06_modele_lightgbm.ipynb
│   ├── 07_evaluation_portefeuilles.ipynb
│   └── 08_comparaison_experiences.ipynb    # voir plus bas
│
├── config.py
├── fenetres.py            # fenêtres d'entraînement glissantes/extensives
├── journal.py              # journal des expériences, voir plus bas
├── pipeline.py             # orchestration automatique 02→03→04/05/06→07, voir plus bas
├── utils.py                 # petits utilitaires partagés par journal.py et pipeline.py
├── modeles/              # modèles entraînés sauvegardés (.joblib)
└── outputs/              # tableaux de résultats, prédictions, graphiques, journal des expériences,
                            # état du pipeline automatisé (etat_pipeline.json)
```

## Comparaison des expériences : `journal.py` + notebook 08

**Le problème que ça résout** : les notebooks 04/05/06 **écrasent** leurs fichiers de
résultats (`outputs/resultats_*.parquet`) à chaque exécution — un seul jeu de résultats à
la fois, celui du dernier lancement (voulu : le notebook 07 doit toujours pouvoir lire
"le" dernier modèle entraîné, sans ambiguïté). Mais ça veut dire qu'on ne peut **pas**
comparer plusieurs lancements entre eux une fois passé au suivant (ex: `expanding` vs
`rolling`, ou deux grilles d'hyperparamètres différentes pour l'Elastic Net) — le premier
résultat est perdu.

`journal.py` ajoute donc un second fichier, `outputs/journal_experiences.parquet`, qui
n'est **jamais écrasé** : chaque exécution de 04/05/06 y **ajoute une ligne** (sans jamais
retoucher aux lignes précédentes), sauf si une expérience strictement identique (même
modèle, mêmes paramètres) y figure déjà — relancer deux fois exactement le même modèle
avec les mêmes paramètres ne crée donc jamais de doublon.

Deux familles de paramètres sont enregistrées pour chaque expérience (voir `config.py`,
sections *"Parametres GENERAUX"* / *"Parametres SPECIFIQUES"*) :
- **Généraux** : prédicteurs choisis, mode de fenêtre (`expanding`/`rolling`), tailles de
  fenêtres, seuils de filtrage de l'univers investissable (`mvel1`, `ill`) — les mêmes
  pour les 3 modèles.
- **Spécifiques** : propres à un seul modèle (ex: `grille_alpha`/`grille_l1_ratio` pour
  l'Elastic Net ; `grille_num_leaves`/`grille_learning_rate`/... pour LightGBM ; aucun
  pour la régression linéaire, un simple OLS).

Le notebook **08** lit ce journal et affiche **un tableau par (modèle, paramètres
spécifiques)** : deux lancements avec les mêmes hyperparamètres spécifiques mais des
paramètres généraux différents tombent dans le **même** tableau (une ligne chacun,
colonnes = paramètres généraux + durée d'entraînement + R²_oos) ; deux lancements avec des
hyperparamètres spécifiques différents donnent **deux tableaux distincts**. Il ajoute
aussi quelques graphiques (classement de toutes les expériences, compromis
précision/temps de calcul, effet des paramètres généraux) et fait ressortir la meilleure
expérience de chaque modèle.

**Comment s'en servir** : modifie un paramètre dans `config.py`, relance le(s)
notebook(s) concerné(s) (04, 05 et/ou 06 — et 03 en plus si tu as changé un seuil de
filtrage), reviens exécuter le notebook 08. Répète avec d'autres paramètres pour enrichir
la comparaison au fil du temps.

ℹ️ **Chaque tableau du notebook 08 inclut aussi les mesures de portefeuille
long-short du notebook 07** (ratio de Sharpe, de Sortino, rendement/volatilité annualisés,
drawdown maximum, t-stat, % de mois positifs). Le **classement** de chaque tableau reste
basé sur le `R²_oos` (jamais sur ces mesures) — mais chaque ligne les affiche aussi,
**dès qu'elles ont été calculées au moins une fois**. Le mécanisme : le
notebook 07 tague chaque modèle avec la même `cle_experience` que celle déjà écrite par
04/05/06 dans `outputs/resultats_*.parquet` (voir `journal.cle_experience_actuelle`), et
**ajoute** (sans jamais rien écraser) ses mesures à
`outputs/historique_performance_portefeuilles.parquet` -- un fichier cumulatif dédupliqué
par `cle_experience`, au même titre que `outputs/journal_experiences.parquet`, que le
notebook 08 fusionne avec le journal (voir `journal.tableaux_par_modele`). Une expérience
GARDE donc ses mesures de portefeuille même après avoir été supplantée par un lancement
plus récent du même modèle : `NaN` uniquement pour les expériences que 07 n'a **encore
jamais** évaluées.

Le notebook **07**, lui, reste la référence pour le détail complet de l'évaluation
économique (portefeuilles long-short, graphiques par décile, richesse cumulée, Sharpe,
drawdown...) du **dernier** modèle entraîné de chaque type — le notebook 08 n'en reprend
qu'un résumé chiffré par expérience, à côté du R²_oos.

## Lancer automatiquement le pipeline : notebook 00 + `pipeline.py`

**Le problème que ça résout** : selon *quel* paramètre de `config.py` on change, il faut
relancer un sous-ensemble différent de notebooks -- changer `TYPE_FENETRE` ne demande de
relancer que 04/05/06 (et 07 en chaîne, puisqu'il dépend des 3) ; changer un seuil de
filtrage (`SEUIL_PERCENTILE_TAILLE`) demande de relancer 03 *puis* 04/05/06 *puis* 07 ;
changer `CARACTERISTIQUES` ou `SEUIL_MAX_PCT_MANQUANT_CARACTERISTIQUES` (voir *"Filtrage
automatique des caractéristiques par valeurs manquantes"* plus bas) demande de tout
relancer depuis 02. Le notebook 00 (`00_lancer_pipeline.ipynb`) automatise cette
détection et l'exécution : c'est le "bouton" — modifie `config.py`, ouvre et exécute ce
notebook, il détermine tout seul l'ensemble **nécessaire et suffisant** de notebooks à
relancer (parmi 02 à 07), dans le bon ordre, et les exécute avec
[papermill](https://papermill.readthedocs.io/).

**Comment ça marche** (implémenté dans `pipeline.py`) :
- Un **graphe de dépendances** entre notebooks : `02 → 03 → {04, 05, 06} → 07`.
- Une liste des **paramètres dont chaque notebook dépend directement** (ex: 03 dépend de
  `SEUIL_PERCENTILE_TAILLE`/`SEUIL_PERCENTILE_LIQUIDITE` ; 06 dépend en plus des grilles
  LightGBM ; 07 dépend de `NB_DECILES`).
- Un **état sauvegardé** (`outputs/etat_pipeline.json`) : pour chaque notebook, les
  valeurs de ses paramètres et l'horodatage de son dernier lancement réussi *via ce
  système*.
- Un notebook est **"sale"** (à re-exécuter) si l'un de ses propres paramètres a changé
  depuis son dernier lancement, **ou** si un notebook dont il dépend est lui-même sale --
  la saleté se propage vers l'aval de la chaîne (ex: changer un seuil de filtrage rend 03
  sale, ce qui rend 04/05/06 sales aussi, même si aucun de LEURS propres paramètres n'a
  changé -- c'est leur entrée qui a changé ; et 07 devient sale à son tour, en chaîne
  derrière eux).

⚠️ **Le notebook 07 fait partie de cette automatisation par défaut** (il se
relance dès que l'un des 3 modèles, ou `NB_DECILES`, change) -- mais comme il dépend des
**TROIS** modèles à la fois, si tu restreins volontairement les `CIBLES` du notebook 00 à
un seul modèle (ex: `('06_modele_lightgbm',)`), retire aussi
`'07_evaluation_portefeuilles'` de `CIBLES`, sans quoi les 2 autres modèles seraient
entraînés quand même juste pour le satisfaire (relance-le à la main séparément dans ce
cas). **Le notebook 08**, lui, est relancé par défaut à la fin (`inclure_08=True`), en
dehors de cette détection : il ne fait que lire le journal des expériences et
l'historique cumulatif `outputs/historique_performance_portefeuilles.parquet` alimenté par
07, donc toujours sans risque.

⚠️ **`pipeline.py` ne voit pas les lancements manuels.** L'état
(`outputs/etat_pipeline.json`) n'est mis à jour QUE par `executer_pipeline` -- si tu
lances un notebook toi-même (Run All dans Jupyter) plutôt que via le notebook 00, il sera
considéré "sale" au prochain passage et relancé inutilement (perte de temps, mais rien de
faux : le journal des expériences ne crée jamais de doublon). Pour éviter ça après un
lancement manuel, synchronise l'état sans rien ré-exécuter :
`pipeline.marquer_a_jour('04_modele_lineaire', '05_modele_elastic_net', '06_modele_lightgbm', '07_evaluation_portefeuilles')`.

**Prérequis** : `pip install papermill` (voir aussi la liste de dépendances plus bas).

⚠️ **Si le projet est dans un dossier synchronisé (OneDrive, Google Drive, Dropbox...)**,
tu peux voir une erreur `PermissionError: [Errno 13] Permission denied` en cours
d'exécution : le service de synchronisation verrouille brièvement le fichier `.ipynb`
pile au moment où papermill essaie de le sauvegarder. `pipeline.py` limite déjà ce risque
au minimum (une seule sauvegarde par notebook, à la toute fin, plutôt qu'après chaque
cellule) et réessaie automatiquement 3 fois en cas de verrou passager -- si l'erreur
persiste malgré tout, ferme le notebook concerné dans Jupyter/VS Code s'il est ouvert
ailleurs, mets la synchronisation en pause le temps du lancement, ou (plus radical mais
plus fiable) travaille dans un dossier local non synchronisé. Les notebooks déjà terminés
avec succès avant l'erreur restent enregistrés comme à jour -- pas besoin de tout
relancer, juste de relancer le notebook 00.

## `config.py` : paramètres généraux vs spécifiques

`config.py` distingue explicitement, avec un bandeau dans le fichier :
- Les **paramètres GÉNÉRAUX** (`PREDICTEURS`, `TYPE_FENETRE`, `ANNEES_*`,
  `SEUIL_PERCENTILE_TAILLE`, `SEUIL_PERCENTILE_LIQUIDITE`,
  `SEUIL_MAX_PCT_MANQUANT_CARACTERISTIQUES`) — communs aux 3 modèles.
  Les deux seuils de filtrage de l'univers investissable (taille `mvel1`, liquidité `ill`)
  sont ici, comme le reste des paramètres partagés.
- Les **paramètres SPÉCIFIQUES** à un modèle (`GRILLE_ALPHA_ELASTIC_NET`,
  `GRILLE_L1_RATIO_ELASTIC_NET`, `MAX_ITER_ELASTIC_NET` pour le notebook 05 ;
  `GRILLE_NUM_LEAVES_LIGHTGBM`, `GRILLE_LEARNING_RATE_LIGHTGBM`,
  `GRILLE_MIN_CHILD_SAMPLES_LIGHTGBM`, `N_ESTIMATORS_LIGHTGBM`, `STOPPING_ROUNDS_LIGHTGBM`
  pour le notebook 06).

Cette distinction correspond exactement à celle utilisée par `journal.py` pour regrouper
les expériences au notebook 08.

`CARACTERISTIQUES` est une simple liste à plat des 94 noms candidats ; la catégorie de
chaque caractéristique (taille, valeur, momentum...) n'est utilisée nulle part dans le
code et reste seulement indiquée en commentaire à côté de chaque nom, à titre indicatif.

## Filtrage automatique des caractéristiques par valeurs manquantes

`config.CARACTERISTIQUES` contient les **94** caractéristiques candidates de Gu, Kelly &
Xiu (2020) au complet — c'est l'univers **candidat**, pas la liste finale utilisée pour la
modélisation.

**Comment le sous-ensemble final est obtenu** : le notebook 02 (partie A, section
**A.3bis**) calcule le taux de valeurs manquantes de chacune des 94 candidates sur la
période retenue (`annee >= ANNEE_DEBUT`), et exclut automatiquement celles au-dessus de
`config.SEUIL_MAX_PCT_MANQUANT_CARACTERISTIQUES` (30 % par défaut) — certaines
caractéristiques (ex : données de R&D, de dette sécurisée...) restent très incomplètes
même après 1980, et les imputer quand même par la médiane du mois reviendrait à remplacer
une grande partie de leurs valeurs par une constante, sans apporter de signal exploitable.

La liste retenue (avec le taux de missing de chaque candidate, gardée ou non) est
sauvegardée dans `data/interim/caracteristiques_retenues.json`. `config.py` expose
`CARACTERISTIQUES_RETENUES` (et `charger_caracteristiques_retenues()`), calculée
automatiquement à partir de ce fichier — c'est cette liste, **pas** `CARACTERISTIQUES`, que
réutilisent le notebook 03 (partie B) et les notebooks 04 à 06 pour construire le panel de
modélisation et `PREDICTEURS` par défaut. Repli automatique sur l'univers candidat complet
si ce fichier n'existe pas encore (avant la toute première exécution du notebook 02), pour
que `import config` ne casse jamais.

## Fenêtres d'entraînement glissantes / extensives

À la Gu, Kelly & Xiu (2020), chaque modèle (04/05/06) est **ré-entraîné plusieurs fois**,
sur une succession de fenêtres qui avancent dans le temps (ré-entraînement annuel par
défaut) :

```
fenetre 0 : train [1980-1997] | validation [1998-2009] | test [2010]
fenetre 1 : train [1980-1998] | validation [1999-2010] | test [2011]   (expanding : le train grandit)
       ou : train [1981-1998] | validation [1999-2010] | test [2011]   (rolling   : le train glisse)
fenetre 2 : train [1980-1999] | validation [2000-2011] | test [2012]
   ...
```

Les prédictions de test de **toutes** les fenêtres sont mises bout à bout pour former une
seule série hors-échantillon continue, utilisée pour le R²_oos final (notebook 07, partie A)
et les portefeuilles par décile (partie B) — beaucoup plus représentatif qu'un seul test sur
une décennie donnée.

- **Où c'est réglé** : `config.py`, section *"Fenêtres d'entraînement"* — `TYPE_FENETRE`
  (`"expanding"` ou `"rolling"`), `ANNEES_TRAIN_INITIAL`, `ANNEES_VALIDATION`,
  `ANNEES_TEST_PAR_FENETRE`.
- **Où c'est implémenté** : `fenetres.py`, à la racine du projet (à côté de `config.py`) —
  génération des fenêtres, standardisation des variables macro *par fenêtre* (indispensable :
  chaque fenêtre a un train différent, donc une standardisation différente), et la fonction
  `r2_oos` partagée par les notebooks 04 à 06.
- **Expanding vs rolling** : `expanding` est le choix par défaut et standard dans la
  littérature (le train ne perd jamais d'historique) ; `rolling` a du sens si on pense que les
  données les plus anciennes ne sont plus pertinentes pour prédire le présent. Change
  simplement `TYPE_FENETRE` dans `config.py` et ré-exécute 04 à 08 pour comparer les deux —
  c'est exactement le genre de comparaison que le notebook 08 automatise (section 5).
- ⚠️ **Notebook 03 ne fait aucun split ni standardisation macro** (un train unique serait
  incompatible avec des fenêtres multiples, chacune ayant son propre train) — voir sa
  partie B.5.

## Format des fichiers : Parquet

Tout le pipeline lit et écrit des fichiers **`.parquet`** plutôt que `.csv`, du premier fichier
brut jusqu'au dernier résultat — plus rapide à charger, plus compact sur disque, et le type de
chaque colonne (ex: `annee_mois` en texte) est préservé automatiquement d'un notebook à l'autre,
sans avoir besoin de le forcer à la lecture.

⚠️ **Les 3 fichiers bruts doivent être en `.parquet` dans `data/raw/`** avant de lancer le
notebook 01 : convertis `datashare.csv`, `StockReturn.csv` et `MacroData.csv` de ton côté, par
exemple :

```python
import pandas as pd
pd.read_csv("datashare.csv").to_parquet("data/raw/datashare.parquet", index=False)
pd.read_csv("StockReturn.csv").to_parquet("data/raw/StockReturn.parquet", index=False)
pd.read_csv("MacroData.csv").to_parquet("data/raw/MacroData.parquet", index=False)
```

Tous les fichiers intermédiaires et finaux (`data/interim/*`, `data/processed/*`,
`outputs/resultats_*`, `outputs/predictions_*`, `outputs/comparaison_modeles`,
`outputs/rendements_portefeuilles_deciles`, `outputs/performance_portefeuilles`,
`outputs/journal_experiences`) sont en `.parquet` — seuls les graphiques (`.png`) et
les modèles sauvegardés (`.joblib`) gardent leur format d'origine. `pip install pyarrow` est
nécessaire (voir plus bas).

## Le pipeline, notebook par notebook

| # | Notebook | Contenu | Entrée | Sortie |
|---|---|---|---|---|
| 00 | Lancer le pipeline | Détecte et exécute automatiquement (via papermill) les notebooks 02 à 07 nécessaires et suffisants pour être à jour avec `config.py`, puis 08 | `config.py`, `outputs/etat_pipeline.json` | ré-exécute 02 à 07 (selon besoin) + 08 |
| 01 | Exploration | Premier coup d'œil aux 3 fichiers bruts, sans rien modifier | `data/raw/*` | — |
| 02 | Nettoyage des données | **Partie A** caractéristiques (+ filtre automatique des candidates trop incomplètes, section A.3bis), **Partie B** rendements, **Partie C** macro — 3 nettoyages indépendants | `data/raw/*` | `data/interim/*` (+ `caracteristiques_retenues.json`) |
| 03 | Construction du panel | **Partie A** fusion des 3 fichiers nettoyés, **Partie B** préparation pour la modélisation (filtres taille/liquidité — seuils dans `config.py` —, imputation, winsorizing, rank transform) | `data/interim/*` | `data/processed/*` |
| 04 | Modèle — Régression linéaire | Benchmark simple, sans hyperparamètre, ré-entraîné à chaque fenêtre ; significativité des variables par Fama-MacBeth (section 6bis) | `panel_pret_modelisation.parquet` | `modeles/regression_lineaire.joblib`, `outputs/predictions_regression_lineaire.parquet`, `outputs/resultats_regression_lineaire*.parquet` (+ `cle_experience`), `outputs/significativite_regression_lineaire.parquet`, + 1 ligne dans `outputs/journal_experiences.parquet` |
| 05 | Modèle — Elastic Net | Linéaire régularisé, hyperparamètres (dans `config.py`) re-choisis sur la `validation` de chaque fenêtre ; stabilité de sélection des variables (section 6bis) | idem | `modeles/elastic_net.joblib`, `outputs/predictions_elastic_net.parquet`, `outputs/resultats_elastic_net*.parquet` (+ `cle_experience`), `outputs/importance_elastic_net.parquet`, + 1 ligne dans le journal |
| 06 | Modèle — LightGBM | Gradient boosting, arrêt anticipé sur la `validation` de chaque fenêtre, grille (dans `config.py`) ; importance gain/split + SHAP (section 6bis) | idem | `modeles/lightgbm.joblib`, `outputs/predictions_lightgbm.parquet`, `outputs/resultats_lightgbm*.parquet` (+ `cle_experience`), `outputs/importance_lightgbm.parquet`, + 1 ligne dans le journal |
| 07 | Évaluation finale (dernier lancement) | **Partie A** comparaison du R²_oos (pooled + évolution par fenêtre), **Partie B** portefeuilles long-short par décile (Sharpe, Sortino, drawdown...) à partir des prédictions déjà sauvegardées, taguées avec `cle_experience` et **ajoutées** (sans rien écraser) à l'historique cumulatif pour le notebook 08, **Synthèse** R²_oos vs Sharpe — toujours le **dernier** modèle entraîné de chaque type ; fait partie du pipeline automatique du notebook 00 | `outputs/resultats_*.parquet`, `outputs/predictions_*.parquet` | `outputs/*` (dont `outputs/performance_portefeuilles.parquet` [instantané] et `outputs/historique_performance_portefeuilles.parquet` [cumulatif], tous deux avec `cle_experience`) |
| 08 | Comparaison des expériences | Compare **tous** les lancements passés de 04/05/06 entre eux (R²_oos, temps d'entraînement), regroupés par modèle et hyperparamètres spécifiques, enrichis des mesures de portefeuille de l'historique cumulatif du notebook 07 dès qu'elles ont été calculées au moins une fois — ne ré-entraîne rien | `outputs/journal_experiences.parquet`, `outputs/historique_performance_portefeuilles.parquet` | (affichage seulement, rien de sauvegardé) |

**Pourquoi ce découpage en 8 notebooks ?** Le nettoyage des 3 sources de données (02) est
soit strictement indépendant (une partie par source), soit strictement séquentiel avec la
construction du panel (03) — les deux sont pensés pour être exécutés l'un après l'autre
sans jamais être ouverts isolément, d'où le regroupement. La comparaison statistique des
modèles (R²_oos) et l'évaluation des portefeuilles long-short évaluent la même question
(quel modèle choisir ?) sous deux angles complémentaires (statistique puis économique) :
les regrouper dans le notebook 07 permet une synthèse finale (le modèle le plus précis
statistiquement est-il aussi le plus rentable ?). Les 3 notebooks de modélisation
(04/05/06) restent volontairement séparés : ce sont 3 expériences indépendantes, avec des
temps d'exécution très différents (LightGBM est nettement plus long que la régression
linéaire), et un lecteur qui veut revoir un seul modèle n'a besoin d'ouvrir qu'un seul
fichier. Le notebook 08 est séparé du 07 pour la même raison qui les distingue
fonctionnellement : 07 évalue en profondeur *un* jeu de modèles (le dernier), 08 compare
*tous* les jeux de paramètres essayés au fil du temps.

## Comment exécuter le pipeline

Dans l'ordre, du notebook 01 au notebook 07 (le 08 est optionnel, à relancer autant de fois
que voulu). Chaque notebook lit uniquement les fichiers produits par les notebooks précédents
(jamais les .ipynb eux-mêmes) — `config.py` centralise tous les chemins et paramètres partagés
(période de départ, liste des caractéristiques gardées, prédicteurs utilisés pour la
modélisation, seuils de filtrage, paramètres des fenêtres d'entraînement, hyperparamètres
spécifiques à l'Elastic Net et à LightGBM, `NB_DECILES`), `fenetres.py` centralise la logique de
construction des fenêtres glissantes/extensives, et `journal.py` centralise la logique du
journal des expériences (notebooks 04/05/06 pour l'écrire, 07 et 08 pour le lire/compléter) —
rien à installer séparément, ce sont juste des fichiers `.py` du projet, importés comme
`config.py`.

```
pip install pandas numpy scikit-learn lightgbm joblib matplotlib scipy pyarrow ipython papermill statsmodels shap
```

(`pyarrow` est le moteur utilisé par pandas pour lire/écrire les fichiers `.parquet` — voir la
section *"Format des fichiers : Parquet"* plus haut. `ipython` est nécessaire pour l'affichage
des tableaux du notebook 08 dans Jupyter. `papermill` est nécessaire pour le notebook 00 -- voir
la section *"Lancer automatiquement le pipeline"* plus haut. `statsmodels` est nécessaire pour
les t-stats de Newey-West de la section 6bis du notebook 04 -- voir *"Significativité et
importance des variables"* plus bas. `shap` est optionnel, seulement pour la partie SHAP de la
section 6bis du notebook 06 -- son absence n'empêche pas le reste du notebook de tourner.)

**Pour comparer plusieurs jeux de paramètres** : après un premier
passage complet 01 → 07, modifie un paramètre dans `config.py` (ex: `TYPE_FENETRE`,
`PREDICTEURS`, une grille d'hyperparamètres, `NB_DECILES`...), puis soit (a) exécute le
notebook 00 pour relancer automatiquement le strict nécessaire (recommandé -- inclut 07 par
défaut), soit (b) relance toi-même uniquement le(s) notebook(s) concerné(s) (04/05/06 — et 03
si tu as changé un seuil de filtrage — puis 07 si tu veux ses portefeuilles à jour) ; dans les
deux cas, termine par le notebook 08 pour comparer ce nouveau lancement aux précédents (déjà
inclus automatiquement si tu es passé par le notebook 00).

## Significativité et importance des variables (section 6bis)

Chaque notebook de modélisation ne se contente pas d'afficher les coefficients/importances
du modèle de la **dernière fenêtre** : la section **6bis** de chacun mesure aussi leur
fiabilité et leur stabilité sur l'ensemble des fenêtres entraînées.

- **Notebook 04 (régression linéaire)** : régressions cross-sectionnelles **mensuelles** à
  la Fama-MacBeth (1973), t-stats de **Newey-West** — la bonne pratique en finance
  empirique pour ce type de panel, où un simple OLS pooled donnerait des t-stats invalides
  (résidus corrélés entre entreprises d'un même mois). Les prédicteurs macro sont
  explicitement exclus de ce test précis (aucune variance cross-sectionnelle intra-mois,
  donc non identifiables dans une régression en coupe). Sauvegardé dans
  `outputs/significativite_regression_lineaire.parquet`.
- **Notebook 05 (Elastic Net)** : sélection de stabilité (*stability selection*,
  Meinshausen & Bühlmann 2010) — fréquence de sélection, magnitude moyenne et cohérence de
  signe de chaque prédicteur, agrégées sur **toutes** les fenêtres entraînées. Sauvegardé
  dans `outputs/importance_elastic_net.parquet`.
- **Notebook 06 (LightGBM)** : importance `gain`/`split` moyennée (avec écart-type) sur
  toutes les fenêtres, plus une analyse SHAP signée du modèle final (dépendance optionnelle
  à `shap`, voir plus haut). Sauvegardé dans `outputs/importance_lightgbm.parquet`.
