# Mémoire Alessio Nocera

Prédiction des rendements boursiers par apprentissage automatique (régression linéaire,
Elastic Net, LightGBM)

## Données

Le dossier `data/raw/` n'est pas versionné et doit être rempli à la main avant le premier
lancement, avec les 3 fichiers sources :

- `datashare.parquet` — les 94 caractéristiques d'entreprise candidates (univers GKX)
- `StockReturn.parquet` — rendements mensuels par titre
- `MacroData.parquet` — les 8 prédicteurs macroéconomiques

Le notebook `01_exploration.ipynb` sert uniquement à inspecter ces fichiers bruts, sans
jamais les modifier. Le nettoyage proprement dit commence au notebook 02 ; voir plus bas.

## Structure du projet

```
Mémoire/
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

## Comment fonctionne le code : `config.py` au centre

`config.py`, à la racine du projet, est la **source unique de vérité** : chemins des
fichiers, paramètres de nettoyage, de fenêtrage, de filtrage de l'univers investissable,
grilles d'hyperparamètres... Tous les notebooks (01 à 08) l'importent, et rien n'est
jamais recopié à la main d'un notebook à l'autre — modifier une valeur ici suffit à la
répercuter partout où elle sert. Il distingue deux familles de paramètres, reprises telle
quelle par `journal.py` (voir plus bas) :

- **Paramètres GÉNÉRAUX** : affectent les 3 modèles de la même façon (choix des
  prédicteurs, mode de fenêtres, seuils de filtrage de l'univers).
- **Paramètres SPÉCIFIQUES** : propres à un seul modèle (ex. la grille d'alpha de
  l'Elastic Net n'a aucun sens pour LightGBM).

Autour de `config.py`, trois fichiers séparent la LOGIQUE (des fonctions) des VALEURS,
pour ne jamais dupliquer le même code dans plusieurs notebooks :

| Fichier | Rôle | Utilisé par |
|---|---|---|
| `fenetres.py` | Construit les fenêtres train/validation/test glissantes ou extensives, standardise les prédicteurs macro par fenêtre, calcule le R²_oos | notebooks 04, 05, 06, 07 |
| `journal.py` | Écrit/lit le journal des expériences et l'historique de performance des portefeuilles (jamais écrasés) | notebooks 04, 05, 06 (écriture), 07 (écriture), 08 (lecture) |
| `pipeline.py` | Détecte quels notebooks sont "sales" et les relance dans le bon ordre via papermill | notebook 00 |
| `utils.py` | Conversion des types numpy → types JSON natifs, utilisée par `journal.py` et `pipeline.py` | journal.py, pipeline.py |

La chaîne de traitement est linéaire :

```
02 (nettoyage) → 03 (construction du panel) → 04 / 05 / 06 (les 3 modèles) → 07 (portefeuilles) → 08 (comparaison)
```

Chaque notebook ne connaît que les fichiers produits par le précédent (tous les chemins
sont dans `config.py`) ; aucun ne ré-exécute la logique d'un autre.

## Premier lancement du projet

1. Installer les dépendances : `pip install -r requirements.txt` (Python 3.13 recommandé).
2. Placer les 3 fichiers bruts dans `data/raw/` (voir section Données).
3. Ouvrir `00_lancer_pipeline.ipynb` et exécuter la cellule d'import (elle crée
   automatiquement `data/interim/`, `data/processed/`, `modeles/` et `outputs/` s'ils
   n'existent pas encore, via `config.assurer_dossiers()`).
4. Laisser `APERCU_SEULEMENT = True` et exécuter la cellule "Le bouton" : le plan
   d'exécution s'affiche (au premier lancement, tout est marqué `[A EXECUTER]`, puisque
   rien n'a encore tourné) sans rien lancer réellement.
5. Une fois le plan vérifié, repasser `APERCU_SEULEMENT = False` et ré-exécuter la
   cellule : le pipeline enchaîne automatiquement `02 → 03 → 04 → 05 → 06 → 07 → 08`,
   dans cet ordre, en s'arrêtant net à la première erreur (le notebook fautif s'ouvre
   dans Jupyter avec la trace complète insérée par papermill à l'endroit exact du
   plantage).


## Le notebook 00 après un changement de paramètre

`pipeline.py` connaît la chaîne de dépendances du projet ainsi que les paramètres de
`config.py` dont dépend **directement** chaque notebook (02 à 07). Un notebook est jugé
"sale" (à ré-exécuter) si :

- l'un de **ses propres** paramètres a changé depuis son dernier lancement réussi via le
  notebook 00, **ou**
- un notebook dont il dépend en amont est lui-même sale (la saleté se propage vers
  l'aval).

Exemples :

- Changer seulement `TYPE_FENETRE` → seuls 04, 05, 06 sont relancés, et 07 avec eux
  (il dépend des trois) ; 02 et 03 restent intacts.
- Changer `SEUIL_PERCENTILE_TAILLE` → 03 devient sale, ce qui rend 04, 05, 06 sales en
  chaîne (leur entrée a changé), et 07 en chaîne derrière eux — même si aucun de leurs
  propres paramètres n'a bougé. 02 reste intact.
- Changer `CARACTERISTIQUES` ou `SEUIL_MAX_PCT_MANQUANT_CARACTERISTIQUES` → 02 devient
  sale, donc **tout** le reste l'est aussi en chaîne.
- Changer seulement `NB_DECILES` → seul 07 est sale.

À chaque relancement du notebook 00, le plan d'exécution est **toujours affiché avant
que quoi que ce soit ne soit réellement lancé** (via `APERCU_SEULEMENT`), pour permettre
de vérifier avant un calcul potentiellement long. Quelques points d'attention :

- **Cibler un seul modèle** : `07_evaluation_portefeuilles` dépend des **trois** modèles
  à la fois. Si `CIBLES` est restreint à un seul modèle, retirer aussi
  `'07_evaluation_portefeuilles'` de `CIBLES`, sinon les deux autres modèles seraient
  entraînés quand même juste pour satisfaire 07.
- **`FORCER`** : liste de notebooks à ré-exécuter inconditionnellement, même si la
  détection les jugerait à jour (utile après une modification manuelle d'un fichier de
  données, ou juste pour régénérer des figures).
- **Lancement manuel (Run All dans Jupyter)** : `outputs/etat_pipeline.json` n'est mis à
  jour que par `pipeline.executer_pipeline` — un notebook lancé à la main sera donc jugé
  sale au prochain passage par le notebook 00, même s'il est déjà à jour. Pour éviter ce
  gaspillage, synchroniser l'état à la main juste après, sans rien ré-exécuter :
  `pipeline.marquer_a_jour('04_modele_lineaire', '05_modele_elastic_net', ...)`.
- **Repartir de zéro** : supprimer `outputs/etat_pipeline.json` — tous les notebooks
  concernés seront alors considérés comme jamais exécutés.
- Le notebook 08 est toujours relancé à la fin (par défaut), sans vérification de
  saleté : il ne fait que lire des fichiers, jamais ré-entraîner quoi que ce soit.

## Le notebook 08 : garder la trace de chaque expérience

Les notebooks 04/05/06 **écrasent** leurs fichiers de résultats
(`outputs/resultats_*.parquet`) à chaque exécution : un seul jeu de résultats à la fois,
celui du dernier lancement (voulu, pour que 07 lise toujours "le" dernier modèle
entraîné sans ambiguïté). Ça veut dire qu'un ancien résultat serait normalement perdu dès
qu'on relance le même modèle avec d'autres paramètres — c'est ce que le système
journal/notebook 08 résout.

**Journal des expériences** (`outputs/journal_experiences.parquet`, écrit par `journal.py`,
jamais écrasé) : à chaque exécution de 04/05/06, une ligne y est ajoutée avec les
paramètres généraux (lus dans `config.py` au moment de l'appel) et spécifiques du
lancement, plus les R²_oos obtenus. Une **clé unique** (hash des paramètres) déduplique
automatiquement : relancer deux fois exactement la même expérience n'ajoute jamais de
doublon.

**Historique de performance des portefeuilles**
(`outputs/historique_performance_portefeuilles.parquet`, écrit par `journal.py` depuis le
notebook 07) fonctionne sur le même principe : à chaque exécution de 07, les mesures
(Sharpe, Sortino, drawdown...) des expériences pas encore vues y sont ajoutées, taguées
par la même clé que dans le journal, sans jamais toucher aux lignes déjà présentes.

Le notebook 08 regroupe ensuite le journal par `(modèle, paramètres spécifiques)` — un
tableau par groupe, une ligne par combinaison de paramètres généraux testée — puis
enrichit chaque ligne avec les mesures de portefeuille de l'historique, en les reliant
par cette clé commune. Une expérience apparaît avec `NaN` sur les colonnes de portefeuille
uniquement si le notebook 07 n'a **jamais encore** tourné avec ses prédictions sur disque ;
sinon elle garde ses mesures même après avoir été supplantée par un lancement plus récent
du même modèle. Ça permet de comparer, par exemple, `expanding` vs `rolling`, ou
plusieurs grilles d'hyperparamètres Elastic Net, sans jamais perdre un résultat passé —
contrairement aux fichiers `resultats_*`/`predictions_*`, qui ne reflètent toujours que le
dernier lancement.

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

