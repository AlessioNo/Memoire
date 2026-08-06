# Mémoire Alessio Nocera

Prédiction des rendements boursiers par apprentissage automatique (régression linéaire,
Elastic Net, LightGBM)

## Principe : les calculs sont dans `scripts/`, les notebooks n'affichent que les résultats

Le projet sépare strictement **calculer** et **montrer** :

- **`scripts/etape02` à `etape06`** font tout le travail lourd (nettoyage, construction du
  panel, entraînement des 3 modèles) et écrivent leurs résultats sur disque. On les lance
  **à la main**, avec `python`, depuis la racine du projet.
- **`notebooks/02` à `06`** ne calculent plus rien : ils **relisent** ces résultats et les
  affichent (tableaux, graphiques, commentaires). Ils sont légers et ré-exécutables à
  volonté (`Run All` en quelques secondes).

Concrètement, le cycle de travail est toujours le même :

```bash
# 1. modifier un paramètre dans config.py
# 2. relancer le(s) script(s) concerné(s)
python scripts/etape03_construction_panel.py
python scripts/etape04_modele_lineaire.py
# 3. ouvrir le notebook correspondant et Run All pour voir le résultat
```

Les notebooks `01` (exploration), `07` (portefeuilles) et `08` (comparaison des
expériences) sont inchangés : ils sont légers ou purement en lecture, et n'ont pas eu
besoin d'être scindés.

## Données

Le dossier `data/raw/` n'est pas versionné et doit être rempli à la main avant le premier
lancement, avec les 3 fichiers sources :

- `datashare.parquet` — les 94 caractéristiques d'entreprise candidates (univers GKX)
- `StockReturn.parquet` — rendements mensuels par titre
- `MacroData.parquet` — les 8 prédicteurs macroéconomiques

Le notebook `01_exploration.ipynb` sert uniquement à inspecter ces fichiers bruts, sans
jamais les modifier. Le nettoyage proprement dit commence à l'étape 02.

## Structure du projet

```
Mémoire/
│
├── data/
│   ├── raw/              # 3 fichiers originaux (datashare, StockReturn, MacroData)
│   ├── interim/          # fichiers nettoyés (un par source), produits par l'étape 02
│   └── processed/        # panel fusionné puis prêt pour la modélisation, étape 03
│
├── scripts/              # LES CALCULS -- lancés à la main avec python
│   ├── etape02_nettoyage_donnees.py
│   ├── etape03_construction_panel.py
│   ├── etape04_modele_lineaire.py
│   ├── etape05_modele_elastic_net.py
│   └── etape06_modele_lightgbm.py
│
├── notebooks/            # L'AFFICHAGE -- aucun calcul lourd
│   ├── 01_exploration.ipynb
│   ├── 02_nettoyage_donnees.ipynb
│   ├── 03_construction_panel.ipynb
│   ├── 04_modele_lineaire.ipynb
│   ├── 05_modele_elastic_net.ipynb
│   ├── 06_modele_lightgbm.ipynb
│   ├── 07_evaluation_portefeuilles.ipynb
│   └── 08_comparaison_experiences.ipynb
│
├── config.py             # source unique de vérité : chemins et paramètres
├── fenetres.py           # fenêtres d'entraînement glissantes/extensives
├── journal.py            # journal des expériences + historique de performance
├── rapports.py           # le pont scripts → notebooks (voir plus bas)
├── utils.py              # petits utilitaires partagés (journal.py, rapports.py)
├── modeles/              # modèles entraînés sauvegardés (.joblib)
└── outputs/              # tableaux de résultats, prédictions, graphiques, journal
    └── rapports/         # rapports d'exécution des scripts, lus par les notebooks
```

## Comment fonctionne le code : `config.py` au centre

`config.py`, à la racine du projet, est la **source unique de vérité** : chemins des
fichiers, paramètres de nettoyage, de fenêtrage, de filtrage de l'univers investissable,
grilles d'hyperparamètres... Tous les scripts et tous les notebooks l'importent, et rien
n'est jamais recopié à la main d'un fichier à l'autre — modifier une valeur ici suffit à la
répercuter partout où elle sert. Il distingue deux familles de paramètres, reprises telles
quelles par `journal.py` :

- **Paramètres GÉNÉRAUX** : affectent les 3 modèles de la même façon (choix des
  prédicteurs, mode de fenêtres, seuils de filtrage de l'univers).
- **Paramètres SPÉCIFIQUES** : propres à un seul modèle (ex. la grille d'alpha de
  l'Elastic Net n'a aucun sens pour LightGBM).

Autour de `config.py`, quatre fichiers séparent la LOGIQUE (des fonctions) des VALEURS,
pour ne jamais dupliquer le même code à plusieurs endroits :

| Fichier | Rôle | Utilisé par |
|---|---|---|
| `fenetres.py` | Construit les fenêtres train/validation/test glissantes ou extensives, standardise les prédicteurs macro par fenêtre, calcule le R²_oos | scripts 04, 05, 06 ; notebook 07 |
| `journal.py` | Écrit/lit le journal des expériences et l'historique de performance des portefeuilles (jamais écrasés) | scripts 04, 05, 06 (écriture), notebook 07 (écriture), notebook 08 (lecture) |
| `rapports.py` | Stocke les diagnostics calculés par les scripts, pour que les notebooks puissent les afficher sans rien recalculer | scripts 02 à 06 (écriture), notebooks 02 à 06 (lecture) |
| `utils.py` | Conversion des types numpy → types JSON natifs | journal.py, rapports.py |

La chaîne de traitement est linéaire :

```
02 (nettoyage) → 03 (panel) → 04 / 05 / 06 (les 3 modèles) → 07 (portefeuilles) → 08 (comparaison)
```

Chaque étape ne connaît que les fichiers produits par la précédente (tous les chemins sont
dans `config.py`) ; aucune ne ré-exécute la logique d'une autre.

## Le pont scripts → notebooks : `rapports.py`

Les gros résultats du projet ont chacun leur fichier attitré dans `config.py`
(`panel_pret_modelisation.parquet`, `resultats_*.parquet`, `predictions_*.parquet`,
`significativite_*`, `importance_*`...). `rapports.py` s'occupe de **tout le reste** : les
compteurs et petits tableaux de diagnostic (nombre de lignes retirées à chaque filtre, taux
de valeurs manquantes, coefficients de la dernière fenêtre, durée d'entraînement...) qui
n'étaient auparavant qu'imprimés au fil des cellules, et qui seraient donc perdus une fois
le calcul déplacé dans un script.

Chaque script écrit un rapport dans `outputs/rapports/` :

```
outputs/rapports/<nom>.json            # les VALEURS (compteurs, listes, textes)
outputs/rapports/<nom>/<clé>.parquet   # les TABLEAUX (DataFrame / Series)
```

Côté script : `rap.valeur('n_lignes', 12345)` / `rap.table('taux_missing', serie)`.
Côté notebook : `rap = rapports.charger('02_nettoyage')` puis `rap.valeur('n_lignes')` /
`rap.table('taux_missing')`.

Si un notebook est ouvert alors que son script n'a jamais tourné, `rapports.charger` lève
une **erreur explicite indiquant la commande exacte à lancer**, plutôt qu'un
`FileNotFoundError` illisible.

## Premier lancement du projet

1. Installer les dépendances : `pip install -r requirements.txt` (Python 3.13 recommandé).
2. Placer les 3 fichiers bruts dans `data/raw/` (voir section Données).
3. Depuis la **racine** du projet, lancer les scripts dans l'ordre :

```bash
python scripts/etape02_nettoyage_donnees.py
python scripts/etape03_construction_panel.py
python scripts/etape04_modele_lineaire.py
python scripts/etape05_modele_elastic_net.py
python scripts/etape06_modele_lightgbm.py      # le plus long
```

Chaque script crée automatiquement `data/interim/`, `data/processed/`, `modeles/`,
`outputs/` et `outputs/rapports/` s'ils n'existent pas encore.

4. Ouvrir les notebooks `02` à `06` (`Run All`) pour visualiser les résultats, puis `07`
   (portefeuilles long-short) et `08` (comparaison des expériences).

## Quel script relancer après quel changement dans `config.py` ?

Les scripts se lancent à la main : ce tableau remplace l'ancienne détection automatique.

| Paramètre modifié | Scripts à relancer |
|---|---|
| `ANNEE_DEBUT`, `CARACTERISTIQUES`, `SEUIL_MAX_PCT_MANQUANT_CARACTERISTIQUES` | 02, puis 03, puis 04 / 05 / 06 (**tout**, en chaîne) |
| `SEUIL_PERCENTILE_TAILLE`, `SEUIL_PERCENTILE_LIQUIDITE` | 03, puis 04 / 05 / 06 |
| `PREDICTEURS`, `TYPE_FENETRE`, `ANNEE_DEBUT_ENTRAINEMENT`, `ANNEES_*` | 04, 05, 06 (02 et 03 restent intacts) |
| `GRILLE_*_ELASTIC_NET`, `MAX_ITER_ELASTIC_NET` | 05 seulement |
| `GRILLE_*_LIGHTGBM`, `STOPPING_ROUNDS_LIGHTGBM` | 06 seulement |
| `NB_DECILES` | aucun script — relancer le notebook 07 |

Dans tous les cas, le notebook 07 est à ré-exécuter après tout ré-entraînement (il dépend
des **trois** modèles à la fois), et le notebook 08 après le 07.

## Deux dates de départ à ne pas confondre

- `ANNEE_DEBUT` (section « Nettoyage ») filtre la **base de données** dès l'étape 02. Les
  années antérieures n'existent plus nulle part ensuite — la changer oblige à tout relancer
  depuis 02.
- `ANNEE_DEBUT_ENTRAINEMENT` (section « Fenêtres ») ne filtre que le panel utilisé pour
  **entraîner les modèles**. Les données restent dans `panel_pret_modelisation.parquet`,
  elles sont simplement ignorées au moment de construire les fenêtres. Tester un démarrage
  plus tardif ne demande donc que de relancer 04/05/06.

⚠️ Retarder `ANNEE_DEBUT_ENTRAINEMENT` raccourcit d'autant la période disponible : il faut
ajuster à la main `ANNEES_TRAIN_INITIAL` et/ou `ANNEES_VALIDATION`, sinon il reste moins
d'années de test (voire aucune fenêtre — `fenetres.py` lève alors une erreur explicite).

## Voir toutes les combinaisons d'hyperparamètres testées

Les scripts 05 et 06 enregistrent le score de **chaque** combinaison de leur grille, sur le
train et sur la validation, pour **chaque** fenêtre (tableau `grille_complete` du rapport).
Les notebooks 05 et 06 l'affichent en section 3bis : une heatmap du R²_oos de validation,
une heatmap de l'écart `R²_train − R²_validation` (diagnostic de sur-apprentissage), et les
tableaux correspondants.

Ces tableaux ne couvrent que le **dernier lancement** de chaque script — la comparaison
entre plusieurs lancements reste le rôle du notebook 08.

La sélection elle-même se fait toujours sur le meilleur R²_oos de validation. Pour la baser
sur l'écart train-validation, il faut changer la ligne `index_meilleur = ...`
(`etape05_modele_elastic_net.py`) ou `meilleur_index = ...` (`etape06_modele_lightgbm.py`).

## Options de relance partielle

Chaque script sauvegarde son rapport **au fur et à mesure** : un plantage tardif ne fait
jamais perdre les étapes déjà terminées. Quelques options en ligne de commande évitent de
tout refaire :

```bash
# rejouer seulement la partie B de l'étape 03 (filtres, imputation, rank transform)
# à partir de panel_final.parquet, sans refaire la fusion
python scripts/etape03_construction_panel.py --partie-b-seulement

# sauter la significativité Fama-MacBeth (section 6bis du notebook 04)
python scripts/etape04_modele_lineaire.py --sans-fama-macbeth

# sauter le calcul SHAP (section 6bis du notebook 06)
python scripts/etape06_modele_lightgbm.py --sans-shap
```

## Garder la trace de chaque expérience (notebook 08)

Les scripts 04/05/06 **écrasent** leurs fichiers de résultats
(`outputs/resultats_*.parquet`) à chaque exécution : un seul jeu de résultats à la fois,
celui du dernier lancement (voulu, pour que le notebook 07 lise toujours « le » dernier
modèle entraîné sans ambiguïté). Un ancien résultat serait donc normalement perdu dès qu'on
relance le même modèle avec d'autres paramètres — c'est ce que le système journal /
notebook 08 résout.

**Journal des expériences** (`outputs/journal_experiences.parquet`, écrit par `journal.py`,
jamais écrasé) : à chaque exécution d'un script 04/05/06, une ligne y est ajoutée avec les
paramètres généraux (lus dans `config.py` au moment de l'appel) et spécifiques du
lancement, plus les R²_oos obtenus. Une **clé unique** (hash des paramètres) déduplique
automatiquement : relancer deux fois exactement la même expérience n'ajoute jamais de
doublon.

**Historique de performance des portefeuilles**
(`outputs/historique_performance_portefeuilles.parquet`, écrit par `journal.py` depuis le
notebook 07) fonctionne sur le même principe : à chaque exécution de 07, les mesures
(Sharpe, Sortino, drawdown...) des expériences pas encore vues y sont ajoutées, taguées par
la même clé que dans le journal, sans jamais toucher aux lignes déjà présentes.

Le notebook 08 regroupe ensuite le journal par `(modèle, paramètres spécifiques)` — un
tableau par groupe, une ligne par combinaison de paramètres généraux testée — puis enrichit
chaque ligne avec les mesures de portefeuille de l'historique, en les reliant par cette clé
commune. Une expérience apparaît avec `NaN` sur les colonnes de portefeuille uniquement si
le notebook 07 n'a **jamais encore** tourné avec ses prédictions sur disque ; sinon elle
garde ses mesures même après avoir été supplantée par un lancement plus récent du même
modèle. Ça permet de comparer, par exemple, `expanding` vs `rolling`, ou plusieurs grilles
d'hyperparamètres Elastic Net, sans jamais perdre un résultat passé.

## Le projet, étape par étape

**01 — Exploration** (notebook seul)
Premier coup d'œil aux 3 fichiers bruts, sans rien modifier.
- Entrée : `data/raw/*` — Sortie : —

**02 — Nettoyage des données**
Partie A caractéristiques (+ filtre automatique des candidates trop incomplètes, section
A.3bis), Partie B rendements, Partie C macro — 3 nettoyages indépendants.
- Script : `scripts/etape02_nettoyage_donnees.py`
- Entrée : `data/raw/*`
- Sortie : `data/interim/*` (+ `caracteristiques_retenues.json`) + rapport `02_nettoyage`

**03 — Construction du panel**
Partie A fusion des 3 fichiers nettoyés, Partie B préparation pour la modélisation
(filtres taille/liquidité, imputation, winsorizing, rank transform).
- Script : `scripts/etape03_construction_panel.py`
- Entrée : `data/interim/*`
- Sortie : `data/processed/*` + rapport `03_panel`

**04 — Modèle : régression linéaire**
Benchmark simple, sans hyperparamètre, ré-entraîné à chaque fenêtre ; significativité des
variables par Fama-MacBeth (section 6bis).
- Script : `scripts/etape04_modele_lineaire.py`
- Entrée : `panel_pret_modelisation.parquet`
- Sortie : `modeles/regression_lineaire.joblib`,
  `outputs/predictions_regression_lineaire.parquet`,
  `outputs/resultats_regression_lineaire*.parquet` (+ `cle_experience`),
  `outputs/significativite_regression_lineaire.parquet`, + 1 ligne dans
  `outputs/journal_experiences.parquet` + rapport `04_regression_lineaire`

**05 — Modèle : Elastic Net**
Linéaire régularisé, hyperparamètres re-choisis sur la `validation` de chaque fenêtre ;
stabilité de sélection des variables (section 6bis).
- Script : `scripts/etape05_modele_elastic_net.py`
- Entrée : `panel_pret_modelisation.parquet`
- Sortie : `modeles/elastic_net.joblib`, `outputs/predictions_elastic_net.parquet`,
  `outputs/resultats_elastic_net*.parquet` (+ `cle_experience`),
  `outputs/importance_elastic_net.parquet`, + 1 ligne dans le journal + rapport
  `05_elastic_net`

**06 — Modèle : LightGBM**
Gradient boosting, arrêt anticipé sur la `validation` de chaque fenêtre, grille
d'hyperparamètres (nombre de feuilles, learning rate, minimum d'observations par feuille,
budget maximal d'arbres) ; importance gain/split + valeurs SHAP (section 6bis).
- Script : `scripts/etape06_modele_lightgbm.py`
- Entrée : `panel_pret_modelisation.parquet`
- Sortie : `modeles/lightgbm.joblib`, `outputs/predictions_lightgbm.parquet`,
  `outputs/resultats_lightgbm*.parquet` (+ `cle_experience`),
  `outputs/importance_lightgbm.parquet`, + 1 ligne dans le journal + rapport `06_lightgbm`
  (valeurs SHAP incluses)

**07 — Évaluation finale (dernier lancement)** (notebook seul)
Partie A comparaison du R²_oos (pooled + évolution par fenêtre), Partie B portefeuilles
long-short par décile (Sharpe, Sortino, drawdown...) à partir des prédictions déjà
sauvegardées, taguées avec `cle_experience` et **ajoutées** (sans rien écraser) à
l'historique cumulatif pour le notebook 08, Synthèse R²_oos vs Sharpe — toujours le
**dernier** modèle entraîné de chaque type.
- Entrée : `outputs/resultats_*.parquet`, `outputs/predictions_*.parquet`
- Sortie : `outputs/*`, dont `outputs/performance_portefeuilles.parquet` (instantané) et
  `outputs/historique_performance_portefeuilles.parquet` (cumulatif)

**08 — Comparaison des expériences** (notebook seul)
Compare **tous** les lancements passés de 04/05/06 entre eux (R²_oos, temps
d'entraînement), regroupés par modèle et hyperparamètres spécifiques, enrichis des mesures
de portefeuille de l'historique cumulatif du notebook 07 — ne ré-entraîne rien.
- Entrée : `outputs/journal_experiences.parquet`,
  `outputs/historique_performance_portefeuilles.parquet`
- Sortie : affichage seulement, rien de sauvegardé
