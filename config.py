"""
Parametres partages entre plusieurs notebooks du projet.

Modifie les valeurs ICI, une seule fois : les notebooks qui les importent
(01 a 08) recevront automatiquement le changement, au lieu d'avoir a le
repeter (et risquer un oubli) dans chaque notebook.

Ce fichier distingue explicitement deux familles de parametres, chacune reperee plus bas par un bandeau
"Parametres GENERAUX" / "Parametres SPECIFIQUES" :
- Les parametres GENERAUX changent le resultat des 3 modeles (04, 05, 06) de la meme
  facon (ex: le choix des predicteurs, le mode de fenetres, les seuils de filtrage de
  l'univers investissable).
- Les parametres SPECIFIQUES a un modele ne concernent qu'un seul modele (ex: la grille
  d'alpha de l'Elastic Net n'a aucun sens pour LightGBM).
Cette distinction est reprise telle quelle par `journal.py`, qui enregistre les deux a
chaque entrainement (04/05/06) pour permettre la comparaison au notebook 08 -- voir sa
docstring pour le detail.
"""

from pathlib import Path
import json
import numpy as np

# ============================================================
# Chemins du projet (utilises dans tous les notebooks, 01 a 08)
#
# Tout est defini a partir de l'emplacement de ce fichier config.py
# (racine du projet), plutot qu'en chemins relatifs "../data/..." repetes
# dans chaque notebook : ca evite les erreurs si un dossier est renomme,
# et ca fonctionne peu importe le repertoire de travail courant depuis
# lequel Jupyter est lance.
# ============================================================
RACINE = Path(__file__).resolve().parent

DATA_RAW = RACINE / "data" / "raw"
DATA_INTERIM = RACINE / "data" / "interim"
DATA_PROCESSED = RACINE / "data" / "processed"
MODELES_DIR = RACINE / "modeles"
OUTPUTS_DIR = RACINE / "outputs"

# Fichiers bruts (notebook 01 : exploration, jamais modifies)
FICHIER_CARACTERISTIQUES_BRUT = DATA_RAW / "datashare.parquet"
FICHIER_RETURNS_BRUT = DATA_RAW / "StockReturn.parquet"
FICHIER_MACRO_BRUT = DATA_RAW / "MacroData.parquet"

# Fichiers intermediaires (produits par le notebook 02, parties A/B/C ;
# consommes par le notebook 03, partie A)
FICHIER_CARACTERISTIQUES_CLEAN = DATA_INTERIM / "characteristics_clean.parquet"
FICHIER_RETURNS_CLEAN = DATA_INTERIM / "returns_clean.parquet"
FICHIER_MACRO_CLEAN = DATA_INTERIM / "macro_clean.parquet"

# Manifeste (JSON) des caracteristiques effectivement retenues apres le filtre de valeurs
# manquantes du notebook 02 (partie A, section A.3bis) -- liste + taux de missing de
# chacune des 94 candidates, sur la population annee >= ANNEE_DEBUT. Voir
# `charger_caracteristiques_retenues()` plus bas : c'est ce manifeste qui alimente
# `CARACTERISTIQUES_RETENUES` (et donc `PREDICTEURS`) automatiquement pour tous les
# notebooks en aval (03 a 06), sans rien recopier a la main.
FICHIER_CARACTERISTIQUES_RETENUES = DATA_INTERIM / "caracteristiques_retenues.json"

# Fichiers finaux (produits par le notebook 03, parties A et B ; consommes par
# le notebook 03 partie B elle-meme, et par les notebooks 04 a 06)
FICHIER_PANEL_FINAL = DATA_PROCESSED / "panel_final.parquet"
FICHIER_PANEL_MODELISATION = DATA_PROCESSED / "panel_pret_modelisation.parquet"

# Modeles entraines sauvegardes (produits par les notebooks 04, 05, 06).
# ⚠️ Avec les fenetres glissantes/extensives, chaque modele est ré-entrainé une
# fois par fenetre : seul le modele de la DERNIERE fenetre (celui qui a vu le
# plus de donnees) est sauvegarde ici, a titre de reference / pour predire
# au-dela des donnees actuelles. Ce n'est PAS ce modele qui sert a evaluer le
# R2_oos ni les portefeuilles (qui reposent sur les predictions pooled de
# TOUTES les fenetres, voir FICHIER_PREDICTIONS_* plus bas).
FICHIER_MODELE_REGRESSION_LINEAIRE = MODELES_DIR / "regression_lineaire.joblib"
FICHIER_MODELE_ELASTIC_NET = MODELES_DIR / "elastic_net.joblib"
FICHIER_MODELE_LIGHTGBM = MODELES_DIR / "lightgbm.joblib"

# Resultats sauvegardes (produits par les notebooks 04, 05, 06 ; consommes par
# le notebook 07, partie A). Ces fichiers contiennent 2 niveaux : un resume
# pooled (une ligne) et un detail par fenetre (une ligne par ré-entrainement),
# voir FICHIER_RESULTATS_*_PAR_FENETRE plus bas.
#
# ⚠️ Ces fichiers *_regression_lineaire / *_elastic_net / *_lightgbm sont ECRASES a
# chaque nouvelle execution de 04/05/06 (un seul jeu de resultats a la fois, celui du
# DERNIER lancement) -- c'est ce que consomme le notebook 07, qui n'affiche donc toujours
# que le dernier modele entraine. Pour GARDER une trace de chaque lancement (et pouvoir
# comparer plusieurs jeux de parametres entre eux), voir FICHIER_JOURNAL_EXPERIENCES
# plus bas, consomme par le notebook 08 -- celui-la n'est jamais ecrase, seulement complete.
FICHIER_RESULTATS_REGRESSION_LINEAIRE = OUTPUTS_DIR / "resultats_regression_lineaire.parquet"
FICHIER_RESULTATS_ELASTIC_NET = OUTPUTS_DIR / "resultats_elastic_net.parquet"
FICHIER_RESULTATS_LIGHTGBM = OUTPUTS_DIR / "resultats_lightgbm.parquet"

FICHIER_RESULTATS_REGRESSION_LINEAIRE_PAR_FENETRE = OUTPUTS_DIR / "resultats_regression_lineaire_par_fenetre.parquet"
FICHIER_RESULTATS_ELASTIC_NET_PAR_FENETRE = OUTPUTS_DIR / "resultats_elastic_net_par_fenetre.parquet"
FICHIER_RESULTATS_LIGHTGBM_PAR_FENETRE = OUTPUTS_DIR / "resultats_lightgbm_par_fenetre.parquet"

# Tableaux de significativite / importance des variables (voir notebooks 04,
# 05, 06, section 6/6bis). Un fichier par modele, ECRASE a chaque nouvelle execution
# (comme les FICHIER_RESULTATS_* et FICHIER_PREDICTIONS_* ci-dessus) : reflete toujours
# le DERNIER lancement de chaque modele, pas un historique. Chacun a ses propres colonnes
# (coefficient moyen + t-stat de Newey-West pour la regression lineaire, frequence de
# selection pour l'Elastic Net, importance gain/split pour LightGBM), voir chaque
# notebook pour le detail.
FICHIER_SIGNIFICATIVITE_REGRESSION_LINEAIRE = OUTPUTS_DIR / "significativite_regression_lineaire.parquet"
FICHIER_IMPORTANCE_ELASTIC_NET = OUTPUTS_DIR / "importance_elastic_net.parquet"
FICHIER_IMPORTANCE_LIGHTGBM = OUTPUTS_DIR / "importance_lightgbm.parquet"

FICHIER_COMPARAISON_PARQUET = OUTPUTS_DIR / "comparaison_modeles.parquet"
FICHIER_COMPARAISON_PNG = OUTPUTS_DIR / "comparaison_modeles.png"
FICHIER_EVOLUTION_R2_PNG = OUTPUTS_DIR / "evolution_r2_oos_par_fenetre.png"

# Journal des experiences (notebook 08) : UNE SEULE ligne par combinaison unique de
# (modele, parametres generaux, parametres specifiques) deja lancee -- jamais ecrase,
# seulement complete (avec deduplication) a chaque execution de 04/05/06. Voir journal.py
# pour toute la logique (c'est ici un simple fichier de VALEURS, comme le reste de
# config.py -- la LOGIQUE d'ecriture/lecture est dans journal.py, comme fenetres.py pour
# les fenetres glissantes).
FICHIER_JOURNAL_EXPERIENCES = OUTPUTS_DIR / "journal_experiences.parquet"

# Etat du pipeline automatise (notebook 00 + pipeline.py) : pour chaque notebook parmi 02 a
# 06, la derniere fois qu'il a ete execute AVEC SUCCES par pipeline.py, et les valeurs des
# parametres de config.py dont il depend a ce moment-la -- sert a detecter automatiquement
# quels notebooks sont "sales" (a re-executer) apres un changement de config.py. Voir
# pipeline.py pour toute la logique. N'est PAS mis a jour si tu executes un notebook
# manuellement (Run All) plutot que via pipeline.py -- dans ce cas, pipeline.py le
# considerera toujours a jour tant que ses parametres n'ont pas change depuis SON dernier
# passage a lui (ce qui reste correct, juste pas au courant du lancement manuel).
FICHIER_ETAT_PIPELINE = OUTPUTS_DIR / "etat_pipeline.json"


# ============================================================
# Portefeuilles long-short par decile (notebook 07, partie B ; consomme les
# predictions deja sauvegardees par 04/05/06 sans rien ré-entrainer)
# ============================================================
NB_DECILES = 10  # decile 1 = predictions les plus faibles, decile NB_DECILES = les plus elevees

FICHIER_RENDEMENTS_PORTEFEUILLES = OUTPUTS_DIR / "rendements_portefeuilles_deciles.parquet"
FICHIER_PERFORMANCE_PORTEFEUILLES = OUTPUTS_DIR / "performance_portefeuilles.parquet"
# ⚠️ FICHIER_PERFORMANCE_PORTEFEUILLES ci-dessus est ECRASE a chaque lancement de 07 (comme
# outputs/resultats_*.parquet pour 04/05/06) : il ne reflete que le DERNIER lancement de
# chaque modele. FICHIER_HISTORIQUE_PERFORMANCE_PORTEFEUILLES ci-dessous, lui, n'est JAMAIS
# ecrase (comme outputs/journal_experiences.parquet) : chaque execution de 07 y AJOUTE les
# mesures de portefeuille des experiences pas encore vues (dedupliquees par cle_experience),
# sans jamais retoucher aux lignes deja presentes -- voir journal.py,
# enregistrer_performances_portefeuilles / charger_historique_performance_portefeuilles.
# C'est ce fichier qu'utilise le notebook 08 pour afficher le Sharpe (etc.) de
# CHAQUE experience du journal, pas seulement de la derniere.
FICHIER_HISTORIQUE_PERFORMANCE_PORTEFEUILLES = OUTPUTS_DIR / "historique_performance_portefeuilles.parquet"
FICHIER_CUMULATIF_PNG = OUTPUTS_DIR / "portefeuilles_richesse_cumulee.png"
FICHIER_DECILES_PNG = OUTPUTS_DIR / "portefeuilles_rendement_par_decile.png"


def assurer_dossiers():
    """Cree les dossiers de sortie du projet s'ils n'existent pas deja.

    Utile au premier lancement du pipeline (ou apres un clone du depot),
    puisque data/interim, data/processed, modeles/ et outputs/ ne contiennent
    pas de fichiers versionnes par defaut et peuvent donc etre absents.
    N'a aucun effet si les dossiers existent deja (appel sans risque a
    chaque notebook)."""
    for dossier in [DATA_INTERIM, DATA_PROCESSED, MODELES_DIR, OUTPUTS_DIR]:
        dossier.mkdir(parents=True, exist_ok=True)


# ============================================================
# Periode de depart (utilisee au notebook 02, parties A et C - elles DOIVENT
# rester coherentes entre elles, d'ou l'interet de ne le definir qu'ici)
# ============================================================
ANNEE_DEBUT = 1980


# ============================================================
# Univers CANDIDAT des 94 caracteristiques d'entreprise de Gu, Kelly & Xiu (2020).
#
# CARACTERISTIQUES est l'univers CANDIDAT (les 94 disponibles dans
# datashare.parquet) ; c'est le notebook 02 (partie A, section A.3bis) qui en
# retient automatiquement un SOUS-ENSEMBLE, sur la base du taux de valeurs
# manquantes de chacune apres ANNEE_DEBUT (voir
# SEUIL_MAX_PCT_MANQUANT_CARACTERISTIQUES juste en dessous) -- plutot qu'une
# selection manuelle figee dans ce fichier.
#
# ============================================================
CARACTERISTIQUES = [
    "mvel1", "beta", "betasq", "chmom", "dolvol", "idiovol", "indmom",
    "mom1m", "mom6m", "mom12m", "mom36m", "pricedelay", "turn",
    "absacc", "acc", "age", "agr", "bm", "bm_ia", "cashdebt", "cashpr",
    "cfp", "cfp_ia", "chatoia", "chcsho", "chempia", "chinv", "chpmia",
    "convind", "currat", "depr", "divi", "divo", "dy", "egr", "ep",
    "gma", "grcapx", "grltnoa", "herf", "hire", "invest", "lev", "lgr",
    "mve_ia", "operprof", "orgcap", "pchcapx_ia", "pchcurrat", "pchdepr",
    "pchgm_pchsale", "pchquick", "pchsale_pchinvt", "pchsale_pchrect",
    "pchsale_pchxsga", "pchsaleinv", "pctacc", "ps", "quick", "rd",
    "rd_mve", "rd_sale", "realestate", "roic", "salecash", "saleinv",
    "salerec", "secured", "securedind", "sgr", "sin", "sp", "tang", "tb",
    "aeavol", "cash", "chtx", "cinvest", "ear", "nincr", "roaq", "roavol",
    "roeq", "rsup", "stdacc", "stdcf", "ms", "baspread", "ill", "maxret",
    "retvol", "std_dolvol", "std_turn", "zerotrade",
]


# ============================================================
# Filtre des caracteristiques par taux de valeurs manquantes
#
# ⚠️ Parametre GENERAL (voir journal.py) : il conditionne quelles caracteristiques
# existent meme, donc il est repercute dans PREDICTEURS -- voir
# `charger_caracteristiques_retenues()` juste en dessous.
# ============================================================
SEUIL_MAX_PCT_MANQUANT_CARACTERISTIQUES = 0.30
# Exclut toute caracteristique manquante sur plus de 30% des lignes (annee >= ANNEE_DEBUT).
# Augmente ce seuil pour etre plus permissif (garder plus de caracteristiques, plus
# imputees), diminue-le pour etre plus strict (panel plus leger, mais moins de
# caracteristiques). 0.30 est un compromis courant dans la litterature empirique --
# ajuste-le et compare le R2_oos (notebook 08) si tu veux challenger ce choix.


def charger_caracteristiques_retenues():
    """Renvoie la liste des caracteristiques REELLEMENT retenues apres le filtre de
    valeurs manquantes ci-dessus, telle que sauvegardee par le notebook 02 (partie A,
    section A.3bis) dans FICHIER_CARACTERISTIQUES_RETENUES.

    Repli (fallback) : si ce fichier n'existe pas encore (avant la toute premiere
    execution du notebook 02, ex. juste apres un clone du depot), renvoie l'univers
    CANDIDAT complet (`CARACTERISTIQUES`, 94 noms) -- pour que `import config` ne casse
    jamais, meme au tout premier lancement. Une fois le notebook 02 execute au moins
    une fois, cette fonction lit systematiquement le manifeste qu'il a produit.
    """
    if FICHIER_CARACTERISTIQUES_RETENUES.exists():
        with open(FICHIER_CARACTERISTIQUES_RETENUES) as f:
            manifeste = json.load(f)
        return manifeste['caracteristiques_retenues']
    return list(CARACTERISTIQUES)


# Calculee une fois a l'import de ce module : c'est CETTE liste (pas CARACTERISTIQUES,
# l'univers candidat complet) que reutilisent le notebook 03 (partie B) et les notebooks
# 04 a 06 pour construire le panel de modelisation et les predicteurs par defaut
# (PREDICTEURS, juste en dessous). Se met a jour automatiquement des que le notebook 02
# est ré-exécuté avec un nouveau SEUIL_MAX_PCT_MANQUANT_CARACTERISTIQUES.
CARACTERISTIQUES_RETENUES = charger_caracteristiques_retenues()


# ============================================================
# Les 8 predicteurs macro de GKX (deja prefixes "macro_" pour eviter toute
# collision avec des caracteristiques d'entreprise du meme nom, ex: bm, ep)
# ============================================================
MACRO_PREDICTEURS = [
    "macro_dp", "macro_ep", "macro_bm", "macro_ntis",
    "macro_tbl", "macro_tms", "macro_dfy", "macro_svar",
]


# ============================================================
# Predicteurs utilises pour la modelisation (notebooks 04, 05, 06)
# Change UNIQUEMENT cette ligne : les 3 notebooks utilisent
# automatiquement le meme choix, pas besoin de toucher a rien d'autre.
#
# ⚠️ Parametre GENERAL (voir journal.py) : il est enregistre a chaque entrainement dans
# outputs/journal_experiences.parquet, comme TYPE_FENETRE ou les seuils de filtrage
# ci-dessous -- deux lancements avec un choix de predicteurs different apparaitront comme
# deux lignes distinctes au notebook 08, meme si tout le reste est identique.
#

# ============================================================
PREDICTEURS = CARACTERISTIQUES_RETENUES


# ------------------------------------------------------------
# Autres recettes possibles pour PREDICTEURS -- decommente UNE SEULE des lignes
# `PREDICTEURS = ...` ci-dessous (en la recopiant apres le bloc ci-dessus, ou en la
# substituant a la ligne active) selon ce que tu veux tester. Toutes utilisent
# CARACTERISTIQUES_RETENUES (le sous-ensemble qui a passe le filtre de valeurs
# manquantes), jamais CARACTERISTIQUES (l'univers candidat complet, potentiellement
# tres incomplet sur certaines colonnes) -- sauf la derniere, qui contourne le filtre
# volontairement.
#
# # Caracteristiques retenues seules (sans les 8 predicteurs macro)
# PREDICTEURS = CARACTERISTIQUES_RETENUES
#
# # Macro seules (8)
# PREDICTEURS = MACRO_PREDICTEURS
#
# # Caracteristiques retenues + macro, MOINS une ou plusieurs variables
# PREDICTEURS = [c for c in CARACTERISTIQUES_RETENUES + MACRO_PREDICTEURS if c not in ['mom1m', 'macro_dp']]
#
# # Caracteristiques retenues seules, MOINS une ou plusieurs variables
# PREDICTEURS = [c for c in CARACTERISTIQUES_RETENUES if c not in ['mom1m', 'mom6m']]
#
# # Macro seules, MOINS une ou plusieurs variables
# PREDICTEURS = [c for c in MACRO_PREDICTEURS if c not in ['macro_dp', 'macro_ep']]
#
# # Liste explicite, ecrite a la main (le sous-ensemble exact que tu veux)
# PREDICTEURS = ['mvel1', 'bm', 'ep', 'mom1m', 'mom6m', 'mom12m', 'ill', 'beta', 'turn', 'agr']
#
# # Univers CANDIDAT complet (94), en ignorant volontairement le filtre de valeurs
# # manquantes -- deconseille (voir SEUIL_MAX_PCT_MANQUANT_CARACTERISTIQUES plus haut),
# # utile seulement pour mesurer l'effet du filtre lui-meme sur le R2_oos.
# PREDICTEURS = CARACTERISTIQUES + MACRO_PREDICTEURS

# ============================================================
# Nom de la variable cible, calculee au notebook 03 (partie A)
# ============================================================
CIBLE = "excess_return"


# ============================================================
# Filtres sur l'univers investissable (notebook 03, partie B, section B.2)
#
# Regroupes ici, a cote du reste des parametres du projet : change-les ICI, PUIS re-execute
# le notebook 03 (au moins la partie B) pour regenerer panel_pret_modelisation.parquet,
# avant de re-entrainer les modeles (04 a 06) sur ce nouvel univers.
#
# ⚠️ Parametres GENERAUX (voir journal.py) : contrairement a TYPE_FENETRE et consorts (qui
# n'affectent que la boucle d'entrainement de 04/05/06), ceux-ci affectent la construction
# meme du panel (notebook 03) -- mais ils sont neanmoins enregistres comme parametres
# generaux dans outputs/journal_experiences.parquet a chaque entrainement, pour garder une
# trace de l'univers utilise par chaque experience.
# ============================================================
SEUIL_PERCENTILE_TAILLE = 0.10
# Filtre taille (mvel1, notebook 03 section B.2.1) : exclut, CHAQUE MOIS, les titres SOUS ce
# percentile de capitalisation boursiere (0.50 = exclut les 2 quartiles inferieurs, la
# moitie la plus petite de l'univers chaque mois).

SEUIL_PERCENTILE_LIQUIDITE = 0.90
# Filtre liquidite (ill, notebook 03 section B.2.2) : exclut, CHAQUE MOIS, les titres
# AU-DESSUS de ce percentile d'illiquidite d'Amihud (0.90 = exclut le decile le plus illiquide).


# ============================================================
# Fenetres d'entrainement glissantes / extensives (notebooks 04 a 08)
#
# Chaque modele est ré-entrainé plusieurs fois,
# sur une succession de fenetres qui avancent dans le temps. Change les
# parametres ICI, les notebooks 04 a 08 recoivent automatiquement le
# changement (la construction des fenetres elle-meme est dans fenetres.py,
# a la racine du projet, a cote de ce fichier).
#
# ⚠️ Parametres GENERAUX (voir journal.py).
# ============================================================
TYPE_FENETRE = "expanding"
# "expanding" : le train GARDE son point de depart d'origine et grandit chaque
#               annee (il n'oublie jamais rien) -- c'est le choix de GKX (2020).
# "rolling"   : le train garde une taille FIXE (ANNEES_TRAIN_INITIAL annees) et
#               glisse dans le temps, en oubliant les annees les plus anciennes.

ANNEES_TRAIN_INITIAL = 18    # nb d'annees d'entrainement de la 1ere fenetre (taille FIXE du train si "rolling")
ANNEES_VALIDATION = 12       # nb d'annees de validation, glisse toujours juste apres le train
ANNEES_TEST_PAR_FENETRE = 4  # nb d'annees de test par fenetre avant de ré-entrainer (1 = ré-entrainement annuel, comme GKX)

# ⚠️ Augmenter ANNEES_TEST_PAR_FENETRE reduit le nombre de fenetres (donc le temps
# de calcul total, surtout pour le notebook 06 LightGBM) au prix d'un ré-entrainement
# moins frequent -- utile si le pipeline est trop lent sur ton PC.


# ============================================================
# Predictions hors-echantillon sauvegardees (produites par 04/05/06, une ligne
# par (permno, annee_mois) de test, sur TOUTES les fenetres mises bout a bout ;
# consommees par le notebook 07 pour la comparaison des R2_oos et les
# portefeuilles par decile -- 07 ne ré-entraine plus aucun modele)
#
# ⚠️ Comme les fichiers FICHIER_RESULTATS_* plus haut, ces fichiers sont ECRASES a chaque
# nouvelle execution : ils ne contiennent toujours que les predictions du DERNIER
# lancement de chaque modele -- c'est voulu, un fichier de predictions complet (une ligne
# par titre/mois) pour CHAQUE experience passee serait bien trop volumineux a conserver.
# Le notebook 08 compare donc les experiences passees sur la base des RESULTATS agreges
# du journal (R2_oos, temps d'entrainement...), pas des predictions brutes.
# ============================================================
FICHIER_PREDICTIONS_REGRESSION_LINEAIRE = OUTPUTS_DIR / "predictions_regression_lineaire.parquet"
FICHIER_PREDICTIONS_ELASTIC_NET = OUTPUTS_DIR / "predictions_elastic_net.parquet"
FICHIER_PREDICTIONS_LIGHTGBM = OUTPUTS_DIR / "predictions_lightgbm.parquet"


# ============================================================
# Hyperparametres SPECIFIQUES a l'Elastic Net (notebook 05 uniquement)
#
# Changes ICI, le notebook 05 les recoit automatiquement (recherche sur grille,
# ré-evaluee une fois par fenetre sur sa propre validation -- voir notebook 05, section 3).
# ⚠️ Parametres SPECIFIQUES (voir journal.py) : contrairement aux parametres generaux
# ci-dessus, ceux-la ne concernent QUE l'Elastic Net -- les enregistrer separement permet
# au notebook 08 de regrouper les runs en un seul tableau par grille testee.
# ============================================================
GRILLE_ALPHA_ELASTIC_NET = np.logspace(-5, -1, 5)        # force de regularisation : 1e-7 a 1e-1 (10 valeurs, echelle log)
GRILLE_L1_RATIO_ELASTIC_NET = [0.1, 0.3, 0.5, 0.7, 0.9]    # equilibre Lasso (1.0) / Ridge (0.0) : 5 valeurs
MAX_ITER_ELASTIC_NET = 1000                                # nb max d'iterations de l'optimiseur, par modele candidat de la grille


# ============================================================
# Hyperparametres SPECIFIQUES a LightGBM (notebook 06 uniquement)
#
# Changes ICI, le notebook 06 les recoit automatiquement (recherche sur grille + arret
# anticipe, ré-evaluee une fois par fenetre sur sa propre validation -- voir notebook 06,
# section 3). Meme remarque que pour l'Elastic Net ci-dessus sur la separation
# generaux/specifiques, cruciale pour le regroupement des tableaux au notebook 08.
# ============================================================
GRILLE_NUM_LEAVES_LIGHTGBM = [7, 15, 31]                 # nb max de feuilles par arbre : 3 valeurs
GRILLE_LEARNING_RATE_LIGHTGBM = [0.005, 0.01, 0.03]      # taux d'apprentissage : 3 valeurs
GRILLE_MIN_CHILD_SAMPLES_LIGHTGBM = [500, 2000, 5000]    # nb min d'observations par feuille : 3 valeurs
N_ESTIMATORS_LIGHTGBM = 1000                             # budget MAXIMUM d'arbres (l'arret anticipe le coupe generalement bien avant)
STOPPING_ROUNDS_LIGHTGBM = 50                            # arret si pas d'amelioration sur la validation depuis N arbres d'affilee
