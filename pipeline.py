"""
Orchestration du pipeline : detecte automatiquement, a partir des parametres ACTUELLEMENT
definis dans config.py, quels notebooks (parmi 02 a 07) doivent etre ré-exécutés -- puis
les exécute dans le bon ordre avec papermill. C'est la logique derriere le notebook
"00_lancer_pipeline.ipynb" (le "bouton").

Pourquoi c'est necessaire : les notebooks 02 a 07 forment une CHAINE (chacun consomme la
sortie du precedent, voir README.md) mais ne dependent pas tous des memes parametres de
config.py :

    02 (nettoyage)          <- ANNEE_DEBUT, CARACTERISTIQUES, SEUIL_MAX_PCT_MANQUANT_CARACTERISTIQUES
    03 (construction panel) <- SEUIL_PERCENTILE_TAILLE, SEUIL_PERCENTILE_LIQUIDITE      (+ sortie de 02)
    04 (regression lineaire)<- PREDICTEURS, TYPE_FENETRE, ANNEES_*                       (+ sortie de 03)
    05 (elastic net)        <- idem 04 + GRILLE_ALPHA/L1_RATIO/MAX_ITER_ELASTIC_NET      (+ sortie de 03)
    06 (lightgbm)           <- idem 04 + GRILLE_*/N_ESTIMATORS/STOPPING_ROUNDS_LIGHTGBM  (+ sortie de 03)
    07 (portefeuilles)      <- NB_DECILES                              (+ sortie de 04 ET 05 ET 06)

Changer un parametre ne rend donc "sale" (a re-executer) que les notebooks qui en
dependent DIRECTEMENT, PLUS tous ceux qui en dependent EN CHAINE (ex: changer
SEUIL_PERCENTILE_TAILLE rend 03 sale, ce qui rend 04/05/06 sales aussi, meme si aucun de
LEURS propres parametres n'a change -- c'est leur ENTREE, panel_pret_modelisation.parquet,
qui a change ; et comme 07 depend a son tour des 3, il devient sale en chaine lui aussi).
C'est exactement ce que verifie `notebook_est_sale` plus bas -- et c'est "necessaire et
suffisant" au sens ou aucun notebook a jour n'est re-execute inutilement, et aucun
notebook sale n'est oublie.

⚠️ Le notebook 07 depend des TROIS modeles (il a besoin de leurs `resultats_*`/
`predictions_*` a la fois) : si tu restreins volontairement `cibles` a un seul modele
(ex: `cibles=('06_modele_lightgbm',)`, voir `executer_pipeline`), n'inclus pas
`'07_evaluation_portefeuilles'` dans `cibles` en meme temps, sans quoi les deux autres
modeles seraient entraines aussi juste pour satisfaire 07 -- ce n'est probablement pas ce
que tu veux dans ce cas ; relance 07 a la main a la place.

Le notebook 08 (comparaison des experiences), lui, ne fait PAS partie de cette detection
au meme titre que 02-07 : il est toujours propose en fin de pipeline par
`executer_pipeline` (inclure_08=True par defaut), sans verification de "saletee", car il
ne ré-entraîne rien et ne fait que lire le journal des experiences (+ l'historique
cumulatif `outputs/historique_performance_portefeuilles.parquet` alimente par 07) :
toujours sans risque a relancer, et c'est generalement ce qu'on veut voir juste apres
avoir change des parametres.
"""

import importlib
import json
import time
from pathlib import Path

import pandas as pd

import config
import utils

try:
    import papermill as pm
except ImportError:
    pm = None


NOTEBOOKS_DIR = config.RACINE / "notebooks"

# ============================================================
# Graphe de dependances entre notebooks : quel(s) notebook(s) EN AMONT chaque notebook
# consomme. L'ORDRE des cles de ce dict EST l'ordre topologique d'execution (02 avant 03,
# 03 avant 04/05/06) -- ne pas le reordonner sans y penser.
# ============================================================
GRAPHE_DEPENDANCES = {
    '02_nettoyage_donnees':  [],
    '03_construction_panel': ['02_nettoyage_donnees'],
    '04_modele_lineaire':    ['03_construction_panel'],
    '05_modele_elastic_net': ['03_construction_panel'],
    '06_modele_lightgbm':    ['03_construction_panel'],
    '07_evaluation_portefeuilles': ['04_modele_lineaire', '05_modele_elastic_net', '06_modele_lightgbm'],
}

# ============================================================
# Parametres de config.py dont depend DIRECTEMENT chaque notebook (sans compter les
# dependances EN CHAINE, deja capturees par GRAPHE_DEPENDANCES ci-dessus).
# ============================================================
_PARAMS_FENETRES = ['TYPE_FENETRE', 'ANNEES_TRAIN_INITIAL', 'ANNEES_VALIDATION', 'ANNEES_TEST_PAR_FENETRE']

PARAMS_PAR_NOTEBOOK = {
    '02_nettoyage_donnees': ['ANNEE_DEBUT', 'CARACTERISTIQUES', 'SEUIL_MAX_PCT_MANQUANT_CARACTERISTIQUES'],
    '03_construction_panel': ['SEUIL_PERCENTILE_TAILLE', 'SEUIL_PERCENTILE_LIQUIDITE'],
    '04_modele_lineaire': ['PREDICTEURS'] + _PARAMS_FENETRES,
    '05_modele_elastic_net': ['PREDICTEURS'] + _PARAMS_FENETRES + [
        'GRILLE_ALPHA_ELASTIC_NET', 'GRILLE_L1_RATIO_ELASTIC_NET', 'MAX_ITER_ELASTIC_NET'],
    '06_modele_lightgbm': ['PREDICTEURS'] + _PARAMS_FENETRES + [
        'GRILLE_NUM_LEAVES_LIGHTGBM', 'GRILLE_LEARNING_RATE_LIGHTGBM', 'GRILLE_MIN_CHILD_SAMPLES_LIGHTGBM',
        'N_ESTIMATORS_LIGHTGBM', 'STOPPING_ROUNDS_LIGHTGBM'],
    # NB_DECILES est le seul parametre dont 07 depend DIRECTEMENT (partie B, construction
    # des portefeuilles) -- tout le reste (R2_oos, Sharpe...) depend des resultats/predictions
    # de 04/05/06, deja capture par GRAPHE_DEPENDANCES ci-dessus.
    '07_evaluation_portefeuilles': ['NB_DECILES'],
}

# Fichier(s) de sortie dont l'EXISTENCE est verifiee avant de considerer un notebook "deja
# execute avec succes" -- absent (premier lancement, ou outputs/ efface a la main) => sale,
# meme si ses parametres n'ont pas change depuis l'etat sauvegarde.
FICHIERS_SORTIE_PAR_NOTEBOOK = {
    '02_nettoyage_donnees': [config.FICHIER_CARACTERISTIQUES_CLEAN, config.FICHIER_RETURNS_CLEAN,
                              config.FICHIER_MACRO_CLEAN, config.FICHIER_CARACTERISTIQUES_RETENUES],
    '03_construction_panel': [config.FICHIER_PANEL_FINAL, config.FICHIER_PANEL_MODELISATION],
    '04_modele_lineaire': [config.FICHIER_RESULTATS_REGRESSION_LINEAIRE],
    '05_modele_elastic_net': [config.FICHIER_RESULTATS_ELASTIC_NET],
    '06_modele_lightgbm': [config.FICHIER_RESULTATS_LIGHTGBM],
    '07_evaluation_portefeuilles': [config.FICHIER_COMPARAISON_PARQUET, config.FICHIER_PERFORMANCE_PORTEFEUILLES,
                                     config.FICHIER_RENDEMENTS_PORTEFEUILLES,
                                     config.FICHIER_HISTORIQUE_PERFORMANCE_PORTEFEUILLES],
}


# ============================================================
# Detection ("est-ce sale ?")
# ============================================================

def _valeurs_params_actuelles(nom_notebook):
    return utils.nettoyer_pour_json({p: getattr(config, p) for p in PARAMS_PAR_NOTEBOOK[nom_notebook]})


def charger_etat():
    """Charge outputs/etat_pipeline.json (parametres + horodatage du dernier lancement
    REUSSI de chaque notebook, via pipeline.py), ou {} si le pipeline n'a jamais tourne
    par ce biais (voir aussi config.FICHIER_ETAT_PIPELINE)."""
    if config.FICHIER_ETAT_PIPELINE.exists():
        with open(config.FICHIER_ETAT_PIPELINE) as f:
            return json.load(f)
    return {}


def _sauvegarder_etat(etat):
    config.assurer_dossiers()
    with open(config.FICHIER_ETAT_PIPELINE, 'w') as f:
        json.dump(etat, f, indent=2, ensure_ascii=False, default=str)


def _enregistrer_notebook_a_jour(etat, nom_notebook):
    """Ecrit dans `etat` (en memoire, pas encore sauvegarde sur disque) que `nom_notebook`
    vient d'etre execute avec succes avec les parametres actuels de config.py.

    Utilise un COMPTEUR ENTIER MONOTONE (`_prochain_notebook` dans l'etat), pas seulement
    un horodatage, pour determiner quel notebook a ete execute "apres" quel autre : un
    horodatage a la seconde pres peut etre identique pour deux ecritures rapprochees (ex:
    deux appels a `marquer_a_jour` dans la meme seconde), ce qui masquerait a tort une
    dependance en chaine (voir `notebook_est_sale`, qui compare ce compteur, pas
    l'horodatage -- l'horodatage reste sauvegarde uniquement a titre informatif/affichage)."""
    etat['_prochain_compteur'] = etat.get('_prochain_compteur', 0) + 1
    etat[nom_notebook] = {
        'horodatage': pd.Timestamp.now().isoformat(timespec='seconds'),
        'compteur': etat['_prochain_compteur'],
        'params': _valeurs_params_actuelles(nom_notebook),
    }


def notebook_est_sale(nom_notebook, etat=None, memo=None):
    """True si `nom_notebook` doit etre ré-exécuté : jamais lance avec succes par
    pipeline.py, fichier(s) de sortie manquant(s), parametres dont il depend DIRECTEMENT
    changes depuis son dernier lancement, OU un notebook dont il depend EN AMONT est
    lui-meme sale (chaine, propagee par recursion)."""
    if etat is None:
        etat = charger_etat()
    if memo is None:
        memo = {}
    if nom_notebook in memo:
        return memo[nom_notebook]

    sale = False
    infos = etat.get(nom_notebook)
    if infos is None:
        sale = True
    elif any(not Path(f).exists() for f in FICHIERS_SORTIE_PAR_NOTEBOOK[nom_notebook]):
        sale = True
    elif _valeurs_params_actuelles(nom_notebook) != infos.get('params'):
        sale = True
    else:
        for amont in GRAPHE_DEPENDANCES[nom_notebook]:
            if notebook_est_sale(amont, etat, memo) or etat[amont].get('compteur', -1) > infos.get('compteur', -1):
                sale = True
                break

    memo[nom_notebook] = sale
    return sale


def plan_execution(cibles=('04_modele_lineaire', '05_modele_elastic_net', '06_modele_lightgbm',
                            '07_evaluation_portefeuilles')):
    """Determine l'ensemble NECESSAIRE ET SUFFISANT de notebooks a executer (dans l'ordre
    topologique 02 -> 03 -> 04/05/06 -> 07), pour que tous les `cibles` demandes soient a
    jour avec les parametres ACTUELS de config.py -- sans re-executer ce qui n'en a pas
    besoin. ⚠️ '07_evaluation_portefeuilles' depend des TROIS modeles a la fois (voir
    GRAPHE_DEPENDANCES) : le garder dans `cibles` (comme dans la valeur par defaut
    ci-dessus) FORCE les 3 modeles a etre a jour, meme si tu ne voulais en cibler qu'un
    seul par ailleurs -- si tu restreins `cibles` a un seul modele (ex:
    `cibles=('06_modele_lightgbm',)`), retire aussi '07_evaluation_portefeuilles' de la
    liste (sans quoi 04 et 05 seraient entraines quand meme, juste pour satisfaire 07) et
    relance 07 a la main si besoin.

    Retourne (a_executer, deja_a_jour), deux listes de noms de notebooks (sans l'extension
    .ipynb), dans l'ordre ou `a_executer` doit etre lance."""
    importlib.reload(config)  # relit config.py au cas ou il aurait ete modifie depuis l'import initial

    etat = charger_etat()
    memo = {}

    concernes = set()

    def ajouter_amonts(nom):
        if nom in concernes:
            return
        concernes.add(nom)
        for amont in GRAPHE_DEPENDANCES[nom]:
            ajouter_amonts(amont)

    for c in cibles:
        ajouter_amonts(c)

    ordre_topologique = [n for n in GRAPHE_DEPENDANCES if n in concernes]

    a_executer, deja_a_jour = [], []
    for nom in ordre_topologique:
        (a_executer if notebook_est_sale(nom, etat, memo) else deja_a_jour).append(nom)
    return a_executer, deja_a_jour


# ============================================================
# Execution (papermill)
# ============================================================

def _executer_notebook(nom_notebook, kernel_name=None, tentatives=3, delai_entre_tentatives=3):
    """Execute un notebook via papermill, en l'ecrasant sur place (sorties a jour dans le
    meme fichier .ipynb).

    Deux precautions specifiques a Windows + OneDrive/Google Drive/Dropbox (ou tout
    antivirus scannant les fichiers a la volee) :
    - `request_save_on_cell_execute=False` : papermill sauvegarde le .ipynb par defaut
      APRES CHAQUE CELLULE (donc des dizaines de fois par notebook) -- sur un dossier
      synchronise, chaque sauvegarde peut tomber pile pendant que OneDrive verrouille
      brievement le fichier pour le synchroniser, ce qui fait planter papermill avec un
      PermissionError meme si rien n'est reellement cassé. On ne sauvegarde donc qu'UNE
      SEULE FOIS, a la toute fin du notebook -- plus rapide, et beaucoup moins de chances
      de tomber sur un verrou.
    - `tentatives` : si malgre tout un PermissionError survient (verrou vraiment tombé au
      mauvais moment), on reessaie automatiquement apres une courte pause, jusqu'a
      `tentatives` fois, avant d'abandonner pour de bon.
    """
    if pm is None:
        raise RuntimeError(
            "papermill n'est pas installe dans cet environnement. Lance `pip install papermill` "
            "(ou `pip install papermill --break-system-packages` selon ton systeme), puis reessaie."
        )
    chemin = NOTEBOOKS_DIR / f"{nom_notebook}.ipynb"
    print(f"-> Execution de {nom_notebook}.ipynb ...", flush=True)
    kwargs = {'kernel_name': kernel_name} if kernel_name else {}

    for tentative in range(1, tentatives + 1):
        try:
            pm.execute_notebook(str(chemin), str(chemin), progress_bar=True,
                                 request_save_on_cell_execute=False, **kwargs)
            break
        except PermissionError as e:
            if tentative == tentatives:
                print(f"\n   ECHEC : impossible d'ecrire {chemin} apres {tentatives} tentatives ({e}).")
                print("   Cause la plus probable : le fichier est verrouille par OneDrive/Google Drive/Dropbox")
                print("   (synchronisation en cours), par un antivirus, ou est ouvert dans un autre programme")
                print("   (Jupyter, VS Code...). Ferme le notebook partout ailleurs, mets la synchronisation")
                print("   en pause le temps du lancement, ou deplace le projet hors du dossier synchronise --")
                print("   puis relance (les notebooks deja termines avec succes ne seront pas repris a zero).")
                raise
            print(f"   Fichier verrouille (tentative {tentative}/{tentatives}), "
                  f"nouvelle tentative dans {delai_entre_tentatives}s...")
            time.sleep(delai_entre_tentatives)

    print(f"   OK -- {nom_notebook}.ipynb termine et sauvegarde (avec ses sorties a jour).")


def marquer_a_jour(*noms_notebooks):
    """Synchronise l'etat (outputs/etat_pipeline.json) pour les notebooks donnes SANS
    RIEN EXECUTER -- a utiliser si tu as lance un ou plusieurs notebooks TOI-MEME (Run All
    dans Jupyter) plutot que via `executer_pipeline`/le notebook 00.

    Pourquoi c'est utile : pipeline.py ne "voit" pas les lancements manuels (l'etat n'est
    mis a jour que par `executer_pipeline`, voir sa docstring). Si tu changes un parametre
    general puis lances 04/05/06 a la main, le prochain passage par le notebook 00 les
    jugera "sales" et les relancera inutilement (pas faux, juste une perte de temps -- le
    journal des experiences, lui, ne cree jamais de doublon quoi qu'il arrive). Appelle
    cette fonction juste apres ton lancement manuel pour eviter ce gaspillage :

        pipeline.marquer_a_jour('04_modele_lineaire', '05_modele_elastic_net', '06_modele_lightgbm')

    ⚠️ Cette fonction NE VERIFIE PAS que les notebooks ont reellement ete executes avec
    succes avec les parametres ACTUELS de config.py -- elle prend ta parole pour argent
    comptant (elle verifie seulement que leur(s) fichier(s) de sortie existent). Ne l'utilise
    qu'immediatement apres avoir toi-meme lance ces notebooks jusqu'au bout, sans erreur.

    ℹ️ Peu importe l'ordre dans lequel tu listes les notebooks en argument : ils sont
    toujours traites dans l'ordre TOPOLOGIQUE du projet (02 avant 03, 03 avant 04/05/06)
    pour que la detection de dependance en chaine (voir `notebook_est_sale`) reste
    correcte -- `marquer_a_jour('03_construction_panel', '02_nettoyage_donnees')` et
    `marquer_a_jour('02_nettoyage_donnees', '03_construction_panel')` ont exactement le
    meme effet.

    Retourne la liste des noms effectivement marques (ceux dont le fichier de sortie
    manque encore sont ignores, avec un avertissement -- ils resteront "sales")."""
    for nom in noms_notebooks:
        if nom not in GRAPHE_DEPENDANCES:
            print(f"⚠️  '{nom}' inconnu (attendu un nom parmi {list(GRAPHE_DEPENDANCES)}) -- ignore.")

    etat = charger_etat()
    marques = []
    for nom in [n for n in GRAPHE_DEPENDANCES if n in noms_notebooks]:  # force l'ordre topologique
        fichiers_manquants = [f for f in FICHIERS_SORTIE_PAR_NOTEBOOK[nom] if not Path(f).exists()]
        if fichiers_manquants:
            print(f"⚠️  '{nom}' : fichier(s) de sortie manquant(s) ({fichiers_manquants}) -- pas marque a jour, "
                  f"il sera relance par le prochain executer_pipeline().")
            continue
        _enregistrer_notebook_a_jour(etat, nom)
        marques.append(nom)
        print(f"'{nom}' marque a jour (parametres actuels de config.py pris comme reference).")

    _sauvegarder_etat(etat)
    return marques


def executer_pipeline(cibles=('04_modele_lineaire', '05_modele_elastic_net', '06_modele_lightgbm',
                               '07_evaluation_portefeuilles'),
                       inclure_08=True, forcer=None, kernel_name=None, executer=True):
    """LE "BOUTON" : detecte et execute automatiquement, dans le bon ordre, les notebooks
    NECESSAIRES ET SUFFISANTS (parmi 02 a 07) pour que `cibles` soient a jour avec les
    parametres actuels de config.py -- puis, par defaut, le notebook 08.

    Parametres
    ----------
    cibles : tuple de noms de notebooks parmi '04_modele_lineaire', '05_modele_elastic_net',
        '06_modele_lightgbm', '07_evaluation_portefeuilles' -- par defaut, les 4 (assure
        que les 3 modeles ET leurs portefeuilles long-short sont a jour). ⚠️ Comme
        '07_evaluation_portefeuilles' depend des 3 modeles a la fois (voir
        GRAPHE_DEPENDANCES), le laisser dans `cibles` en meme temps qu'un sous-ensemble
        des modeles force quand meme TOUS les modeles a etre a jour -- si tu veux ne t'occuper
        que d'un seul modele (ex: cibles=('06_modele_lightgbm',)), retire aussi
        '07_evaluation_portefeuilles' de `cibles`, et relance-le a la main plus tard si besoin.
    inclure_08 : bool, True par defaut -- execute aussi le notebook 08 (comparaison des
        experiences) a la fin. Toujours sans risque (lecture seule du journal + de
        l'historique cumulatif outputs/historique_performance_portefeuilles.parquet
        alimente par 07).
    forcer : liste optionnelle de notebooks (parmi 02 a 07) a re-executer
        INCONDITIONNELLEMENT, meme si la detection les jugerait a jour -- utile si tu as
        modifie un fichier de donnees a la main, ou juste pour regenerer des figures.
    kernel_name : nom du kernel Jupyter a utiliser (None = kernel par defaut de papermill).
    executer : bool, True par defaut. Mets False pour voir SEULEMENT le plan d'execution
        (quels notebooks seraient lances, dans quel ordre) SANS rien executer -- pratique
        pour verifier avant un calcul potentiellement long (LightGBM notamment).

    Retourne la liste des notebooks effectivement executes (nom sans .ipynb), dans l'ordre
    -- liste vide si executer=False.
    """
    a_executer_detectes, deja_a_jour = plan_execution(cibles)
    forcer = list(forcer or [])
    ordre_complet = list(GRAPHE_DEPENDANCES)
    a_executer = [n for n in ordre_complet if n in a_executer_detectes or n in forcer]
    deja_a_jour = [n for n in deja_a_jour if n not in forcer]

    print("Plan d'execution (d'apres les parametres ACTUELS de config.py) :")
    for n in a_executer:
        raison = "force manuellement" if (n in forcer and n not in a_executer_detectes) else "parametres/entree modifies (ou jamais execute par pipeline.py)"
        print(f"  [A EXECUTER]  {n:<28s} -- {raison}")
    for n in deja_a_jour:
        print(f"  [deja a jour] {n}")
    if inclure_08:
        print(f"  [A EXECUTER]  08_comparaison_experiences   -- toujours relance (lecture seule, sans risque)")
    print()

    if not executer:
        print("executer=False : rien n'a ete lance, plan affiche seulement.")
        return []

    if not a_executer and not inclure_08:
        print("Rien a faire : tout est deja a jour.")
        return []

    etat = charger_etat()
    notebooks_lances = []
    try:
        for nom in a_executer:
            _executer_notebook(nom, kernel_name=kernel_name)
            _enregistrer_notebook_a_jour(etat, nom)
            _sauvegarder_etat(etat)  # sauvegarde apres CHAQUE notebook (pas seulement a la fin) : en cas
            notebooks_lances.append(nom)  # d'echec plus loin, le travail deja fait n'est pas reperdu au prochain lancement
    except Exception:
        print(f"\nECHEC pendant l'execution -- pipeline arrete la.")
        print(f"Notebook(s) execute(s) avec succes avant l'echec : {notebooks_lances or '(aucun)'}")
        print("Ouvre le notebook fautif dans Jupyter pour voir la trace complete "
              "(papermill y insere une cellule d'erreur a l'endroit exact ou ca a plante).")
        raise

    if inclure_08:
        _executer_notebook('08_comparaison_experiences', kernel_name=kernel_name)
        notebooks_lances.append('08_comparaison_experiences')

    print(f"\nTermine : {len(notebooks_lances)} notebook(s) execute(s) -- {notebooks_lances}")
    print("Ouvre 07_evaluation_portefeuilles.ipynb pour le detail des portefeuilles long-short "
          "(Sharpe, drawdown...) du dernier lancement de chaque modele, ou "
          "08_comparaison_experiences.ipynb pour comparer toutes les experiences passees "
          "(R2_oos, et ces memes mesures de portefeuille pour chaque experience "
          "deja evaluee par 07).")
    return notebooks_lances
