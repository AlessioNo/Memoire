"""
Fenetres d'entrainement glissantes / extensives, et utilitaires partages par
les notebooks 04 a 07.

Pourquoi un fichier a part (et pas directement dans config.py) ? config.py ne
contient que des VALEURS (chemins, parametres) ; celui-ci contient de la
LOGIQUE (des fonctions) reutilisee par 4 notebooks differents (04 a 07) --
les regrouper ici evite de copier-coller les memes fonctions dans chacun, et
garantit qu'un futur changement (ex: une correction dans le calcul du R2_oos)
se repercute automatiquement partout, comme config.py le fait deja pour les
parametres.

Rappel du principe (voir aussi config.py, section "Fenetres d'entrainement") :
au lieu d'un seul decoupage train/validation/test fixe pour tout le projet, le
modele est ré-entrainé plusieurs fois, sur une succession de fenetres qui
avancent dans le temps :

    fenetre 0 : train [1980-1997] | validation [1998-2009] | test [2010]
    fenetre 1 : train [1980-1998] | validation [1999-2010] | test [2011]   (expanding)
        ou   : train [1981-1998] | validation [1999-2010] | test [2011]   (rolling)
    fenetre 2 : train [1980-1999] | validation [2000-2011] | test [2012]   (expanding)
        ...

A chaque fenetre, le modele est ré-entrainé sur son train, ses hyperparametres
(s'il en a) sont choisis sur sa validation, puis on predit sur son test -- une
seule fois, jamais reutilise pour choisir quoi que ce soit. Les predictions de
test de TOUTES les fenetres sont ensuite mises bout a bout pour former une
seule serie hors-echantillon continue, exactement comme dans Gu, Kelly & Xiu
(2020) -- c'est cette serie mise bout a bout qui sert au R2_oos final et aux
portefeuilles par decile du notebook 07, pas une fenetre individuelle.
"""

import numpy as np
import pandas as pd


def r2_oos(y_true, y_pred):
    """R2 hors-echantillon a la Gu, Kelly & Xiu (2020).

    Contrairement au R2 habituel (qui compare aux erreurs d'une prediction
    naive egale a la MOYENNE de y), celui-ci compare aux erreurs d'une
    prediction naive de ZERO -- plus strict, et plus adapte aux rendements
    financiers (voir notebook 04, section 2, pour la discussion complete).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum(y_true ** 2)


def generer_fenetres(mois_disponibles, type_fenetre, annees_train_initial,
                      annees_validation, annees_test_par_fenetre):
    """Construit la liste des fenetres d'entrainement/validation/test.

    Parametres
    ----------
    mois_disponibles : iterable d'annee_mois (str, format "AAAAMM"), toutes les
        periodes presentes dans le panel (pas besoin d'etre triees ni uniques).
    type_fenetre : "expanding" ou "rolling" (voir config.py, TYPE_FENETRE).
    annees_train_initial : nb d'annees d'entrainement de la toute premiere
        fenetre (et taille FIXE du train a chaque fenetre si type_fenetre="rolling").
    annees_validation : nb d'annees de validation (fenetre glissante, dans les
        2 modes -- seul le train differe entre "expanding" et "rolling").
    annees_test_par_fenetre : nb d'annees de test evaluees avant de ré-entrainer
        (1 = ré-entrainement annuel, comme Gu, Kelly & Xiu 2020).

    Retourne
    --------
    Une liste de dicts (un par fenetre), avec les cles :
        'numero'     : numero de la fenetre (0, 1, 2, ...)
        'train'      : liste des annee_mois d'entrainement de cette fenetre
        'validation' : liste des annee_mois de validation de cette fenetre
        'test'       : liste des annee_mois de test de cette fenetre
        'annee_test' : annee(s) couverte(s) par le test, pour l'affichage
                       (ex: "2010" ou "2010-2011" si annees_test_par_fenetre > 1)

    On s'arrete des que la fenetre de test suivante depasserait le dernier mois
    disponible -- la toute derniere fenetre peut donc couvrir moins d'annees de
    test que les autres si le nombre total d'annees ne tombe pas juste.
    """
    mois_tries = sorted(set(mois_disponibles))
    if not mois_tries:
        raise ValueError("mois_disponibles est vide : impossible de construire des fenetres.")

    annees_disponibles = sorted(set(int(m[:4]) for m in mois_tries))
    premiere_annee = annees_disponibles[0]
    derniere_annee = annees_disponibles[-1]

    mois_par_annee = {}
    for m in mois_tries:
        mois_par_annee.setdefault(int(m[:4]), []).append(m)

    def mois_des_annees(annees):
        resultat = []
        for a in annees:
            resultat.extend(mois_par_annee.get(a, []))
        return resultat

    liste_fenetres = []
    numero = 0
    debut_train = premiere_annee
    fin_train = premiere_annee + annees_train_initial - 1  # inclus

    while True:
        debut_validation = fin_train + 1
        fin_validation = debut_validation + annees_validation - 1
        debut_test = fin_validation + 1
        fin_test = debut_test + annees_test_par_fenetre - 1

        if debut_test > derniere_annee:
            break  # plus aucune annee de test disponible : on s'arrete

        fin_test = min(fin_test, derniere_annee)  # la toute derniere fenetre peut etre plus courte

        mois_train = mois_des_annees(range(debut_train, fin_train + 1))
        mois_validation = mois_des_annees(range(debut_validation, fin_validation + 1))
        mois_test = mois_des_annees(range(debut_test, fin_test + 1))

        if mois_train and mois_validation and mois_test:
            liste_fenetres.append({
                'numero': numero,
                'train': mois_train,
                'validation': mois_validation,
                'test': mois_test,
                'annee_test': f"{debut_test}" if debut_test == fin_test else f"{debut_test}-{fin_test}",
            })
            numero += 1

        if fin_test >= derniere_annee:
            break

        # On avance tout de annees_test_par_fenetre, mais le TRAIN differe selon le mode :
        if type_fenetre == "expanding":
            fin_train += annees_test_par_fenetre        # le debut ne bouge pas : le train grandit
        elif type_fenetre == "rolling":
            debut_train += annees_test_par_fenetre       # le debut avance aussi : le train glisse (taille fixe)
            fin_train += annees_test_par_fenetre
        else:
            raise ValueError(
                f"type_fenetre inconnu : {type_fenetre!r} (attendu 'expanding' ou 'rolling')"
            )

    if not liste_fenetres:
        raise ValueError(
            "Aucune fenetre generee : la periode disponible "
            f"({premiere_annee}-{derniere_annee}, {len(annees_disponibles)} ans) est trop courte pour "
            f"annees_train_initial={annees_train_initial} + annees_validation={annees_validation} "
            f"+ annees_test_par_fenetre={annees_test_par_fenetre}. Reduis ces parametres dans config.py."
        )

    return liste_fenetres


def preparer_fenetre(panel, fenetre, predicteurs, macro_predicteurs, cible):
    """Isole les lignes de train/validation/test d'une fenetre, et standardise
    les predicteurs macro sur cette fenetre.

    ⚠️ Point cle (fuite de donnees) : la moyenne et l'ecart-type de chaque
    predicteur macro sont recalcules sur le TRAIN DE CETTE FENETRE UNIQUEMENT,
    jamais sur sa validation ni son test : chaque fenetre a un train different,
    donc chacune a besoin de sa PROPRE standardisation pour ne jamais laisser
    filtrer une information du futur (voir aussi notebook 03, partie B, sur le
    meme principe de standardisation train-only).

    Les caracteristiques d'entreprise, elles, n'ont besoin d'aucun retraitement
    ici : le rank transform du notebook 03 (partie B) est deja calcule mois par
    mois, sans aucune fuite -- il reste valable tel quel, quelle que soit la
    fenetre.

    Retourne un dict avec X_train/y_train, X_validation/y_validation,
    X_test/y_test, et id_test (permno + annee_mois du test, necessaire au
    notebook 07 pour reconstruire des portefeuilles par decile mois par mois).
    """
    train = panel[panel['annee_mois'].isin(fenetre['train'])].copy()
    validation = panel[panel['annee_mois'].isin(fenetre['validation'])].copy()
    test = panel[panel['annee_mois'].isin(fenetre['test'])].copy()

    moyennes_train = train[macro_predicteurs].mean()
    ecarts_types_train = train[macro_predicteurs].std()

    for jeu in (train, validation, test):
        for col in macro_predicteurs:
            ecart = ecarts_types_train[col]
            jeu[col] = (jeu[col] - moyennes_train[col]) / ecart if ecart > 0 else 0.0

    return {
        'X_train': train[predicteurs], 'y_train': train[cible],
        'X_validation': validation[predicteurs], 'y_validation': validation[cible],
        'X_test': test[predicteurs], 'y_test': test[cible],
        'id_test': test[['permno', 'annee_mois']].reset_index(drop=True),
    }


def resumer_fenetres(liste_fenetres):
    """Petit tableau recapitulatif (une ligne par fenetre) -- pratique pour un
    print() rapide en debut de notebook 04/05/06, avant de lancer les boucles
    d'entrainement (qui peuvent etre longues, surtout pour LightGBM)."""
    lignes = []
    for f in liste_fenetres:
        lignes.append({
            'fenetre': f['numero'],
            'train_debut': f['train'][0], 'train_fin': f['train'][-1], 'n_mois_train': len(f['train']),
            'validation_debut': f['validation'][0], 'validation_fin': f['validation'][-1],
            'test_debut': f['test'][0], 'test_fin': f['test'][-1],
            'annee_test': f['annee_test'],
        })
    return pd.DataFrame(lignes)
