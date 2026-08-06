"""
ETAPE 04 -- Modele de reference : regression lineaire simple (ex-notebook 04).

Ce script contient TOUT le calcul qui se trouvait auparavant dans les cellules du
notebook `04_modele_lineaire.ipynb` : boucle d'entrainement sur les fenetres,
R2_oos pooled, significativite Fama-MacBeth, sauvegardes et enregistrement au journal
des experiences. Le notebook ne fait plus que LIRE et AFFICHER ces resultats.

Lancement, depuis la RACINE du projet :

    python scripts/etape04_modele_lineaire.py
    python scripts/etape04_modele_lineaire.py --sans-fama-macbeth   # saute la section 6bis

⚠️ Pre-requis : `python scripts/etape03_construction_panel.py` deja lance.

Parametres, tous dans config.py : PREDICTEURS, TYPE_FENETRE, ANNEE_DEBUT_ENTRAINEMENT,
ANNEES_TRAIN_INITIAL, ANNEES_VALIDATION, ANNEES_TEST_PAR_FENETRE.

Fichiers produits (chemins definis dans config.py) :
  - modeles/regression_lineaire.joblib             (modele de la DERNIERE fenetre)
  - outputs/predictions_regression_lineaire.parquet (toutes fenetres bout a bout)
  - outputs/resultats_regression_lineaire.parquet          (resume pooled + cle_experience)
  - outputs/resultats_regression_lineaire_par_fenetre.parquet
  - outputs/significativite_regression_lineaire.parquet     (Fama-MacBeth / Newey-West)
  - 1 ligne dans outputs/journal_experiences.parquet        (sans doublon)
  - le rapport '04_regression_lineaire' (outputs/rapports/) pour le notebook
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

import config
import fenetres
import journal
import rapports

MODELE = 'Regression lineaire'


# ============================================================
# Sections 1 a 5 -- entrainement fenetre par fenetre + R2_oos pooled
# ============================================================

def entrainer(panel, rap):
    predicteurs = config.PREDICTEURS
    macro_predicteurs = config.MACRO_PREDICTEURS
    cible = config.CIBLE

    liste_fenetres = fenetres.generer_fenetres(
        panel['annee_mois'].unique(),
        type_fenetre=config.TYPE_FENETRE,
        annees_train_initial=config.ANNEES_TRAIN_INITIAL,
        annees_validation=config.ANNEES_VALIDATION,
        annees_test_par_fenetre=config.ANNEES_TEST_PAR_FENETRE,
    )
    resume = fenetres.resumer_fenetres(liste_fenetres)
    rap.valeur('type_fenetre', config.TYPE_FENETRE)
    rap.valeur('n_fenetres', len(liste_fenetres))
    rap.table('resume_fenetres', resume)
    print(f"Mode : {config.TYPE_FENETRE} | {len(liste_fenetres)} fenetres generees")

    r2_oos = fenetres.r2_oos

    debut_entrainement = time.perf_counter()

    resultats_par_fenetre = []
    predictions_toutes_fenetres = []
    pool_train = {'y': [], 'pred': []}
    pool_validation = {'y': [], 'pred': []}
    modele_final = None

    for f in liste_fenetres:
        donnees = fenetres.preparer_fenetre(panel, f, predicteurs, macro_predicteurs, cible)

        modele = LinearRegression()
        modele.fit(donnees['X_train'], donnees['y_train'])

        pred_train = modele.predict(donnees['X_train'])
        pred_validation = modele.predict(donnees['X_validation'])
        pred_test = modele.predict(donnees['X_test'])

        resultats_par_fenetre.append({
            'fenetre': f['numero'],
            'annee_test': f['annee_test'],
            'n_train': len(donnees['y_train']),
            'r2_oos_train': r2_oos(donnees['y_train'], pred_train),
            'r2_oos_validation': r2_oos(donnees['y_validation'], pred_validation),
            'r2_oos_test': r2_oos(donnees['y_test'], pred_test),
        })

        preds = donnees['id_test'].copy()
        preds[cible] = donnees['y_test'].values
        preds['prediction'] = pred_test
        preds['fenetre'] = f['numero']
        predictions_toutes_fenetres.append(preds)

        pool_train['y'].append(donnees['y_train'].values); pool_train['pred'].append(pred_train)
        pool_validation['y'].append(donnees['y_validation'].values); pool_validation['pred'].append(pred_validation)

        modele_final = modele  # a la fin de la boucle : celui de la derniere fenetre

        print(f"Fenetre {f['numero']:2d} (test {f['annee_test']:>9s}) : "
              f"n_train={len(donnees['y_train']):>8,} | "
              f"R2_oos test = {resultats_par_fenetre[-1]['r2_oos_test']:.4f}")

    duree_entrainement_secondes = time.perf_counter() - debut_entrainement

    resultats_par_fenetre = pd.DataFrame(resultats_par_fenetre)
    predictions_toutes_fenetres = pd.concat(predictions_toutes_fenetres, ignore_index=True)

    # --- R2_oos pooled (toutes les fenetres mises bout a bout) ---
    r2_train = r2_oos(np.concatenate(pool_train['y']), np.concatenate(pool_train['pred']))
    r2_validation = r2_oos(np.concatenate(pool_validation['y']), np.concatenate(pool_validation['pred']))
    r2_test = r2_oos(predictions_toutes_fenetres[cible], predictions_toutes_fenetres['prediction'])

    print(f"\nR2_oos train      (pooled) : {r2_train:.4f}")
    print(f"R2_oos validation (pooled) : {r2_validation:.4f}")
    print(f"R2_oos test       (pooled) : {r2_test:.4f}")
    print(f"Duree totale d'entrainement (toutes fenetres) : {duree_entrainement_secondes:.1f} s")

    rap.valeur('duree_entrainement_secondes', float(duree_entrainement_secondes))
    rap.valeur('n_predictions_test', int(len(predictions_toutes_fenetres)))
    rap.table('apercu_predictions', predictions_toutes_fenetres.head())

    # --- Coefficients du modele de la derniere fenetre (section 6 du notebook) ---
    coefficients = pd.Series(modele_final.coef_, index=predicteurs).sort_values(key=abs, ascending=False)
    rap.table('coefficients_derniere_fenetre', coefficients.rename('coefficient'))
    rap.valeur('n_coefficients_a_zero', int((coefficients == 0).sum()))
    rap.valeur('n_predicteurs', len(predicteurs))

    return {
        'liste_fenetres': liste_fenetres,
        'resultats_par_fenetre': resultats_par_fenetre,
        'predictions': predictions_toutes_fenetres,
        'modele_final': modele_final,
        'r2': (r2_train, r2_validation, r2_test),
        'duree': duree_entrainement_secondes,
    }


# ============================================================
# Section 6bis -- Significativite Fama-MacBeth (1973) + Newey-West
# ============================================================

def fama_macbeth(panel, rap):
    """Une regression cross-sectionnelle PAR MOIS sur les caracteristiques uniquement
    (les predicteurs macro sont exclus : constants dans un mois donne, donc colineaires
    avec la constante de la regression cross-sectionnelle -- coefficient non identifiable).

    Utilise TOUT l'historique du panel, pas seulement le train d'une fenetre : l'objectif
    est descriptif (quelles caracteristiques comptent en moyenne, de facon stable dans le
    temps), pas une nouvelle evaluation hors-echantillon.
    """
    print("\n--- Section 6bis : significativite Fama-MacBeth ---")
    predicteurs = config.PREDICTEURS
    caracteristiques = config.CARACTERISTIQUES_RETENUES
    cible = config.CIBLE

    predicteurs_fm = [p for p in predicteurs if p in caracteristiques]

    coefs_mensuels = []
    mois_utilises = []
    for mois, g in panel.groupby('annee_mois', sort=True, observed=True):
        # Securite : il faut strictement plus d'observations que de parametres a estimer
        if len(g) <= len(predicteurs_fm) + 1:
            continue
        X_design = np.column_stack([np.ones(len(g)), g[predicteurs_fm].values])
        beta, *_ = np.linalg.lstsq(X_design, g[cible].values, rcond=None)
        coefs_mensuels.append(beta[1:])  # on jette la constante, on garde les pentes
        mois_utilises.append(mois)

    coefs_mensuels = pd.DataFrame(coefs_mensuels, columns=predicteurs_fm, index=mois_utilises)
    print(f"{len(coefs_mensuels)} regressions cross-sectionnelles mensuelles estimees "
          f"({len(predicteurs_fm)} predicteurs, periode {mois_utilises[0]} a {mois_utilises[-1]}).")

    # Moyenne temporelle + t-stat de Newey-West
    T_mois = len(coefs_mensuels)
    nb_lags_nw = max(1, int(np.floor(4 * (T_mois / 100) ** (2 / 9))))
    print(f"Nombre de lags Newey-West : {nb_lags_nw} (regle de Newey-West 1994, T={T_mois} mois)")

    lignes = []
    for p in predicteurs_fm:
        serie = coefs_mensuels[p].values
        resultat_nw = sm.OLS(serie, np.ones(len(serie))).fit(cov_type='HAC', cov_kwds={'maxlags': nb_lags_nw})
        lignes.append({
            'predicteur': p,
            'coef_moyen_fm': resultat_nw.params[0],
            'erreur_type_nw': resultat_nw.bse[0],
            't_stat_nw': resultat_nw.tvalues[0],
            'p_value_nw': resultat_nw.pvalues[0],
            'pct_mois_signe_positif': (serie > 0).mean() * 100,
        })

    significativite = pd.DataFrame(lignes).set_index('predicteur')
    significativite['significatif_5pct'] = significativite['p_value_nw'] < 0.05
    significativite = significativite.sort_values('t_stat_nw', key=abs, ascending=False)

    nb_significatifs = int(significativite['significatif_5pct'].sum())
    print(f"{nb_significatifs} / {len(significativite)} predicteurs significatifs a 5%.")

    significativite.to_parquet(config.FICHIER_SIGNIFICATIVITE_REGRESSION_LINEAIRE)
    print("Tableau de significativite sauvegarde :", config.FICHIER_SIGNIFICATIVITE_REGRESSION_LINEAIRE)

    rap.valeur('fm_n_regressions', int(len(coefs_mensuels)))
    rap.valeur('fm_n_predicteurs', len(predicteurs_fm))
    rap.valeur('fm_periode', [str(mois_utilises[0]), str(mois_utilises[-1])])
    rap.valeur('fm_nb_lags_nw', nb_lags_nw)
    rap.valeur('fm_nb_significatifs', nb_significatifs)
    rap.valeur('fm_n_teste', int(len(significativite)))


# ============================================================
# Section 7/8 -- Sauvegardes et journal des experiences
# ============================================================

def sauvegarder(sortie, rap):
    r2_train, r2_validation, r2_test = sortie['r2']

    joblib.dump(sortie['modele_final'], config.FICHIER_MODELE_REGRESSION_LINEAIRE)
    print("\nModele final (derniere fenetre) sauvegarde :", config.FICHIER_MODELE_REGRESSION_LINEAIRE)

    sortie['predictions'].to_parquet(config.FICHIER_PREDICTIONS_REGRESSION_LINEAIRE, index=False)
    print("Predictions (toutes fenetres) sauvegardees :", config.FICHIER_PREDICTIONS_REGRESSION_LINEAIRE)

    params_specifiques = {}  # OLS : aucun hyperparametre propre a ce modele

    # Meme cle que celle calculee par enregistrer_experience ci-dessous : c'est elle que
    # le notebook 07 relit pour taguer ses mesures de portefeuille, et le notebook 08
    # pour relier Sharpe/Sortino a la bonne ligne de son tableau.
    cle_experience = journal.cle_experience_actuelle(MODELE, params_specifiques)

    resultats_finaux = pd.DataFrame([{
        'modele': MODELE,
        'type_fenetre': config.TYPE_FENETRE,
        'n_fenetres': len(sortie['liste_fenetres']),
        'r2_oos_train': r2_train,
        'r2_oos_validation': r2_validation,
        'r2_oos_test': r2_test,
        'cle_experience': cle_experience,
    }])

    resultats_finaux.to_parquet(config.FICHIER_RESULTATS_REGRESSION_LINEAIRE, index=False)
    sortie['resultats_par_fenetre'].to_parquet(
        config.FICHIER_RESULTATS_REGRESSION_LINEAIRE_PAR_FENETRE, index=False)
    print("Resultats pooled sauvegardes      :", config.FICHIER_RESULTATS_REGRESSION_LINEAIRE)
    print("Resultats par fenetre sauvegardes :", config.FICHIER_RESULTATS_REGRESSION_LINEAIRE_PAR_FENETRE)

    ajoutee = journal.enregistrer_experience(
        modele=MODELE,
        params_specifiques=params_specifiques,
        resultats={
            'n_fenetres': len(sortie['liste_fenetres']),
            'r2_oos_train': r2_train,
            'r2_oos_validation': r2_validation,
            'r2_oos_test': r2_test,
        },
        duree_entrainement_secondes=sortie['duree'],
    )

    rap.valeur('cle_experience', cle_experience)
    rap.valeur('experience_ajoutee_au_journal', bool(ajoutee))
    rap.valeur('params_specifiques', params_specifiques)


# ============================================================
def main():
    sans_fm = '--sans-fama-macbeth' in sys.argv

    config.assurer_dossiers()
    rapports.assurer_dossier()
    rap = rapports.Rapport('04_regression_lineaire')

    panel = pd.read_parquet(config.FICHIER_PANEL_MODELISATION)
    print(f"Panel complet : {panel.shape}")
    print(f"Periode couverte : {panel['annee_mois'].min()} a {panel['annee_mois'].max()}")
    rap.valeur('shape_panel', list(panel.shape))
    rap.valeur('periode_panel', [str(panel['annee_mois'].min()), str(panel['annee_mois'].max())])
    rap.valeur('n_caracteristiques', len(config.CARACTERISTIQUES_RETENUES))
    rap.valeur('n_macro_predicteurs', len(config.MACRO_PREDICTEURS))

    panel = fenetres.restreindre_debut_entrainement(panel, config.ANNEE_DEBUT_ENTRAINEMENT)
    rap.valeur('annee_debut_entrainement', config.ANNEE_DEBUT_ENTRAINEMENT)
    rap.valeur('shape_panel_entrainement', list(panel.shape))
    rap.valeur('periode_entrainement', [str(panel['annee_mois'].min()), str(panel['annee_mois'].max())])

    sortie = entrainer(panel, rap)
    sauvegarder(sortie, rap)
    rap.sauvegarder()  # l'entrainement n'est pas perdu si Fama-MacBeth plante

    if sans_fm:
        print("\n(--sans-fama-macbeth : section 6bis sautee)")
        rap.valeur('fama_macbeth_execute', False)
    else:
        fama_macbeth(panel, rap)
        rap.valeur('fama_macbeth_execute', True)

    rap.sauvegarder()

    print("\n" + "=" * 70)
    print("ETAPE 04 TERMINEE.")
    print("  -> ouvre notebooks/04_modele_lineaire.ipynb pour visualiser les resultats")
    print("=" * 70)


if __name__ == "__main__":
    main()
