"""
ETAPE 06 -- Modele LightGBM (ex-notebook 06).

Ce script contient TOUT le calcul qui se trouvait auparavant dans les cellules du
notebook `06_modele_lightgbm.ipynb` : grille d'hyperparametres avec arret anticipe sur
la validation de CHAQUE fenetre, R2_oos pooled, importance gain/split agregee, valeurs
SHAP du modele final, sauvegardes et enregistrement au journal. Le notebook ne fait
plus qu'AFFICHER (y compris le graphique SHAP, trace a partir des valeurs deja calculees
ici -- il ne relance jamais l'explainer).

⚠️ C'est le script le plus long a tourner du projet (grille x fenetres). Lancement,
depuis la RACINE du projet :

    python scripts/etape06_modele_lightgbm.py
    python scripts/etape06_modele_lightgbm.py --sans-shap   # saute le calcul SHAP

⚠️ Pre-requis : `python scripts/etape03_construction_panel.py` deja lance.

Parametres, tous dans config.py :
  - generaux   : PREDICTEURS, TYPE_FENETRE, ANNEE_DEBUT_ENTRAINEMENT, ANNEES_*
                 (partages avec 04 et 05)
  - specifiques: GRILLE_NUM_LEAVES_LIGHTGBM, GRILLE_LEARNING_RATE_LIGHTGBM,
                 GRILLE_MIN_CHILD_SAMPLES_LIGHTGBM, GRILLE_N_ESTIMATORS_LIGHTGBM,
                 STOPPING_ROUNDS_LIGHTGBM

Fichiers produits (chemins definis dans config.py) :
  - modeles/lightgbm.joblib                          (modele de la DERNIERE fenetre)
  - outputs/predictions_lightgbm.parquet
  - outputs/resultats_lightgbm.parquet                (+ cle_experience)
  - outputs/resultats_lightgbm_par_fenetre.parquet
  - outputs/importance_lightgbm.parquet               (gain/split agreges)
  - 1 ligne dans outputs/journal_experiences.parquet  (sans doublon)
  - le rapport '06_lightgbm' (outputs/rapports/), valeurs SHAP incluses
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

import config
import fenetres
import journal
import rapports

MODELE = 'LightGBM'


# ============================================================
# Section 3 a 5 -- grille + arret anticipe, fenetre par fenetre
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
    rap.valeur('type_fenetre', config.TYPE_FENETRE)
    rap.valeur('n_fenetres', len(liste_fenetres))
    rap.table('resume_fenetres', fenetres.resumer_fenetres(liste_fenetres))
    print(f"Mode : {config.TYPE_FENETRE} | {len(liste_fenetres)} fenetres generees")

    r2_oos = fenetres.r2_oos

    # Parametres SPECIFIQUES a LightGBM (config.py)
    grille_num_leaves = config.GRILLE_NUM_LEAVES_LIGHTGBM
    grille_learning_rate = config.GRILLE_LEARNING_RATE_LIGHTGBM
    grille_min_child_samples = config.GRILLE_MIN_CHILD_SAMPLES_LIGHTGBM
    grille_n_estimators = config.GRILLE_N_ESTIMATORS_LIGHTGBM
    stopping_rounds = config.STOPPING_ROUNDS_LIGHTGBM

    n_combinaisons = (len(grille_num_leaves) * len(grille_learning_rate)
                      * len(grille_min_child_samples) * len(grille_n_estimators))
    print(f"Grille : {n_combinaisons} combinaisons x {len(liste_fenetres)} fenetres "
          f"= {n_combinaisons * len(liste_fenetres)} entrainements. Sois patient.")
    rap.valeur('n_combinaisons_grille', n_combinaisons)

    debut_entrainement = time.perf_counter()

    resultats_par_fenetre = []
    predictions_toutes_fenetres = []
    pool_train = {'y': [], 'pred': []}
    pool_validation = {'y': [], 'pred': []}
    importances_gain_par_fenetre = []
    importances_split_par_fenetre = []
    grilles_par_fenetre = []       # LA grille complete de chaque fenetre (toutes combinaisons)
    modele_final = None

    for f in liste_fenetres:
        donnees = fenetres.preparer_fenetre(panel, f, predicteurs, macro_predicteurs, cible)

        # --- recherche d'hyperparametres (avec arret anticipe) sur la validation de CETTE fenetre ---
        # On garde le score de CHAQUE combinaison, sur le train ET sur la validation :
        # le tableau complet est sauvegarde dans le rapport et affiche au notebook 06
        # (heatmap + tableau), y compris l'ecart train-validation, qui mesure le
        # sur-apprentissage de chaque combinaison.
        recherche = []
        modeles_candidats = []
        for num_leaves in grille_num_leaves:
            for learning_rate in grille_learning_rate:
                for min_child_samples in grille_min_child_samples:
                    for n_estimators in grille_n_estimators:
                        modele = lgb.LGBMRegressor(
                            n_estimators=n_estimators,   # budget max d'arbres, arret anticipe possible
                            num_leaves=num_leaves,
                            learning_rate=learning_rate,
                            min_child_samples=min_child_samples,
                            feature_fraction=0.7,
                            bagging_fraction=0.7,
                            bagging_freq=1,
                            reg_alpha=0.1,
                            reg_lambda=0.1,
                            importance_type='gain',
                            random_state=0,
                            verbose=-1,
                        )
                        modele.fit(
                            donnees['X_train'], donnees['y_train'],
                            eval_set=[(donnees['X_validation'], donnees['y_validation'])],
                            eval_metric='l2',
                            callbacks=[lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False)],
                        )
                        nb_arbres = modele.best_iteration_ or n_estimators
                        score_train = r2_oos(donnees['y_train'], modele.predict(donnees['X_train']))
                        score_validation = r2_oos(donnees['y_validation'], modele.predict(donnees['X_validation']))
                        recherche.append({
                            'fenetre': f['numero'],
                            'annee_test': f['annee_test'],
                            'num_leaves': num_leaves,
                            'learning_rate': learning_rate,
                            'min_child_samples': min_child_samples,
                            'n_estimators': n_estimators,
                            'nb_arbres_utilises': nb_arbres,
                            'budget_atteint': nb_arbres >= n_estimators,
                            'r2_oos_train': score_train,
                            'r2_oos_validation': score_validation,
                            'ecart_train_validation': score_train - score_validation,
                        })
                        modeles_candidats.append(modele)

        recherche = pd.DataFrame(recherche)
        meilleur_index = recherche['r2_oos_validation'].idxmax()
        recherche['selectionnee'] = False
        recherche.loc[meilleur_index, 'selectionnee'] = True
        grilles_par_fenetre.append(recherche)

        modele = modeles_candidats[meilleur_index]
        meilleure_ligne = recherche.loc[meilleur_index]

        pred_train = modele.predict(donnees['X_train'])
        pred_validation = modele.predict(donnees['X_validation'])
        pred_test = modele.predict(donnees['X_test'])

        resultats_par_fenetre.append({
            'fenetre': f['numero'],
            'annee_test': f['annee_test'],
            'n_train': len(donnees['y_train']),
            'num_leaves': meilleure_ligne['num_leaves'],
            'learning_rate': meilleure_ligne['learning_rate'],
            'min_child_samples': meilleure_ligne['min_child_samples'],
            'n_estimators': meilleure_ligne['n_estimators'],
            'nb_arbres_utilises': meilleure_ligne['nb_arbres_utilises'],
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

        # Importance 'gain' et 'split' de CETTE fenetre -- le gain est normalise a 100 par
        # fenetre pour rester comparable malgre des nombres d'arbres differents.
        gain_fenetre = modele.booster_.feature_importance(importance_type='gain').astype(float)
        if gain_fenetre.sum() > 0:
            gain_fenetre = gain_fenetre / gain_fenetre.sum() * 100
        split_fenetre = modele.booster_.feature_importance(importance_type='split').astype(float)
        importances_gain_par_fenetre.append(dict(zip(predicteurs, gain_fenetre)))
        importances_split_par_fenetre.append(dict(zip(predicteurs, split_fenetre)))

        modele_final = modele

        print(f"Fenetre {f['numero']:2d} (test {f['annee_test']:>9s}) : "
              f"num_leaves={int(meilleure_ligne['num_leaves']):>2d}, "
              f"lr={meilleure_ligne['learning_rate']:.3f}, "
              f"arbres={int(meilleure_ligne['nb_arbres_utilises']):>4d} | "
              f"R2_oos test = {resultats_par_fenetre[-1]['r2_oos_test']:.4f}")

    duree_entrainement_secondes = time.perf_counter() - debut_entrainement

    resultats_par_fenetre = pd.DataFrame(resultats_par_fenetre)
    predictions_toutes_fenetres = pd.concat(predictions_toutes_fenetres, ignore_index=True)

    # --- R2_oos pooled ---
    r2_train = r2_oos(np.concatenate(pool_train['y']), np.concatenate(pool_train['pred']))
    r2_validation = r2_oos(np.concatenate(pool_validation['y']), np.concatenate(pool_validation['pred']))
    r2_test = r2_oos(predictions_toutes_fenetres[cible], predictions_toutes_fenetres['prediction'])

    print(f"\nR2_oos train      (pooled) : {r2_train:.4f}")
    print(f"R2_oos validation (pooled) : {r2_validation:.4f}")
    print(f"R2_oos test       (pooled) : {r2_test:.4f}")
    print(f"Duree totale (recherche d'hyperparametres incluse) : {duree_entrainement_secondes:.1f} s")

    rap.valeur('duree_entrainement_secondes', float(duree_entrainement_secondes))
    rap.valeur('n_predicteurs', len(predicteurs))
    rap.table('apercu_predictions', predictions_toutes_fenetres.head())

    # Grille complete, toutes fenetres empilees : une ligne par (fenetre, combinaison).
    # C'est ce tableau qu'affiche le notebook 06 (section 3bis).
    grille_complete = pd.concat(grilles_par_fenetre, ignore_index=True)
    rap.table('grille_complete', grille_complete)
    rap.valeur('n_modeles_entraines_grille', int(len(grille_complete)))
    # Diagnostic : si le budget d'arbres n'est JAMAIS atteint, toutes les valeurs de
    # GRILLE_N_ESTIMATORS_LIGHTGBM donnent le meme modele (l'arret anticipe coupe avant).
    rap.valeur('pct_budget_arbres_atteint', float(grille_complete['budget_atteint'].mean() * 100))

    # --- Importance du modele de la derniere fenetre (section 6 du notebook) ---
    importances = pd.Series(modele_final.feature_importances_, index=predicteurs).sort_values(ascending=False)
    rap.table('importance_derniere_fenetre', importances.rename('importance_gain'))

    return {
        'liste_fenetres': liste_fenetres,
        'resultats_par_fenetre': resultats_par_fenetre,
        'predictions': predictions_toutes_fenetres,
        'importances_gain_par_fenetre': importances_gain_par_fenetre,
        'importances_split_par_fenetre': importances_split_par_fenetre,
        'modele_final': modele_final,
        'r2': (r2_train, r2_validation, r2_test),
        'duree': duree_entrainement_secondes,
        'params_specifiques': {
            'grille_num_leaves': grille_num_leaves,
            'grille_learning_rate': grille_learning_rate,
            'grille_min_child_samples': grille_min_child_samples,
            'grille_n_estimators': grille_n_estimators,
            'stopping_rounds': stopping_rounds,
        },
    }


# ============================================================
# Section 6bis -- Importance agregee sur toutes les fenetres + SHAP
# ============================================================

def importance_agregee(sortie, rap):
    print("\n--- Section 6bis : importance agregee sur toutes les fenetres ---")

    index_fenetres = sortie['resultats_par_fenetre']['fenetre']
    gain_par_fenetre_df = pd.DataFrame(sortie['importances_gain_par_fenetre'], index=index_fenetres)
    split_par_fenetre_df = pd.DataFrame(sortie['importances_split_par_fenetre'], index=index_fenetres)

    importance = pd.DataFrame({
        'gain_moyen_pct': gain_par_fenetre_df.mean(),
        'gain_ecart_type_pct': gain_par_fenetre_df.std(),
        'split_moyen': split_par_fenetre_df.mean(),
        'split_ecart_type': split_par_fenetre_df.std(),
        'jamais_utilisee_pct_fenetres': (split_par_fenetre_df == 0).mean() * 100,
    }).sort_values('gain_moyen_pct', ascending=False)

    n_jamais = int((importance['jamais_utilisee_pct_fenetres'] == 100).sum())
    print(f"Importance agregee sur {len(gain_par_fenetre_df)} fenetres.")
    print(f"{n_jamais} / {len(importance)} predicteurs ne sont utilises dans AUCUN arbre "
          "d'aucune fenetre.")

    importance.to_parquet(config.FICHIER_IMPORTANCE_LIGHTGBM)
    print("Tableau d'importance sauvegarde :", config.FICHIER_IMPORTANCE_LIGHTGBM)

    rap.valeur('imp_n_jamais_utilisees', n_jamais)
    rap.valeur('imp_n_predicteurs', int(len(importance)))
    rap.table('gain_par_fenetre', gain_par_fenetre_df)
    rap.table('split_par_fenetre', split_par_fenetre_df)


def calculer_shap(panel, sortie, rap):
    """Calcule les valeurs SHAP du modele de la derniere fenetre sur un echantillon de
    son test, et les enregistre dans le rapport. Le notebook se contente ensuite de
    tracer le summary_plot a partir de ces valeurs, sans relancer l'explainer (couteux).
    """
    print("\n--- Section 6bis : valeurs SHAP (modele de la derniere fenetre) ---")
    try:
        import shap
    except ImportError:
        print("Le package 'shap' n'est pas installe -- calcul SHAP saute "
              "(le reste de l'etape n'est pas affecte). Installe-le avec : pip install shap")
        rap.valeur('shap_calcule', False)
        return

    predicteurs = config.PREDICTEURS
    derniere_fenetre = sortie['liste_fenetres'][-1]
    donnees_derniere = fenetres.preparer_fenetre(
        panel, derniere_fenetre, predicteurs, config.MACRO_PREDICTEURS, config.CIBLE)

    X_test_derniere = donnees_derniere['X_test']
    taille_echantillon = min(2000, len(X_test_derniere))
    X_echantillon = X_test_derniere.sample(n=taille_echantillon, random_state=0)

    explainer = shap.TreeExplainer(sortie['modele_final'])
    valeurs_shap = explainer.shap_values(X_echantillon)

    rap.valeur('shap_calcule', True)
    rap.valeur('shap_taille_echantillon', int(taille_echantillon))
    rap.valeur('shap_fenetre', int(derniere_fenetre['numero']))
    rap.table('shap_valeurs', pd.DataFrame(valeurs_shap, columns=predicteurs,
                                           index=X_echantillon.index))
    rap.table('shap_echantillon_X', X_echantillon)
    print(f"Valeurs SHAP calculees sur {taille_echantillon} lignes du test de la derniere fenetre.")


# ============================================================
# Sections 7/8 -- Sauvegardes et journal des experiences
# ============================================================

def sauvegarder(sortie, rap):
    r2_train, r2_validation, r2_test = sortie['r2']

    joblib.dump(sortie['modele_final'], config.FICHIER_MODELE_LIGHTGBM)
    print("\nModele final (derniere fenetre) sauvegarde :", config.FICHIER_MODELE_LIGHTGBM)

    sortie['predictions'].to_parquet(config.FICHIER_PREDICTIONS_LIGHTGBM, index=False)
    print("Predictions (toutes fenetres) sauvegardees :", config.FICHIER_PREDICTIONS_LIGHTGBM)

    params_specifiques = sortie['params_specifiques']
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

    resultats_finaux.to_parquet(config.FICHIER_RESULTATS_LIGHTGBM, index=False)
    sortie['resultats_par_fenetre'].to_parquet(config.FICHIER_RESULTATS_LIGHTGBM_PAR_FENETRE, index=False)
    print("Resultats pooled sauvegardes      :", config.FICHIER_RESULTATS_LIGHTGBM)
    print("Resultats par fenetre sauvegardes :", config.FICHIER_RESULTATS_LIGHTGBM_PAR_FENETRE)

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
    sans_shap = '--sans-shap' in sys.argv

    config.assurer_dossiers()
    rapports.assurer_dossier()
    rap = rapports.Rapport('06_lightgbm')

    panel = pd.read_parquet(config.FICHIER_PANEL_MODELISATION)
    print(f"Panel complet : {panel.shape}")
    rap.valeur('shape_panel', list(panel.shape))
    rap.valeur('periode_panel', [str(panel['annee_mois'].min()), str(panel['annee_mois'].max())])

    panel = fenetres.restreindre_debut_entrainement(panel, config.ANNEE_DEBUT_ENTRAINEMENT)
    rap.valeur('annee_debut_entrainement', config.ANNEE_DEBUT_ENTRAINEMENT)
    rap.valeur('shape_panel_entrainement', list(panel.shape))
    rap.valeur('periode_entrainement', [str(panel['annee_mois'].min()), str(panel['annee_mois'].max())])

    sortie = entrainer(panel, rap)
    sauvegarder(sortie, rap)
    rap.sauvegarder()  # l'entrainement (le plus long) n'est pas perdu si la suite plante

    importance_agregee(sortie, rap)
    rap.sauvegarder()

    if sans_shap:
        print("\n(--sans-shap : calcul SHAP saute)")
        rap.valeur('shap_calcule', False)
    else:
        calculer_shap(panel, sortie, rap)

    rap.sauvegarder()

    print("\n" + "=" * 70)
    print("ETAPE 06 TERMINEE.")
    print("  -> ouvre notebooks/06_modele_lightgbm.ipynb pour visualiser les resultats")
    print("=" * 70)


if __name__ == "__main__":
    main()
