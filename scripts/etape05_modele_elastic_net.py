"""
ETAPE 05 -- Modele Elastic Net (ex-notebook 05).

Ce script contient TOUT le calcul qui se trouvait auparavant dans les cellules du
notebook `05_modele_elastic_net.ipynb` : recherche d'hyperparametres sur la validation
de CHAQUE fenetre, entrainement, R2_oos pooled, stabilite de selection des variables,
sauvegardes et enregistrement au journal. Le notebook ne fait plus qu'AFFICHER.

Lancement, depuis la RACINE du projet :

    python scripts/etape05_modele_elastic_net.py

⚠️ Pre-requis : `python scripts/etape03_construction_panel.py` deja lance.

Parametres, tous dans config.py :
  - generaux   : PREDICTEURS, TYPE_FENETRE, ANNEE_DEBUT_ENTRAINEMENT, ANNEES_*
                 (partages avec 04 et 06)
  - specifiques: GRILLE_ALPHA_ELASTIC_NET, GRILLE_L1_RATIO_ELASTIC_NET, MAX_ITER_ELASTIC_NET

Fichiers produits (chemins definis dans config.py) :
  - modeles/elastic_net.joblib                      (modele de la DERNIERE fenetre)
  - outputs/predictions_elastic_net.parquet
  - outputs/resultats_elastic_net.parquet            (+ cle_experience)
  - outputs/resultats_elastic_net_par_fenetre.parquet
  - outputs/importance_elastic_net.parquet           (stability selection)
  - 1 ligne dans outputs/journal_experiences.parquet (sans doublon)
  - le rapport '05_elastic_net' (outputs/rapports/) pour le notebook
"""

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

import config
import fenetres
import journal
import rapports

warnings.filterwarnings('ignore', category=UserWarning)

MODELE = 'Elastic Net'


# ============================================================
# Section 3 a 5 -- recherche d'hyperparametres + entrainement, fenetre par fenetre
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

    # Parametres SPECIFIQUES a l'Elastic Net (config.py)
    grille_alpha = config.GRILLE_ALPHA_ELASTIC_NET
    grille_l1_ratio = config.GRILLE_L1_RATIO_ELASTIC_NET
    max_iter = config.MAX_ITER_ELASTIC_NET

    debut_entrainement = time.perf_counter()

    resultats_par_fenetre = []
    predictions_toutes_fenetres = []
    pool_train = {'y': [], 'pred': []}
    pool_validation = {'y': [], 'pred': []}
    coefficients_par_fenetre = []  # un dict {predicteur: coefficient} par fenetre
    grilles_par_fenetre = []       # LA grille complete de chaque fenetre (toutes combinaisons)
    modele_final = None

    for f in liste_fenetres:
        donnees = fenetres.preparer_fenetre(panel, f, predicteurs, macro_predicteurs, cible)

        # --- recherche d'hyperparametres sur la validation de CETTE fenetre ---
        # On garde le score de CHAQUE combinaison, sur le train ET sur la validation :
        # le tableau complet est sauvegarde dans le rapport et affiche au notebook 05
        # (heatmap + tableau), y compris l'ecart train-validation, qui mesure le
        # sur-apprentissage de chaque combinaison.
        recherche = []
        for alpha in grille_alpha:
            for l1_ratio in grille_l1_ratio:
                modele = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=0)
                modele.fit(donnees['X_train'], donnees['y_train'])
                score_train = r2_oos(donnees['y_train'], modele.predict(donnees['X_train']))
                score_validation = r2_oos(donnees['y_validation'], modele.predict(donnees['X_validation']))
                recherche.append({
                    'fenetre': f['numero'],
                    'annee_test': f['annee_test'],
                    'alpha': alpha,
                    'l1_ratio': l1_ratio,
                    'r2_oos_train': score_train,
                    'r2_oos_validation': score_validation,
                    'ecart_train_validation': score_train - score_validation,
                    'n_coefficients_non_nuls': int((modele.coef_ != 0).sum()),
                })

        recherche = pd.DataFrame(recherche)
        # Combinaison retenue pour CETTE fenetre : le meilleur R2_oos de validation
        index_meilleur = recherche['r2_oos_validation'].idxmax()
        recherche['selectionnee'] = False
        recherche.loc[index_meilleur, 'selectionnee'] = True
        grilles_par_fenetre.append(recherche)

        meilleur_alpha = recherche.loc[index_meilleur, 'alpha']
        meilleur_l1_ratio = recherche.loc[index_meilleur, 'l1_ratio']

        # --- modele final de cette fenetre ---
        modele = ElasticNet(alpha=meilleur_alpha, l1_ratio=meilleur_l1_ratio,
                            max_iter=max_iter, random_state=0)
        modele.fit(donnees['X_train'], donnees['y_train'])

        pred_train = modele.predict(donnees['X_train'])
        pred_validation = modele.predict(donnees['X_validation'])
        pred_test = modele.predict(donnees['X_test'])

        resultats_par_fenetre.append({
            'fenetre': f['numero'],
            'annee_test': f['annee_test'],
            'n_train': len(donnees['y_train']),
            'alpha': meilleur_alpha,
            'l1_ratio': meilleur_l1_ratio,
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
        coefficients_par_fenetre.append(dict(zip(predicteurs, modele.coef_)))

        modele_final = modele

        print(f"Fenetre {f['numero']:2d} (test {f['annee_test']:>9s}) : "
              f"alpha={meilleur_alpha:.1e}, l1_ratio={meilleur_l1_ratio:.1f} | "
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
    # C'est ce tableau qu'affiche le notebook 05 (section 3bis).
    grille_complete = pd.concat(grilles_par_fenetre, ignore_index=True)
    rap.table('grille_complete', grille_complete)
    rap.valeur('n_combinaisons_grille', len(grille_alpha) * len(grille_l1_ratio))
    rap.valeur('n_modeles_entraines_grille', int(len(grille_complete)))

    # --- Coefficients de la derniere fenetre (section 6 du notebook) ---
    coefficients = pd.Series(modele_final.coef_, index=predicteurs).sort_values(key=abs, ascending=False)
    rap.table('coefficients_derniere_fenetre', coefficients.rename('coefficient'))
    rap.valeur('n_coefficients_a_zero', int((coefficients == 0).sum()))

    return {
        'liste_fenetres': liste_fenetres,
        'resultats_par_fenetre': resultats_par_fenetre,
        'predictions': predictions_toutes_fenetres,
        'coefficients_par_fenetre': coefficients_par_fenetre,
        'modele_final': modele_final,
        'r2': (r2_train, r2_validation, r2_test),
        'duree': duree_entrainement_secondes,
        'params_specifiques': {
            'grille_alpha': grille_alpha,
            'grille_l1_ratio': grille_l1_ratio,
            'max_iter': max_iter,
        },
    }


# ============================================================
# Section 6bis -- Stabilite de la selection de variables (toutes les fenetres)
# ============================================================

def stabilite_selection(sortie, rap):
    """Stability selection (Meinshausen & Buhlmann, 2010) : sur TOUTES les fenetres
    entrainees, a quelle frequence chaque predicteur est-il retenu (coefficient != 0),
    avec quelle magnitude moyenne et quelle coherence de signe."""
    print("\n--- Section 6bis : stabilite de la selection de variables ---")

    coefficients_par_fenetre_df = pd.DataFrame(
        sortie['coefficients_par_fenetre'],
        index=sortie['resultats_par_fenetre']['fenetre'],
    )
    print(f"Coefficients sauvegardes pour {len(coefficients_par_fenetre_df)} fenetres "
          f"x {coefficients_par_fenetre_df.shape[1]} predicteurs.")

    frequence_selection = (coefficients_par_fenetre_df != 0).mean() * 100

    def moyenne_si_selectionne(colonne):
        non_nuls = colonne[colonne != 0]
        return non_nuls.mean() if len(non_nuls) > 0 else 0.0

    coef_moyen_selectionne = coefficients_par_fenetre_df.apply(moyenne_si_selectionne)

    def signe_majoritaire(colonne):
        non_nuls = colonne[colonne != 0]
        if len(non_nuls) == 0:
            return 0
        return 1 if (non_nuls > 0).mean() >= 0.5 else -1

    signe_dominant = coefficients_par_fenetre_df.apply(signe_majoritaire)

    def pct_accord_signe(colonne):
        non_nuls = colonne[colonne != 0]
        if len(non_nuls) == 0:
            return np.nan
        majorite = 1 if (non_nuls > 0).mean() >= 0.5 else -1
        return (np.sign(non_nuls) == majorite).mean() * 100

    coherence_signe = coefficients_par_fenetre_df.apply(pct_accord_signe)

    stabilite = pd.DataFrame({
        'frequence_selection_pct': frequence_selection,
        'coef_moyen_si_selectionne': coef_moyen_selectionne,
        'signe_dominant': signe_dominant,
        'coherence_signe_pct': coherence_signe,
    }).sort_values(['frequence_selection_pct', 'coef_moyen_si_selectionne'], key=abs, ascending=False)

    n_toujours = int((stabilite['frequence_selection_pct'] == 100).sum())
    n_jamais = int((stabilite['frequence_selection_pct'] == 0).sum())
    print(f"{n_toujours} / {len(stabilite)} predicteurs selectionnes dans TOUTES les fenetres.")
    print(f"{n_jamais} / {len(stabilite)} predicteurs jamais selectionnes.")

    stabilite.to_parquet(config.FICHIER_IMPORTANCE_ELASTIC_NET)
    print("Tableau de stabilite sauvegarde :", config.FICHIER_IMPORTANCE_ELASTIC_NET)

    rap.valeur('stab_n_toujours_selectionnes', n_toujours)
    rap.valeur('stab_n_jamais_selectionnes', n_jamais)
    rap.valeur('stab_n_predicteurs', int(len(stabilite)))
    # Coefficients bruts par fenetre : utile pour un diagnostic fin dans le notebook
    rap.table('coefficients_par_fenetre', coefficients_par_fenetre_df)


# ============================================================
# Sections 7/8 -- Sauvegardes et journal des experiences
# ============================================================

def sauvegarder(sortie, rap):
    r2_train, r2_validation, r2_test = sortie['r2']

    joblib.dump(sortie['modele_final'], config.FICHIER_MODELE_ELASTIC_NET)
    print("\nModele final (derniere fenetre) sauvegarde :", config.FICHIER_MODELE_ELASTIC_NET)

    sortie['predictions'].to_parquet(config.FICHIER_PREDICTIONS_ELASTIC_NET, index=False)
    print("Predictions (toutes fenetres) sauvegardees :", config.FICHIER_PREDICTIONS_ELASTIC_NET)

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

    resultats_finaux.to_parquet(config.FICHIER_RESULTATS_ELASTIC_NET, index=False)
    sortie['resultats_par_fenetre'].to_parquet(config.FICHIER_RESULTATS_ELASTIC_NET_PAR_FENETRE, index=False)
    print("Resultats pooled sauvegardes      :", config.FICHIER_RESULTATS_ELASTIC_NET)
    print("Resultats par fenetre sauvegardes :", config.FICHIER_RESULTATS_ELASTIC_NET_PAR_FENETRE)

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
    config.assurer_dossiers()
    rapports.assurer_dossier()
    rap = rapports.Rapport('05_elastic_net')

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
    rap.sauvegarder()

    stabilite_selection(sortie, rap)
    rap.sauvegarder()

    print("\n" + "=" * 70)
    print("ETAPE 05 TERMINEE.")
    print("  -> ouvre notebooks/05_modele_elastic_net.ipynb pour visualiser les resultats")
    print("=" * 70)


if __name__ == "__main__":
    main()
