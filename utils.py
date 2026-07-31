"""
Petits utilitaires partages par journal.py et pipeline.py.

Rien a voir avec la logique metier du projet -- juste de la plomberie commune aux deux
fichiers, pour eviter de dupliquer la meme fonction a deux endroits (meme esprit que la
separation config.py / fenetres.py / journal.py : une seule version de chaque bout de
logique, reutilisee partout ou elle sert).
"""

import numpy as np


def nettoyer_pour_json(valeur):
    """Convertit recursivement les types numpy (array, float64, int64...) en types
    natifs Python, pour pouvoir serialiser en JSON. Utilise a chaque fois qu'on doit
    comparer ou sauvegarder des parametres issus de config.py :
    - journal.py : cle de deduplication des experiences, colonne 'params_specifiques_json'
    - pipeline.py : detection des parametres changes depuis le dernier lancement d'un notebook
    """
    if isinstance(valeur, dict):
        return {cle: nettoyer_pour_json(v) for cle, v in valeur.items()}
    if isinstance(valeur, (list, tuple, np.ndarray)):
        return [nettoyer_pour_json(v) for v in valeur]
    if isinstance(valeur, np.integer):
        return int(valeur)
    if isinstance(valeur, np.floating):
        return float(valeur)
    return valeur
