from typing import Any

import numpy as np
import pandas as pd


def clean_for_json(obj: Any) -> Any:
    """
    Nettoie récursivement un objet pour le rendre compatible avec le format JSON strict.

    - Convertit les NaN, +Inf, -Inf en None (null en SQL).
    - Convertit les types natifs NumPy (int64, float32) en types Python standards.
    - Parcourt les dictionnaires et les listes.
    """
    if isinstance(obj, dict):
        return {key: clean_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_for_json(i) for i in obj]

    # Gestion des valeurs numériques non finies (NaN, Inf)
    # pd.isna() détecte None, np.nan, et les NA de pandas.
    if pd.isna(obj) or (isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj))):
        return None

    # Gestion des scalaires NumPy (ex: np.int64 -> int)
    # On vérifie si c'est un type NumPy pour appeler .item() en toute sécurité
    if isinstance(obj, np.generic):
        return obj.item()

    # Retour par défaut pour les types natifs (str, int, float valides, bool)
    return obj
