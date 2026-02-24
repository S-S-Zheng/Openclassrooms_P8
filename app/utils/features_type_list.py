from typing import List, Tuple

import numpy as np
import pandas as pd

# ===========================================================================


# Fonction pour lister les features numériques et les catégorielles
def features_type(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Créée deux listes à partir d'une dataframe:
    num_list: liste des indicateurs à valeur numérique
    cat_list: liste des indicateurs categorielles
    """

    num_list = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_list = df.select_dtypes(exclude=[np.number]).columns.tolist()

    return num_list, cat_list
