# Imports
from typing import Any, List

import pandas as pd

from app.utils.monitoring.profiling import get_profile

# ===========================================================


class InputPreproc:
    def __init__(
        self,
        model_full_feature_names: List[str],
    ):
        """
        Prépare la donnée d'entrée afin que même lorsque l'utilisateur ne rentre que les features
        obligatoires, le reste apparaissent (df.reindex) et dans le bon ordre avec une valeur par
        défaut (365243.0 pour les numérique et Unknown pour les numériques)

        Args:
            model_full_feature_names(List[str]):
                Liste des features avec lsquels le modèle s'est entrainé
        """
        self.feature_names = model_full_feature_names

    @get_profile
    def process(self, data: Any) -> pd.DataFrame:
        """
        Prépare les données d'entrées.

        Args:
            data(Any): Données d'entrées

        Returns:
            pd.DataFrame
        """
        # Conversion consistante en DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        # Alignement strict sur les 800+ features
        df = df.reindex(columns=self.feature_names)

        # Séparer le traitement pour ne pas polluer les strings
        # On définit les colonnes que le modèle considère comme catégorielles
        cat_cols_model = [
            "NAME_CONTRACT_TYPE",
            "CODE_GENDER",
            "NAME_TYPE_SUITE",
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "OCCUPATION_TYPE",
            "WEEKDAY_APPR_PROCESS_START",
            "ORGANIZATION_TYPE",
            "FONDKAPREMONT_MODE",
            "HOUSETYPE_MODE",
            "WALLSMATERIAL_MODE",
            "EMERGENCYSTATE_MODE",
        ]

        for col in df.columns:
            if col in cat_cols_model:
                # IMPORTANT : Le OneHotEncoder du modèle a besoin de STRINGS
                # On remplace les NaNs par une chaîne, pas par un nombre
                df[col] = df[col].fillna("Unknown").astype(str)
            else:
                # Pour le numérique : on garde tes 365243.0
                # Mais on utilise errors='coerce' pour être sûr de ne pas avoir de restes de texte
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(365243.0)
        return df
