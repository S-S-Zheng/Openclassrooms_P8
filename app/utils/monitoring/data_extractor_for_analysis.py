from pathlib import Path

# from sqlalchemy import create_engine
from typing import Optional, Union

import pandas as pd

from app.utils.save_load_datas import load_datas


class DataProvider:
    """
    Fournisseur de données pour l'analyse de drift.
    Centralise l'extraction SQL et le chargement des référentiels.
    """

    def __init__(self, db_url: Optional[Union[str, Path]] = None):
        """
        Initialise la connexion à la base de données.\n
        REMARQUE: Pour éviter de complexifier pour le moment, on ne passera pas par l'url.

        Args:
            db_url (str): URL de connexion PostgreSQL (Supabase).
        """
        # self.engine = create_engine(db_url)
        pass

    # def fetch_datas_from_db(self, limit: int = 1000) -> pd.DataFrame:
    #     """
    #     Récupère les dernières données de production depuis Supabase.

    #     Args:
    #         limit(int): . Défaut 1000

    #     Returns:
    #         pd.DataFrame: Les features, prédictions et scores de confiance.
    #     """
    #     # On récupère les données pour l'analyse
    #     query = f"SELECT * FROM predictions ORDER BY created_at DESC LIMIT {limit}"
    #     return pd.read_sql(query, self.engine)
    def load_potential_data_drift(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Charge le dataset courant qui pourrait avoir subi du data drifting.
        """
        return load_datas(file_path)[0]

    def load_reference_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Charge le dataset de référence (données d'entraînement).
        """
        return load_datas(file_path)[0]

    def align_datasets(
        self, reference: pd.DataFrame, current: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Assure que les deux datasets ont exactement les mêmes colonnes.
        Indispensable avant de les passer à Evidently.
        """
        common_cols = list(set(reference.columns) & set(current.columns))
        # On trie pour garantir l'ordre des colonnes
        common_cols.sort()
        return reference[common_cols], current[common_cols]
