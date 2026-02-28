"""
Module d'importation de données historiques depuis un fichier CSV vers PostgreSQL.

Ce module permet d'étoffer la base de données avec des datasets pré-existants.
Il intègre une logique de dédoublonnage par hachage (SHA-256) pour éviter les
insertions redondantes et assure la traçabilité de l'opération via le système
de logging applicatif.
"""

# imports

import time
from pathlib import Path
from typing import Union

import pandas as pd
from sqlalchemy import inspect

from app.db.database import get_db_contextmanager
from app.db.models_db import PredictionRecord
from app.utils.hash_id import generate_feature_hash
from app.utils.logger_db import closing_log, init_log
from app.utils.save_load_datas import load_datas

# =============================


def import_historical_data(file_path: Union[Path, str]):
    """
    Lit un fichier csv/parquet et importe les enregistrements uniques dans
    la table PredictionRecord.

    Le processus suit les étapes suivantes :
    1. Chargement du fichier via Pandas et conversion des NaN en 'None' pour compatibilité JSON.
    2. Initialisation d'un log d'activité pour l'endpoint virtuel '/import'.
    3. Pour chaque ligne :
        extraction de la target, génération d'un hash ID unique sur les features.
    4. Vérification de l'existence de l'ID en base pour ignorer les doublons.
    5. Construction et insertion massive (bulk insert) des nouveaux enregistrements.

    Args:
        file_path (str):
            Chemin local vers le fichier csv/parquet contenant les données historiques.

    Raises:
        Exception: En cas d'erreur de lecture, de hachage ou de contrainte d'intégrité SQL,
            une annulation (rollback) est effectuée.

    Note:
        - La 'confidence' est fixée à 1.0 car il s'agit de données historiques observées.
        - Le statut HTTP 201 est loggé en cas de succès avec insertion.
        - Le statut HTTP 204 est loggé si aucun nouvel enregistrement n'a été trouvé.
    """
    start_time = time.time()
    path = Path(file_path)  # On s'assure d'avoir un format Path
    df, _ = load_datas(path)  # S'assurer que c'est un pd.DataFrame

    # Remplacer les NaN par None (car NaN n'est pas un JSON valide)
    df = df.where(pd.notnull(df), None)

    with get_db_contextmanager() as db:
        try:
            # L'endpoint de l'import n'existe pas
            log_entry = init_log(db, f"/import/{path.suffix[1:]}")
            print("Importation de données historique ...")

            new_records = []
            for _, row in df.iterrows():
                features = row.to_dict()
                # Conversion pour éviter les erreurs de sérialisation NumPy -> JSON
                features = {
                    key: (value.item() if hasattr(value, "item") else value)
                    for key, value in features.items()
                }
                # on retire la target pour ne hasher que les features
                target = features.pop("TARGET", None)

                unique_id = generate_feature_hash(features)
                # Vérification si cet ID existe déjà
                if db.get(PredictionRecord, unique_id):
                    continue

                # On constitue un dictionnaire complet qui répond aux exigences de l'UML
                assemble = {
                    "id": unique_id,
                    # "EXT_SOURCE_1" : features.get('EXT_SOURCE_1'),
                    # "EXT_SOURCE_2" : features.get('EXT_SOURCE_2'),
                    # "EXT_SOURCE_3" : features.get('EXT_SOURCE_3'),
                    "EXT_SOURCE_COUNT": features.get("EXT_SOURCE_COUNT"),
                    "AMT_ANNUITY": features.get("AMT_ANNUITY"),
                    # "AMT_CREDIT" : features.get('AMT_CREDIT'),
                    "PAYMENT_RATE": features.get("PAYMENT_RATE"),
                    # "AMT_CREDIT_SUM_DEBT" : features.get('AMT_CREDIT_SUM_DEBT'),
                    "BUREAU_AMT_CREDIT_SUM_DEBT_MEAN": features.get(
                        "BUREAU_AMT_CREDIT_SUM_DEBT_MEAN"
                    ),
                    "DAYS_BIRTH": features.get("DAYS_BIRTH"),
                    "DAYS_EMPLOYED": features.get("DAYS_EMPLOYED"),
                    "OWN_CAR_AGE": features.get("OWN_CAR_AGE"),
                    # "DAYS_LAST_DUE_1ST_VERSION" : features.get('DAYS_LAST_DUE_1ST_VERSION'),
                    "PREV_DAYS_LAST_DUE_1ST_VERSION_MAX": features.get(
                        "PREV_DAYS_LAST_DUE_1ST_VERSION_MAX"
                    ),
                    "NAME_FAMILY_STATUS": features.get("NAME_FAMILY_STATUS"),
                    "inputs": features,
                    "prediction": int(target) if target is not None else None,
                    "confidence": 1.0,  # données historique donc forcement 1.0
                    "class_name": "insolvable" if target == 1 else "solvable",
                    "TARGET": (int(target) if target is not None else None),
                    "log_id": log_entry.id,
                }
                # --------- Dangereux pour 800+ features ----------
                # assemble.update(features)

                # # On unpack suivant le model UML
                # record = PredictionRecord(**assemble)
                # new_records.append(record)
                # -------------------------------------------
                # OPTIMISATION : On ne garde que ce qui appartient au modèle SQL
                # Rend l'existence des features dynamique, inspect va checker PredictionRecord
                # et s'adapter par exemple si une nouvelle colonne est ajouté,
                # il la prendra en compte.
                inspection = inspect(PredictionRecord)
                columns = [col_attr.key for col_attr in inspection.mapper.column_attrs]
                # On ne prend que les clés qui sont des colonnes SQL
                valid_data = {key: val for key, val in assemble.items() if key in columns}
                # On ajoute aussi les features si elles correspondent à des colonnes SQL
                sql_features = {key: val for key, val in features.items() if key in columns}
                valid_data.update(sql_features)

                record = PredictionRecord(**valid_data)
                new_records.append(record)

            if new_records:
                db.add_all(new_records)
                db.flush()  # permettra d'anticper les erreurs SQL
                # On logue avec inference_time=0 car c'est un import, pas une prédiction
                closing_log(db, log_entry, start_time, status_code=201, inference_time=0.0)
                print("Importation réussie.")
            else:
                # code 204 = No content
                closing_log(db, log_entry, start_time, status_code=204, inference_time=0.0)
                print("Pas d'importation nécéssaire")

        except Exception as e:
            db.rollback()
            print(f"Erreur lors de l'importation : {e}")
            raise e


if __name__ == "__main__":  # pragma: no cover
    from pathlib import Path

    # On remonte a root/
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "datas" / "raw"

    for filename in ["Xy_hist.csv", "Xy_hist.parquet"]:
        file_to_import = DATA_DIR / filename
        if file_to_import.exists():
            print(f"\n--- Lancement de l'import : {filename} ---")
            import_historical_data(file_to_import)
        else:
            print(f"Fichier non trouvé : {file_to_import}")
