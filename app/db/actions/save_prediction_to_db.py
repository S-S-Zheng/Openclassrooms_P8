"""
Module de persistance des prédictions dans la base de données.

Ce module assure l'enregistrement des résultats d'inférence, la gestion
de la concurrence pour éviter les doublons accidentels et la traçabilité
des opérations en liant les enregistrements aux logs de requêtes.
"""

# imports

from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from app.db.models_db import PredictionRecord
from app.utils.hash_id import generate_feature_hash
from app.utils.logger_db import link_log

# ==========================


def save_prediction(db: Session, features: dict, pred_data: tuple, log_id: int):
    """
    Enregistre une nouvelle prédiction et la lie à un log d'activité.

    Cette fonction effectue les étapes suivantes :
    1. Calcule un hash SHA-256 unique à partir des features d'entrées (la requete).
    2. Vérifie une dernière fois l'existence de l'ID (protection contre les 'race conditions').
    3. Si nouveau, crée un enregistrement 'PredictionRecord' en dépaquetant les features.
    4. Établit une relation de traçabilité entre la prédiction et le log via 'link_log'.

    Args:
        db (Session): Session SQLAlchemy active pour les transactions.
        features (dict): Dictionnaire des variables d'entrée utilisées pour la prédiction.
        pred_data (tuple): Un tuple contenant (valeur_prediction, confiance, nom_classe).
            Exemple : (1, 0.85, "Démissionnaire").
        log_id (int): Identifiant unique de la ligne de log (RequestLog) à l'origine de l'action.

    Returns:
        str: L'identifiant unique (hash hexadécimal) de la requête enregistrée.

    Raises:
        Exception: Relance toute exception survenant lors de la transaction après avoir
            effectué un rollback pour maintenir l'intégrité de la base.

    Note:
        L'utilisation de `db.flush()` permet d'envoyer l'objet à la base de données
        pour obtenir une confirmation sans pour autant clore la transaction globale
        (permettant le rollback en cas d'échec de la liaison des logs).
    """
    prediction, confidence, class_name = pred_data
    # Generation ID de caractères hexadecimaux
    request_id = generate_feature_hash(features)

    # Rend l'existence des features dynamique, inspect va checker PredictionRecord et s'adapter
    # par exemple si une nouvelle colonne est ajouté, il la prendra en compte.
    inspection = inspect(PredictionRecord)
    columns = [col_attr.key for col_attr in inspection.mapper.column_attrs]
    # Filtrage pour l'unpacking (**features)
    sql_features = {key: val for key, val in features.items() if key in columns}

    try:
        # Si deux requetes par ex étaient lancé en meme temps, get_prediction se ferait avoir
        existing = db.scalars(
            select(PredictionRecord).where(PredictionRecord.id == request_id)
        ).first()

        if not existing:
            new_record = PredictionRecord(
                id=request_id,
                inputs=features,  # On y inscrit l'ensemble des features
                prediction=int(prediction),
                confidence=float(confidence),
                class_name=class_name,
                **sql_features,  # On unpack les features définies
            )
            db.add(new_record)
            db.flush()

        # Liaison avec le log pour la traça
        link_log(db, log_id, request_id)

        return request_id

    except Exception as e:
        db.rollback()
        # On relance l'erreur pour que l'API puisse la gérer ou la logger
        raise e
