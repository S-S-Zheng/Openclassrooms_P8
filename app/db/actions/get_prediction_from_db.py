"""
Module de gestion des doublons dans la base de données.

Ce module permet de vérifier l'existence d'une prédiction avant toute nouvelle
insertion afin de garantir l'intégrité des données et d'optimiser les performances
en évitant les calculs redondants.
"""

# imports

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models_db import PredictionRecord
from app.utils.hash_id import generate_feature_hash


# ==========================
# Garde-fou pour empêcher de save plusieurs fois même requete dans db
def get_prediction_id(db: Session, features: dict) -> PredictionRecord:
    """
    Recherche une prédiction existante basée sur le hash des caractéristiques.

    Cette fonction calcule un identifiant unique (SHA-256) à partir des 'features'
    fournies, puis interroge la table 'PredictionRecord' pour voir si cet ID
    est déjà présent. Cela sert de mécanisme de cache et de protection contre les doublons.

    Args:
        db (Session): La session active de la base de données SQLAlchemy.
        features (dict): Le dictionnaire contenant les caractéristiques de l'employé
            (ex: age, salaire, poste, etc.).

    Returns:
        PredictionRecord | None: Retourne l'objet de prédiction complet si trouvé,
            sinon None si aucune requête identique n'a été enregistrée.

    Example:
        >>> result = get_prediction_id(db_session, {"age": 30, "salaire": 5000})
        >>> if result:
        ...     print(f"Prédiction trouvée : {result.prediction}")
    """
    # On recalcul le hash des features recues
    feature_id = generate_feature_hash(features)

    return db.scalars(select(PredictionRecord).where(PredictionRecord.id == feature_id)).first()
