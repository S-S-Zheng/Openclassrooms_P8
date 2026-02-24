"""
Module de définition des modèles de données ORM (Object-Relational Mapping).

Ce module contient les schémas SQL pour PostgreSQL via SQLAlchemy. Il définit
l'organisation des données stockées, incluant les enregistrements de prédictions
détaillés et le système de journalisation (logging) pour la traçabilité des requêtes.
"""

# Pydantic définit la forme des données qui entrent/sortent,
# SQLAlchemy définit la forme des données qui dorment en base.

# imports
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base import Base

# ============== Tables ========================


# Traçabilité
class RequestLog(Base):
    """
    Modèle représentant les logs d'activité de l'API.

    Stocke les métadonnées de chaque requête entrante pour permettre l'audit de performance
    et la traçabilité des erreurs.
    Chaque log peut être lié à un enregistrement de prédiction spécifique.

    Attributes:
        id (int): Clé primaire auto-incrémentée.
        created_at (datetime): Horodatage de la requête (géré par le serveur SQL).
        endpoint (str): Le point d'entrée API sollicité (ex: '/predict').
        status_code (int): Le code de statut HTTP retourné (ex: 200, 422, 500).
        response_time_ms (float): Temps de traitement de la requête en millisecondes.
        inference_time_ms (float): Temps d'inférence en ms
        prediction_id (str): Clé étrangère pointant vers l'ID unique de la prédiction associée.
        prediction_record (relationship): Relation ORM vers l'objet PredictionRecord correspondant.
    """

    __tablename__ = "request_logs"

    # Identifications
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Intéractions
    endpoint = Column(String, default="/predict")
    status_code = Column(Integer)

    # Temps total (Route + DB + Modèle)
    response_time_ms = Column(Float)
    # Temps spécifique au calcul ML
    inference_time_ms = Column(Float)

    # Métriques Système
    cpu_usage = Column(Float)  # % CPU global
    memory_usage = Column(Float)  # RAM utilisée par le process en Mo

    # Relations
    # Crée une dépendance des ID avec la table predictions (permet la jointure)
    prediction_id = Column(String(64), ForeignKey("predictions.id"))
    # Créée une relation bidirectionnelle entre log et record
    prediction_record = relationship("PredictionRecord", back_populates="logs")


# ===================================================================


# Requete utilisateur
class PredictionRecord(Base):
    """
    Modèle représentant une prédiction stockée et ses caractéristiques d'entrée.

    Cette table contient l'ensemble des variables métier (features) envoyées par l'utilisateur,
    le résultat du modèle ML, et une version sérialisée (JSONB) pour la flexibilité.
    L'ID est un hash SHA-256 des entrées servant de mécanisme de dédoublonnage.

    Attributes:
        id (str): Hash unique des features servant de clé primaire.
        created_at (datetime): Date d'enregistrement.
        inputs (JSONB): Copie de sauvegarde de l'intégralité des entrées au format JSON.
        prediction (int): Classe prédite par le modèle (0 ou 1).
        confidence (float): Score de probabilité associé à la prédiction.
        class_name (str): Traduction textuelle de la classe (ex: 'Employé', 'Démissionnaire').
        model_version (str): Version du modèle utilisé lors de l'inférence.
        logs (relationship): Liste des logs de requêtes ayant sollicité cette prédiction précise.
    """

    __tablename__ = "predictions"

    # Identifications
    id = Column(String(64), primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Features
    # EXT_SOURCE_1 = Column(Float)
    # EXT_SOURCE_2 = Column(Float)
    # EXT_SOURCE_3 = Column(Float)
    # AMT_CREDIT = Column(Float)
    # AMT_CREDIT_SUM_DEBT = Column(Float)
    # DAYS_LAST_DUE_1ST_VERSION = Column(Float)
    EXT_SOURCE_COUNT = Column(Float)
    DAYS_EMPLOYED = Column(Float)
    DAYS_BIRTH = Column(Integer)
    AMT_ANNUITY = Column(Float)
    PAYMENT_RATE = Column(Float)
    PREV_DAYS_LAST_DUE_1ST_VERSION_MAX = Column(Float)
    BUREAU_AMT_CREDIT_SUM_DEBT_MEAN = Column(Float)
    OWN_CAR_AGE = Column(Float)
    NAME_FAMILY_STATUS = Column(String)

    # Target
    TARGET = Column(Integer, nullable=True)

    # Inputs condensé
    inputs = Column(JSONB)

    # Prédiction
    prediction = Column(Integer)
    confidence = Column(Float)
    class_name = Column(String)

    model_version = Column(String, default="v1.0.0")

    logs = relationship("RequestLog", back_populates="prediction_record")
