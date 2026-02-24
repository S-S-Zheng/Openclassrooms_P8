"""
Module de définition des schémas Pydantic pour l'API.

Ce module définit les contrats d'interface (Data Transfer Objects) pour les requêtes
et les réponses de l'API FastAPI. Il inclut une couche de validation métier robuste
pour garantir que les données envoyées au modèle CatBoost respectent les plages
de valeurs attendues.

Schémas principaux :
    - PredictionInput : Données d'entrée pour l'inférence avec validation métier.
    - PredictionOutput : Résultat de la prédiction (classe et confiance).
    - ModelInfoOutput : Métadonnées et configuration du modèle ML.

| Endpoint               | Méthode ML                       |
| ---------------------- | -------------------------------- |
| `/predict`             | `MLModel.predict`                |
| `/health`              | `ml.model is not None`           |
| `/model-info`          | attributs du modèle              |
"""

from typing import List, Literal, Optional

# Imports
from pydantic import BaseModel, ConfigDict, Field

# ========================= PREDICTION ===========================


# Schéma pour les entrées de prédiction
# L'ordre des features est garantie par feature_names dans MLModel
class PredictionInput(BaseModel):
    """
    Schéma d'entrée pour la prédiction d'attrition.
    """

    # ------------------------- VARIABLES OBLIGATOIRES ----------------------------
    # Variables numériques classiques
    # EXT_SOURCE_1: float = Field(
    #     ...,
    #     ge=0,
    #     le=1,
    #     description="Score normalisé d'une source de données externe 1"
    # )
    # EXT_SOURCE_2: float = Field(
    #     ...,
    #     ge=0,
    #     le=1,
    #     description="Score normalisé d'une source de données externe 2"
    # )
    # EXT_SOURCE_3: float = Field(
    #     ...,
    #     ge=0,
    #     le=1,
    #     description="Score normalisé d'une source de données externe 3"
    # )
    EXT_SOURCE_COUNT: float = Field(
        ...,
        ge=0.0,
        le=3.0,
        description="Score normalisé de la somme (3) des sources de données externe",
    )

    AMT_ANNUITY: float = Field(..., gt=0, description="Montant des annuités/mensualités")

    PAYMENT_RATE: float = Field(..., gt=0, description="Rapidité de remboursement du client")

    # AMT_CREDIT:float = Field(
    #     ...,
    #     gt=0,
    #     description="Montant total du crédit demandé"
    # )

    # AMT_CREDIT_SUM_DEBT:float= Field(
    #     ...,
    #     gt=0,
    #     description="Montant de dette restante pour un crédit déclaré a Bureau Credit"
    # )

    BUREAU_AMT_CREDIT_SUM_DEBT_MEAN: float = Field(
        ..., description="Total des dette moyenné rapporté au bureau"
    )

    # Variables temporelles (souvent en jours négatifs dans Kaggle Home Credit)
    DAYS_BIRTH: int = Field(
        ...,
        ge=-25229,  # 69 ans
        le=-7489,  # 20 ans
        description="Âge en jours (valeur négative)",
    )

    DAYS_EMPLOYED: float = Field(
        ...,
        ge=-17912.0,  # 49 ans d'emploi
        le=0,
        description="Nombre de jours d'emploi (valeur négative)",
    )

    OWN_CAR_AGE: float = Field(
        ..., ge=0, description="Âge du véhicule (en année, nan si pas de voiture)"
    )

    # DAYS_LAST_DUE_1ST_VERSION: float = Field(
    #     ...,
    #     ge=-17912, # 49 ans
    #     description="Nb de j depuis dernier paiement (V1 du contrat). 0 < passé, 365243 = nan"
    # )
    PREV_DAYS_LAST_DUE_1ST_VERSION_MAX: float = Field(
        ..., description="Nb de j depuis dernier paiement (V1 du contrat)."
    )

    # Variables catégorielles (à envoyer en chaînes de caractères)
    NAME_FAMILY_STATUS: Literal[
        "Married", "Single / not married", "Civil marriage", "Separated", "Widow", "Unknown"
    ] = Field(
        ...,
        description=(
            'Situation familiale ("Married","Single / not married",'
            '"Civil marriage","Separated","Widow","Unknown")'
        ),
    )

    # ------------------------- VARIABLES OPTIONNELLES ----------------------------
    # Configuration Pydantic v2 pour autoriser les 400+ autres features
    model_config = ConfigDict(
        extra="allow",  # Autorise les données supplémentaires sans erreur
        json_schema_extra={
            "example":
            # {
            #     "EXT_SOURCE_1": 1, "EXT_SOURCE_2": 1, "EXT_SOURCE_3": 1,
            #     "AMT_ANNUITY": 24700.5, "AMT_CREDIT": 406597.5, "AMT_CREDIT_SUM_DEBT": 0.0,
            #     "DAYS_BIRTH": -9461, "DAYS_EMPLOYED": -637, "OWN_CAR_AGE": 4,
            #     "DAYS_LAST_DUE_1ST_VERSION": -500, "NAME_FAMILY_STATUS": "Married",
            #     "OTHER_FEATURE_1": 0.5, "OTHER_FEATURE_400": 12.0 # Exemples de champs extra
            # }
            {
                "EXT_SOURCE_COUNT": 1.0,
                "DAYS_EMPLOYED": -637.0,
                "DAYS_BIRTH": -9461,
                "AMT_ANNUITY": 24700.5,
                "PAYMENT_RATE": 0.1,
                "PREV_DAYS_LAST_DUE_1ST_VERSION_MAX": 0.0,
                "BUREAU_AMT_CREDIT_SUM_DEBT_MEAN": 100000.0,
                "OWN_CAR_AGE": 40.0,
                "NAME_FAMILY_STATUS": "Married",
                "OTHER_FEATURE_1": 0.5,
                "OTHER_FEATURE_400": 12.0,
            }
        },
    )


# Schéma pour les sorties de prédiction
class PredictionOutput(BaseModel):
    """
    Schéma de réponse après une prédiction.
    """

    prediction: int = Field(..., description="Classe prédite (0: solvable, 1: insolvable)")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Probabilité associée à la prédiction"
    )
    class_name: str = Field(..., description="solvable,insolvable")
    latency_ms: Optional[float] = None


# ========================= METADATAS ===========================


# Métadatas du modèle
class ModelInfoOutput(BaseModel):
    """
    Schéma de réponse détaillant la configuration interne du modèle ML.
    """

    model_type: str = Field(..., description="Algorithme utilisé (ex: CatBoostClassifier)")
    n_features: int = Field(..., description="Nombre total de variables d'entrée")
    feature_names: List[str] = Field(..., description="Noms des variables dans l'ordre attendu")
    classes: List[str] = Field(..., description="Labels des classes de prédiction")
    threshold: float = Field(..., description="Seuil de décision optimisé")


# Erreur standardisée
class ErrorResponse(BaseModel):
    """
    Schéma standardisé pour les messages d'erreur de l'API.
    """

    error: str = Field(..., description="Type ou titre de l'erreur")
    detail: str | None = Field(None, description="Détails supplémentaires sur la cause de l'erreur")
