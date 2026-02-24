"""
Module de définition du router pour l'accès aux métadonnées du modèle.

Ce module expose un endpoint permettant de consulter les caractéristiques
techniques et la configuration du modèle en production. Il fournit une
transparence sur les variables attendues, le type d'algorithme et les
seuils de décision appliqués lors de l'inférence.
"""

# ====================== Imports ========================
from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import ModelInfoOutput

router = APIRouter(prefix="/model-info", tags=["Model informations"])

# ===================== Initialisation du modele =========================


@router.get("/", response_model=ModelInfoOutput)
async def model_info(request: Request):
    """
    Récupère les métadonnées et la configuration technique du modèle ML.

    Cet endpoint permet de vérifier l'état du modèle chargé en mémoire et
    d'obtenir des informations cruciales pour l'intégration client, telles que
    la liste ordonnée des variables (features) et le seuil de classification
    (threshold) utilisé pour séparer les classes.

    Args:
        request (Request): Objet requête FastAPI permettant d'accéder au 'state'
            global de l'application où réside l'instance du modèle.

    Returns:
        ModelInfoOutput: Un objet contenant le type de modèle, le nombre de features,
            les noms des variables (catégorielles et numériques), les labels de
            classes et le seuil de décision.

    Raises:
        HTTPException (503): Si l'instance du modèle n'est pas initialisée dans
            le state de l'application.
        HTTPException (422): Si une erreur de validation survient lors de la
            récupération des informations du modèle.
        HTTPException (500): En cas d'erreur interne imprévue sur le serveur.
    """

    # Récupération de l'instance du modèle depuis le state de l'application
    model_instance = getattr(request.app.state, "model", None)

    if model_instance is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    try:
        info = model_instance.get_model_info()

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return info
