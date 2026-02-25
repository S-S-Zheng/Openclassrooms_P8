"""
Module de définition du router pour les prédictions d'attrition.

Ce module constitue le point d'entrée principal de l'intelligence artificielle.
Il orchestre le flux de données complet : réception de la requête, journalisation
initiale, vérification de l'existence d'un cache en base de données, exécution
de l'inférence par le modèle CatBoost, persistance du résultat et retour de la
réponse à l'utilisateur.
"""

# ====================== Imports ========================
import io
import time
from typing import cast

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from app.api.schemas import PredictionInput, PredictionOutput
from app.db.database import get_db
from app.utils.clean_for_json_db import clean_for_json
from app.utils.inference_process import batch_prediction_pipeline, prediction_pipeline
from app.utils.input_preproc import InputPreproc
from app.utils.logger_db import closing_log, init_log

router = APIRouter(prefix="/predict", tags=["Prediction"])


# ===================== Initialisation du modele =========================
@router.post("/manual", response_model=PredictionOutput)
async def predict_manual(request: Request, payload: PredictionInput, db: Session = Depends(get_db)):
    """
    Gère une requête manuelle unique avec validation Pydantic et Cache.

    Cette méthode suit la pipeline suivante :

    1. **Logging** : Initialisation d'un enregistrement dans 'request_logs'.
    2. **Mise en cache** : Vérification si les caractéristiques ont déjà été traitées
       (via un hash SHA-256) pour retourner un résultat instantané.
    3. **Inférence** : Si nouveau, appel de la méthode predict du modèle chargé.
    4. **Persistance** : Sauvegarde des entrées et de la sortie dans 'predictions'.
    5. **Finalisation** : Calcul du temps de réponse et mise à jour du log.

    Args:
        request (Request): Objet requête FastAPI pour accéder au modèle global.
        payload (PredictionInput): Dictionnaire entré manuellement
        db (Session): Session de base de données injectée par dépendance.

    Returns:
        PredictionOutput: Résultat comprenant la prédiction (0/1), le score de
        confiance et le nom de la classe.

    Raises:
        HTTPException: 503 si le modèle n'est pas chargé.
        HTTPException: 422 en cas d'erreur de valeur lors de l'inférence.
        HTTPException: 500 pour les erreurs serveur imprévues.
    """
    # On initialise le temps et le log
    start_time = time.time()
    log_entry = init_log(db, "/predict/manual")

    # Récupération de l'instance du modèle depuis le state de l'application
    model_instance = getattr(request.app.state, "model", None)
    if not model_instance:
        closing_log(db, log_entry, start_time, status_code=503)
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    try:
        # Instancie l'objet InputPreproc avec injection de la liste des features et l'ordre
        preproc = InputPreproc(model_instance.feature_names)

        # NETTOYAGE JSON des données entrantes (pour éviter l'erreur NaN en DB)
        raw_data = clean_for_json(payload.model_dump())

        # Pipeline de l'inférence avec préprocessing du dictionnaire d'entrée payload
        # .model_dump() récupère tout (champs obligatoires + extras autorisés)
        output, request_id, inference_time = prediction_pipeline(
            db=db,
            model_instance=model_instance,
            preproc=preproc,
            raw_data=raw_data,
            log_id=cast(int, log_entry.id),
        )
        closing_log(
            db,
            log_entry,
            start_time,
            status_code=200,
            prediction_id=request_id,
            inference_time=inference_time,
        )
        return output

    except ValueError as exc:
        db.rollback()
        closing_log(db, log_entry, start_time, status_code=422)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as e:
        db.rollback()
        closing_log(db, log_entry, start_time, status_code=500)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ==========================================================================


@router.post("/upload", response_model=list[PredictionOutput])
async def predict_file(
    request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)
):
    """
    Inférence de masse via fichier CSV ou Parquet. Si il y a trop d'inférence à réaliser (on a
    choisi arbitrairement 100), le router passe en mode batch predict il va insérer par paquet
    (Bulk Insert) et faire de l'inférence vectorisée.
    """
    # On initialise le temps et le log
    start_time = time.time()
    log_entry = init_log(db, "/predict/upload")

    # Récupération de l'instance du modèle depuis le state de l'application
    model_instance = getattr(request.app.state, "model", None)
    if model_instance is None:
        closing_log(db, log_entry, start_time, status_code=503)
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    try:
        # ------------  Lecture Async du fichier ---------------
        contents = await file.read()
        df = (
            pd.read_csv(io.BytesIO(contents))
            if file.filename and file.filename.endswith(".csv")  # Pylance et le None de filename
            else pd.read_parquet(io.BytesIO(contents))
        )
        # -----------------------------------------------------

        preproc = InputPreproc(model_instance.feature_names)

        # ============= Sequentiel simple ===================
        if len(df) <= 100:
            results = []
            total_inference_time = 0.0  # Pour cumuler le temps
            for _, row in df.iterrows():
                # data_dict = row.to_dict()
                # NETTOYAGE JSON de la ligne avant envoi au pipeline
                data_dict = clean_for_json(row.to_dict())

                # Utilisation du pipeline pour chaque ligne du fichier
                output, _, inference_time = prediction_pipeline(
                    db=db,
                    model_instance=model_instance,
                    preproc=preproc,
                    raw_data=data_dict,  # type:ignore
                    log_id=cast(int, log_entry.id),
                )
                results.append(output)
                total_inference_time += inference_time

        # ====================== Batch predict =========================
        else:
            results, total_inference_time = batch_prediction_pipeline(
                db=db,
                model_instance=model_instance,
                preproc=preproc,
                df=df,
                log_id=cast(int, log_entry.id),
            )

        closing_log(
            db,
            log_entry,
            start_time,
            status_code=200,
            inference_time=total_inference_time,  # temps total d'inference
        )
        return results

    except Exception as e:
        db.rollback()
        closing_log(db, log_entry, start_time, status_code=400)
        raise HTTPException(status_code=400, detail=f"Erreur fichier: {str(e)}") from e
