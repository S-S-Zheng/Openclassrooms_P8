# imports
import time
from typing import Any, Dict, Tuple, cast

from sqlalchemy.orm import Session

from app.api.schemas import PredictionOutput
from app.db.actions.get_prediction_from_db import get_prediction_id
from app.db.actions.save_prediction_to_db import save_prediction
from app.utils.input_preproc import InputPreproc


def prediction_pipeline(
    db: Session, model_instance: Any, preproc: InputPreproc, raw_data: Dict[str, Any], log_id: int
) -> Tuple[PredictionOutput, str, float]:
    """
    Cache Check avec hash_id, préproc et predict de la donnée d'entrée,
    sauvegarde la prediction et la liaison log et retourne la prédiction + l'ID

    C'est le HAPPY PATH d'une prediction (quelque soit l'entrée: manuel ou chargement)

    Args:
        db (Session): Connexion DB.
        model_instance: Instance du modèle ML (HGBM).
        preproc: Instance de la classe InputPreproc.
        raw_data (Dict): Données brutes (11 ou 400+ features).
        log_id (int): ID du log de requête associé.

    Returns:
        Tuple[PredictionOutput, str, float]:
            output=[pred,conf, class], log_ID et le temps d'inference
    """
    # Cache Check (Hash des données)
    cached = get_prediction_id(db, raw_data)
    if cached:
        output = PredictionOutput(
            prediction=cast(int, cached.prediction),
            confidence=cast(float, cached.confidence),
            class_name=cast(str, cached.class_name),
        )
        return output, cast(str, cached.id), 0.0  # 0ms si c'est du cache

    # Transformation des données brutes en DataFrame aligné via la classe instanciée preproc
    df_prepared = preproc.process(raw_data)

    # Inférence via le wrapper ML
    inf_start = time.perf_counter()
    prediction, confidence, class_name = model_instance.predict(df_prepared.head(1))

    # Temps d'inférence
    inference_time = (time.perf_counter() - inf_start) * 1000

    # Sauvegarde(Associe la prédiction au log_id)
    request_id = save_prediction(db, raw_data, (prediction, confidence, class_name), log_id=log_id)

    output = PredictionOutput(prediction=prediction, confidence=confidence, class_name=class_name)

    return output, request_id, inference_time
