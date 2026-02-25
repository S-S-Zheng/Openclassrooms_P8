# imports
import time
from typing import Any, Dict, List, Tuple, cast

import pandas as pd
from sqlalchemy.orm import Session

from app.api.schemas import PredictionOutput
from app.db.actions.get_prediction_from_db import get_prediction_id
from app.db.actions.save_prediction_to_db import save_prediction
from app.db.models_db import PredictionRecord
from app.utils.clean_for_json_db import clean_for_json
from app.utils.hash_id import generate_feature_hash
from app.utils.input_preproc import InputPreproc


# =========================== PREDICTION_SEQUENTIAL =====================
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


# =============== BATCH_PREDICTION =================================


def batch_prediction_pipeline(
    db: Session, model_instance: Any, preproc: InputPreproc, df: pd.DataFrame, log_id: int
) -> Tuple[List[PredictionOutput], float]:
    """
    Inférence de masse avec Cache Check par ligne et Inférence Vectorisée.

    IMPORTANT:
        IL Y A UN CHANGEMENT DE PARADIGME QUI NOUS EMPECHE DE FACILEMENT EMPLOYER LA MÉTHODE
        DU WRAPPER (en terme de propreté il faudrait implémenter une méthode batch_predict
        dans HGBM plutot que de la fusionner ici mais pour des questions de simplicité
        et de temps...). MAIS PENSER A LE FAIRE PLUS TARD.

    Logique alignée sur le séquentiel :
    1. Génération des Hashs et vérification du Cache.
    2. Inférence vectorisée UNIQUEMENT sur ce qui n'est pas en cache.
    3. Construction des résultats (mélange Cache + Nouveau).
    4. Bulk Insert des nouveaux records uniquement.
    """
    # Initialisation de la liste avec le bon type pour satisfaire le linter
    results: List[PredictionOutput] = []
    # On utilise un dictionnaire temporaire pour reconstruire l'ordre final facilement
    final_ordered_results: Dict[int, PredictionOutput] = {}

    records_to_save: List[PredictionRecord] = []
    indices_to_predict: List[int] = []
    rows_to_predict: List[Dict[str, Any]] = []

    # --- ÉTAPE 1 : CACHE CHECK & HASHING ---
    for i in range(len(df)):
        raw_row = df.iloc[i].to_dict()
        # Le nettoyage JSON est fait ici comme dans le router pour le séquentiel
        clean_row = clean_for_json(raw_row)

        cached = get_prediction_id(db, clean_row)
        if cached:
            final_ordered_results[i] = PredictionOutput(
                prediction=cast(int, cached.prediction),
                confidence=cast(float, cached.confidence),
                class_name=cast(str, cached.class_name),
            )
        else:
            indices_to_predict.append(i)
            rows_to_predict.append(clean_row)

    # --- ÉTAPE 2 : INFÉRENCE VECTORISÉE (Si nécessaire) ---
    inference_time_ms = 0.0
    if indices_to_predict:
        df_to_predict = pd.DataFrame(rows_to_predict)
        df_prepared = preproc.process(df_to_predict)

        # Inférence SANS LE WRAPPER car la méthode n'est pas adapté pour du batching!!
        inf_start = time.perf_counter()
        probas = model_instance.model.predict_proba(df_prepared)[:, 1]
        preds = (probas >= model_instance.threshold).astype(int)

        # Temps d'inférence
        inference_time_ms = (time.perf_counter() - inf_start) * 1000

        # Création des résultats et des records SQL
        for idx, original_idx in enumerate(indices_to_predict):
            current_clean_data = rows_to_predict[idx]
            # On réutilise le hash pour l'ID de la table 'predictions'
            unique_id = generate_feature_hash(current_clean_data)

            p_out = PredictionOutput(
                prediction=int(preds[idx]),
                confidence=float(probas[idx]),
                class_name="insolvable" if preds[idx] == 1 else "solvable",
            )

            final_ordered_results[original_idx] = p_out

            records_to_save.append(
                PredictionRecord(
                    id=unique_id,
                    inputs=current_clean_data,
                    prediction=p_out.prediction,
                    confidence=p_out.confidence,
                    class_name=p_out.class_name,
                    model_version=model_instance.version,
                )
            )

    # --- ÉTAPE 3 : PERSISTANCE GROUPÉE ---
    if records_to_save:
        # On insère les prédictions (Bulk Insert)
        # Note: add_all gère les doublons si l'ID (hash) existe déjà grâce au merge interne
        # ou via une gestion d'erreur, mais ici on va au plus simple :
        for rec in records_to_save:
            db.merge(rec)  # merge est plus sûr que add si plusieurs lignes du CSV ont le même hash

        # On lie TOUTES ces prédictions au log actuel dans la table RequestLog
        # Puisque c'est un upload de masse, le log de la requête doit pointer
        # vers les IDs de prédiction créés (si ta table de log permet le multi-liens
        # ou via une table pivot)
        db.commit()

    # Reconstruction de la liste dans l'ordre original du DataFrame
    results = [final_ordered_results[i] for i in range(len(df))]

    return results, inference_time_ms
