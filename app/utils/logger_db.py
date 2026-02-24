"""
Module de gestion de la journalisation (logging) en base de données.

Ce module fournit les outils nécessaires pour suivre le cycle de vie d'une requête HTTP.
Il permet de mesurer la performance (temps de réponse), de capturer les codes de statut
et d'établir un lien de parenté entre un log technique et une prédiction métier.
Ces données sont essentielles pour le monitoring et l'audit du service.

Afin d'éviter les warning Pylance du a SQLAlchemy:
setattr(obj, name, val) est une fonction Python standard que les analyseurs de type
ne valident pas de la même manière qu'une assignation directe (obj.name = val).
"""

# imports
import os
import time
from typing import Optional

import psutil
from sqlalchemy.orm import Session

from app.db.models_db import RequestLog

# ============== Initalise le log ======================


def init_log(db: Session, endpoint: str) -> RequestLog:
    """
    Initialise une entrée de log dans la table 'request_logs'.

    Crée l'objet de log au début de la requête. L'utilisation de `db.flush()`
    permet de récupérer l'ID auto-incrémenté généré par PostgreSQL sans pour autant
    clore la transaction SQL, permettant ainsi d'associer cet ID à d'autres opérations.

    Args:
        db (Session): Session SQLAlchemy active.
        endpoint (str): Le chemin de l'URL sollicité (ex: '/predict').

    Returns:
        RequestLog: L'instance du log nouvellement créée.
    """
    new_log = RequestLog(endpoint=endpoint, status_code=200)
    db.add(new_log)
    db.flush()  # Pour obtenir l'ID sans committer
    return new_log


# ================ Finalise le log =======================


def closing_log(
    db: Session,
    log_obj: RequestLog,
    start_time: float,
    status_code: Optional[int] = None,
    prediction_id: Optional[str] = None,
    inference_time: Optional[float] = None,
):
    """
    Finalise et persiste le log en base de données.

    Cette fonction calcule la latence totale de la requête, met à jour le code
    de statut final (200, 422, 500, etc.) et valide la transaction (commit).

    Args:
        db (Session): Session SQLAlchemy active.
        log_obj (RequestLog): L'objet log initialisé par `init_log`.
        start_time (float): Le timestamp de début (provenant de `time.time()`).
        status_code (int, optional): Le code HTTP final. Si None, conserve la valeur initiale.
        prediction_id (str, optional): L'identifiant unique (hash) de la prédiction associée.
    """
    # Code status
    if status_code is not None:
        # log_obj.status_code = int(status_code) # erreur Pylance VS SQLAlchemy
        log_obj.status_code = int(status_code)

    # Calcul de la latence totale de l'API (Réseau + DB + Inférence)
    duration = (time.time() - start_time) * 1000
    # log_obj.response_time_ms = float(duration)
    log_obj.response_time_ms = float(duration)

    # Latence modèle
    if inference_time is not None:
        # log_obj.inference_time_ms = float(inference_time)
        log_obj.inference_time_ms = float(inference_time)

    # Capture de l'utilisation CPU globale au moment T
    # On ne met pas d'intervalle pour ne pas bloquer l'API (non-blocking)
    cpu_usage = psutil.cpu_percent(interval=None)
    # log_obj.cpu_usage = float(cpu_usage)
    log_obj.cpu_usage = float(cpu_usage)

    # Métriques GPU (uniquement si NVIDIA et pynvml installé)
    # try:
    #     import pynvml
    #     pynvml.nvmlInit()
    #     handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    #     info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    #     setattr(log_obj, 'gpu_usage', float(info.used / info.total * 100))
    # except:
    #     pass

    # RAM utilisée par le worker API (en Mo)
    process = psutil.Process(os.getpid())
    log_obj.memory_usage = process.memory_info().rss / (1024 * 1024)  # type:ignore

    if prediction_id:
        # log_obj.prediction_id = str(prediction_id)
        log_obj.prediction_id = str(prediction_id)

    db.commit()


# =============== Link log avec prediction =================


def link_log(db: Session, log_id: int, prediction_id: str):
    """
    Établit un lien a posteriori entre un log technique et un enregistrement métier.

    Cette fonction est utile car elle garantie l'intégrité de la
    relation entre les tables 'request_logs' et 'predictions'.

    Args:
        db (Session): Session SQLAlchemy active.
        log_id (int): Identifiant numérique du log.
        prediction_id (str): Hash SHA-256 de la prédiction.
    """
    log_entry = db.get(RequestLog, log_id)
    if log_entry:
        # log_entry.prediction_id = str(prediction_id)
        log_entry.prediction_id = str(prediction_id)
        db.flush()
