"""
Suite de tests d'intégration (globalement) pour la couche de persistance et de base de données.

Ce module valide l'intégralité du cycle de vie des données au sein du système :
1. La gestion des sessions SQLAlchemy et du contexte de base de données.
2. La persistance des prédictions (modèle métier) et des logs (modèle technique).
3. L'intégrité référentielle, l'idempotence des enregistrements (hachage unique)
    et la robustesse des transactions (rollbacks en cas de crash).
4. Les scripts d'administration : initialisation du schéma, réinitialisation des
    tables et importation de données historiques depuis des fichiers CSV.

Chaque test utilise une base de données PostgreSQL de test isolée pour garantir
qu'aucun effet de bord n'affecte les données de production.
"""

# imports
import importlib
import logging
import os
from typing import cast
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, delete, func, inspect, select
from sqlalchemy.orm import Session, sessionmaker  # noqa: F401

from app.db import database
from app.db.actions.save_prediction_to_db import save_prediction
from app.db.create_db import init_db
from app.db.import_dataset_to_db import import_historical_data
from app.db.models_db import PredictionRecord, RequestLog
from app.utils.hash_id import generate_feature_hash
from app.utils.logger_db import init_log

# ========================== DATABASE =======================


@pytest.mark.unit
def test_get_db(monkeypatch, TestingSession):
    """Vérifie le cycle de vie du générateur de session et sa fermeture correcte."""

    # On force database.SessionLocal à utiliser la session de test
    monkeypatch.setattr(database, "SessionLocal", TestingSession)

    # On teste le générateur (@contextmanager ==> on l'utilise avec 'with')
    with database.get_db_contextmanager() as session:
        assert isinstance(session, Session)
        assert session.is_active

    # On vérifie que la session est fermée
    assert session.get_transaction() is None


# =========================== PERSISTENCE ==========================


@pytest.mark.integration
def test_get_prediction(client, mock_ml_model, func_sample, db_session_for_tests):
    """
    Vérifie le mécanisme de cache en base de données.
    L'API doit retourner une prédiction existante si les features sont identiques,
    sans solliciter le modèle ML.
    """
    payload = func_sample
    expected_hash_id = generate_feature_hash(payload)

    # On insère manuellement une prédiction lié a fun_sample dans la db
    existing_pred = PredictionRecord(
        id=expected_hash_id,
        inputs=payload,
        prediction=0,
        confidence=0.99,
        class_name="solvable",
        model_version="v1_test",
    )
    db_session_for_tests.add(existing_pred)
    db_session_for_tests.commit()

    # On configure le ML pour qu'il réponde l'inverse de la DB MAIS l'API doit renvoyer
    # ce qui est dans la db
    mock_ml_model(should_fail=False)
    response = client.post("/predict/manual", json=payload)

    data = response.json()
    # On confirme que la DB a été utilisée
    assert data["prediction"] == existing_pred.prediction
    assert data["class_name"] == existing_pred.class_name


# ======================== REMPLISSAGE DES TABLES ===========================


# Happy path
@pytest.mark.integration
def test_save_prediction(client, mock_ml_model, func_sample, db_session_for_tests):
    """Valide l'insertion complète d'un résultat d'inférence via l'endpoint API."""
    mock_ml_model(should_fail=False)
    payload = func_sample
    expected_hash_id = generate_feature_hash(payload)

    # Premier appel: l'API va utiliser sa propre session via Depends(get_db)
    response = client.post("/predict/manual", json=payload)
    assert response.status_code == 200

    saved_pred = db_session_for_tests.scalars(
        select(PredictionRecord).filter_by(id=expected_hash_id)
    ).first()

    assert saved_pred is not None
    assert saved_pred.id == expected_hash_id
    assert saved_pred.inputs == payload
    # La prédiction doit correspondre aux résultats du mock_ml_model.predict
    assert saved_pred.class_name == "insolvable"
    assert saved_pred.prediction == 1
    assert saved_pred.confidence == 0.8


# =====================================================


# Gestion de l'idempotence
@pytest.mark.integration
def test_save_prediction_unique_id(db_session_for_tests, func_sample):
    """
    Vérifie l'idempotence basée sur le hash des features.
    Deux appels avec des entrées identiques ne doivent créer qu'un seul
    enregistrement 'PredictionRecord'.
    """
    payload = func_sample
    pred_data = (1, 0.85, "insolvable")

    # On instancie deux log
    log_1 = init_log(db_session_for_tests, "/predict")
    log_2 = init_log(db_session_for_tests, "/predict")

    # On injecte une premère fois la donnée qui doit bien se passer
    request_id_1 = save_prediction(
        db_session_for_tests, payload, pred_data, log_id=cast("int", log_1.id)
    )
    # Deuxième enregistrement (mêmes données, log différent)
    request_id_2 = save_prediction(
        db_session_for_tests, payload, pred_data, log_id=cast("int", log_2.id)
    )

    assert request_id_1 == request_id_2  # L'ID doit être le même (basée sur les features)
    # On vérifie qu'il n'y a pas de doublon
    count = db_session_for_tests.scalar(select(func.count()).select_from(PredictionRecord))
    assert count == 1
    # On vérifie que les deux ID log pointent vers le même ID (features)
    assert cast("str", log_1.prediction_id) == request_id_1
    assert cast("str", log_2.prediction_id) == request_id_2


# ================================================================


# Crash en cours d'opération
@pytest.mark.integration
def test_save_prediction_crash(db_session_broken_for_tests, func_sample):
    """Vérifie que la transaction SQL est annulée (rollback) en cas d'erreur serveur."""
    payload = func_sample
    pred_data = (1, 0.85, "insolvable")
    log_1 = 1

    with pytest.raises(Exception) as excinfo:
        save_prediction(db_session_broken_for_tests, payload, pred_data, log_id=log_1)
    assert "Unexpected Crash" in str(excinfo.value)
    # On vérifie que le rollback a bien fonctionné
    assert db_session_broken_for_tests.rollback.called


# ====================== INITIALISATION DB ===========================


# Happy path
@pytest.mark.integration
def test_init_db(TestingEngine):
    """Vérifie que le schéma SQL est correctement créé au démarrage de l'application."""
    init_db(reset_tables=False, engine=TestingEngine)

    # Vérification physique de la présence de la table via l'inspecteur
    inspector = inspect(TestingEngine)
    tables = inspector.get_table_names()
    assert "request_logs" in tables
    assert "predictions" in tables


# =============================================================


# Reset fonctionnel
@pytest.mark.integration
def test_init_db_reset_table(TestingEngine, func_sample, TestingSession):
    """Vérifie que l'option de réinitialisation vide effectivement toutes les tables."""
    payload = func_sample

    # On créée une session parallèle pour constituer une base non vide et
    # s'assurer que la donnée est présente
    with TestingSession() as temp_session:
        log_entry = init_log(temp_session, "/predict")
        save_prediction(
            temp_session,
            payload,
            (0, 0.9, "solvable"),
            log_id=cast("int", log_entry.id),
        )
        temp_session.commit()

    # Reset
    init_db(reset_tables=True, engine=TestingEngine)

    # Vérification physique de la présence de la table via l'inspecteur
    inspector = inspect(TestingEngine)
    tables = inspector.get_table_names()
    assert "request_logs" in tables
    assert "predictions" in tables

    # Vérifications logiques (vidage des données)
    with TestingSession() as new_session:
        count_pred = new_session.scalar(select(func.count()).select_from(PredictionRecord))
        count_logs = new_session.scalar(select(func.count()).select_from(RequestLog))

        assert count_pred == 0
        assert count_logs == 0


# =================================================================


# Echec de création de base
@pytest.mark.integration
def test_init_db_failed_init():
    """Vérifie la levée d'exception en cas de paramètres de connexion erronés."""
    fake_engine = create_engine("postgresql://user:wrong@localhost:9999/fake")

    with pytest.raises(Exception):
        init_db(reset_tables=False, engine=fake_engine)


# =================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "host, expected_suffix", [("db.supabase.com", "?sslmode=require"), ("localhost", "")]
)
def test_database_url_ssl_logic(host, expected_suffix):
    with patch.dict(os.environ, {"SB_HOST": host}):
        importlib.reload(database)
        assert database.DATABASE_URL.endswith(expected_suffix)


# ========================================================================


# Password manquant
@pytest.mark.unit
def test_database_password_missing_warning(caplog: pytest.LogCaptureFixture) -> None:
    """
    Vérifie le déclenchement d'un warning si SB_PASSWORD est absent.

    Ce test simule un environnement sans mot de passe, force le rechargement
    du module database et vérifie la présence du message attendu dans les logs.

    Args:
        caplog (pytest.LogCaptureFixture): Fixture Pytest pour intercepter les logs.
    """
    # mdp vide
    # patch.dict permet de modifier os.environ uniquement pendant le bloc 'with'
    with patch.dict(os.environ, {"SB_PASSWORD": ""}):
        # caplog sur WARNING
        with caplog.at_level(logging.WARNING):
            # database a la racine ==> il a déjà été exécuté donc
            # on force la ré-exécution pour voir le warning.
            importlib.reload(database)

            # Assertions
            expected_msg = "Attention : Aucun mot de passe de base de données trouvé (SB_PASSWORD)."
            assert expected_msg in caplog.text

            assert any(record.levelname == "WARNING" for record in caplog.records)


# ======================= IMPORTATION DE DONNEES =======================


# Happy path
@pytest.mark.integration
def test_import_csv_success(monkeypatch, TestingSession, TestingEngine, fake_csv):
    """
    Valide le pipeline d'ingestion de données historiques.
    Vérifie le mapping des colonnes CSV vers les champs de la base de données.
    """
    # On import chemin et données
    fake_file_path, fake_data = fake_csv

    # On court-circuite get_db
    monkeypatch.setattr(database, "SessionLocal", TestingSession)

    # On importe
    import_historical_data(str(fake_file_path))

    with TestingSession() as session:
        # vérification métier
        records = session.scalars(select(PredictionRecord)).all()

        assert len(records) == len(fake_data["TARGET"])

        # Vérification du mapping
        classification = session.scalars(
            select(PredictionRecord).filter_by(class_name="insolvable")
        ).first()

        assert classification is not None
        assert classification.prediction == 1
        assert classification.confidence == 1.0

        # Vérification du log
        log = session.scalars(
            select(RequestLog).where(RequestLog.endpoint.contains("/import"))
        ).first()

        assert log is not None
        assert log.status_code == 201
        assert log.response_time_ms > 0


# ==============================================================


# Unicité
@pytest.mark.integration
def test_import_csv_unique(fake_csv, TestingSession, monkeypatch, TestingEngine):
    """Vérifie que l'importation massive gère correctement les doublons de données."""
    # On import chemin et données
    fake_file_path, fake_data = fake_csv

    # On court-circuite get_db
    monkeypatch.setattr(database, "SessionLocal", TestingSession)

    # On clean les tables avant de lancer le test (à cause de la fixture init)
    with TestingSession() as session:
        session.execute(delete(RequestLog))
        session.execute(delete(PredictionRecord))
        session.commit()

    # On importe deux fois, le premier devrait etre bon seulement
    import_historical_data(str(fake_file_path))
    import_historical_data(str(fake_file_path))

    with TestingSession() as session:
        # Un seul import features
        records = session.scalars(select(PredictionRecord)).all()

        assert len(records) == len(fake_data["TARGET"])

        # Mais deux logs d'import
        log = session.scalars(select(RequestLog)).all()

        assert len(log) == 2


# =============================================================


# Crash immédiat
@pytest.mark.integration
def test_import_csv_crash_instant(db_session_broken_for_tests, fake_csv, monkeypatch):
    """Vérifie la sécurité transactionnelle lors d'un échec d'importation groupée."""
    fake_file_path, _ = fake_csv
    monkeypatch.setattr(database, "SessionLocal", lambda: db_session_broken_for_tests)

    with pytest.raises(Exception) as excinfo:
        import_historical_data(str(fake_file_path))

    assert "Unexpected Crash" in str(excinfo.value)
    # On vérifie que le rollback a bien fonctionné
    assert db_session_broken_for_tests.rollback.called
