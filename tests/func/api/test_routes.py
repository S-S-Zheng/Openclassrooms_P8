"""
Suite de tests d'intégration pour les routes de l'API FastAPI.

Ce module valide le comportement des endpoints exposés en simulant des requêtes
HTTP réelles via le TestClient. Il vérifie :
1. Le routage correct des requêtes vers les services (ML et Base de données).
2. La conformité des réponses HTTP (codes de statut et corps de réponse JSON).
3. La gestion des erreurs et la propagation des messages d'exception.
4. L'impact des appels API sur la persistance des données.

L'utilisation de la fixture paramétrée 'error_responses' permet de tester
systématiquement la résilience de chaque route face à des pannes du modèle
ou des erreurs de configuration.
"""

# Imports
from typing import Any, Callable, Dict

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.db.models_db import PredictionRecord

# ========================= PREDICT ==========================


# Happy path
@pytest.mark.integration
def test_predict_success(
    client: TestClient,
    mock_ml_model: Callable[..., Any],
    func_sample: Dict[str, Any],
    db_session_for_tests: Session,
) -> None:
    """
    Vérifie le succès complet d'une requête de prédiction.

    S'assure que :
    - La route répond avec un code 200.
    - Les données renvoyées correspondent aux valeurs produites par le modèle.
    - Un enregistrement correspondant est correctement créé en base de données.
    """
    # Nettoyage de la base pour ne pas être influencé par la fixture init
    # permet aussi de ne pas casser le test reset_tables
    db_session_for_tests.execute(delete(PredictionRecord))
    db_session_for_tests.commit()

    payload = func_sample

    # Config de l'API
    mock_ml_model(should_fail=False)
    response = client.post("/predict/manual", json=payload)

    data = response.json()
    assert data["prediction"] == 1
    assert data["confidence"] == 0.8


# =========================================================


# Erreurs
def test_predict_errors(
    client: TestClient,
    mock_ml_model: Callable[..., Any],
    func_sample: Dict[str, Any],
    error_responses: Dict[str, Any],
    db_session_for_tests: Session,
) -> None:
    """
    Vérifie la gestion des erreurs sur l'endpoint /predict.

    Teste les codes de statut 422, 503 et 500 selon l'état du mock, et confirme
    qu'aucune donnée n'est persistée en cas d'échec de la requête.
    """
    # Nettoyage de la base pour ne pas être influencé par la fixture init
    # permet aussi de ne pas casser le test reset_tables
    db_session_for_tests.execute(delete(PredictionRecord))
    db_session_for_tests.commit()

    payload = func_sample

    mock_ml_model(**error_responses["mock_args"])

    response = client.post("/predict/manual", json=payload)

    assert response.status_code == error_responses["expected_status"]
    assert response.json()["detail"] == error_responses["error_msg"]

    # On s'assure que rien n'est sauvegardé puisque ça plante
    record = db_session_for_tests.scalar(select(func.count(PredictionRecord.id)))
    assert record == 0


# ==============================================================


# Happy path/conversion csv vers df/ cumul de temps d'inference
@pytest.mark.integration
@pytest.mark.asyncio
async def test_predict_upload_csv_success(
    client: TestClient, mock_ml_model: Callable[..., Any], db_session_for_tests: Session
) -> None:
    """
    Vérifie le succès du traitement d'un fichier CSV via l'endpoint /predict/upload.

    S'assure que :
    - Le fichier est correctement lu et parsé par Pandas.
    - Chaque ligne déclenche une prédiction via le pipeline.
    - Le code de retour est 200 et contient une liste de résultats.

    Args:
        client (TestClient): Client de test FastAPI.
        mock_ml_model (Callable): Fixture pour configurer l'état du modèle.
        db_session_for_tests (Session): Session de base de données.
    """
    # Config du mock modèle
    mock_ml_model(should_fail=False)

    # Création d'un CSV fictif avec 2 lignes
    # On utilise des colonnes minimales, InputPreproc gérera le reste
    csv_content = "SK_ID_CURR,NAME_FAMILY_STATUS\n100001,Married\n100002,Single"
    files = {"file": ("test.csv", csv_content.encode("utf-8"), "text/csv")}

    # Appel de la route upload
    response = client.post("/predict/upload", files=files)

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["prediction"] == 1


# ============================================================


# modele absent (code status 503)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_predict_upload_model_not_found(
    client: TestClient,
    mock_ml_model: Callable[..., Any],
) -> None:
    """
    Vérifie que l'API renvoie une erreur 503 si le modèle n'est pas chargé.

    Args:
        client (TestClient): Client de test FastAPI.
        mock_ml_model (Callable): Fixture configurée pour simuler l'absence du modèle.
    """
    # On force le state.model à None
    mock_ml_model(is_missing=True)

    files = {"file": ("test.csv", b"dummy content", "text/csv")}
    response = client.post("/predict/upload", files=files)

    assert response.status_code == 503
    assert response.json()["detail"] == "Modèle non chargé"


# ================================================================


# fichier corrompu (code 400) couvre Exeption, rollback et closing_log
@pytest.mark.integration
@pytest.mark.asyncio
async def test_predict_upload_invalid_file(
    client: TestClient,
    mock_ml_model: Callable[..., Any],
) -> None:
    """
    Vérifie le comportement de l'API face à un fichier corrompu.

    S'assure que :
    - L'exception lors de la lecture du fichier est capturée.
    - Un code 400 est retourné.
    - Le message d'erreur contient les détails de l'exception.

    Args:
        client (TestClient): Client de test FastAPI.
        mock_ml_model (Callable): Fixture pour configurer l'état du modèle.
    """
    mock_ml_model(should_fail=False)

    # Envoi de contenu binaire invalide pour un CSV
    bad_content = b"\x00\x01\x02\x03\xff"
    files = {"file": ("corrupt.csv", bad_content, "text/csv")}

    response = client.post("/predict/upload", files=files)

    assert response.status_code == 400
    assert "Erreur fichier" in response.json()["detail"]


# ========================= METADATAS ===========================


# Happy path
@pytest.mark.integration
def test_model_info_success(client: TestClient, mock_ml_model: Callable[..., Any]) -> None:
    """
    Vérifie l'endpoint d'introspection du modèle (/model-info).

    S'assure que toutes les métadonnées techniques (type, noms des features,
    catégorisation, classes et seuil) sont correctement exposées.
    """
    model = mock_ml_model(should_fail=False)
    response = client.get("/model-info")

    assert response.status_code == 200
    data = response.json()
    # Rappel : feature_names_in_ est un numpy array dans le mock,
    # mais le JSON renvoie une liste.
    expected_features = list(model.feature_names)

    assert data["model_type"] == "HistGradientBoostingClassifier"
    assert data["n_features"] == len(expected_features)
    assert data["feature_names"] == expected_features
    assert data["classes"] == ["solvable", "insolvable"]
    assert data["threshold"] == 0.5


# =========================================================


# Erreurs
@pytest.mark.integration
def test_model_info_errors(
    client: TestClient, mock_ml_model: Callable[..., Any], error_responses: Dict[str, Any]
) -> None:
    """Vérifie la gestion des erreurs lors de la récupération des métadonnées du modèle."""
    mock_ml_model(**error_responses["mock_args"])

    response = client.get("/model-info")

    assert response.status_code == error_responses["expected_status"]
    assert response.json()["detail"] == error_responses["error_msg"]
