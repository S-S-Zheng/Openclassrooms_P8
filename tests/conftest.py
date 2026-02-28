"""
Module de configuration globale et de définition des fixtures pour Pytest.

Ce fichier est le pilier de la suite de tests. Il centralise la configuration
des environnements de test, notamment :
1. La gestion d'une base de données PostgreSQL de test isolée.
2. Le mockage du modèle et des dépendances système.
3. L'injection de dépendances pour le client de test FastAPI.
4. La génération de jeux de données fictifs (profils YAML, fichiers CSV temporaires).

L'utilisation de fixtures permet de garantir l'indépendance des tests et la
reproductibilité des scénarios de succès et d'échec.
"""

import os

# le noqa permet d'indiquer explicitement a isort d'ignorer les lignes
import urllib.parse  # Import indispensable pour les caractères spéciaux
from pathlib import Path
from typing import Any, Callable, List, Union
from unittest.mock import (
    MagicMock,
    create_autospec,  # noqa: F401
)

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from sklearn.ensemble import HistGradientBoostingClassifier
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.database import get_db
from app.db.models_db import PredictionRecord, RequestLog  # noqa: F401
from app.main import app
from app.ml.model import HGBM
from app.utils.input_preproc import InputPreproc
from app.utils.monitoring.data_extractor_for_analysis import DataProvider

# Empeche le profiling pendant les tests
os.environ["ENABLE_PROFILING"] = "false"
# Force le chargement explicite avant toute initialisation de moteur DB
load_dotenv(".env.test", override=True)

# ========================== MOCK/FIXTURE==========================

# =========================== DATAS ===========================


# Les features obligatoires du projet
@pytest.fixture
def base_features():
    return {
        "EXT_SOURCE_COUNT": 1.0,
        "DAYS_EMPLOYED": -637.0,
        "DAYS_BIRTH": -9461,
        "AMT_ANNUITY": 24700.5,
        "PAYMENT_RATE": 0.1,
        "PREV_DAYS_LAST_DUE_1ST_VERSION_MAX": 0.0,
        "BUREAU_AMT_CREDIT_SUM_DEBT_MEAN": 100000.0,
        "OWN_CAR_AGE": 40.0,
        "NAME_FAMILY_STATUS": "Married",
    }


# dict de test
@pytest.fixture
def fake_dict():
    return {"f1": 1.0, "f2": "deux", "f3": 3.0, "f4": 4.0, "f5": 5.0}


# df de test
@pytest.fixture
def fake_df(fake_dict):
    return pd.DataFrame([fake_dict])


# Fonction imbriqué nécéssaire car sinon on peut pas vérif que le wrapper transmet
# correctement l'information (profiling)
@pytest.fixture
def fake_func():
    def _func(x=1):
        return x

    return _func


@pytest.fixture
def input_preproc_mocked():
    """Fixture pour initialiser le préprocesseur avec quelques features."""
    return InputPreproc(
        model_full_feature_names=[
            "EXT_SOURCE_COUNT",
            "OWN_CAR_AGE",
            "AMT_ANNUITY",
            "DAYS_BIRTH",
            "NAME_FAMILY_STATUS",
        ]
    )


# dict de test partie fonctionnelle
@pytest.fixture
def func_sample(base_features):
    """
    Fournit un dictionnaire complet de caractéristiques valides.
    Utilisé pour tester les payloads de l'endpoint /predict.
    """
    sample = base_features.copy()
    sample["OTHER_FEATURE_99"] = 1.0  # Test de la flexibilité extra=allow
    return sample


@pytest.fixture(params=["service_unavailable", "value_error", "unexpected_error"])
def error_responses(request):
    """
    On définit un dictionnaire où les clés correspondent aux params
    """
    datas = {
        "service_unavailable": {
            "mock_args": {"is_missing": True},
            "expected_status": 503,
            "error_msg": "Modèle non chargé",
        },
        "value_error": {
            "mock_args": {"should_fail": True, "error_type": "value"},
            "expected_status": 422,
            "error_msg": "Modèle non chargé",
        },
        "unexpected_error": {
            "mock_args": {"should_fail": True, "error_type": "exception"},
            "expected_status": 500,
            "error_msg": "Erreur interne critique",
        },
    }
    return datas[request.param]


# Charge des profils fonctionnels
@pytest.fixture
def functionnal_profile(request):
    """
    Charge dynamiquement un profil YAML selon le paramètre fourni.
    """
    # 'request.param' contiendra le nom du profil (ex: 'happy_path')
    profile_name = request.param
    file_path = Path(__file__).parent / "fixtures" / f"fake_profile_{profile_name}.yml"

    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        data["_profile_name"] = profile_name
        return data


# fake csv
@pytest.fixture
def fake_csv(tmp_path, base_features):
    """
    Crée un fichier CSV temporaire simulant un dataset historique.
    Utilisé pour tester les scripts d'importation.
    """
    # faux chemin avec fichier csv
    fake_file = tmp_path / "hist_datas"
    fake_file.mkdir()
    fake_file_path = fake_file / "test_data.csv"

    # On crée 2 lignes avec toutes les colonnes requises
    fake_data = {key: [val, val * 2] for key, val in base_features.items()}
    fake_data["TARGET"] = [1, 0]  # On ajoute la target pour l'import historique

    pd.DataFrame(fake_data).to_csv(fake_file_path, index=False)

    return fake_file_path, fake_data


# ========================= UNIT (ML MOCKING) ==========================


# Mock l'instanciation du modèle
@pytest.fixture
def hgbm_instance_mock(base_features: dict) -> MagicMock:
    """
    Crée une instance isolée de MagicMock simulant un HistGradientBoostingClassifier.
    Respecte l'interface Scikit-Learn attendue par le wrapper HGBM.
    """
    mock_model = MagicMock(spec=HistGradientBoostingClassifier)

    # Configuration des attributs Scikit-Learn
    features_list = list(base_features.keys())
    mock_model.feature_names_in_ = np.array(features_list)
    mock_model.n_features_in_ = len(features_list)

    # Configuration des retours d'inférence (Probabilité classe 1 = 80%)
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
    mock_model.predict.return_value = np.array([1])

    return mock_model


# Helper pour faker joblib et charger la donnée
def joblib_loader(path: Union[str, Path], hgbm_mock: MagicMock, features_list: List[str]) -> Any:
    """
    Simule le comportement de la lecture sur disque (renvoie le modèle,
    les features ou le seuil selon le fichier).

    Args:
        path: Chemin du fichier intercepté.
        hgbm_mock: Instance du modèle à renvoyer si 'model' est dans le chemin.
        features_list: Liste des features à renvoyer si 'features' est dans le chemin.
    """
    path_str = str(path).lower()
    if "model" in path_str:
        return hgbm_mock
    if "feature_names" in path_str:
        return list(features_list)
    if "thresh" in path_str:
        return 0.6
    return MagicMock()


@pytest.fixture
def mock_ml_infrastructure(
    monkeypatch: pytest.MonkeyPatch, hgbm_instance_mock: MagicMock, base_features: dict
) -> None:
    """
    Configure l'environnement système pour le chargement du modèle (Modulaire).

    1. Intercepte pathlib.Path.exists pour simuler la présence des artefacts.
    2. Intercepte joblib.load pour injecter les mocks via joblib_loader_side_effect.
    3. Intercepte l'instanciation de HistGradientBoostingClassifier.
    """
    # On fait croire que le path exst
    monkeypatch.setattr("pathlib.Path.exists", lambda _: True)

    features_list = list(base_features.keys())

    # On mocke joblib.load au niveau global ET dans le namespace du module model
    monkeypatch.setattr(
        joblib, "load", lambda path: joblib_loader(path, hgbm_instance_mock, features_list)
    )
    monkeypatch.setattr(
        "app.ml.model.joblib.load",
        lambda path: joblib_loader(path, hgbm_instance_mock, features_list),
    )

    # mock sklearn
    monkeypatch.setattr(
        "sklearn.ensemble.HistGradientBoostingClassifier", lambda **kwargs: hgbm_instance_mock
    )


# l'instance wrapper entièrement mockée (le produit final 100% d'impact sur model.py)
@pytest.fixture
def ml_model_mocked(mock_ml_infrastructure: None) -> HGBM:
    """
    Fournit une instance du wrapper HGBM initialisée et chargée.
    L'appel à .load() passera par les mocks configurés dans mock_ml_infrastructure.
    """
    model_wrapper = HGBM(
        model_path="fake_model.joblib",
        feature_names_path="fake_features.joblib",
        threshold_path="fake_threshold.joblib",
    )
    model_wrapper.load()
    return model_wrapper


# =========================== API & INTEGRATION ===========================


# Client FastAPI
@pytest.fixture
def client():
    """
    Fournit un TestClient FastAPI configuré avec une session de base de données de test.
    Gère le cycle de vie (lifespan) de l'application pour chaque test.
    """
    """
    Fournit un client HTTP configuré pour tester les endpoints de l'API.

    Cette fixture utilise le 'TestClient' de FastAPI au sein d'un gestionnaire
    de contexte afin de déclencher les événements 'lifespan' (startup/shutdown).
    Cela permet notamment au modèle ML d'être chargé en mémoire avant l'exécution
    des tests.

    Note:
        L'injection de la base de données n'est plus gérée ici car elle est
        assurée globalement par la fixture 'override_db'.

    Yields:
        TestClient: Une instance de client capable d'effectuer des requêtes (GET, POST, etc.).
    """
    with TestClient(app) as test_client:
        yield test_client


# MLModel pour tester la relation API/ML (Aucun impact sur model.py)
@pytest.fixture
def mock_ml_model(ml_model_mocked: HGBM, hgbm_instance_mock: MagicMock) -> Callable:
    """
    Factory de mocks pour simuler différents états du modèle ML au sein de l'application.

    Cette fixture retourne une fonction usine permettant de configurer dynamiquement
    le comportement du modèle (succès, erreur, ou absence) et de l'injecter
    directement dans l'état de l'application FastAPI (`app.state.model`).

    Args:
        func_sample (dict): Fixture fournissant un exemple de features pour
            calculer les métadonnées de réponse.

    Returns:
        Callable: Une fonction `_factory` acceptant les paramètres suivants :
            - is_missing (bool): Si True, simule l'absence totale de modèle chargé.
            - should_fail (bool): Si True, les méthodes du modèle lèveront une exception.
            - error_type (str): Type d'erreur à lever ('value' pour ValueError,
                sinon Exception générique).

    Note:
        Le mock généré simule l'intégralité de l'interface wrapper dans app.ml.model :
        - `predict()` : Retourne un tuple (classe, confiance, nom).
        - `get_model_info()` : Retourne un dictionnaire de métadonnées.
    """

    def _factory(
        is_missing: bool = False, should_fail: bool = False, error_type: str = "value"
    ) -> Union[HGBM, None]:
        # Cas 1 : Simulation du modèle non initialisé (Lifespan défaillant)
        if is_missing:
            app.state.model = None
            return None
        # Cas 2 : Simulation de pannes durant l'inférence ou l'analyse
        if should_fail:
            error = (
                ValueError("Modèle non chargé")
                if error_type == "value"
                else Exception("Erreur interne critique")
            )
            # Blocage des predictions
            hgbm_instance_mock.predict_proba.side_effect = error
            hgbm_instance_mock.predict.side_effect = error
            # blocage du model info
            ml_model_mocked.get_model_info = MagicMock(side_effect=error)
        # Cas 3 : Simulation d'un fonctionnement nominal (Happy Path)
        else:
            # Reset des side_effects
            hgbm_instance_mock.predict_proba.side_effect = None
            hgbm_instance_mock.predict.side_effect = None
            # On restaure le comportement normal (on enlève le side_effect d'erreur)
            ml_model_mocked.get_model_info = MagicMock(
                return_value={
                    "model_type": "HistGradientBoostingClassifier",
                    "n_features": len(ml_model_mocked.feature_names),
                    "feature_names": ml_model_mocked.feature_names,
                    "classes": ["solvable", "insolvable"],
                    "threshold": 0.5,
                }
            )
        # Injection de la VRAIE instance (mais mockée en périphérie) dans FastAPI
        app.state.model = ml_model_mocked
        return ml_model_mocked

    return _factory


# Dossier temporaire pour les rapports
@pytest.fixture
def mock_reports_dir(tmp_path, monkeypatch):
    """
    Prépare un dossier de rapports temporaire et redirige la constante
    REPORT_DIR du module ddl vers ce dossier.
    """
    # Création du dossier temporaire
    report_dir = tmp_path / "reports_test"
    report_dir.mkdir()

    # Redirection de la constante dans le module
    import_path = "app.api.routes.ddl.REPORT_DIR"
    monkeypatch.setattr(import_path, report_dir)

    return report_dir


@pytest.fixture
def provider():
    """Initialise le DataProvider pour les tests."""
    return DataProvider()


# ========================= DB (POSTGRES TEST) ==============================

# hors docker l'adresse = localhost != dans docker = db ==> os.getenv()
# ====================== Variables d'environnement ==============
# Récupération sécurisée
db_user_test = os.getenv("POSTGRES_USER_TEST", "postgres")
db_pass_test = urllib.parse.quote_plus(os.getenv("POSTGRES_PASSWORD_TEST", "password"))
db_host_test = os.getenv("POSTGRES_HOST_TEST", "localhost")
db_port_test = os.getenv("POSTGRES_PORT_TEST", "5432")  # Port souvent différent du prod
db_name_test = os.getenv("POSTGRES_DB_TEST", "test_db")

#
DATABASE_URL_TEST = (
    f"postgresql+psycopg2://{db_user_test}:{db_pass_test}@{db_host_test}:"
    f"{db_port_test}/{db_name_test}"
)
# DATABASE_URL_TEST = os.getenv("DATABASE_URL_TEST")

# ENGINE: point de départ de SQLAlchemy
test_engine = create_engine(DATABASE_URL_TEST)


@pytest.fixture
def TestingEngine():
    return test_engine


# SessionLocal est une factory à sessions pour les routes
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture
def TestingSession():
    return TestingSessionLocal


# fixture lancer automatiquement et pour la durée de la session de test
# Pour eviter de recréer les tables a chaque test les utilisant
@pytest.fixture(scope="session", autouse=True)
def init_db_for_tests():
    """
    Fixture de session : Crée les tables au démarrage et les supprime à la fin de la session.
    Utilisation automatique de la fixture.
    """
    try:
        # On vérifie si on peut se connecter avant de tenter le create_all
        test_engine.connect()
        Base.metadata.create_all(bind=test_engine)
        yield
        # Vide la base db à la fin de la session != rollback()
        Base.metadata.drop_all(bind=test_engine)
    except OperationalError:
        # pytest.skip("Base de données de test non disponible")
        # On ne fait rien : les tests d'intégration DB échoueront d'eux-mêmes
        # mais les tests unitaires et de routes mockées passeront !
        yield


@pytest.fixture
def db_session_for_tests():
    """
    Fournit une session de base de données isolée pour chaque test.
    Utilise une transaction SQL pour effectuer un rollback systématique à la fin
    du test, garantissant une base propre pour le test suivant.
    """
    connection = test_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    yield session

    session.close()
    # Annule l'insertion pour le test suivant plus efficace de delete
    if transaction.is_active:
        transaction.rollback()
    connection.close()


# Evite au routes d'utiliser la session normale plutot que le test à cause de Depend(get_db)
# ==> les client.post, get etc seront automatiquement envoyées vers le test.
@pytest.fixture(autouse=True)
def override_db(db_session_for_tests):
    """
    Assure l'isolation systématique de la base de données pour l'application
    ==> Remplace la dépendance get_db par la session de test

    En utilisant 'autouse=True', cette fixture garantit que n'importe quelle route
    faisant appel à la dépendance 'get_db' recevra la session de test en cours,
    évitant ainsi toute écriture accidentelle dans la base de production ou de dev.

    Args:
        db_session_for_tests: La session SQLAlchemy transactionnelle définie plus haut.
    """
    app.dependency_overrides[get_db] = lambda: db_session_for_tests
    yield
    app.dependency_overrides.clear()


# Session db qui saute pendant une transaction
@pytest.fixture
def db_session_broken_for_tests(db_session_for_tests):
    """
    Simule une panne de base de données (OperationalError).
    Utilisé pour tester la robustesse des rollbacks et la gestion des erreurs API.
    """
    # Le flush() entrainera un crash
    db_session_for_tests.flush = MagicMock(
        side_effect=OperationalError("Unexpected Crash", params=None, orig=None)  # type:ignore
    )
    # On "espionne" le rollback
    db_session_for_tests.rollback = MagicMock(wraps=db_session_for_tests.rollback)
    return db_session_for_tests
