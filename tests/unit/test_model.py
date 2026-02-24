"""
Suite de tests unitaires pour la classe MLModel.

Ce module valide le comportement interne du wrapper de modèle CatBoost.
Il utilise intensivement le mockage pour simuler les artefacts du modèle
(fichiers .cbm, .pkl) et l'objet CatBoostClassifier lui-même, garantissant
que la logique de chargement, de prédiction et d'analyse est robuste
face à diverses configurations et erreurs de fichiers.
"""

import logging
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np  # noqa: F401
import pytest

from app.ml.model import HGBM

# =============================== INIT =======================================


# test methode init() ()
@pytest.mark.unit
def test_init(ml_model_mocked):
    """Vérifie l'état initial par défaut d'une nouvelle instance."""
    assert ml_model_mocked.model is not None
    assert ml_model_mocked.feature_names is not None
    assert ml_model_mocked.threshold == 0.6 or ml_model_mocked.threshold == 1.0
    assert ml_model_mocked.classes == ["solvable", "insolvable"]


# ======================== GET MODEL INFO =================================


# Happy path + threshold limite
@pytest.mark.unit
@pytest.mark.parametrize("test_threshold", [0.6, 0.5])
def test_get_model_info(hgbm_instance_mock, test_threshold, ml_model_mocked):
    """
    Vérifie la compilation correcte des métadonnées du modèle.
    Teste la structure du dictionnaire retourné avec et sans seuil personnalisé.
    """
    ml_model_mocked.model = hgbm_instance_mock
    ml_model_mocked.feature_names = list(ml_model_mocked.model.feature_names_in_)
    ml_model_mocked.classes = ["solvable", "insolvable"]
    ml_model_mocked.threshold = test_threshold

    model_info = ml_model_mocked.get_model_info()

    assert isinstance(model_info, dict)
    # Rq: create_autospec(spec:nom_modele) implique que
    # le nom du type doit être celui de la classe mockée
    assert model_info["model_type"] == "HistGradientBoostingClassifier"
    assert model_info["n_features"] == len(ml_model_mocked.feature_names)
    assert "EXT_SOURCE_COUNT" in model_info["feature_names"]
    assert model_info["classes"] == ["solvable", "insolvable"]
    # Avec le décorateur on test les deux valeurs
    assert model_info["threshold"] == test_threshold


# =====================================================================


# vérifie que si ce n'est pas un mock, on aille bien regarder la classe réelle du modèle
@pytest.mark.unit
def test_get_model_info_real_name(ml_model_mocked) -> None:
    """
    Vérifie que le nom réel du modèle est correctement extrait.

    Args:
        ml_model_mocked (HGBM): Instance de la classe HGBM.
    """
    # On injecte le vrai modèle
    ml_model_mocked.model = HGBM()

    info = ml_model_mocked.get_model_info()

    # Assertion
    assert info["model_type"] == type(ml_model_mocked.model).__name__


# =====================================================================


# Echec de chargement du modele
@pytest.mark.unit
def test_get_model_info_not_loaded(ml_model_mocked):
    """Vérifie qu'une erreur est levée si l'on demande les infos d'un modèle non chargé."""
    ml_model_mocked.model = None

    with pytest.raises(ValueError):
        ml_model_mocked.get_model_info()


# ================================= LOAD ======================================


# test de la méthode load() (model, features, seuil)
# load - CAS nominal
@pytest.mark.unit
def test_load_model_success(tmp_path, hgbm_instance_mock, ml_model_mocked):
    """
    Teste le scénario nominal de chargement complet du modèle.
    Valide l'intégration entre le binaire, les noms de features et le seuil.
    """
    ml_model_mocked.load()
    assert isinstance(ml_model_mocked.feature_names, list)
    # On compare des listes
    assert ml_model_mocked.feature_names == list(hgbm_instance_mock.feature_names_in_)


# =======================================================================


# load - CAS model manquant
@pytest.mark.unit
def test_load_model_failed(
    tmp_path,
):
    """Vérifie que l'instance reste à None si le fichier binaire du modèle est introuvable."""
    model_file = tmp_path / "absent.pkl"
    ml = HGBM(model_file)
    ml.load()

    assert ml.model is None


# =======================================================================


# chargement du pkl si feature_names_in_ absent
@pytest.mark.unit
def test_load_features_from_pkl_fallback(
    ml_model_mocked, base_features, hgbm_instance_mock
) -> None:
    """
    Vérifie le chargement des features depuis un fichier .pkl si le modèle est incomplet.

    S'assure que :
    - Le fallback vers le fichier de features fonctionne quand 'feature_names_in_' manque.
    - L'extraction du premier dictionnaire (isinstance dict) est correcte.

    Args:
        ml_model_mocked (HGBM): Instance de la classe HGBM.
    """
    # Faux modèle sans l'attribut feature_names_in_
    fake_model = hgbm_instance_mock
    if hasattr(fake_model, "feature_names_in_"):
        del fake_model.feature_names_in_
    fake_features_data = {"data": list(base_features.keys())}
    fake_threshold = 0.6

    with patch("pathlib.Path.exists", return_value=True), patch("joblib.load") as mock_joblib:
        # Le premier appel pour le modèle, le second pour les features, le troisieme pour le thresh
        mock_joblib.side_effect = [fake_model, fake_features_data, fake_threshold]

        ml_model_mocked.load()

        # Assertion
        assert ml_model_mocked.feature_names == list(base_features.keys())


# =======================================================================


# fichier liste des features absente
@pytest.mark.unit
def test_load_feature_names_absent(caplog, monkeypatch):
    """Vérifie le comportement et la journalisation d'erreur si le fichier de features manque."""
    # On mocke exists pour qu'il renvoie True
    monkeypatch.setattr(Path, "exists", lambda self: "feature_names" not in str(self))
    monkeypatch.setattr(joblib, "load", lambda x: None)

    ml = HGBM(feature_names_path="feature_names.pkl")
    with caplog.at_level(logging.ERROR):
        ml.load()

    assert "Fichier features absent" in caplog.text


# ================================ PREDICT =============================


@pytest.mark.unit
def test_predict_model(ml_model_mocked, base_features):
    # On s'assure que le wrapper connaît les features
    ml_model_mocked.feature_names = list(base_features.keys())
    # ml_model_mocked_mocked est déjà une instance de HGBM chargée
    result = ml_model_mocked.predict(base_features)

    assert result[0] == 1
    assert result[1] == 0.8


# =======================================================================


# test predict() (model nn chargé, nb features diff, seuil par défaut ou opt)
# predict - CAS modele non chargé
@pytest.mark.unit
def test_predict_model_not_loaded(base_features, ml_model_mocked):
    """S'assure qu'une prédiction est impossible sans chargement préalable."""
    ml_model_mocked.model = None  # On décharge explicitement
    with pytest.raises(ValueError):
        ml_model_mocked.predict(base_features)


# =======================================================================


# predict - CAS seuil de validation
@pytest.mark.unit
@pytest.mark.parametrize("thresh", [0.6, None])
def test_predict_with_threshold(hgbm_instance_mock, thresh, ml_model_mocked, base_features):
    """
    Valide la logique de prédiction binaire.
    Compare le comportement entre le seuil par défaut du modèle et le seuil optimisé.
    """
    ml_model_mocked.model = hgbm_instance_mock
    ml_model_mocked.feature_names = list(ml_model_mocked.model.feature_names_in_)
    ml_model_mocked.threshold = thresh

    pred, conf, class_name = ml_model_mocked.predict(base_features)

    assert pred == 1.0
    assert conf == 0.8
    assert class_name == "insolvable"
