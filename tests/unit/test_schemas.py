"""
Suite de tests unitaires pour la validation des schémas Pydantic.

Ce module vérifie que les contrats d'interface de l'API sont respectés et que
la logique métier intégrée aux validateurs fonctionne comme prévu. Il couvre :
1. La validation des données d'entrée (PredictionInput) incluant les types,
    les champs obligatoires et les contraintes métier (outliers).
2. La conformité des modèles de réponse (Output) et leur capacité de coercion.
3. La robustesse des schémas de métadonnées et de gestion d'erreurs.

L'objectif est de garantir qu'aucune donnée corrompue ne puisse atteindre
le modèle ML et que les sorties de l'API soient toujours prévisibles.
"""

# Imports
import pytest
from pydantic import ValidationError

from app.api.schemas import (
    ErrorResponse,
    ModelInfoOutput,
    PredictionInput,
    PredictionOutput,
)

# ===================== PredictionInput =======================


# Happy path
def test_prediction_input_valid(func_sample):
    """Vérifie qu'un dictionnaire de caractéristiques complet et valide est accepté."""
    obj = PredictionInput(**func_sample)

    assert obj.EXT_SOURCE_COUNT == func_sample["EXT_SOURCE_COUNT"]
    assert obj.NAME_FAMILY_STATUS == "Married"


# features manquante
def test_prediction_input_missing_features(base_features):
    """Vérifie que l'absence totale de données déclenche une erreur de validation."""
    payload = base_features.copy()
    payload.pop("EXT_SOURCE_COUNT")
    with pytest.raises(ValidationError):
        PredictionInput(**payload)


# type de feature incorrect
def test_prediction_input_wrong_type(base_features):
    """Vérifie que des types de données incorrects (ex: liste au lieu d'int) sont rejetés."""
    payload = base_features.copy()
    payload["EXT_SOURCE_COUNT"] = [0.5]  # Invalide
    with pytest.raises(ValidationError):
        PredictionInput(**payload)


# Champs obligatoires manquant
@pytest.mark.parametrize("mandatory_features", ["EXT_SOURCE_COUNT", "AMT_ANNUITY", "DAYS_BIRTH"])
def test_prediction_mandatory_input_missing_business_rules(mandatory_features, base_features):
    """
    Vérifie la présence obligatoire de chaque caractéristique métier critique.
    Teste systématiquement l'absence
    """
    # On test que la suppr de la feature mandatory_features donne bien une erreur
    payload = base_features.copy()
    payload.pop(mandatory_features)
    with pytest.raises(ValidationError) as excinfo:
        PredictionInput(**payload)
    # Vérification que l'erreur mentionne le champ manquant
    assert mandatory_features in str(excinfo.value)


# Outliers
@pytest.mark.parametrize(
    "invalid_features, expected_keywords_error_msg",
    [
        ({"EXT_SOURCE_COUNT": 12.0}, ["EXT_SOURCE_COUNT", "3"]),
        ({"DAYS_BIRTH": -30000}, ["DAYS_BIRTH", "-25229"]),
        ({"DAYS_BIRTH": -500}, ["DAYS_BIRTH", "7489"]),
        (
            {"NAME_FAMILY_STATUS": "Alone"},
            [
                "NAME_FAMILY_STATUS",
                "Married",
                "Single / not married",
                "Civil marriage",
                "Separated",
                "Widow",
                "Unknown",
            ],
        ),
    ],
)
def test_prediction_input_invalid_business_rules(
    invalid_features, expected_keywords_error_msg, base_features
):
    """
    Valide les contraintes de domaine métier (Custom Validators).
    Vérifie le rejet des valeurs aberrantes pour l'âge, les heures supp. et le revenu.
    """
    payload = base_features.copy()
    payload.update(invalid_features)

    with pytest.raises(ValidationError) as excinfo:
        PredictionInput(**payload)

    # On vérifie que TOUS les mots-clés attendus sont dans l'erreur
    error_str = str(excinfo.value)
    for keyword in expected_keywords_error_msg:
        # print(f"DEBUG: {str(excinfo.value)}") # pour debut
        assert keyword in error_str


# ===================== PredictionOutput =======================


# Happy path
def test_prediction_output_valid():
    """Vérifie qu'une sortie de prédiction standard respecte le schéma de réponse."""
    out = PredictionOutput(
        prediction=0,
        confidence=0.9,
        class_name="solvable",
    )

    assert out.prediction == 0
    assert out.class_name == "solvable"


# confidence hors bornes
# décorateur parametrize permet d'executer même test plusieurs fois
# permettant de tester les deux bornes de confidence
@pytest.mark.parametrize("confidence", [-0.1, 1.5])
def test_prediction_output_invalid_confidence(confidence):
    """Vérifie que le score de confiance est strictement borné entre 0.0 et 1.0."""
    with pytest.raises(ValidationError):
        PredictionOutput(
            prediction=1,
            confidence=confidence,
            class_name="insolvable",
        )


# ===================== Métadatas du modèle =====================
# Test de santé pas de précision, on s'attend juste a ce que toutes les
# données soient présntes et dans avec le bon type
def test_model_info_output():
    """Vérifie l'exhaustivité et le typage des métadonnées descriptives du modèle."""
    obj = ModelInfoOutput(
        model_type="HistGradientBoostingClassifier",
        n_features=11,
        feature_names=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11"],
        classes=["solvable", "insolvable"],
        threshold=0.5,
    )

    assert obj.model_type == "HistGradientBoostingClassifier"
    assert obj.n_features == 11
    assert obj.feature_names == ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11"]
    assert obj.classes == ["solvable", "insolvable"]
    assert obj.threshold == 0.5


# ===================== Erreur standardisée =====================
# On garantie que l'erreur est stable et que l'API ne casse pas quand
# detail est absent
def test_error_response_optional_detail():
    """S'assure que le champ 'detail' des erreurs est bien optionnel et n'entraîne pas de crash."""
    err = ErrorResponse(error="Model Error", detail="Check logs")
    assert err.error == "Model Error"
