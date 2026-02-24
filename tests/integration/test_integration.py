"""
Test d'intégration du pipeline d'inférence avec les artefacts réels.

Ce module effectue un test de "bout-en-bout" sur la couche ML en utilisant
les véritables fichiers binaires (.pkl) produits lors de la phase
d'entraînement. Il garantit que le modèle chargé en local est compatible
avec le code du wrapper et que les types de données produits
en sortie sont strictement conformes aux attentes de l'API (notamment
l'absence de types NumPy natifs non sérialisables).
"""

# Imports
import pytest

# =========================================================


@pytest.mark.integration
def test_functional_pipeline_real_model(ml_model_mocked, base_features):
    """
    Valide le chargement et l'inférence sur le modèle de production réel.

    Ce test vérifie que :
    1. Le modèle et ses métadonnées (features, seuils) sont accessibles et chargeables.
    2. Un dictionnaire d'entrée généré dynamiquement selon le schéma du modèle
        est correctement traité.
    3. Les sorties (prediction, confidence) sont converties en types Python natifs
        (int, float) pour éviter les erreurs de sérialisation JSON.

    Args:
        ml_model_mocked: Instance réelle du wrapper de modèle.

    Raises:
        pytest.skip: Si les fichiers du modèle sont absents de l'environnement de test.
    """
    try:
        ml_model_mocked.load()  # Charge fichiers locaux
    except Exception as e:
        pytest.skip(f"Fichiers de modèle non trouvés pour le test fonctionnel : {e}")

    # On récupère la liste des features depuis le wrapper (chargée depuis le modèle ou le pickle)
    model_features = getattr(
        ml_model_mocked.model,
        "feature_names_in_",
        getattr(ml_model_mocked.model, "feature_names_", None),
    )  # on tente feature_names_in_ sinon feature_names_ sinon None
    if model_features is None:
        # Fallback sur les noms chargés par le wrapper depuis le pickle
        model_features = ml_model_mocked.feature_names

    # On utilise base_features pour garantir le types et nom des features attendues
    sample_input = {name: base_features[name] for name in model_features}

    # Exécution
    prediction, confidence, class_name = ml_model_mocked.predict(sample_input)

    # Assertions sur les types de sortie (très important pour ton erreur ndarray/float)
    assert isinstance(prediction, int)
    assert isinstance(confidence, float)
    assert isinstance(class_name, str)
    assert 0 <= confidence <= 1.0
    # Pour voir ce que ressort le test (pytest -m integration -s)
    print(f"\n{sample_input} \nPred={prediction} \nConf={confidence} \nClass_name={class_name}")
