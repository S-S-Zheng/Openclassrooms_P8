"""
Suite de tests fonctionnels de bout-en-bout (E2E) pour l'API de prédiction.

Ce module utilise des profils YAML prédéfinis (fixtures dynamiques) pour simuler
des cas d'utilisation réels et complexes. Il valide la capacité de l'API à
traiter différents scénarios métier, allant du parcours nominal (Happy Path)
aux erreurs de validation structurelles ou statistiques (Outliers).

L'utilisation du décorateur 'indirect=True' permet de charger chaque profil
comme une configuration isolée, garantissant que l'API réagit conformément
aux spécifications fonctionnelles pour chaque type d'employé testé.
"""

# imports
import pytest

# ======================== test des profiles =========================


@pytest.mark.parametrize(
    "functionnal_profile",
    ["happy_path", "missing_features", "outliers", "over_featured"],
    indirect=True,
)
def test_predict_functionnal(client, functionnal_profile):
    """
    Exécute un test d'intégration complet basé sur des profils YAML.

    Ce test vérifie le comportement global du système :
    1. Parsing du payload JSON par FastAPI.
    2. Validation des contraintes métier par Pydantic.
    3. Inférence par le modèle ML (ou récupération via le cache DB).
    4. Formatage de la réponse HTTP finale.

    Scénarios testés :
    - 'happy_path' : Données conformes, attend un code 200.
    - 'missing_features' : Absence de variables clés, attend un code 422.
    - 'outliers' : Valeurs hors bornes (ex: âge > 65), attend un code 422.
    - 'over_featured' : Surplus de données non attendues, attend un code 422.

    Args:
        client (TestClient): Client de test FastAPI.
        functionnal_profile (dict): Dictionnaire chargé depuis un fichier YAML
            contenant les features et le code de statut attendu.
    """
    # On récupère le nom du profil utilisé pour ce test précis sinon on attribut 200
    expected_status = functionnal_profile.get("expected_status", 200)
    payload = functionnal_profile["features"]

    response = client.post("/predict/manual", json=payload)

    assert response.status_code == expected_status, (
        f"Échec pour le profil {functionnal_profile.get('_profile_name')}: {response.text}"
    )

    # Vérification du contenu (uniquement si le test doit réussir)
    if expected_status == 200:
        json_data = response.json()
        assert "prediction" in json_data
        assert 0 <= json_data["confidence"] <= 1.0
        assert json_data["class_name"] in ["solvable", "insolvable"]
