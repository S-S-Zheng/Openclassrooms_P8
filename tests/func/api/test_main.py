"""
Suite de tests pour le point d'entrée principal (Main) et le cycle de vie de l'application.

Ce module valide les fonctionnalités de base de l'infrastructure FastAPI :
1. La disponibilité opérationnelle via le endpoint de santé (Healthcheck).
2. La redirection de l'URL racine vers l'interface de documentation Swagger.
3. Le bon fonctionnement du 'Lifespan', garantissant que le modèle ML est
    correctement chargé en mémoire au démarrage du serveur.
"""

# Imports
import pytest

# =================== Health =======================


@pytest.mark.integration
# On s'assure que /health est fonctionnelle: code 200
# On s'assure que la réponse est bien status:ok
def test_healthcheck(client):
    """
    Vérifie que le point d'entrée de santé est opérationnel.

    Indispensable pour les sondes de disponibilité (Liveness/Readiness probes)
    dans les environnements de déploiement type Docker ou Kubernetes.
    """
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# =================== Root =======================


@pytest.mark.integration
def test_root_redirects_to_docs(client):
    """
    Vérifie la redirection automatique de la racine.

    S'assure que tout utilisateur accédant à l'URL de base est immédiatement
    orienté vers la documentation interactive de l'API (Swagger UI).
    """
    # follow_redirects=False permet de vérifier le code 307 de redirection
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


# =================== Lifespan =======================


@pytest.mark.integration
def test_lifespan_startup(client):
    """
    Valide l'initialisation du contexte de l'application.

    Vérifie que le mécanisme 'lifespan' a correctement injecté l'instance du
    modèle ML dans l'état global de l'application (`app.state.model`),
    permettant son utilisation par les routes d'inférence.
    """
    # Accès à l'état global de l'application FastAPI
    app_state = client.app.state

    # Vérifie que l'instance a bien été créée et attachée
    # injection du modele dans l'app.state ok?
    assert hasattr(client.app.state, "model")
    assert client.app.state.model is not None
    # Présence des features?
    assert app_state.model.feature_names is not None
