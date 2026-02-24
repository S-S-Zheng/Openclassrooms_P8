"""
Point d'entrée principal de l'application FastAPI.

Ce module assemble les différents composants de l'architecture :
1. Orchestre le cycle de vie de l'application (Lifespan) pour le chargement du modèle.
2. Centralise l'inclusion des routeurs spécialisés (Inférence, Importance, Métadonnées).
3. Définit les endpoints de base comme la vérification de l'état (Healthcheck).
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.api.routes.model_info import router as model_info_router
from app.api.routes.predict import router as predict_router
from app.ml.model import HGBM

# ==================== Imports ==========================


# assynccontextmanager est un décorateur qui permet de définir une fonction
# capable de gérer une phase avant de démarrage et une après d'arrêt.
# Ici, tout ce qui est écrit avant yield s'éxé UNE SEULE FOIS au lancement
# du serveur ce qui permet de maintenit l'état tant que le serveur est en ON
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le cycle de vie de l'application (Démarrage et Arrêt).

    Tout le code situé avant l'instruction 'yield' est exécuté une seule fois
    lors du lancement du serveur. Cela permet d'initialiser les ressources lourdes
    (modèle ML) et de les maintenir en mémoire RAM tout au long de la session.

    Actions au démarrage :
        - Instanciation de la classe MLModel.
        - Chargement des artefacts (modèle, features, seuil).
        - Injection de l'instance dans 'app.state' pour un accès global via les requêtes.

    Args:
        app (FastAPI): L'instance de l'application.
    """
    # ============== Phase de démarrage ================
    # on charge une seule fois les données pour optimiser la RAM
    model_instance = HGBM()
    model_instance.load()
    # Stockage de l'app pour qu'il soit accessible partout
    app.state.model = model_instance

    yield
    # ================== Phase d'arrêt =================
    # On coupe les connexions avec la DB


# ==================== API =============================


app = FastAPI(
    title="ML Prediction API",
    description="API REST pour Prêt à depenser",
    version="1.0.0",
    lifespan=lifespan,
)


# ==================== Montage des Routers ==========================


# Inclusion des modules de routes pour une architecture modulaire et scalable
app.include_router(predict_router)
app.include_router(model_info_router)


# ==================== Endpoints Génériques ========================


# /health
# Test auto CI/CD, debug rapide
# FONDAMENTAL + NE DOIT JAMAIS DEPENDRE DU ML OU DE LA DB
@app.get("/health", tags=["Health"])
def healthcheck():
    """
    Vérifie la disponibilité opérationnelle du service.

    Ce endpoint est crucial pour les outils de monitoring.
    Il doit rester indépendant des ressources externes pour
    isoler les pannes réseau/modèle de la panne serveur.

    Returns:
        dict: Un dictionnaire indiquant le statut opérationnel.
    """
    return {"status": "ok"}


# / (root)
# Feedback immédiat, debug, UX minimale
@app.get("/", tags=["Root"], include_in_schema=False)
def root():
    """
    Point d'entrée racine.

    Redirige automatiquement l'utilisateur vers la documentation Swagger
    interactive (/docs) pour faciliter l'exploration de l'API.

    Returns:
        RedirectResponse: Redirection vers l'interface utilisateur Swagger.
    """
    return RedirectResponse(url="/docs")
