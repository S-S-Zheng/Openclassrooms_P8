"""
Module de configuration et de gestion de la connexion PostgreSQL.

Ce module constitue le cœur de l'infrastructure de données. Il gère la construction
sécurisée de l'URL de connexion (notamment le traitement des caractères spéciaux
et l'activation du SSL pour Supabase), configure le moteur SQLAlchemy (Engine)
et fournit des générateurs de sessions pour l'API et les scripts utilitaires.
"""

# C'est le cœur de l'infrastructure de données.
# Il doit être accessible à la fois par les routes et par les scripts de création.
# Configuration de la connexion (Engine, SessionLocal)

import logging
import os

# imports
import urllib.parse  # Import indispensable pour les caractères spéciaux
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

load_dotenv()
logger = logging.getLogger(__name__)

# =================== Mise en place ===========================

# ====================== Variables d'environnement ==============
# Récupération sécurisée + val par défaut pour éviter la ValueError
# (on préfèrera la ConnectionError) qui et moins sévère avec accès aux logs)
db_user = os.getenv("SB_USER", "postgres")
raw_password = os.getenv("SB_PASSWORD", "")
if not raw_password:
    logger.warning("Attention : Aucun mot de passe de base de données trouvé (SB_PASSWORD).")
db_pass = urllib.parse.quote_plus(raw_password)  # Sécurise le password
db_host = os.getenv("SB_HOST", "localhost")
db_port = os.getenv("SB_PORT", "5432")  # val défaut crucial pour eviter le crash
db_name = os.getenv("SB_DB", "postgres")

# On sécurise la connexion HF/Supabase par ssl
options = ""
if db_host and "supabase.com" in db_host:
    options = "?sslmode=require"
DATABASE_URL = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}{options}"
# DATABASE_URL = os.getenv("DATABASE_URL")

# ENGINE: point de départ de SQLAlchemy
# -----------------
# Accepte 3 connexions simultanées et jusqu'à 10 en cas de pic temporaire
# Vérif la connexion toujours valide (indispensable en Cloud)
# base_engine = create_engine(
#     DATABASE_URL,
#     pool_size=3,
#     max_overflow=7,
#     pool_pre_ping=True,
#     connect_args={"connect_timeout": 10}
# )
# ------------------
# If using Transaction Pooler or Session Pooler,
# we want to ensure we disable SQLAlchemy client side pooling -
# https://docs.sqlalchemy.org/en/20/core/pooling.html#switching-pool-implementations
base_engine = create_engine(DATABASE_URL, poolclass=NullPool, connect_args={"connect_timeout": 15})

# SessionLocal est une factory à sessions pour les routes
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=base_engine)


# Fonction utilitaire pour récupérer une session de base de données
def get_db_generator():
    """
    Générateur de session de base de données.

    Crée une nouvelle session SQLAlchemy pour une opération unique et garantit
    sa fermeture systématique après utilisation, même en cas d'exception.

    Yields:
        Session: Une instance de session SQLAlchemy (SessionLocal).

    Note:
        Utilisé principalement comme dépendance injectée dans les routes FastAPI.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# contextmanager est redondant pour FastAPI mais nécéssaire pour les with get_db du coup
# FASTAPI
get_db = get_db_generator

# Adaptateur pour l'utilisation via l'instruction 'with' dans les scripts Python
get_db_contextmanager = contextmanager(get_db_generator)
