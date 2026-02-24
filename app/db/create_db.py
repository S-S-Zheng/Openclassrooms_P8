"""
Module d'initialisation du schéma de la base de données.

Ce module fournit les outils nécessaires pour synchroniser les modèles ORM avec
la base de données physique. Il permet la création des tables ainsi que leur
réinitialisation (drop & create) pour les environnements de test ou de développement.
"""

# Placer dans app/ permet d'importer facilement models_db et database pour créer les tables.

# imports
import sys
from pathlib import Path

# Ajout du dossier racine au path pour permettre les imports relatifs
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

# IMPORT OBLIGATOIRE des modèles pour l'enregistrement des métadonnées
# Même si non utilisés directement, cela permet à Base.metadata de les "voir" et d'initier le lien
# Sinon le registre db restera vide et metadata.create_all ne fera rien et sans lever d'erreur!
from app.db.base import Base  # noqa: E402 # Ruff veut les imports avant tout, pb ici
from app.db.database import base_engine  # noqa: E402

# ====================== Création de la DB ============================


def init_db(reset_tables=False, engine=base_engine):
    """
    Initialise la structure de la base de données PostgreSQL.

    Cette fonction utilise les métadonnées de SQLAlchemy pour générer le schéma
    SQL correspondant aux classes héritant de 'Base'. Elle peut être configurée
    pour supprimer les tables existantes avant la création.

    Args:
        reset_tables (bool, optional): Si True, supprime toutes les tables existantes
            avant de les recréer. Utile pour repartir d'une base vierge. Par défaut à False.
        engine (sqlalchemy.engine.Engine, optional): L'instance du moteur de base de données
            à utiliser. Par défaut, utilise 'base_engine' configuré pour Supabase/PostgreSQL.

    Raises:
        Exception: Relance toute exception survenant lors de la communication avec
            le serveur de base de données (ex: erreur de connexion, droits insuffisants).

    Note:
        L'import de 'PredictionRecord' et 'RequestLog' est nécessaire ici, même s'ils
        ne sont pas appelés explicitement, afin qu'ils soient enregistrés dans le registre
        'Base.metadata' avant l'exécution de 'create_all'.
    """
    if reset_tables:
        print("Suppression des anciennes tables...")
        Base.metadata.drop_all(bind=engine)
    print("Initialisation de la base de données...")
    try:
        # Crée toutes les tables définies dans models_db qui héritent de Base
        Base.metadata.create_all(bind=engine)
        print("Tables créées avec succès:")
        print(f" - Tables créées : {list(Base.metadata.tables.keys())}")
    except Exception as e:
        print(f"Erreur lors de la création de la base : {e}")
        raise e


# Empeche le script de se lancer par erreur si appelé par un autre script
# Inutile à tester puisque c'est juste un déclencheur conditionnel
if __name__ == "__main__":
    init_db()
