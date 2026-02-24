"""
Module de définition de la base déclarative SQLAlchemy.

Ce module centralise l'initialisation de la classe 'Base' pour l'ORM.
L'isolation de cet objet dans un fichier dédié est une pratique de conception
visant à prévenir les imports circulaires lors de la définition de multiples
modèles répartis dans différents fichiers du projet.
"""

# imports
from sqlalchemy.orm import declarative_base

# Instance de base pour la définition des modèles ORM.
# Base est la classe mère dont hériteront tous les modèles SQL
Base = declarative_base()
