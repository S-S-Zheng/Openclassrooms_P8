"""
Module de génération d'identifiants uniques par hachage cryptographique.

Ce module fournit une fonction permettant de transformer un dictionnaire de
caractéristiques (features) en une empreinte numérique unique (ID). Ce mécanisme
est la pierre angulaire du système pour :
1. Identifier de manière déterministe chaque profil d'employé.
2. Éviter les doublons lors des imports de données historiques.
3. Implémenter un système de cache pour les requêtes API en temps réel.
"""

# imports

import hashlib
import json

# ======================


def generate_feature_hash(features: dict) -> str:
    """
    Génère un identifiant unique (SHA-256) à partir d'un dictionnaire de caractéristiques.

    Le processus garantit le déterminisme (le même dictionnaire produit toujours
    le même hash) en suivant deux étapes critiques :
    1. Tri alphabétique des clés du dictionnaire pour neutraliser l'ordre d'insertion.
    2. Sérialisation JSON et encodage en UTF-8 avant le passage dans l'algorithme SHA-256.

    Args:
        features (dict): Le dictionnaire contenant les variables d'entrée de la requête.

    Returns:
        str: Une chaîne de 64 caractères hexadécimaux représentant l'empreinte unique.

    Example:
        >>> features = {"age": 30, "poste": "Manager"}
        >>> generate_feature_hash(features)
        '7a1b...8f2e'
    """
    # 1. Trier les clés pour que {"a":1, "b":2} donne le même résultat que {"b":2, "a":1}
    encoded_features = json.dumps(features, sort_keys=True).encode("utf-8")
    # 2. Hasher
    return hashlib.sha256(encoded_features).hexdigest()
