#!/bin/bash

# Pour initialiser la db et lancer l'API
# Créer les tables et importer le dataset
python -m app.db.create_db
python -m app.db.import_dataset_to_db

# Lancer l'API
uvicorn app.main:app --host 127.0.0.1 --port 7860