# --- Étape 1 : Build (Installation des dépendances) ---
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_VIRTUALENVS_CREATE=true

# Installation des dépendances système nécessaires à la compilation (gcc, libpq, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installation de Poetry 2.0
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

# Copie des fichiers de config uniquement (pour le cache Docker)
COPY pyproject.toml ./ 
# Note : on ne copie pas encore le code pour optimiser le cache des layers
RUN /root/.local/bin/poetry install --only main --no-root

# --- Étape 2 : Runtime (Image finale légère) ---
FROM python:3.12-slim AS runtime

# Installation des dépendances système pour l'exécution (libgomp1 pour CatBoost, libpq pour Postgres)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Création de l'utilisateur non-root pour HF Spaces
RUN useradd -m -u 1000 user
USER user
ENV PATH="/app/.venv/bin:/home/user/.local/bin:${PATH}" \
    PYTHONPATH="/app/src"

WORKDIR /app

# On récupère l'environnement virtuel créé à l'étape précédente
COPY --from=builder --chown=user /app/.venv /app/.venv
# On copie le code source
COPY --chown=user . .

EXPOSE 7860

# ---------------- EN LOCAL ----------------
# # Commande de lancement
# CMD python -m app.db.create_db && \
#     python -m app.db.import_dataset_to_db && \
#     uvicorn app.main:app --host 0.0.0.0 --port 7860
# On ne garde que le lancement de l'API.
# ---------------- AVEC SUPABASE ---------------------
# L'initialisation de la DB se fait une seule fois manuellement ou via une migration.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]