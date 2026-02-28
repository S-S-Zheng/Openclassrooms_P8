"""
Module de définition du router pour télécharger le rapport de profilage.

Ce module expose un endpoint permettant de rechercher le rapport de profilage et de le récupérer
soit via son nom si on le connait soit le plus récent.
"""

# ====================== Imports ========================
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/reports", tags=["Profiling"])

# ===================== Initialisation du modele =========================

# Définition du chemin absolu de sauvegarde du html
ROOT_DIR = Path(__file__).resolve().parents[3]  # equivalent a parent.parent.parent.parent
REPORT_DIR = ROOT_DIR / "datas" / "results" / "profiling_reports"


# Deux router get car filename est optionnel
@router.get("/")
@router.get("/{filename}")
async def download_report(filename: Optional[str] = None):
    """
    Télécharge un rapport.
    - Sans nom : télécharge le plus récent.
    - Avec nom : télécharge le fichier spécifié (ajoute .html si besoin).
    """
    # ========== Cas du dossier de rapport inexistant ===================
    if not REPORT_DIR.exists():
        raise HTTPException(status_code=404, detail="Dossier de rapports inexistant.")

    target_file: Optional[Path] = None
    # =============== Cas l'utilisateur donne un nom de fichier ============
    if filename:
        # On cherche le fichier précis demandé
        clean_name = filename if filename.endswith(".html") else f"{filename}.html"
        target_file = REPORT_DIR / clean_name
    else:
        # ============== Cas l'utilisateur n'a rien mis, on cherche le plus récent ==========
        files = list(REPORT_DIR.glob("*.html"))
        if not files:
            raise HTTPException(status_code=404, detail="Aucun fichier HTML trouvé.")
        target_file = max(files, key=os.path.getmtime)

    # Validation et envoi
    if target_file and target_file.is_file():
        return FileResponse(
            path=target_file,
            filename=target_file.name,
            media_type="application/octet-stream",  # Force le téléchargement
        )
    raise HTTPException(status_code=404, detail="Rapport non trouvé.")
