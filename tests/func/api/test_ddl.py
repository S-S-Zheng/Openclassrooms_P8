# Improts
from pathlib import Path

from fastapi.testclient import TestClient

# ===========================================================


# Verifie que la logique du fichier le plus recent est fonctionnel
def test_download_latest_logic(client: TestClient, mock_reports_dir):
    """
    Teste la récupération du fichier le plus récent.
    """
    # On crée deux fichiers avec des "dates différentes"
    file_old = mock_reports_dir / "old.html"
    file_old.write_text("vieux")

    file_new = mock_reports_dir / "recent.html"
    file_new.write_text("récent")

    # Assertions
    response = client.get("/reports/")
    assert response.status_code == 200
    # On vérifie que c'est bien le contenu du plus récent
    assert response.headers["content-disposition"] == 'attachment; filename="recent.html"'


# ==================================================================


# Dossier pas encore créé
def test_download_no_directory(client: TestClient, monkeypatch):
    """
    'Dossier de rapports inexistant.' (HTTP 404)
    """
    # On mock un chemin qui mene nulle part
    monkeypatch.setattr("app.api.routes.ddl.REPORT_DIR", Path("/tmp/non_existent_path_999"))

    # Assertions
    response = client.get("/reports/")
    assert response.status_code == 404
    assert response.json()["detail"] == "Dossier de rapports inexistant."


# ====================================================================


# Dossier existant mais vide
def test_download_no_files_in_directory(client: TestClient, mock_reports_dir):
    """
    'Aucun fichier HTML trouvé.' (HTTP 404)
    """
    # Assertions
    response = client.get("/reports/")
    assert response.status_code == 404
    assert response.json()["detail"] == "Aucun fichier HTML trouvé."


# =========================================================================


# Le nom de fichier ne correspond a rien
def test_download_specific_file_not_found(client: TestClient, mock_reports_dir):
    """
    'Rapport non trouvé.' (HTTP 404) quand le fichier précis manque
    """
    # Demande de chemin inconnu
    file_non_existent = mock_reports_dir / "unknown"
    file_non_existent.write_text("unknown")

    # Assertions
    # On demande 'unknown.html'
    response = client.get("/reports/unknown")
    assert response.status_code == 404
    assert response.json()["detail"] == "Rapport non trouvé."
