# imports
import cProfile
import datetime
import functools
import io
import os
import pstats
from pathlib import Path
from typing import Any, Callable, cast

# Au cas où, notamment si la fonction est utilisé de façon isolée
from dotenv import load_dotenv

from app.utils.save_load_datas import save_datas

load_dotenv()


# ============================ INTERNAL FONCTIONAL ================================
def _htmling(func: Callable[..., Any], content: str) -> None:
    """
    Sauvegarde le contenu du profiling dans un fichier html.
    """
    # Définition du chemin absolu de sauvegarde du html
    root = Path(__file__).resolve().parent.parent.parent.parent
    save_dir = root / "datas" / "results"
    # Conversion minimaliste en HTML pour la lisibilité
    style = (
        "body { font-family: monospace; background-color: #1e1e1e; "
        "color: #d4d4d4; padding: 20px; }\n"
        "h2 { color: #569cd6; }\n"
        "pre { background-color: #252526; padding: 15px; "
        "border-radius: 5px; border: 1px solid #3e3e42; }"
    )
    html_report: str = f"""
        <html>
        <head>
            <title>Profiling: {func.__name__}</title>
            <style>{style}</style>
        </head>
        <body>
            <h2>Rapport de performance : {func.__name__}</h2>
            <p>Généré le : {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <pre>{content}</pre>
        </body>
        </html>
        """

    # Sauvegarde du rapport
    timestamp: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_datas(
        data=html_report,
        folder_path=Path(save_dir),
        subs="profiling_reports",
        filename=f"profile_{func.__name__}_{timestamp}",
        format="html",
    )


# ========================= Profiling ===========================


def get_profile(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Décorateur permettant de profiler une fonction et de sauvegarder
    le rapport de performance dans un fichier log formaté.

    Args:
        func (Callable): La fonction à analyser.

    Returns:
        Callable: La fonction enveloppée avec capture des métriques.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Profilage désactivé
        if os.getenv("ENABLE_PROFILING", "false").lower() != "true":
            return func(*args, **kwargs)

        # Instanciation de l'objet
        profiling = cProfile.Profile()
        # Flag d'activité
        is_profiling_active = False
        try:
            # Activation de cProfile
            profiling.enable()
            is_profiling_active = True
        except ValueError:
            # Si on arrive ici, c'est qu'un profiler tourne déjà au-dessus
            # On se contente d'exécuter la fonction sans rien faire d'autre
            return func(*args, **kwargs)

        try:
            # Exécution de la fonction a profiler
            result = func(*args, **kwargs)
            return result
        finally:
            if is_profiling_active:
                # Désactivation de cProfile
                profiling.disable()

                # Instanciation d'un fichier texte virtuel EN RAM
                stream: io.StringIO = io.StringIO()
                # Extraction et tri des statistiques par temps cumulé
                stats: pstats.Stats = pstats.Stats(profiling, stream=stream)
                stats.sort_stats(pstats.SortKey.CUMULATIVE)
                # On prend les 20 premières lignes pour plus de détail
                stats.print_stats(20)

                # Récupération du rapport textuel
                report_content: str = stream.getvalue()

                # Sauvegarde dans un fichier log daté
                _htmling(func, report_content)

    return cast(Callable[..., Any], wrapper)
