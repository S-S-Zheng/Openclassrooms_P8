# imports
import cProfile
import datetime
import functools
import inspect
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
    # timestamp: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = str(f"profile_{func.__name__}")  # _{timestamp}"
    print(f"--- PROFILING: Tentative de sauvegarde de {filename} ---")
    # Définition du chemin absolu de sauvegarde du html
    ROOT_DIR = Path(__file__).resolve().parents[3]
    SAVE_DIR = ROOT_DIR / "datas" / "results"
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
    save_datas(
        data=html_report,
        folder_path=Path(SAVE_DIR),
        subs="profiling_reports",
        filename=filename,
        format="html",
    )


def _finalize_profiling(profiling: cProfile.Profile, func: Callable[..., Any]) -> None:
    """
    Cumul les dernières étapes du profilage
    """
    # Désactivation de cProfile
    profiling.disable()
    # Instanciation d'un fichier texte virtuel EN RAM
    stream: io.StringIO = io.StringIO()
    # Extraction et tri des statistiques par temps cumulé
    stats: pstats.Stats = pstats.Stats(profiling, stream=stream)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    # On prend les 30 premières lignes pour plus de détail
    stats.print_stats(30)
    # Récupération du rapport textuel
    report_content: str = stream.getvalue()
    # Sauvegarde dans un fichier log daté
    _htmling(func, report_content)


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
            # SI ASYNC : On définit une fonction coroutine pour attendre le résultat
            if inspect.iscoroutine(result):

                async def wait_for_result():
                    try:
                        return await result
                    finally:
                        if is_profiling_active:
                            _finalize_profiling(profiling, func)

                return wait_for_result()
            # SI SYNC : On retourne le résultat normalement
            return result
        finally:
            # On ne finalise ici QUE si ce n'est pas un objet async
            # (car l'async finalisera dans wait_for_result)
            if is_profiling_active and not inspect.iscoroutine(result):
                _finalize_profiling(profiling, func)

    return cast(Callable[..., Any], wrapper)
