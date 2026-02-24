from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class BaseMLModel(ABC):  # pragma: no cover
    """
    Classe de base abstraite définissant l'interface pour tous les modèles ML.
    Respecte le principe de substitution de Liskov (LSP).
    """

    def __init__(
        self,
        model_path: Union[Path, str],
        feature_names_path: Union[Path, str],
        threshold_path: Optional[Union[Path, str]],
    ):
        """
        Initialise les chemins vers les artefacts du modèle.

        Args:
            model_path (Union[Path,str]): Chemin vers le modele.
            feature_names_path (Union[Path,str]):
                Chemin vers le fichier contenant le nom des features.
            threshold_path (Optional[Union[Path,str]]):
                Chemin vers le file de seuil de décision optimisé.
        """
        self.model_path = model_path
        self.feature_names_path = feature_names_path
        self.threshold_path = threshold_path

        self.model: Any = None
        self.feature_names: List[str] = []
        self.threshold: float = 0.5
        self.classes = [0, 1]

    @abstractmethod
    def load(self) -> None:
        """
        Charge les artefacts du modèle.
        """
        pass

    @abstractmethod
    def predict(self, features: Dict[str, Any]) -> Tuple[int, float, str]:
        """
        Réalise une inférence à partir d'un dictionnaire de caractéristiques.

        Args:
            features (Dict[str, any]): Dictionnaire des variables d'entrée.

        Returns:
            Tuple[int, float, str]: Un tuple contenant :
                - prediction (int): 0 ou 1.
                - confidence (float): Score de probabilité (0.0 à 1.0).
                - class_name (str): Nom des classes
        """
        pass
