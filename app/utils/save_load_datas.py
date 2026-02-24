# imports
import json
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, Union

import joblib
import pandas as pd
import yaml

# ===========================================================================


def save_datas(
    data: Any,
    folder_path: Path,
    subs: str = "",
    filename: str = "",
    format: Literal["parquet", "joblib", "json", "csv", "yml", "yaml"] = "csv",
) -> Optional[Path]:
    """
    Eporte la dataframe vers le chemin spécifié

    Args:
        data(Any): donnée a sauvegarder
        folder_path(Path): chemin du dossier a destination
        subs(str): sous-dossier
        filename(str): nom du fichier
        format(Literal["parquet","joblib","json","csv","yml","yaml"]):
            format de sauvegarde. par défaut: csv

    Returns:
        path(Path): chemin
    """
    path = Path(folder_path / subs)

    # Création du dossier si inexistant
    path.mkdir(parents=True, exist_ok=True)

    # On définit l'extension du fichier a sauvegarder
    actual_extension = format if not format.startswith(".") else format[1:]
    filename_path = path / f"{filename}.{actual_extension}"

    try:
        if format == "parquet":
            data.to_parquet(filename_path, index=False)
        elif format == "joblib":
            joblib.dump(data, filename_path)
        elif format == "json":
            if isinstance(data, pd.DataFrame):
                data.to_json(filename_path, orient="records", indent=4)
            else:
                with open(filename_path, "w") as f:
                    json.dump(data, f, indent=4)
        elif format == "csv":
            data.to_csv(filename_path, index=False)
        elif format == "yaml" or format == "yml":
            with open(filename_path, "w") as f:
                yaml.dump(data, f, indent=4)
        else:
            return None
    except Exception as e:
        print(f"Erreur, sauvegarde avortée: {e}")
        return None
    return path


# ===========================================================================


def load_datas(
    file_path: Union[Path, str],
) -> Tuple[Any, str]:
    """
    Importe la donnée que ce soit sous format csv, parquet, joblib, json ou yaml/yml

    Args:
        file_path(Union[Path,str]): donnée a chargé

    Returns:
        data,suffix(Tuple[Any,str]): Fichier chargé et son format
    """

    path = Path(file_path)  # On s'assure d'avoir un format Path
    suffix = path.suffix

    if suffix == ".csv":
        data = pd.read_csv(path)
    elif suffix == ".parquet":
        data = pd.read_parquet(path)
    elif suffix == ".joblib":  # or suffix == '.pkl': # On évite suffixe pour la sécurité
        data = joblib.load(path)
    elif suffix == ".json":
        try:
            data = pd.read_json(path)
        except Exception:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
    elif suffix == ".yaml" or suffix == ".yml":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(f"Format de fichier non supporté : {suffix}")
    return data, suffix
