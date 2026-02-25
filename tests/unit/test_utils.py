import json

import numpy as np
import pandas as pd
import pytest

from app.utils.clean_for_json_db import clean_for_json
from app.utils.features_type_list import features_type
from app.utils.hash_id import generate_feature_hash
from app.utils.save_load_datas import load_datas, save_datas

# ==================== FEATURES_TYPE_LIST ========================================


# Verifie si les features sont correctement splitté entre num et cat_list
@pytest.mark.unit
def test_features_type_mixed_columns():
    """Vérifie la séparation correcte entre colonnes numériques et catégorielles."""
    df = pd.DataFrame(
        {
            "age": [25, 30],  # int64
            "score": [1.5, 2.2],  # float64
            "city": ["Paris", "Lyon"],  # object
            "is_client": [True, False],  # bool (exclu par np.number généralement)
        }
    )

    num, cat = features_type(df)

    assert "age" in num
    assert "score" in num
    assert "city" in cat
    assert "is_client" in cat  # Les booléens sont catégoriels ici
    assert len(num) == 2
    assert len(cat) == 2


# =========================================================================


# Vérifie si la fonction reste stable même si cat_list vide (ou num)
@pytest.mark.unit
def test_features_type_only_numeric():
    """Vérifie le comportement avec 100% de colonnes numériques."""
    df = pd.DataFrame({"f1": [1, 2], "f2": [3.5, 4.5]})

    num, cat = features_type(df)

    assert len(num) == 2
    assert len(cat) == 0


# =================================================================


# Vérifie ausi la stabilité ais avec un df vide
@pytest.mark.unit
def test_features_type_empty_df():
    """Vérifie que la fonction ne crash pas avec un DataFrame vide."""
    df = pd.DataFrame()
    num, cat = features_type(df)

    assert num == []
    assert cat == []


# ===========================================================================


# Vérifie le type spéciifique
@pytest.mark.unit
@pytest.mark.parametrize(
    "dtype, expected_in_num",
    [(np.int64, True), (np.float64, True), (object, False), ("category", False)],
)
def test_features_type_specific_dtypes(dtype, expected_in_num):
    """Teste la détection précise de types spécifiques via paramétrage."""
    df = pd.DataFrame({"col": [1, 2]}).astype({"col": dtype})
    num, cat = features_type(df)

    if expected_in_num:
        assert "col" in num
    else:
        assert "col" in cat


# ========================== HASH_ID =======================================


# Vérifier le determinisme du hashing
@pytest.mark.unit
def test_hash_determinism():
    """Vérifie que le même dictionnaire produit toujours le même hash."""
    features = {"A": 1, "B": 2}
    hash1 = generate_feature_hash(features)
    hash2 = generate_feature_hash(features)

    assert hash1 == hash2
    assert len(hash1) == 64


# ================================================================


# Vériie que l'ordre importe peu pour le hashing
@pytest.mark.unit
def test_hash_order_insensitivity():
    """
    Vérifie que l'ordre des clés dans le dictionnaire n'influence pas le hash.
    C'est la propriété 'sort_keys=True' qui est testée ici.
    """
    features_1 = {"A": 1, "B": 2}
    features_2 = {"B": 2, "A": 1}

    assert generate_feature_hash(features_1) == generate_feature_hash(features_2)


# ================================================================


# Vrifie l'unicité des hash
@pytest.mark.unit
def test_hash_sensitivity():
    """Vérifie qu'une modification minime produit un hash totalement différent."""
    features_1 = {"A": 1, "B": 2}
    features_2 = {"A": 1, "B": 2.000000000001}

    assert generate_feature_hash(features_1) != generate_feature_hash(features_2)


# ===============================================================


# Vérifie la stabilité de la fonction
@pytest.mark.unit
def test_hash_empty_dict():
    """Vérifie que la fonction gère un dictionnaire vide sans crash."""
    h = generate_feature_hash({})
    assert isinstance(h, str)
    assert len(h) == 64


# ========================= INPUT_PREPROC =========================================


# Verifie que la tf en df sur list fonctionne
@pytest.mark.unit
def test_process_input_as_list(input_preproc_mocked):
    """
    Cible la branche : elif isinstance(data, list)
    Vérifie que le traitement par lot (batch) fonctionne.
    """
    fake_data = [
        {"EXT_SOURCE_COUNT": 1.56, "OWN_CAR_AGE": 10.0},
        {"EXT_SOURCE_COUNT": 2.31, "OWN_CAR_AGE": 60.5},
    ]

    df_result = input_preproc_mocked.process(fake_data)

    assert isinstance(df_result, pd.DataFrame)
    assert len(df_result) == 2
    assert list(df_result.columns) == input_preproc_mocked.feature_names


# ==============================================================


# cas data = pd.DataFrame
@pytest.mark.unit
def test_process_input_as_dataframe(input_preproc_mocked):
    """
    Cible la branche : else (df.copy())
    Vérifie que si on passe déjà un DataFrame, il est copié et nettoyé.
    """
    fake_input_df = pd.DataFrame({"EXT_SOURCE_COUNT": [1.56, 2.31], "OWN_CAR_AGE": [10.0, 60.5]})

    df_result = input_preproc_mocked.process(fake_input_df)

    assert df_result is not fake_input_df  # Vérifie que c'est une copie (ID différent)
    assert len(df_result) == 2
    assert "NAME_FAMILY_STATUS" in df_result.columns  # Vérifie le reindex


# ==========================================================================


# Vérifie l'application des valeurs par défaut: num = 365243.0 et cat = "Unknown"
@pytest.mark.unit
def test_process_numeric_and_categorical_filling(input_preproc_mocked):
    """
    Vérifie que les valeurs par défaut sont correctement appliquées
    selon la nature de la colonne.
    """
    # On envoie un dictionnaire vide pour forcer les NaNs partout
    fake_data = {}
    df_result = input_preproc_mocked.process(fake_data)

    # Assertions
    # Vérification numérique (AMT_ANNUITY n'est pas dans cat_cols_model)
    assert df_result["AMT_ANNUITY"].iloc[0] == 365243.0
    # Vérification catégorielle (NAME_FAMILY_STATUS est dans cat_cols_model)
    assert df_result["NAME_FAMILY_STATUS"].iloc[0] == "Unknown"
    assert isinstance(df_result["NAME_FAMILY_STATUS"].iloc[0], str)


# ====================================================================


# Vérifie la correction des données suivant leur typage
@pytest.mark.unit
def test_process_invalid_numeric_coercion(input_preproc_mocked):
    """Vérifie que le texte dans une colonne numérique est 'coerced' en 365243.0."""
    data = {"DAYS_BIRTH": "35ans"}
    df_result = input_preproc_mocked.process(data)

    assert df_result["DAYS_BIRTH"].iloc[0] == 365243.0


# ======================== SAVE_LOAD_DATAS =======================================


# Vérifie la sauvegarde et le chargement d'un df
@pytest.mark.unit
@pytest.mark.parametrize("fmt", ["csv", "parquet", "joblib", "json"])
def test_save_load_dataframe_formats(tmp_path, fake_df, fmt):
    """Teste le cycle complet sauvegarde/chargement pour un DataFrame."""
    filename = "test_file"

    # Sauvegarde
    saved_path = save_datas(fake_df, tmp_path, filename=filename, format=fmt)

    # Chargement
    # On reconstruit le chemin complet avec l'extension
    full_path = tmp_path / f"{filename}.{fmt}"
    loaded_data, suffix = load_datas(full_path)

    # Assertions
    assert saved_path is not None  # sauvegarde
    assert suffix == f".{fmt}"
    # check_**=False permet de passer les probleme de typage, precision... post chargement
    pd.testing.assert_frame_equal(
        loaded_data, fake_df, check_dtype=False, check_exact=False, check_index_type=False
    )


# =========================================================================


# save et load aevc un dictionnaire
@pytest.mark.unit
@pytest.mark.parametrize("fmt", ["json", "yaml", "yml"])
def test_save_load_dict_formats(tmp_path, fake_dict, fmt):
    """Teste le cycle complet pour des données de type dictionnaire."""
    filename = "test_dict"

    save_datas(fake_dict, tmp_path, filename=filename, format=fmt)

    full_path = tmp_path / f"{filename}.{fmt}"
    loaded_data, suffix = load_datas(full_path)

    assert loaded_data == fake_dict
    assert suffix == f".{fmt}"


# =============================================================


# Vérifie qu'un format non reconnu renvoie none
@pytest.mark.unit
def test_save_datas_invalid_format(tmp_path, fake_df):
    """Vérifie que None est retourné pour un format non supporté."""
    # pas besoin de pytest.raises car la fonction gère l'erreur en interne
    result = save_datas(fake_df, tmp_path, filename="fail", format="txt")  # type:ignore
    assert result is None


# =================================================================


# Verifie le bloc Exception quand la data n'est pas correct
@pytest.mark.unit
def test_save_datas_exception_handling(tmp_path):
    """Force une exception pour tester le bloc try/except (ex: data n'est pas une DF)."""
    # Essayer de sauver une string comme un parquet (to_parquet n'existe pas sur str)
    result = save_datas("pas_un_df", tmp_path, filename="error", format="parquet")
    assert result is None


# ==================================================================


# suffix inconnue pour l'import
@pytest.mark.unit
def test_load_datas_unsupported_suffix(tmp_path):
    """Vérifie qu'une ValueError est levée pour une extension inconnue."""
    unsupported_file = tmp_path / "data.txt"
    unsupported_file.write_text("hello")

    with pytest.raises(ValueError, match="Format de fichier non supporté"):
        load_datas(unsupported_file)


# =========================================================================


# Vérifie le fallback ur le chargement json
@pytest.mark.unit
def test_load_json_fallback_to_dict(tmp_path, fake_dict):
    """
    Teste spécifiquement le bloc try/except de load_datas pour le JSON.
    Force le passage vers json.load si pd.read_json échoue.
    """
    path = tmp_path / "simple_dict.json"
    with open(path, "w") as f:
        json.dump(fake_dict, f)

    data, _ = load_datas(path)
    assert data == fake_dict


# =========================== CLEAN_FOR_JSON_DB ======================================


# Verifie que seul les NaN, inf... sont converti en None
def test_clean_for_json_simple_nan():
    data = {"a": 1, "b": np.nan, "c": float("inf")}
    cleaned = clean_for_json(data)
    assert cleaned["a"] == 1
    assert cleaned["b"] is None
    assert cleaned["c"] is None


# =========================================================================


# Converti les numpy en python standard
def test_clean_for_json_numpy_types():
    data = {"int": np.int64(42), "float": np.float32(10.5)}
    cleaned = clean_for_json(data)
    assert isinstance(cleaned["int"], int)
    assert isinstance(cleaned["float"], float)
    assert cleaned["int"] == 42
