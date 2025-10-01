from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib


# --------- Public API ---------------------------------------------------------

# --- in load_artifacts(...) ---
def load_artifacts(project_root: Path | None = None):
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
    models_dir = root / "models"

    # read features + label_map from one JSON
    km_meta = _load_json(models_dir / "kmeans_features.json")

    artifacts = {
        "model": joblib.load(models_dir / "lightgbm_model.pkl"),
        "feature_columns": _load_json(models_dir / "feature_columns.json")["columns"],
        "model_freq_map": _load_json(models_dir / "model_freq_map.json"),
        "kmeans": joblib.load(models_dir / "kmeans.pkl"),
        "kmeans_scaler": joblib.load(models_dir / "kmeans_scaler.pkl"),
        "kmeans_features": km_meta.get("features", []),
        "kmeans_label_map": km_meta.get("label_map", {}),   # <— add this
    }
    return artifacts




def predict_price(input_data: Dict[str, Any], artifacts: Dict[str, Any]) -> float:
    """
    Predict price for a single vehicle dict using the trained LightGBM model.
    """
    X = _build_model_features(input_data, artifacts)
    pred = float(artifacts["model"].predict(X)[0])
    return pred


# --- replace assign_cluster(...) entirely ---
def assign_cluster(input_data: Dict[str, Any], artifacts: Dict[str, Any]) -> Tuple[int, str]:
    Xc = _build_cluster_features(input_data, artifacts)
    scaler = artifacts["kmeans_scaler"]
    km = artifacts["kmeans"]

    Xc_scaled = scaler.transform(Xc)
    label = int(km.predict(Xc_scaled)[0])

    # Prefer the saved human-friendly names
    names = artifacts.get("kmeans_label_map", {})
    # keys in json may be strings; try both
    tier = names.get(str(label)) or names.get(label) or f"Cluster {label}"
    return label, tier




def predict_and_cluster(input_data: Dict[str, Any], artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience wrapper that returns both price and cluster.
    """
    price = predict_price(input_data, artifacts)
    label, tier = assign_cluster(input_data, artifacts)
    return {"predicted_price": price, "cluster_label": label, "cluster_name": tier}


# --------- Internal helpers ---------------------------------------------------

def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _options_to_flags(options_str: Any, feature_columns: list[str]) -> Dict[str, int]:
    """
    Recreate OptionsCount and opt_* flags from a free-text options string.
    We infer which flags to build by scanning feature_columns for 'opt_'.
    """
    text = "" if options_str is None else str(options_str)
    tokens = [t.strip().lower() for t in text.split(",") if t.strip()]
    flags = {"OptionsCount": len(tokens)}

    # any opt_* that existed during training will be rebuilt here
    for col in feature_columns:
        if col.startswith("opt_"):
            # training column name uses underscores; convert back to spaced token
            look_for = col[4:].replace("_", " ")
            flags[col] = int(look_for in tokens)

    return flags


def _build_model_features(input_data: Dict[str, Any], artifacts: Dict[str, Any]) -> pd.DataFrame:
    """
    Turn a single input dict into the exact feature vector the model expects.
    Steps mirror your training code:
      - derive OptionsCount + opt_* flags
      - Model -> Model_freq using saved freq map
      - one-hot encode other categoricals with drop_first=True
      - align to saved feature_columns
    """
    feat_cols = artifacts["feature_columns"]
    model_freq = artifacts.get("model_freq_map", {})
    row = dict(input_data)  # copy

    # ---- Options engineering
    row.update(_options_to_flags(row.get("Options", ""), feat_cols))
    row.pop("Options", None)  # raw text was dropped during training

    # ---- Model frequency encoding
    model_val = str(row.get("Model", "") or "")
    row["Model_freq"] = float(model_freq.get(model_val, 0.0))
    row.pop("Model", None)  # ‘Model’ was dropped after freq encoding

    # Build a one-row DataFrame
    df = pd.DataFrame([row])

    # Make sure obvious numeric columns are numeric (ignore if missing)
    numeric_suspects = [
        "Year", "Mileage(km)", "EngineSize(L)", "Horsepower", "Torque",
        "Doors", "Seats", "FuelEfficiency(L/100km)", "OptionsCount",
        "PricePerKm", "Price($)"
    ]
    for c in numeric_suspects:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Identify categoricals (objects) except the removed 'Model'
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if "Model" in cat_cols:
        cat_cols.remove("Model")

    # One-hot encode (drop_first=True to match training)
    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Finally align to the training feature list
    X = df_enc.reindex(columns=feat_cols, fill_value=0)

    return X.astype(float)


def _build_cluster_features(input_data: Dict[str, Any], artifacts: Dict[str, Any]) -> pd.DataFrame:
    """
    Build the numeric feature set the KMeans scaler/model expect.
    Uses only the features listed in artifacts['kmeans_features'].
    """
    feats  = artifacts["kmeans_features"]          # ["Mileage(km)","Year","Horsepower","EngineSize(L)"]
    scaler = artifacts["kmeans_scaler"]

    row = {f: pd.to_numeric(input_data.get(f, np.nan), errors="coerce") for f in feats}
    Xc  = pd.DataFrame([row], columns=feats)

    # Impute any missing with the scaler's training means (keeps vector valid)
    means = pd.Series(scaler.mean_, index=feats)
    Xc = Xc.fillna(means)

    return Xc
