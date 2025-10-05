# src/inference.py
from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib


# =========================
# Public API
# =========================

def load_artifacts(project_root: Path | None = None) -> Dict[str, Any]:
    """
    Load all persisted artifacts needed for prediction & clustering.
    Files expected (graceful if some are missing):
      models/
        - lightgbm_model.pkl
        - feature_columns.json          -> {"columns": [...]}
        - model_freq_map.json           -> {"model_name": freq, ...}
        - kmeans.pkl
        - kmeans_scaler.pkl
        - kmeans_features.json          -> {"features":[...], "label_map":{...}}  (optional label_map)
        - price_bins.json               -> {"q33": float, "q66": float} (OPTION A naming)
    """
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
    models_dir = root / "models"

    km_meta = _load_json(models_dir / "kmeans_features.json")
    price_bins = _load_json(models_dir / "price_bins.json")  # may be {}

    artifacts: Dict[str, Any] = {
        "model": _safe_joblib(models_dir / "lightgbm_model.pkl"),
        "feature_columns": _load_json(models_dir / "feature_columns.json").get("columns", []),
        "model_freq_map": _load_json(models_dir / "model_freq_map.json"),  # may be {}
        "kmeans": _safe_joblib(models_dir / "kmeans.pkl"),
        "kmeans_scaler": _safe_joblib(models_dir / "kmeans_scaler.pkl"),
        "kmeans_features": km_meta.get("features", []),
        "kmeans_label_map": km_meta.get("label_map", {}),  # string or int keys are both fine
        "price_bins": price_bins,  # {"q33": ..., "q66": ...} if present
    }
    return artifacts


def predict_price(input_data: Dict[str, Any], artifacts: Dict[str, Any]) -> float:
    """
    Predict price for a single vehicle dict using the trained LightGBM model.
    """
    _ensure_model(artifacts["model"], need="LightGBM model")
    X = _build_model_features(input_data, artifacts)
    pred = float(artifacts["model"].predict(X)[0])
    return pred


def assign_cluster(input_data: Dict[str, Any],
                   artifacts: Dict[str, Any],
                   price_for_naming: float | None = None) -> Tuple[int, str]:
    """
    Compute the KMeans cluster label. Name the human-friendly segment by:
      - If price bins are available: map the PREDICTED price to Budget/Mid-range/Luxury (Option A)
      - Else: fall back to any saved kmeans_label_map
    Returns (numeric_label, human_name).
    """
    _ensure_model(artifacts["kmeans"], need="KMeans")
    _ensure_model(artifacts["kmeans_scaler"], need="KMeans scaler")

    Xc = _build_cluster_features(input_data, artifacts)
    Xc_scaled = artifacts["kmeans_scaler"].transform(Xc)
    label = int(artifacts["kmeans"].predict(Xc_scaled)[0])

    # Prefer price-band naming for human segment (Option A)
    if price_for_naming is not None and artifacts.get("price_bins"):
        tier = _segment_name_from_price(price_for_naming, artifacts["price_bins"])
    else:
        # fallback to label map saved with kmeans (if any)
        names = artifacts.get("kmeans_label_map", {})
        tier = names.get(str(label)) or names.get(label) or f"Cluster {label}"

    return label, tier


def predict_and_cluster(input_data: Dict[str, Any], artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience wrapper that returns both price and cluster.
    - Predict price with LightGBM
    - Get KMeans numeric label
    - Name the segment using price bands if present; else fallback to kmeans label_map
    """
    price = predict_price(input_data, artifacts)
    label, tier = assign_cluster(input_data, artifacts, price_for_naming=price)
    return {"predicted_price": price, "cluster_label": label, "cluster_name": tier}


# =========================
# Internal helpers
# =========================

def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_joblib(path: Path):
    """Return joblib.load(path) or None if file missing."""
    if path.exists():
        return joblib.load(path)
    return None


def _ensure_model(obj, need: str):
    if obj is None:
        raise RuntimeError(f"{need} not found in artifacts. Make sure it is saved in models/.")


def _segment_name_from_price(price: float, bins: Dict[str, float]) -> str:
    """
    Map a price to Budget / Mid-range / Luxury using precomputed quantile bins.
    Expected: bins = {"q33": float, "q66": float}
    """
    try:
        q33 = float(bins["q33"])
        q66 = float(bins["q66"])
    except Exception:
        # bins malformed → fallback generic
        return "Unknown"
    if price < q33:
        return "Budget"
    elif price < q66:
        return "Mid-range"
    else:
        return "Luxury"


def _build_model_features(input_data: Dict[str, Any], artifacts: Dict[str, Any]) -> pd.DataFrame:
    """
    Build the exact model feature vector:
      - Model -> Model_freq using saved freq map
      - One-hot encode other categoricals with drop_first=True
      - Align to saved feature_columns
    NOTE: This version is aligned to the FINAL 14 features (no Options/Doors/Seats/etc.).
    """
    feat_cols: list[str] = artifacts.get("feature_columns", [])
    model_freq = artifacts.get("model_freq_map", {})
    row = dict(input_data)  # copy

    # --- Model frequency encoding (keep this!)
    model_val = str(row.get("Model", "") or "")
    row["Model_freq"] = float(model_freq.get(model_val, 0.0))
    row.pop("Model", None)

    # Build a one-row DataFrame
    df = pd.DataFrame([row])

    # Coerce obvious numeric columns (ignore if missing)
    numeric_candidates = [
        "Year", "Mileage(km)", "EngineSize(L)", "Horsepower", "Torque",
        "FuelEfficiency(L/100km)", "Price($)", "Model_freq"
    ]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Identify remaining categoricals
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    # One-hot encode (drop_first=True to match training)
    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Align to the training feature list
    if not feat_cols:
        raise RuntimeError("feature_columns list is missing; cannot align features.")
    X = df_enc.reindex(columns=feat_cols, fill_value=0.0)

    return X.astype(float)


def _build_cluster_features(input_data: Dict[str, Any], artifacts: Dict[str, Any]) -> pd.DataFrame:
    """
    Build the numeric feature set the KMeans scaler/model expect.
    (Typically: ["Mileage(km)", "Year", "Horsepower", "EngineSize(L)"])
    """
    feats: list[str] = artifacts.get("kmeans_features", [])
    if not feats:
        raise RuntimeError("kmeans_features are missing; cannot build cluster features.")

    row = {f: pd.to_numeric(input_data.get(f, np.nan), errors="coerce") for f in feats}
    df = pd.DataFrame([row], columns=feats)

    # Simple safe fill: if any required field missing, fill with scaler means if available,
    # otherwise with the row medians.
    scaler = artifacts.get("kmeans_scaler")
    if scaler is not None and getattr(scaler, "mean_", None) is not None:
        means = pd.Series(scaler.mean_, index=feats)
        df = df.fillna(means)
    else:
        df = df.fillna(df.median(numeric_only=True))

    return df
