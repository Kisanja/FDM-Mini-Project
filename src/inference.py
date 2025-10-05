# src/inference.py
from __future__ import annotations

"""
Inference/Clustering utilities.

Key change for Streamlit Cloud reliability:
- We *attempt* to import joblib but fall back to pickle automatically.
  This prevents ModuleNotFoundError at import-time on environments that
  don't have joblib wheels for the selected Python version.
- All model loads go through a single _load_pickle() helper.

Public API preserved:
  - load_artifacts(...)
  - predict_price(...)
  - assign_cluster(...)
  - predict_and_cluster(...)
  - _build_model_features(...)        # used by your pages (debug/visibility)
  - _build_cluster_features(...)      # used by your pages
"""

from pathlib import Path
from typing import Any, Dict, Tuple
import json

import numpy as np
import pandas as pd

# -------------------------
# Safe (deferred) pickle/joblib loader
# -------------------------
# Prefer joblib (faster, handles memmaps), but never fail import-time if it's missing.
# If joblib isn't available, fall back to stdlib pickle.
try:
    import joblib as _joblib  # type: ignore
    _HAVE_JOBLIB = True

    def _load_pickle(path: Path) -> Any:
        # joblib can read both .pkl and .joblib created by joblib.dump
        return _joblib.load(path)

except Exception:
    import pickle as _pickle  # type: ignore
    _HAVE_JOBLIB = False

    def _load_pickle(path: Path) -> Any:
        # Works for most sklearn artifacts saved via joblib (they're pickled under the hood).
        # If an artifact truly requires joblib (e.g., memmap arrays), this will raise;
        # we catch and re-raise with a clear message where needed.
        with open(path, "rb") as f:
            return _pickle.load(f)


# -------------------------
# Paths & helpers
# -------------------------

def _repo_root() -> Path:
    # src/  -> repo root
    return Path(__file__).resolve().parents[1]


def _find_models_dir(project_root: Path) -> Path:
    """
    Prefer app/models (your project layout), fall back to models/.
    """
    candidates = [project_root / "app" / "models", project_root / "models"]
    for c in candidates:
        if c.exists():
            return c
    # If nothing exists, still return the primary location so the error message is clear
    return candidates[0]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_load_artifact(path: Path) -> Any | None:
    """
    Load an artifact if the file exists; return None if missing.
    Raises a clear error if loading fails due to incompatible serialization.
    """
    if not path.exists():
        return None
    try:
        return _load_pickle(path)
    except Exception as e:
        # Provide an actionable error message with context.
        hint = (
            "This artifact could not be loaded. If it was saved using joblib with memmapped arrays, "
            "ensure 'joblib' is installed on the deployment (or re-save as a plain pickle)."
        )
        raise RuntimeError(f"Failed to load artifact: {path.name}. {_loader_summary()} | {hint}") from e


def _ensure_present(obj, need: str):
    if obj is None:
        raise RuntimeError(f"{need} not found. Please ensure it is saved in app/models/.")


def _loader_summary() -> str:
    return f"joblib_available={_HAVE_JOBLIB}"


# -------------------------
# Public API
# -------------------------

def load_artifacts(project_root: Path | None = None) -> Dict[str, Any]:
    """
    Load all persisted artifacts needed for prediction & clustering.

    Expected (under app/models/):
        - lightgbm_model.pkl        (required)
        - feature_columns.json      -> {"columns": [...] }     (required)
        - model_freq_map.json       -> {"model": count, ...}   (optional but recommended)
        - kmeans.pkl                (required for clustering pages)
        - kmeans_scaler.pkl         (required for clustering pages)
        - kmeans_features.json      -> {"features":[...], "label_map":{...}} (optional label_map)
        - price_bins.json           -> {"q33": float, "q66": float} (optional; for segment naming)
    """
    root = Path(project_root) if project_root else _repo_root()
    models_dir = _find_models_dir(root)

    # Core
    model = _safe_load_artifact(models_dir / "lightgbm_model.pkl")
    feature_meta = _read_json(models_dir / "feature_columns.json")
    feature_columns = feature_meta.get("columns", [])

    # Optional
    model_freq_map = _read_json(models_dir / "model_freq_map.json")
    kmeans = _safe_load_artifact(models_dir / "kmeans.pkl")
    kmeans_scaler = _safe_load_artifact(models_dir / "kmeans_scaler.pkl")
    km_meta = _read_json(models_dir / "kmeans_features.json")
    kmeans_features = km_meta.get("features", [])
    kmeans_label_map = km_meta.get("label_map", {})
    price_bins = _read_json(models_dir / "price_bins.json")

    # Validate required bits for prediction
    _ensure_present(model, "LightGBM model (lightgbm_model.pkl)")
    if not feature_columns:
        raise RuntimeError("feature_columns.json missing or malformed (no 'columns').")

    return {
        "model": model,
        "feature_columns": feature_columns,
        "model_freq_map": model_freq_map,
        "kmeans": kmeans,
        "kmeans_scaler": kmeans_scaler,
        "kmeans_features": kmeans_features,
        "kmeans_label_map": kmeans_label_map,
        "price_bins": price_bins,
        "models_dir": str(models_dir),            # helpful for debugging
        "joblib_available": _HAVE_JOBLIB,         # surface loader state for UI/debug
    }


def predict_price(input_data: Dict[str, Any], artifacts: Dict[str, Any]) -> float:
    """
    Predict price for a single vehicle dict using the trained LightGBM model.
    """
    _ensure_present(artifacts.get("model"), "LightGBM model")
    X = _build_model_features(input_data, artifacts)
    return float(artifacts["model"].predict(X)[0])


def assign_cluster(
    input_data: Dict[str, Any],
    artifacts: Dict[str, Any],
    price_for_naming: float | None = None,
) -> Tuple[int, str]:
    """
    Compute KMeans label and return (numeric_label, human_name).
    If price bins exist and price_for_naming is provided, use price bands
    (Budget/Mid-range/Luxury). Otherwise fall back to kmeans label_map.
    """
    _ensure_present(artifacts.get("kmeans"), "KMeans model (kmeans.pkl)")
    _ensure_present(artifacts.get("kmeans_scaler"), "KMeans scaler (kmeans_scaler.pkl)")

    Xc = _build_cluster_features(input_data, artifacts)
    Xc_scaled = artifacts["kmeans_scaler"].transform(Xc)
    label = int(artifacts["kmeans"].predict(Xc_scaled)[0])

    # Prefer price-band naming (if we have bins and a price)
    bins = artifacts.get("price_bins") or {}
    if price_for_naming is not None and "q33" in bins and "q66" in bins:
        name = _segment_name_from_price(price_for_naming, bins)
    else:
        names = artifacts.get("kmeans_label_map", {}) or {}
        name = names.get(str(label)) or names.get(label) or f"Cluster {label}"

    return label, name


def predict_and_cluster(input_data: Dict[str, Any], artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience wrapper: predict price, then assign a cluster and human-readable segment.
    """
    price = predict_price(input_data, artifacts)
    label, name = assign_cluster(input_data, artifacts, price_for_naming=price)
    return {"predicted_price": price, "cluster_label": label, "cluster_name": name}


# -------------------------
# Internal feature builders
# -------------------------

def _build_model_features(input_data: Dict[str, Any], artifacts: Dict[str, Any]) -> pd.DataFrame:
    """
    Build the exact model feature vector:
      - Encode 'Model' as 'Model_freq' using the saved frequency map
      - One-hot encode remaining categoricals (drop_first=True)
      - Align to saved feature_columns
    """
    feat_cols: list[str] = artifacts.get("feature_columns", [])
    model_freq_map = artifacts.get("model_freq_map", {}) or {}

    row = dict(input_data)  # copy

    # --- Model frequency encoding (keep this!)
    model_val = str(row.get("Model", "") or "")
    row["Model_freq"] = float(model_freq_map.get(model_val, 0.0))
    row.pop("Model", None)

    df = pd.DataFrame([row])

    # Coerce obvious numeric columns (ignore safely if missing)
    numeric_candidates = [
        "Year",
        "Mileage(km)",
        "EngineSize(L)",
        "Horsepower",
        "Torque",
        "FuelEfficiency(L/100km)",
        "Price($)",
        "Model_freq",
    ]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # One-hot encode remaining categoricals
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    if not feat_cols:
        raise RuntimeError("feature_columns list is missing; cannot align features.")

    X = df_enc.reindex(columns=feat_cols, fill_value=0.0)
    return X.astype(float)


def _build_cluster_features(input_data: Dict[str, Any], artifacts: Dict[str, Any]) -> pd.DataFrame:
    """
    Build the numeric features expected by the KMeans/scaler.
    Typically: ["Mileage(km)", "Year", "Horsepower", "EngineSize(L)"].
    """
    feats: list[str] = artifacts.get("kmeans_features", []) or []
    if not feats:
        raise RuntimeError("kmeans_features are missing; cannot build cluster features.")

    row = {f: pd.to_numeric(input_data.get(f, np.nan), errors="coerce") for f in feats}
    df = pd.DataFrame([row], columns=feats)

    scaler = artifacts.get("kmeans_scaler")
    if scaler is not None and getattr(scaler, "mean_", None) is not None:
        means = pd.Series(scaler.mean_, index=feats)
        df = df.fillna(means)
    else:
        df = df.fillna(df.median(numeric_only=True))

    return df


def _segment_name_from_price(price: float, bins: Dict[str, Any]) -> str:
    """
    Map price into Budget / Mid-range / Luxury using precomputed quantiles:
    bins = {"q33": float, "q66": float}
    """
    try:
        q33 = float(bins["q33"])
        q66 = float(bins["q66"])
    except Exception:
        return "Unknown"

    if price < q33:
        return "Budget"
    elif price < q66:
        return "Mid-range"
    else:
        return "Luxury"
