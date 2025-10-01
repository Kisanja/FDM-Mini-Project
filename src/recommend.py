# src/recommend.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.inference import load_artifacts, predict_price, _options_to_flags  # reuse what you already have


# =========================
# Public API
# =========================

DISPLAY_COLS = [
    "Brand","Model","Year","Condition","Mileage(km)","EngineSize(L)","FuelType",
    "Horsepower","Transmission","DriveType","BodyType","Doors","Seats","Color",
    "Interior","City","Price($)","OptionsCount"
]

def _format_recommendations(df: pd.DataFrame, predicted_price: float | None = None) -> pd.DataFrame:
    df = df.copy()

    # reverse log1p for columns that were log-transformed in preprocessing
    for col in ["Doors", "Horsepower"]:
        if col in df.columns:
            df[col] = np.expm1(df[col])  # back to original scale
            # tidy types
            if col == "Doors":
                df[col] = df[col].round(0).astype(int)
            else:
                df[col] = df[col].round(0).astype(int)

    # tidy other numerics
    if "EngineSize(L)" in df.columns:
        df["EngineSize(L)"] = df["EngineSize(L)"].round(1)
    if "Mileage(km)" in df.columns:
        df["Mileage(km)"] = df["Mileage(km)"].round(0).astype(int)
    if "Price($)" in df.columns:
        df["Price($)"] = df["Price($)"].round(0).astype(int)
    if "similarity" in df.columns:
        df["similarity"] = df["similarity"].round(3)

    # nice column order
    order = ["rec_id","Brand","Model","Year","Condition","Price($)","Mileage(km)","EngineSize(L)",
             "Horsepower","Doors","Seats","FuelType","Transmission","DriveType","BodyType",
             "Color","Interior","City","similarity"]
    df = df[[c for c in order if c in df.columns]]

    if predicted_price is not None:
        df.insert(1, "PredPriceUser", round(predicted_price, 0))

    return df


def load_catalog(project_root: Path) -> pd.DataFrame:
    """
    Load the recommendation catalog. For now we use processed TRAIN as 'inventory'.
    """
    path = Path(project_root) / "data" / "processed" / "train.csv"
    df = pd.read_csv(path)
    # lightweight id for reference in UI
    if "rec_id" not in df.columns:
        df = df.reset_index(drop=True).copy()
        df.insert(0, "rec_id", np.arange(len(df)))
    return df


def budget_recommendations(
    user_input: Dict[str, Any],
    catalog: pd.DataFrame,
    artifacts: Dict[str, Any],
    pct: float = 0.15,
    top_n: int = 10,
    extra_filters: Dict[str, Any] | None = None
) -> pd.DataFrame:
    pred_price = predict_price(user_input, artifacts)
    low, high = pred_price * (1 - pct), pred_price * (1 + pct)

    pool = catalog[(catalog["Price($)"] >= low) & (catalog["Price($)"] <= high)].copy()
    if extra_filters:
        for col, val in extra_filters.items():
            if col in pool.columns and val not in (None, ""):
                pool = pool[pool[col] == val]
    if pool.empty:
        pool = catalog.copy()

    X_user = _encode_one(user_input, artifacts)
    X_pool = _encode_many(pool, artifacts)

    sims = cosine_similarity(X_user, X_pool).ravel()
    pool = pool.assign(similarity=sims).sort_values("similarity", ascending=False)

    cols = ["rec_id", "similarity"] + [c for c in DISPLAY_COLS if c in pool.columns]
    out = pool[cols].head(top_n).reset_index(drop=True)
    out.insert(1, "predicted_price_user", pred_price)

    return _format_recommendations(out, predicted_price=pred_price)



def similar_items(
    user_input: Dict[str, Any],
    catalog: pd.DataFrame,
    artifacts: Dict[str, Any],
    top_n: int = 10,
    extra_filters: Dict[str, Any] | None = None
) -> pd.DataFrame:
    pool = catalog.copy()
    if extra_filters:
        for col, val in extra_filters.items():
            if col in pool.columns and val not in (None, ""):
                pool = pool[pool[col] == val]
    if pool.empty:
        pool = catalog.copy()

    X_user = _encode_one(user_input, artifacts)
    X_pool = _encode_many(pool, artifacts)

    sims = cosine_similarity(X_user, X_pool).ravel()
    pool = pool.assign(similarity=sims).sort_values("similarity", ascending=False)

    cols = ["rec_id", "similarity"] + [c for c in DISPLAY_COLS if c in pool.columns]
    out = pool[cols].head(top_n).reset_index(drop=True)

    return _format_recommendations(out)



# =========================
# Internal helpers
# =========================

def _encode_one(row: Dict[str, Any], artifacts: Dict[str, Any]) -> np.ndarray:
    """
    Build one feature vector aligned to artifacts['feature_columns'].
    Vectorized clone of your training logic; mirrors inference._build_model_features
    but returns a numpy array for similarity.
    """
    feat_cols = artifacts["feature_columns"]
    model_freq = artifacts.get("model_freq_map", {})
    r = dict(row)

    # options engineering
    r.update(_options_to_flags(r.get("Options", ""), feat_cols))
    r.pop("Options", None)

    # model -> Model_freq
    r["Model_freq"] = float(model_freq.get(str(r.get("Model","")), 0.0))
    r.pop("Model", None)

    df = pd.DataFrame([r])

    # numeric coercion (safe)
    numeric_suspects = [
        "Year", "Mileage(km)", "EngineSize(L)", "Horsepower", "Torque",
        "Doors", "Seats", "FuelEfficiency(L/100km)", "OptionsCount",
        "PricePerKm", "Price($)"
    ]
    for c in numeric_suspects:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # one-hot encode remaining categoricals (drop_first=True to match training)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if "Model" in cat_cols:
        cat_cols.remove("Model")
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # align to training features
    X = df.reindex(columns=feat_cols, fill_value=0.0).astype(float).to_numpy()
    return X


def _encode_many(df: pd.DataFrame, artifacts: Dict[str, Any]) -> np.ndarray:
    """
    Vectorized encoding for a whole DataFrame, consistent with training.
    """
    feat_cols = artifacts["feature_columns"]
    model_freq = artifacts.get("model_freq_map", {})

    work = df.copy()

    # ensure OptionsCount exists (you already saved it in preprocessing; if not, rebuild)
    if "OptionsCount" not in work.columns:
        work["Options"] = work["Options"].fillna("").astype(str)
        work["OptionsCount"] = work["Options"].apply(lambda s: len([t for t in s.split(",") if t.strip()]))

    # model -> Model_freq then drop raw 'Model'
    work["Model_freq"] = work["Model"].astype(str).map(model_freq).fillna(0.0)
    work = work.drop(columns=["Model"], errors="ignore")

    # numeric coercion
    numeric_suspects = [
        "Year", "Mileage(km)", "EngineSize(L)", "Horsepower", "Torque",
        "Doors", "Seats", "FuelEfficiency(L/100km)", "OptionsCount",
        "PricePerKm", "Price($)"
    ]
    for c in numeric_suspects:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    # one-hot encode remaining categoricals (drop_first=True)
    cat_cols = work.select_dtypes(include="object").columns.tolist()
    if "Model" in cat_cols:
        cat_cols.remove("Model")
    work = pd.get_dummies(work, columns=cat_cols, drop_first=True)

    # align to model features
    X = work.reindex(columns=feat_cols, fill_value=0.0).astype(float).to_numpy()
    return X


# =========================
# Simple recommendation modes (add these)
# =========================

def recommend_by_budget_only(budget_usd: float,
                             catalog: pd.DataFrame,
                             pct: float = 0.15,
                             top_n: int = 10,
                             filters: dict | None = None) -> pd.DataFrame:
    low, high = budget_usd * (1 - pct), budget_usd * (1 + pct)
    pool = catalog[(catalog["Price($)"] >= low) & (catalog["Price($)"] <= high)].copy()
    if filters:
        for k, v in filters.items():
            if k in pool.columns and v not in (None, ""):
                pool = pool[pool[k] == v]
    if pool.empty:
        pool = catalog.copy()

    # simple rank by closeness to budget
    pool["similarity"] = 1.0 - (pool["Price($)"] - budget_usd).abs() / (budget_usd + 1e-9)
    out = pool.sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)
    # ✅ format before returning
    return _format_recommendations(out, predicted_price=budget_usd)


def recommend_minimal(min_input: dict,
                      catalog: pd.DataFrame,
                      artifacts: dict,
                      top_n: int = 10,
                      filters: dict | None = None) -> pd.DataFrame:
    # build a simple similarity using the four clustering numerics
    feats = [f for f in ["Mileage(km)", "Year", "Horsepower", "EngineSize(L)"] if f in catalog.columns]
    if filters:
        pool = catalog.copy()
        for k, v in filters.items():
            if k in pool.columns and v not in (None, ""):
                pool = pool[pool[k] == v]
        if pool.empty:
            pool = catalog.copy()
    else:
        pool = catalog.copy()

    user_vec = pd.DataFrame([{f: pd.to_numeric(min_input.get(f, np.nan), errors="coerce") for f in feats}]).fillna(pool[feats].median(numeric_only=True))
    # cosine similarity in that 4D space
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(user_vec.to_numpy(), pool[feats].to_numpy()).ravel()
    pool = pool.assign(similarity=sims).sort_values("similarity", ascending=False)
    out = pool.head(top_n).reset_index(drop=True)
    # ✅ format before returning
    return _format_recommendations(out)

