# src/recommend.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .inference import load_artifacts, predict_price

# =========================
# Public API
# =========================

# Only show columns we keep in the 14-feature setup + Price for display
DISPLAY_COLS = [
    "Brand", "Model", "Year", "Condition",
    "Mileage(km)", "EngineSize(L)", "FuelType",
    "Horsepower", "Torque", "Transmission",
    "DriveType", "BodyType",
    "FuelEfficiency(L/100km)", "AccidentHistory",
    "Price($)"
]

def _format_recommendations(df: pd.DataFrame, predicted_price: float | None = None) -> pd.DataFrame:
    df = df.copy()

    def pretty_int(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce").round(0).astype("Int64")
        # show blanks instead of <NA> without downcasting warning
        s_obj = s.astype("object")
        return s_obj.mask(s_obj.isna(), "")

    def pretty_float(series: pd.Series, ndigits: int = 1) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce").round(ndigits)
        s_obj = s.astype("object")
        return s_obj.mask(s_obj.isna(), "")

    for c in ["Year", "Horsepower", "Torque"]:
        if c in df.columns:
            df[c] = pretty_int(df[c])

    if "Mileage(km)" in df.columns:
        df["Mileage(km)"] = pretty_int(df["Mileage(km)"])

    if "EngineSize(L)" in df.columns:
        df["EngineSize(L)"] = pretty_float(df["EngineSize(L)"], 1)

    if "FuelEfficiency(L/100km)" in df.columns:
        df["FuelEfficiency(L/100km)"] = pretty_float(df["FuelEfficiency(L/100km)"], 1)

    if "Price($)" in df.columns:
        df["Price($)"] = pretty_int(df["Price($)"])

    if "similarity" in df.columns:
        df["similarity"] = pd.to_numeric(df["similarity"], errors="coerce").round(3)

    order = ["rec_id", "similarity"] + [c for c in DISPLAY_COLS if c in df.columns]
    df = df[[c for c in order if c in df.columns]]

    if predicted_price is not None:
        df.insert(1, "predicted_price_user", round(float(predicted_price), 0))

    return df


def load_catalog(project_root: Path) -> pd.DataFrame:
    """Load the recommendation catalog (use processed TRAIN as 'inventory')."""
    path = Path(project_root) / "data" / "processed" / "train.csv"
    df = pd.read_csv(path)
    if "rec_id" not in df.columns:
        df = df.reset_index(drop=True).copy()
        df.insert(0, "rec_id", np.arange(len(df)))
    return df

# ---------- cosine helpers ----------
def _cosine_std(Xu: np.ndarray, Xp: np.ndarray) -> np.ndarray:
    """Cosine similarity after standardizing by the pool statistics; returns values in [-1, 1]."""
    mu  = Xp.mean(axis=0, keepdims=True)
    std = Xp.std(axis=0, keepdims=True) + 1e-9
    Xu_n = (Xu - mu) / std
    Xp_n = (Xp - mu) / std
    return cosine_similarity(Xu_n, Xp_n).ravel()

def _cosine_score_0_1(Xu: np.ndarray, Xp: np.ndarray) -> np.ndarray:
    """Standardized cosine mapped to [0,1] for ranking/UI."""
    sims = _cosine_std(Xu, Xp)           # [-1, 1]
    return (sims + 1.0) / 2.0            # [0, 1]

def budget_recommendations(
    user_input: Dict[str, Any],
    catalog: pd.DataFrame,
    artifacts: Dict[str, Any],
    pct: float = 0.15,
    top_n: int = 10,
    extra_filters: Dict[str, Any] | None = None
) -> pd.DataFrame:
    """
    1) Predict user's price with the trained model
    2) Filter catalog to ±pct around that price
    3) Rank by standardized cosine similarity in the model feature space
    """
    pred_price = predict_price(user_input, artifacts)
    low, high = pred_price * (1 - pct), pred_price * (1 + pct)

    pool = catalog[(catalog["Price($)"] >= low) & (catalog["Price($)"] <= high)].copy()
    if extra_filters:
        for col, val in extra_filters.items():
            if col in pool.columns and val not in (None, ""):
                pool = pool[pool[col] == val]
    if pool.empty:
        pool = catalog.copy()

    X_user = _encode_one(user_input, artifacts)   # (1, D)
    X_pool = _encode_many(pool, artifacts)        # (N, D)

    score = _cosine_score_0_1(X_user, X_pool)     # [0,1]
    pool = pool.assign(similarity=score).sort_values("similarity", ascending=False)

    cols = ["rec_id", "similarity"] + [c for c in DISPLAY_COLS if c in pool.columns]
    out = pool[cols].head(top_n).reset_index(drop=True)
    return _format_recommendations(out, predicted_price=pred_price)

def similar_items(
    user_input: Dict[str, Any],
    catalog: pd.DataFrame,
    artifacts: Dict[str, Any],
    top_n: int = 10,
    extra_filters: Dict[str, Any] | None = None
) -> pd.DataFrame:
    """Pure similarity search in the model feature space (standardized cosine)."""
    pool = catalog.copy()
    if extra_filters:
        for col, val in extra_filters.items():
            if col in pool.columns and val not in (None, ""):
                pool = pool[pool[col] == val]
    if pool.empty:
        pool = catalog.copy()

    X_user = _encode_one(user_input, artifacts)
    X_pool = _encode_many(pool, artifacts)

    score = _cosine_score_0_1(X_user, X_pool)     # [0,1]
    pool = pool.assign(similarity=score).sort_values("similarity", ascending=False)

    cols = ["rec_id", "similarity"] + [c for c in DISPLAY_COLS if c in pool.columns]
    out = pool[cols].head(top_n).reset_index(drop=True)
    return _format_recommendations(out)

# =========================
# Internal helpers (encoding = same as training)
# =========================

def _encode_one(row: Dict[str, Any], artifacts: Dict[str, Any]) -> np.ndarray:
    feat_cols  = artifacts["feature_columns"]
    model_freq = artifacts.get("model_freq_map", {})

    r = dict(row)
    # Model -> Model_freq (keep this feature!)
    r["Model_freq"] = float(model_freq.get(str(r.get("Model", "")), 0.0))
    r.pop("Model", None)

    df = pd.DataFrame([r])

    # safe numeric coercion
    for c in ["Year","Mileage(km)","EngineSize(L)","Horsepower","Torque",
              "FuelEfficiency(L/100km)","Price($)","Model_freq"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # one-hot categoricals (drop_first=True)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # align to training features
    X = df.reindex(columns=feat_cols, fill_value=0.0).astype(float).to_numpy()
    return X

def _encode_many(df: pd.DataFrame, artifacts: Dict[str, Any]) -> np.ndarray:
    feat_cols  = artifacts["feature_columns"]
    model_freq = artifacts.get("model_freq_map", {})

    work = df.copy()
    if "Model" in work.columns:
        work["Model_freq"] = work["Model"].astype(str).map(model_freq).fillna(0.0)
        work = work.drop(columns=["Model"], errors="ignore")
    else:
        work["Model_freq"] = 0.0

    for c in ["Year","Mileage(km)","EngineSize(L)","Horsepower","Torque",
              "FuelEfficiency(L/100km)","Price($)","Model_freq"]:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    cat_cols = work.select_dtypes(include="object").columns.tolist()
    work = pd.get_dummies(work, columns=cat_cols, drop_first=True)

    X = work.reindex(columns=feat_cols, fill_value=0.0).astype(float).to_numpy()
    return X

# =========================
# Simple recommendation modes (budget & minimal inputs)
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
    pool["similarity"] = 1.0 - (pool["Price($)"] - budget_usd).abs() / (abs(budget_usd) + 1e-9)
    out = pool.sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)
    return _format_recommendations(out, predicted_price=budget_usd)

def recommend_minimal(min_input: dict,
                      catalog: pd.DataFrame,
                      artifacts: dict,
                      top_n: int = 10,
                      filters: dict | None = None) -> pd.DataFrame:
    # Similarity in the simple 4D numeric space used for clustering
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

    user_vec = pd.DataFrame([{f: pd.to_numeric(min_input.get(f, np.nan), errors="coerce") for f in feats}])
    user_vec = user_vec.fillna(pool[feats].median(numeric_only=True))

    sims = cosine_similarity(user_vec.to_numpy(), pool[feats].to_numpy()).ravel()
    pool = pool.assign(similarity=sims).sort_values("similarity", ascending=False)
    out = pool.head(top_n).reset_index(drop=True)
    return _format_recommendations(out)
