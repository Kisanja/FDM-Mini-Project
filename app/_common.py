# app/_common.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import sys
import json

import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@st.cache_resource
def get_artifacts_and_catalog():
    from src.inference import load_artifacts
    from src.recommend import load_catalog
    art = load_artifacts(PROJECT_ROOT)
    cat = load_catalog(PROJECT_ROOT)
    return art, cat


def intfmt(x):
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return x


def pctfmt(x):
    try:
        return f"{float(x)*100:.1f}%"
    except Exception:
        return x


def show_version_sidebar(show_artifacts: bool = False) -> None:
    """
    Render the sidebar. By default, the detailed Artifacts block is hidden.
    Set show_artifacts=True to display it (for debugging/demos).
    """
    with st.sidebar:
        st.markdown("### App")
        # (put lightweight info/links here if you like)

        if not show_artifacts:
            return  # 🔕 hide artifacts by default

        # --- Optional Artifacts block (only if show_artifacts=True) ---
        vpath = PROJECT_ROOT / "models" / "version.json"
        if vpath.exists():
            meta = json.load(open(vpath))
            st.markdown("### Artifacts")
            st.write(
                f"**Model**: {meta.get('model_type')}  "
                f"| **Features**: {meta.get('feature_count')}  "
                f"| **Hash**: `{meta.get('feature_list_hash')}`"
            )
            st.caption(
                f"Trained: {meta.get('train_date')} • Seed: {meta.get('random_seed')} • "
                f"Rows: {meta.get('training_rows')} train / {meta.get('test_rows')} test"
            )
        else:
            st.info("version.json not found")


def get_price_bins(project_root: Path, catalog: pd.DataFrame, return_source: bool = False):
    """
    Prefer static cutoffs from models/price_bins.json.
    Fallback to catalog quantiles only if the file is missing/broken.
    """
    p = Path(project_root) / "models" / "price_bins.json"
    try:
        with open(p, "r") as f:
            d = json.load(f)
        q33 = float(d["q33"])
        q66 = float(d["q66"])
        src = "file"
    except Exception:
        # Fallback — compute from catalog
        q = pd.to_numeric(catalog["Price($)"], errors="coerce").dropna().quantile([0.33, 0.66])
        q33, q66 = float(q.loc[0.33]), float(q.loc[0.66])
        src = "catalog"

    # Sanity: ensure ascending
    if q33 > q66:
        q33, q66 = q66, q33

    return (q33, q66, src) if return_source else (q33, q66)


def segment_name_by_price(price: float, q33: float, q66: float) -> str:
    if not (price == price):  # NaN check
        return "Unknown"
    if price <= q33:
        return "Budget"
    if price <= q66:
        return "Mid-range"
    return "Luxury"
