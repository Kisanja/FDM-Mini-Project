# app/1_🚗_Price_Predictor.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import sys

# Add app directory to path for imports
app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))
    
# Add project root to path for src imports
project_root = app_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# project helpers
from app._common import get_artifacts_and_catalog, intfmt, show_version_sidebar

# backend (your existing inference & recommend code)
from src.inference import (
    predict_and_cluster,
    _build_model_features,      # debug only
    _build_cluster_features,    # debug only
)
from src.recommend import similar_items

# ---------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------
st.set_page_config(page_title="Price Predictor", page_icon="🚗", layout="wide")
show_version_sidebar()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# nicer CSS (lifted/adapted from your friend’s style)
st.markdown("""
<style>
    :root {
        --primary: #667eea;
        --secondary: #764ba2;
    }
    .hero {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        padding: 1.6rem 2rem;
        border-radius: 14px;
        color: #fff;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(0,0,0,.15);
    }
    .hero h1 { margin: 0 0 .35rem 0; font-weight: 800; }
    .hero p { margin: 0; opacity: .92; }

    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem 1.25rem;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(0,0,0,.08);
    }
    .metric-card h4 { margin: 0; font-size: .95rem; color: #374151; }
    .metric-card .big {
        font-size: 1.6rem; font-weight: 800; margin-top: .25rem;
        color: #0f172a;
    }
    .muted { color: #64748b; font-size: .9rem; }
</style>
""", unsafe_allow_html=True)

# header
st.markdown("""
<div class="hero">
  <h1>🏷️ Car Price Prediction</h1>
  <p>Give a few key details, and we’ll estimate price & market segment — with similar cars you can browse.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Load artifacts + catalog (cached by your helper)
# ---------------------------------------------------------------------
art, cat = get_artifacts_and_catalog()
st.session_state.setdefault("artifacts", art)
st.session_state.setdefault("catalog", cat)
art = st.session_state.artifacts
cat = st.session_state.catalog

# ---------------------------------------------------------------------
# Helpers: compute segment statistics once (for segment average card)
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_segment_stats(_cat: pd.DataFrame, _art: dict) -> pd.DataFrame:
    """Assign your saved KMeans to catalog and aggregate per segment."""
    df = _cat.copy()
    feats = _art["kmeans_features"]           # ["Mileage(km)", "Year", "Horsepower", "EngineSize(L)"]
    scaler = _art["kmeans_scaler"]
    km     = _art["kmeans"]
    label_map = _art.get("kmeans_label_map", {})

    Xc = df[feats].apply(pd.to_numeric, errors="coerce")
    Xc = Xc.fillna(Xc.median(numeric_only=True))
    labels = km.predict(scaler.transform(Xc))
    names  = [label_map.get(str(l), label_map.get(l, f"Cluster {l}")) for l in labels]
    df["Segment"] = names

    grp = (df
           .groupby("Segment")
           .agg({
               "Price($)": "mean",
               "Mileage(km)": "mean",
               "Year": "mean",
               "Horsepower": "mean",
               "EngineSize(L)": "mean",
           })
           .rename(columns={
               "Price($)": "avg_price",
               "Mileage(km)": "avg_mileage",
               "Year": "avg_year",
               "Horsepower": "avg_horsepower",
               "EngineSize(L)": "avg_engine",
           })
           .reset_index())

    return grp

seg_stats = compute_segment_stats(cat, art)

# ---------------------------------------------------------------------
# Input UI
# ---------------------------------------------------------------------
st.subheader("Required basics")
c1, c2, c3 = st.columns(3)
with c1:
    brand = st.text_input("Brand", "Toyota")
    model = st.text_input("Model", "Corolla")
with c2:
    year  = st.number_input("Year", min_value=1990, max_value=2025, value=2018, step=1)
    mileage = st.number_input("Mileage (km)", min_value=0, max_value=500_000, value=84_000, step=1_000)
with c3:
    engine = st.number_input("Engine Size (L)", min_value=0.6, max_value=8.0, value=1.6, step=0.1, format="%.1f")
    hp     = st.number_input("Horsepower", min_value=40, max_value=1100, value=132, step=5)

with st.expander("Optional details (improves accuracy)"):
    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        condition = st.selectbox("Condition", ["Used","New","Damaged"], index=0)
        fuel      = st.selectbox("Fuel Type", ["Gasoline","Diesel","Hybrid","Electric"], index=0)
        torque    = st.number_input("Torque", min_value=50, max_value=1500, value=128, step=5)
    with oc2:
        trans = st.selectbox("Transmission", ["Automatic","Manual"], index=0)
        drive = st.selectbox("Drive Type", ["FWD","RWD","AWD"], index=0)
        body  = st.selectbox("Body Type", ["Sedan","Hatchback","SUV","Coupe","Pickup","Convertible"], index=0)
    with oc3:
        doors = st.number_input("Doors", min_value=2, max_value=6, value=4, step=1)
        seats = st.number_input("Seats", min_value=2, max_value=9, value=5, step=1)
        fe    = st.number_input("Fuel Efficiency (L/100km)", min_value=1.0, max_value=30.0, value=6.8, step=0.1, format="%.1f")

    oc4, oc5, oc6 = st.columns(3)
    with oc4:
        color = st.text_input("Color", "White")
        interior = st.selectbox("Interior", ["Cloth","Leather"], index=0)
    with oc5:
        city = st.text_input("City", "Berlin")
        accident = st.selectbox("Accident History", ["No","Yes"], index=0)
    with oc6:
        insurance = st.selectbox("Insurance", ["Valid","Expired"], index=0)
        reg = st.selectbox("Registration Status", ["Complete","Incomplete"], index=0)

    options = st.text_area("Options (comma-separated)",
                           "bluetooth, rear camera, navigation")

# assemble input row (make sure to keep the exact training names)
user = {
    "Brand": brand, "Model": model, "Year": int(year), "Condition": condition,
    "Mileage(km)": int(mileage), "EngineSize(L)": float(engine), "FuelType": fuel,
    "Horsepower": int(hp), "Torque": int(torque), "Transmission": trans, "DriveType": drive,
    "BodyType": body, "Doors": int(doors), "Seats": int(seats), "Color": color, "Interior": interior,
    "City": city, "AccidentHistory": accident, "Insurance": insurance, "RegistrationStatus": reg,
    "FuelEfficiency(L/100km)": float(fe), "Options": options,
}

# action buttons
left, right = st.columns([1, 1.2])
debug = st.checkbox("Show debug info", value=False)

with left:
    predict_clicked = st.button("🔮 Predict Price", use_container_width=True)
with right:
    similar_clicked = st.button("🔍 Show similar cars (top 10)", use_container_width=True)

# ---------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------
if predict_clicked:
    try:
        out = predict_and_cluster(user, art)
        price   = float(out["predicted_price"])
        seg     = str(out["cluster_name"])
        seg_row = seg_stats.loc[seg_stats["Segment"] == seg]

        seg_avg = float(seg_row["avg_price"].iloc[0]) if not seg_row.empty else np.nan
        delta_str = f"{'+' if price - seg_avg >= 0 else ''}{intfmt(price - seg_avg)} vs segment avg" if not np.isnan(seg_avg) else "n/a"

        m1, m2 = st.columns(2)
        with m1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("<h4>💰 Predicted Price</h4>", unsafe_allow_html=True)
            st.markdown(f'<div class="big">${intfmt(price)}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="muted">{delta_str}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with m2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("<h4>📊 Market Segment</h4>", unsafe_allow_html=True)
            st.markdown(f'<div class="big">{seg}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="muted">cluster #{out["cluster_label"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        

        # store for “similar”
        st.session_state["last_user_input"] = user

        if debug:
            st.subheader("Debug panel")
            X = _build_model_features(user, art)
            nonzero = X.columns[X.iloc[0].ne(0)].tolist()
            st.write("Non-zero model features (first 25):", nonzero[:25])

            Xc = _build_cluster_features(user, art)
            st.write("Cluster features row:")
            st.dataframe(Xc, use_container_width=True)
            st.write("Any NaN in cluster features?", bool(Xc.isna().any().any()))

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ---------------------------------------------------------------------
# Similar cars
# ---------------------------------------------------------------------
if similar_clicked:
    if "last_user_input" not in st.session_state:
        st.warning("Run a prediction first.")
    else:
        try:
            recs = similar_items(st.session_state["last_user_input"], cat, art, top_n=10)
            st.dataframe(recs, use_container_width=True)
        except Exception as e:
            st.error(f"Similarity error: {e}")

