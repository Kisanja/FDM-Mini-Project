# app/1_🚗_Price_Predictor.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from _common import get_artifacts_and_catalog, intfmt, show_version_sidebar
from src.inference import (
    predict_and_cluster,
    _build_model_features,      # debug only
    _build_cluster_features,    # debug only
)

from src.recommend import similar_items

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Price Predictor", page_icon="🚗", layout="wide")
show_version_sidebar()  # artifacts block is hidden by default per your updated _common.py

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# -----------------------------
# Load artifacts & catalog
# -----------------------------
art, cat = get_artifacts_and_catalog()
st.session_state.setdefault("artifacts", art)
st.session_state.setdefault("catalog", cat)
art = st.session_state.artifacts
cat = st.session_state.catalog

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
:root { --primary:#667eea; --secondary:#764ba2; }
.hero{
  background:linear-gradient(135deg,var(--primary) 0%,var(--secondary) 100%);
  padding:1.6rem 2rem; border-radius:14px; color:#fff; margin-bottom:1rem;
  box-shadow:0 8px 24px rgba(0,0,0,.15);
}
.hero h1{ margin:0 0 .35rem 0; font-weight:800; }
.metric-card{
  background:linear-gradient(135deg,#f5f7fa 0%,#c3cfe2 100%);
  padding:1rem 1.25rem; border-radius:12px; box-shadow:0 6px 18px rgba(0,0,0,.08);
}
.metric-card h4{ margin:0; font-size:.95rem; color:#374151; }
.metric-card .big{ font-size:1.6rem; font-weight:800; margin-top:.25rem; color:#0f172a; }
.muted{ color:#64748b; font-size:.9rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🏷️ Car Price Prediction</h1>
  <p>We estimate price and determine the market segment using our saved K-Means model.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Helper: assign K-Means segment to catalog once
# -----------------------------
@st.cache_data(show_spinner=False)
def kmeans_segment_summary(_catalog: pd.DataFrame, _art: dict) -> pd.DataFrame:
    feats = _art["kmeans_features"]           # e.g., ["Mileage(km)","Year","Horsepower","EngineSize(L)"]
    scaler = _art["kmeans_scaler"]
    km     = _art["kmeans"]
    label_map = _art.get("kmeans_label_map", {})

    Xc = _catalog[feats].apply(pd.to_numeric, errors="coerce")
    # Use scaler means for any missing values (vectors must be valid)
    means = pd.Series(getattr(scaler, "mean_", np.nan), index=feats)
    Xc = Xc.fillna(means)

    labels = km.predict(scaler.transform(Xc))
    names  = [label_map.get(str(l), label_map.get(l, f"Cluster {l}")) for l in labels]

    df = _catalog.copy()
    df["Segment"] = names
    # Pre-compute average price per segment for quick lookup
    seg_avg = df.groupby("Segment", as_index=False)["Price($)"].mean().rename(columns={"Price($)":"avg_price"})
    return seg_avg  # columns: Segment, avg_price

seg_summary = kmeans_segment_summary(cat, art)  # cached

# -----------------------------
# Inputs — EXACT 14 FEATURES
# -----------------------------
st.subheader("Required basics")
c1, c2, c3 = st.columns(3)
with c1:
    brand = st.text_input("Brand", "Toyota")
    model = st.text_input("Model", "Corolla")
with c2:
    year = st.number_input("Year", 1990, 2025, 2018, 1)
    mileage = st.number_input("Mileage (km)", 0, 500_000, 84_000, 1_000)
with c3:
    engine = st.number_input("Engine Size (L)", 0.6, 8.0, 1.6, 0.1, format="%.1f")
    hp = st.number_input("Horsepower", 40, 1100, 132, 5)

oc1, oc2, oc3 = st.columns(3)
with oc1:
    condition = st.selectbox("Condition", ["Used", "New", "Damaged"], index=0)
    fuel = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Hybrid", "Electric"], index=0)
with oc2:
    torque = st.number_input("Torque", 50, 1500, 128, 5)
    trans = st.selectbox("Transmission", ["Automatic", "Manual"], index=0)
with oc3:
    drive = st.selectbox("Drive Type", ["FWD", "RWD", "AWD"], index=0)
    body = st.selectbox("Body Type", ["Sedan", "Hatchback", "SUV", "Coupe", "Pickup", "Convertible"], index=0)

c4, c5 = st.columns(2)
with c4:
    fe = st.number_input("Fuel Efficiency (L/100km)", 1.0, 30.0, 6.8, 0.1, format="%.1f")
with c5:
    accident = st.selectbox("Accident History", ["No", "Yes"], index=0)

# Build payload (ONLY the 14 kept features)
user = {
    "Brand": brand, "Model": model, "Year": int(year), "Condition": condition,
    "Mileage(km)": int(mileage), "EngineSize(L)": float(engine), "FuelType": fuel,
    "Horsepower": int(hp), "Torque": int(torque), "Transmission": trans,
    "DriveType": drive, "BodyType": body,
    "FuelEfficiency(L/100km)": float(fe), "AccidentHistory": accident
}

left, right = st.columns([1, 1.2])
debug = st.checkbox("Show debug info", value=False)
with left:
    predict_clicked = st.button("🔮 Predict Price", use_container_width=True)
with right:
    similar_clicked = st.button("🔍 Show similar cars (top 10)", use_container_width=True)

# -----------------------------
# Predict & display (K-MEANS ONLY)
# -----------------------------
if predict_clicked:
    try:
        out = predict_and_cluster(user, art)  # price via model; segment via saved K-Means
        price   = float(out["predicted_price"])
        seg_km  = str(out["cluster_name"])
        seg_id  = int(out["cluster_label"])

        # Segment average price based on K-Means segment in catalog
        seg_row = seg_summary.loc[seg_summary["Segment"] == seg_km]
        seg_avg = float(seg_row["avg_price"].iloc[0]) if not seg_row.empty else np.nan
        delta_str = (
            f"{'+' if price - seg_avg >= 0 else ''}{intfmt(price - seg_avg)} vs segment avg"
            if not np.isnan(seg_avg) else "n/a"
        )

        # Cards
        m1, m2 = st.columns(2)
        with m1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("<h4>💰 Predicted Price</h4>", unsafe_allow_html=True)
            st.markdown(f'<div class="big">${intfmt(price)}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="muted">{delta_str}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with m2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("<h4>🏷️ Segment (K-Means)</h4>", unsafe_allow_html=True)
            st.markdown(f'<div class="big">{seg_km}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="muted">cluster #{seg_id}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # store for “similar”
        st.session_state["last_user_input"] = user
        st.session_state["last_pred_price"] = price

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

# -----------------------------
# Similar cars
# -----------------------------
if similar_clicked:
    if "last_user_input" not in st.session_state:
        st.warning("Run a prediction first.")
    else:
        try:
            recs = similar_items(st.session_state["last_user_input"], cat, art, top_n=10)
            st.dataframe(recs, use_container_width=True)
        except Exception as e:
            st.error(f"Similarity error: {e}")
