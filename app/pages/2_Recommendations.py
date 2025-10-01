# app/2_🧭_Recommendations.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from _common import get_artifacts_and_catalog, intfmt, show_version_sidebar

# ---------- Page setup ----------
st.set_page_config(page_title="Recommendations", page_icon="🧭", layout="wide")
show_version_sidebar()

# ---------- Styling to match the rest of the app ----------
st.markdown("""
<style>
:root { --primary:#667eea; --secondary:#764ba2; }
.hero{
  background: linear-gradient(135deg,var(--primary) 0%,var(--secondary) 100%);
  padding: 1.6rem 2rem; border-radius: 16px; color:#fff;
  margin-bottom: 1rem; box-shadow:0 10px 26px rgba(0,0,0,.15);
}
.hero h1{margin:0 0 .35rem 0; font-weight:800;}
.hero p{margin:0; opacity:.95;}

.card{
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  border-radius: 14px; padding: 1rem 1.25rem;
  box-shadow: 0 8px 22px rgba(0,0,0,.08);
}
.info{
  background: linear-gradient(135deg,#a8edea 0%,#fed6e3 100%);
  border-radius: 12px; padding:.9rem 1.1rem; border:1px solid rgba(0,0,0,.06);
  margin:.25rem 0 1rem 0;
}
.small{ color:#64748b; font-size:.92rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🧭 Recommendations</h1>
  <p>Find great matches by budget or by your minimal specs. Results come from the catalog in <code>data/processed/train.csv</code>.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Data/artifacts ----------
art, cat = get_artifacts_and_catalog()

# helper for dropdowns
@st.cache_data(show_spinner=False)
def _choices(col: str) -> list[str]:
    if col not in cat.columns:
        return []
    vals = cat[col].dropna().astype(str).unique().tolist()
    vals.sort()
    return vals

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["💰 By Budget", "🔍 Similar Cars"])

# =====================================================================
# TAB 1 — By Budget
# =====================================================================
with tab1:
    st.markdown('<div class="info">💡 <b>Tip:</b> Use either a single budget with a ± window or switch to Min–Max mode.</div>', unsafe_allow_html=True)

    mode = st.radio("Budget mode", ["Single + window", "Min–Max range"], horizontal=True)

    col1, col2 = st.columns(2)
    if mode == "Single + window":
        with col1:
            budget = st.number_input("Your budget (USD)", 1000, 300_000, 18_000, step=500)
        with col2:
            pct = st.slider("± Window (%)", 5, 40, 15, step=5)
        # compute low/high just for display
        low, high = int(budget * (1 - pct/100)), int(budget * (1 + pct/100))
        st.caption(f"Search window ≈ ${intfmt(low)} to ${intfmt(high)}")

    else:
        with col1:
            min_price = st.number_input("Min price ($)", 0, 300_000, 8_000, step=500)
        with col2:
            max_price = st.number_input("Max price ($)", min_price, 300_000, 22_000, step=500)
        # map min–max to our backend API (midpoint + pct window)
        budget = (min_price + max_price) / 2.0
        pct = 100 * (max_price - budget) / budget if budget > 0 else 15

    st.markdown("### ⚙️ Optional filters")
    f1, f2, f3 = st.columns(3)
    with f1:
        f_brand = st.selectbox("Brand", [""] + _choices("Brand"))
    with f2:
        f_fuel  = st.selectbox("FuelType", [""] + _choices("FuelType"))
    with f3:
        f_cond  = st.selectbox("Condition", [""] + _choices("Condition"))

    filters = {}
    if f_brand: filters["Brand"] = f_brand
    if f_fuel:  filters["FuelType"] = f_fuel
    if f_cond:  filters["Condition"] = f_cond

    if st.button("🔍 Find budget recommendations", use_container_width=True):
        from src.recommend import recommend_by_budget_only
        try:
            df = recommend_by_budget_only(float(budget), cat, pct=float(pct)/100.0, top_n=12, filters=filters)
            if df.empty:
                st.info("No matches found. Try widening the window or removing filters.")
            else:
                # Tidy display (if similarity present keep it; else just show main columns)
                show = df.copy()
                if "similarity" in show.columns:
                    show["Match %"] = (show["similarity"] * 100).round(1)
                    show = show.drop(columns=["similarity"])
                st.dataframe(show, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Recommendation error: {e}")

# =====================================================================
# TAB 2 — Similar Cars (minimal inputs)
# =====================================================================
with tab2:
    st.markdown('<div class="info">💡 <b>Tip:</b> Just the 4 core numeric fields are enough. Add an optional body type filter if you like.</div>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        year = st.number_input("Year", 1990, 2025, 2018, step=1)
        mileage = st.number_input("Mileage (km)", 0, 500_000, 85_000, step=1_000)
    with right:
        hp = st.number_input("Horsepower", 40, 1_100, 130, step=5)
        engine = st.number_input("Engine Size (L)", 0.5, 8.0, 1.6, step=0.1)

    body = st.selectbox("BodyType (optional filter)", [""] + _choices("BodyType"))

    if st.button("🔎 Find similar cars", use_container_width=True):
        from src.recommend import recommend_minimal
        try:
            min_input = {
                "Year": int(year),
                "Mileage(km)": int(mileage),
                "Horsepower": int(hp),
                "EngineSize(L)": float(engine),
            }
            filt = {"BodyType": body} if body else None
            df = recommend_minimal(min_input, cat, art, top_n=12, filters=filt)
            if df.empty:
                st.info("No similar cars found. Try relaxing filters.")
            else:
                show = df.copy()
                # present similarity as a percentage if present
                if "similarity" in show.columns:
                    show["Match %"] = (show["similarity"] * 100).round(1)
                    show = show.drop(columns=["similarity"])
                st.dataframe(show, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Similarity error: {e}")
