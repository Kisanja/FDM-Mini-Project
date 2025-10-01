# app/Home.py
from __future__ import annotations
from pathlib import Path
import streamlit as st
from _common import show_version_sidebar

# ----- Page setup -----
st.set_page_config(page_title="Used Car Price — Demo", page_icon="🚘", layout="wide")
show_version_sidebar()

# ----- Lightweight styling (matches your new pages) -----
st.markdown("""
<style>
:root {
  --primary: #667eea;
  --secondary: #764ba2;
}
.hero {
  background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
  padding: 2rem;
  border-radius: 16px;
  color: #fff;
  margin-bottom: 1.25rem;
  box-shadow: 0 10px 28px rgba(0,0,0,0.15);
}
.hero h1 { margin: 0 0 .35rem 0; font-weight: 800; letter-spacing: .2px; }
.hero p  { margin: 0; opacity: .95; font-size: 1.05rem; }

.card {
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  border-radius: 14px;
  padding: 1.1rem 1.25rem;
  box-shadow: 0 8px 22px rgba(0,0,0,.08);
}
.card h3 { margin: .1rem 0 .35rem 0; font-size: 1.05rem; }
.card p  { margin: 0; color: #334155; }

.callout {
  background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
  padding: .9rem 1.1rem;
  border-radius: 12px;
  border: 1px solid rgba(0,0,0,.06);
  margin: .25rem 0 1rem 0;
}

.small { color: #64748b; font-size: .92rem; }
</style>
""", unsafe_allow_html=True)

# ----- Hero -----
st.markdown("""
<div class="hero">
  <h1>🚘 Used Car Price Prediction & Recommendations</h1>
  <p>
    End-to-end demo: LightGBM regression for price, K-Means market segments, 
    budget & similarity recommendations, and explainability. Artifacts are read from <code>models/</code>.
  </p>
</div>
""", unsafe_allow_html=True)

# ----- Quick links (uses st.page_link when available) -----
# Falls back to text if older Streamlit, but most recent versions support it.
links = [
    ("app/1_🚗_Price_Predictor.py", "🚗 Price Predictor"),
    ("app/2_🎯_Recommendations.py", "🎯 Recommendations"),
    ("app/3_📊_Market_Segments.py", "📊 Market Segments"),
    ("app/4_🔎_Explainability.py", "🔎 Explainability"),
]
st.markdown('<div class="callout"><b>Quick start:</b> jump straight to a page ↓</div>', unsafe_allow_html=True)
cols = st.columns(4)
for col, (path, label) in zip(cols, links):
    with col:
        try:
            st.page_link(path, label=label, use_container_width=True)
        except Exception:
            # Older Streamlit fallback
            st.button(label, use_container_width=True, help=f"Open {label} via sidebar")

st.divider()

# ----- What’s inside (feature cards) -----
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 💰 Price prediction")
    st.markdown("""
- LightGBM regression (saved to `models/lightgbm_model.pkl`)
- Robust preprocessing (OHE, frequency encoding, options parsing)
- Single-car inference + clean UI form
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧭 Market segments")
    st.markdown("""
- K-Means on numeric features (scaled)
- Clusters labeled by median price (Budget / Mid-range / Luxury)
- Segment profile table in **Market Segments** page
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Recommendations")
    st.markdown("""
- Budget-only and minimal-inputs modes
- Cosine similarity in model feature space
- Catalog powered from `data/processed/train.csv`
    """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("")

# ----- Explainability + where artifacts live -----
st.markdown("### 🔍 Explainability & artifacts")
st.markdown("""
- Feature importance (tree-based + permutation) on the **Explainability** page  
- Exported figures expected in `reports/figures/`  
- All runtime artifacts are loaded from `models/` (model, features list, k-means, scaler, label map)
""")

st.markdown('<p class="small">Tip: if a page says a file is missing, re-run the notebook cells that save it into the expected folder.</p>', unsafe_allow_html=True)
