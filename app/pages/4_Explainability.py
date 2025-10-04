# app/4_🔎_Explainability.py
from __future__ import annotations
from pathlib import Path
import streamlit as st
import sys

# Add app directory to path for imports
app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from app._common import show_version_sidebar

# ---------- Page setup ----------
st.set_page_config(page_title="Explainability", page_icon="🔎", layout="wide")
show_version_sidebar()

# ---------- Styling (same vibe as other pages) ----------
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
.small{ color:#64748b; font-size:.92rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🔎 Model Explainability</h1>
  <p>See which features matter most and why some columns were dropped or excluded.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Find <project_root>/reports/figures robustly ----------
this_file = Path(__file__).resolve()
candidates = [
    this_file.parent.parent,  # <repo>/
    this_file.parent,         # <repo>/app/
    Path.cwd(),               # current working dir
]
FIG_DIR = None
for root in candidates:
    p = root / "reports" / "figures"
    if p.exists():
        FIG_DIR = p
        break
# final fallback: repo-wide search
if FIG_DIR is None:
    hits = list(this_file.parents[2].rglob("reports/figures"))
    if hits:
        FIG_DIR = hits[0]

st.caption(f"Figures folder resolved to: `{FIG_DIR}` | Exists? **{bool(FIG_DIR and FIG_DIR.exists())}**")

# ---------- Show importance figures if present ----------
st.subheader("Feature Importance")

files = [
    ("feature_importance_tree_top15.png", "Top 15 Feature Importances (model / gain)"),
    ("feature_importance_permutation_top15.png", "Top 15 Permutation Importances (↓MAE when shuffled)"),
]

found_any = False
if FIG_DIR and FIG_DIR.exists():
    for fname, title in files:
        path = FIG_DIR / fname
        if path.exists():
            st.image(str(path), caption=title, use_container_width=True)
            found_any = True

if not found_any:
    st.info(
        "No importance figures found. Generate them in your modeling notebook and save to "
        "`<project_root>/reports/figures/` as:\n"
        "• `feature_importance_tree_top15.png`\n"
        "• `feature_importance_permutation_top15.png`"
    )

st.markdown("---")
st.subheader("Why certain columns were dropped/excluded")
st.markdown("""
- **CarAge** was dropped as redundant with `Year` (they’re perfectly collinear: `CarAge = current_year − Year`).
- **PricePerKm** was **excluded from training** to prevent **data leakage** because it’s derived from the target (`Price($) ÷ Mileage(km)`).
- Free-text **Options** were engineered into numeric signals (`OptionsCount` plus one-hot `opt_*` flags) so the model can use them.
""")