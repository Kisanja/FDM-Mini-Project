# app/_common.py
from __future__ import annotations
from pathlib import Path
import sys
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
    try: return f"{int(round(float(x))):,}"
    except: return x

def pctfmt(x):
    try: return f"{float(x)*100:.1f}%"
    except: return x

def show_version_sidebar():
    import json
    vpath = PROJECT_ROOT / "models" / "version.json"
    st.sidebar.markdown("### Artifacts")
    if vpath.exists():
        meta = json.load(open(vpath))
        st.sidebar.write(
            f"**Model**: {meta.get('model_type')}  "
            f" | **Features**: {meta.get('feature_count')}  "
            f" | **Hash**: `{meta.get('feature_list_hash')}`"
        )
        st.sidebar.caption(
            f"Trained: {meta.get('train_date')} • Seed: {meta.get('random_seed')} • "
            f"Rows: {meta.get('training_rows')} train / {meta.get('test_rows')} test"
        )
    else:
        st.sidebar.info("version.json not found")
