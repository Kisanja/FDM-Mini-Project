# app/3_📊_Market_Segments.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st
import altair as alt

from _common import show_version_sidebar

# ---------------- Page setup ----------------
st.set_page_config(page_title="Market Segments", page_icon="📊", layout="wide")
show_version_sidebar()

# ---------------- Styling (match other pages) ----------------
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
  <h1>📊 Market Segments Profile</h1>
  <p>Clusters learned via K-Means on numeric attributes and labeled by median price (Budget / Mid-range / Luxury).</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Locate profile CSV robustly ----------------
this_file = Path(__file__).resolve()
candidates = [
    this_file.parent.parent,   # <repo>/
    this_file.parent,          # <repo>/app/
    Path.cwd(),                # current working dir
]
profile_path = None
for root in candidates:
    p = root / "reports" / "results" / "cluster_profile.csv"
    if p.exists():
        profile_path = p
        break
# final fallback: repo-wide search
if profile_path is None:
    hits = list(this_file.parents[2].rglob("cluster_profile.csv"))
    if hits:
        profile_path = hits[0]

# Debug hint (can comment out if you prefer)
st.caption(f"Looking for: `{profile_path}`")
st.caption(f"Exists here? **{bool(profile_path and profile_path.exists())}**")

# ---------------- Load & display ----------------
if profile_path and profile_path.exists():
    # Try without index_col first; fall back if file has an index column
    try:
        df = pd.read_csv(profile_path)
        # if a stray unnamed index sneaks in, drop it
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
    except Exception:
        df = pd.read_csv(profile_path, index_col=0).reset_index(drop=True)

    # Ensure Segment is present and first
    if "Segment" in df.columns:
        cols = ["Segment"] + [c for c in df.columns if c != "Segment"]
        df = df[cols]

    st.subheader("Segment table")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download profile CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="cluster_profile.csv",
        use_container_width=True
    )

    # ---------------- Quick visuals (flexible to available columns) ----------------
    st.markdown("### Visual summary")
    num_cols = [c for c in df.columns if c != "Segment" and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        st.info("No numeric columns to chart. Generate profile with numeric summaries in your notebook.")
    else:
        # If an average price column exists, show a bar chart for it
        price_cols = [c for c in num_cols if c.lower() in {"price($)", "avg_price", "price"}]
        col1, col2 = st.columns(2)

        if price_cols:
            price_col = price_cols[0]
            with col1:
                bar = (
                    alt.Chart(df)
                    .mark_bar()
                    .encode(
                        x=alt.X("Segment:N", sort="-y", title="Segment"),
                        y=alt.Y(f"{price_col}:Q", title="Average Price ($)"),
                        color="Segment:N",
                        tooltip=["Segment"] + price_cols
                    )
                    .properties(height=300)
                )
                st.altair_chart(bar, use_container_width=True)
        else:
            st.caption("No price column found in profile (e.g., `avg_price` or `Price($)`). Skipping price chart.")

        # Radar-like normalized chart for a few key numeric metrics if present
        keys = [k for k in ["avg_year", "avg_horsepower", "avg_engine", "avg_mileage"] if k in df.columns]
        if not keys:
            # try generic names if your CSV used raw column names
            alt_keys = [k for k in ["Year", "Horsepower", "EngineSize(L)", "Mileage(km)"] if k in df.columns]
            keys = alt_keys

        if keys:
            # Normalize for display (0–1) per column
            plot_df = df[["Segment"] + keys].copy()
            for k in keys:
                col_min, col_max = plot_df[k].min(), plot_df[k].max()
                if col_max > col_min:
                    plot_df[k] = (plot_df[k] - col_min) / (col_max - col_min)
                else:
                    plot_df[k] = 0.5  # fallback when constant

            long_df = plot_df.melt(id_vars="Segment", var_name="Metric", value_name="Normalized")

            with col2:
                line = (
                    alt.Chart(long_df)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("Metric:N", title=None),
                        y=alt.Y("Normalized:Q", scale=alt.Scale(domain=[0,1])),
                        color="Segment:N",
                        tooltip=["Segment","Metric","Normalized"]
                    )
                    .properties(height=300)
                )
                st.altair_chart(line, use_container_width=True)
        else:
            st.caption("No suitable numeric columns found for the radar-style summary.")
else:
    st.info(
        "cluster_profile.csv not found. Generate it in your modeling notebook "
        "(clustering section). Expected at reports/results/cluster_profile.csv."
    )

st.markdown("---")
st.caption(
    "This profile is computed from your training catalog. "
    "Segments come from K-Means (k=3 by default), labeled by median price per cluster."
)
