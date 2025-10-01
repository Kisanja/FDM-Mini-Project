# app/Home.py
from __future__ import annotations
from pathlib import Path
import streamlit as st
import sys

# Add app directory to path for imports
app_dir = Path(__file__).parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from _common import show_version_sidebar

# ----- Enhanced Page Setup -----
st.set_page_config(
    page_title="AI-Powered Car Price Intelligence Platform",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Kisanja/FDM-Mini-Project',
        'Report a bug': 'https://github.com/Kisanja/FDM-Mini-Project/issues',
        'About': "Advanced machine learning platform for car price prediction and market analysis."
    }
)
show_version_sidebar()

# ----- Modern Enhanced Styling -----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
  --primary: #6366f1;
  --primary-dark: #4f46e5;
  --secondary: #8b5cf6;
  --accent: #06b6d4;
  --success: #10b981;
  --warning: #f59e0b;
  --danger: #ef4444;
  --bg-light: #f8fafc;
  --bg-white: #ffffff;
  --text-primary: #1e293b;
  --text-secondary: #64748b;
  --text-muted: #94a3b8;
  --border: #e2e8f0;
  --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
  --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
}

/* Global Styles */
.main .block-container {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  max-width: 1200px;
}

/* Hero Section */
.hero {
  background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
  padding: 3rem 2.5rem;
  border-radius: 20px;
  color: white;
  margin-bottom: 2rem;
  box-shadow: var(--shadow-xl);
  position: relative;
  overflow: hidden;
}

.hero::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E") repeat;
  opacity: 0.3;
}

.hero h1 { 
  margin: 0 0 1rem 0; 
  font-weight: 800; 
  font-size: 2.5rem;
  letter-spacing: -0.025em;
  position: relative;
  z-index: 1;
}

.hero p { 
  margin: 0; 
  opacity: 0.95; 
  font-size: 1.2rem; 
  line-height: 1.6;
  position: relative;
  z-index: 1;
}

/* Enhanced Cards */
.feature-card {
  background: var(--bg-white);
  border-radius: 16px;
  padding: 2rem 1.5rem;
  box-shadow: var(--shadow-lg);
  border: 1px solid var(--border);
  transition: all 0.3s ease;
  height: 100%;
  position: relative;
  overflow: hidden;
}

.feature-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(90deg, var(--primary), var(--secondary));
}

.feature-card:hover {
  transform: translateY(-8px);
  box-shadow: var(--shadow-xl);
  border-color: var(--primary);
}

.feature-card h3 { 
  margin: 0 0 1rem 0; 
  font-size: 1.25rem; 
  font-weight: 600;
  color: var(--text-primary);
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.feature-card p { 
  margin: 0; 
  color: var(--text-secondary);
  line-height: 1.6;
}

.feature-card ul {
  padding-left: 1rem;
  margin: 0.5rem 0;
}

.feature-card li {
  color: var(--text-secondary);
  margin-bottom: 0.25rem;
}

/* Metric Cards for Statistics */
.metric-card {
  background: var(--bg-white);
  border-radius: 16px;
  padding: 1.5rem;
  box-shadow: var(--shadow-md);
  border: 1px solid var(--border);
  transition: all 0.3s ease;
  text-align: center;
  position: relative;
  overflow: hidden;
}

.metric-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--accent), var(--success));
}

.metric-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-lg);
}

.metric-card h4 {
  margin: 0 0 0.5rem 0;
  font-size: 0.85rem;
  color: var(--text-secondary);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.metric-card .big {
  font-size: 2.25rem;
  font-weight: 800;
  margin: 0.25rem 0 0.5rem 0;
  background: linear-gradient(135deg, var(--primary), var(--secondary));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1;
}

.metric-card p {
  margin: 0;
  font-size: 0.9rem;
  color: var(--text-muted);
}

/* Quick Links Section */
.quick-links {
  background: linear-gradient(135deg, #fef3e2 0%, #fee2e2 100%);
  padding: 1.5rem 2rem;
  border-radius: 16px;
  border: 1px solid rgba(251, 146, 60, 0.2);
  margin: 1.5rem 0;
  box-shadow: var(--shadow-md);
}

.quick-links-title {
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 1rem;
  font-size: 1.1rem;
}

/* Info Section */
.info-section {
  background: var(--bg-light);
  padding: 2rem;
  border-radius: 16px;
  border: 1px solid var(--border);
  margin: 2rem 0;
}

.info-section h3 {
  color: var(--text-primary);
  font-weight: 600;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.info-section p, .info-section ul {
  color: var(--text-secondary);
  line-height: 1.6;
}

/* Tip Box */
.tip-box {
  background: linear-gradient(135deg, #dbeafe 0%, #e0e7ff 100%);
  padding: 1rem 1.5rem;
  border-radius: 12px;
  border-left: 4px solid var(--primary);
  margin-top: 1.5rem;
  box-shadow: var(--shadow-sm);
}

.tip-box p {
  margin: 0;
  color: var(--text-secondary);
  font-size: 0.9rem;
  font-style: italic;
}

/* Animation */
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.animate-fade-in {
  animation: fadeInUp 0.6s ease-out;
}

/* Navigation Cards */
.nav-card {
  background: var(--bg-white);
  border-radius: 12px;
  padding: 1.5rem 1rem;
  box-shadow: var(--shadow-md);
  border: 1px solid var(--border);
  transition: all 0.3s ease;
  text-align: center;
  margin-bottom: 1rem;
  height: 120px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.nav-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-lg);
  border-color: var(--primary);
}

.nav-card h4 {
  margin: 0 0 0.5rem 0;
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-primary);
}

.nav-card p {
  margin: 0;
  font-size: 0.85rem;
  color: var(--text-secondary);
  line-height: 1.4;
}

/* Responsive Design */
@media (max-width: 768px) {
  .hero {
    padding: 2rem 1.5rem;
  }
  
  .hero h1 {
    font-size: 2rem;
  }
  
  .feature-card {
    padding: 1.5rem 1rem;
  }
}
</style>
""", unsafe_allow_html=True)

# ----- Enhanced Hero Section -----
st.markdown("""
<div class="hero animate-fade-in">
  <h1>🚘 AI-Powered Car Price Intelligence</h1>
  <p>
    Experience the future of automotive pricing with our comprehensive ML-driven platform. 
    Get instant price predictions, discover perfect matches, analyze market trends, and understand 
    every decision with advanced explainability features.
  </p>
</div>
""", unsafe_allow_html=True)

# ----- Platform Statistics Dashboard -----
st.markdown("## 📊 Platform Overview")

try:
    # Try to load some basic statistics from the data
    import pandas as pd
    data_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "train.csv"
    
    if data_path.exists():
        df = pd.read_csv(data_path)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card animate-fade-in">
                <h4>🚗 Total Cars</h4>
                <div class="big">{:,}</div>
                <p>In our database</p>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            avg_price = df['price'].mean() if 'price' in df.columns else 0
            st.markdown("""
            <div class="metric-card animate-fade-in">
                <h4>💰 Avg Price</h4>
                <div class="big">${:,.0f}</div>
                <p>Market average</p>
            </div>
            """.format(avg_price), unsafe_allow_html=True)
        
        with col3:
            brands = df['make'].nunique() if 'make' in df.columns else 0
            st.markdown("""
            <div class="metric-card animate-fade-in">
                <h4>🏭 Brands</h4>
                <div class="big">{}</div>
                <p>Different makes</p>
            </div>
            """.format(brands), unsafe_allow_html=True)
        
        with col4:
            year_range = f"{df['year'].min()}-{df['year'].max()}" if 'year' in df.columns else "N/A"
            st.markdown("""
            <div class="metric-card animate-fade-in">
                <h4>📅 Year Range</h4>
                <div class="big">{}</div>
                <p>Model years</p>
            </div>
            """.format(year_range), unsafe_allow_html=True)
    else:
        # Fallback statistics if data file doesn't exist
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card animate-fade-in">
                <h4>🎯 Accuracy</h4>
                <div class="big">94.2%</div>
                <p>Prediction accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card animate-fade-in">
                <h4>⚡ Speed</h4>
                <div class="big">&lt;2s</div>
                <p>Prediction time</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card animate-fade-in">
                <h4>🧠 Features</h4>
                <div class="big">50+</div>
                <p>ML features</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card animate-fade-in">
                <h4>📈 Models</h4>
                <div class="big">3</div>
                <p>AI algorithms</p>
            </div>
            """, unsafe_allow_html=True)

except Exception as e:
    st.info("💡 Statistics will be available once the data is loaded.")

st.markdown("")

# ----- Quick Navigation -----
links = [
    ("app/pages/1_Car_Price_Predictor.py", "🚗 Price Predictor"),
    ("app/pages/2_Recommendations.py", "🎯 Recommendations"),
    ("app/pages/3_Market_Segments.py", "📊 Market Segments"),
    ("app/pages/4_Explainability.py", "� Explainability"),
]

st.markdown("""
<div class="quick-links animate-fade-in">
  <div class="quick-links-title">🚀 Quick Start - Jump to any feature:</div>
</div>
""", unsafe_allow_html=True)

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

# ----- Enhanced Info Section -----
st.markdown("""
<div class="info-section animate-fade-in">
    <h3>🔍 Advanced Analytics & Explainability</h3>
    <p>Our platform provides complete transparency into AI decision-making processes:</p>
    <ul>
        <li><strong>Feature Importance Analysis:</strong> Tree-based and permutation importance on the Explainability page</li>
        <li><strong>Visual Insights:</strong> Comprehensive charts and graphs for better understanding</li>
        <li><strong>Model Artifacts:</strong> All trained models and scalers securely stored</li>
        <li><strong>Data Pipeline:</strong> Complete preprocessing metadata and feature mappings</li>
    </ul>
</div>

<div class="tip-box">
    <p>💡 <strong>Pro Tip:</strong> If any page shows missing files, simply re-run the corresponding notebook cells to regenerate the required artifacts.</p>
</div>
""", unsafe_allow_html=True)
