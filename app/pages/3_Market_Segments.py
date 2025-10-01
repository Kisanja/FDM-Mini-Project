# app/3_📊_Market_Segments.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

# Add app directory to path for imports
app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from app._common import show_version_sidebar

# ---------------- Page setup ----------------
st.set_page_config(page_title="Market Segments", page_icon="📊", layout="wide")
show_version_sidebar()

# ---------------- Enhanced Styling ----------------
st.markdown("""
<style>
:root { --primary:#667eea; --secondary:#764ba2; --success:#10b981; --warning:#f59e0b; --info:#3b82f6; }
.hero{
  background: linear-gradient(135deg,var(--primary) 0%,var(--secondary) 100%);
  padding: 1.6rem 2rem; border-radius: 16px; color:#fff;
  margin-bottom: 1rem; box-shadow:0 10px 26px rgba(0,0,0,.15);
}
.hero h1{margin:0 0 .35rem 0; font-weight:800;}
.hero p{margin:0; opacity:.95;}

.segment-card{
  background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
  border-radius: 12px; padding: 1.5rem;
  box-shadow: 0 4px 12px rgba(0,0,0,.1);
  border-left: 4px solid var(--primary);
  margin-bottom: 1rem;
}

.segment-card.budget{
  border-left-color: var(--success);
  background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
}

.segment-card.midrange{
  border-left-color: var(--warning);
  background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
}

.segment-card.luxury{
  border-left-color: var(--info);
  background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
}

.metric-row{
  display: flex; gap: 1rem; align-items: center; margin: 0.5rem 0;
}

.metric-label{
  font-weight: 600; color: #374151; min-width: 120px;
}

.metric-value{
  color: #1f2937; font-weight: 500;
}

.insight-box{
  background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
  border-radius: 12px; padding: 1.5rem;
  border-left: 4px solid #3b82f6;
  margin: 1rem 0;
  box-shadow: 0 4px 12px rgba(0,0,0,.12);
  border: 1px solid #e2e8f0;
}

.insight-box.success{
  background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
  border-left-color: #10b981;
}

.insight-box.warning{
  background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
  border-left-color: #f59e0b;
}

.insight-box.info{
  background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
  border-left-color: #3b82f6;
}

.chart-container{
  background: #ffffff;
  border-radius: 8px;
  padding: 1rem;
  box-shadow: 0 2px 8px rgba(0,0,0,.05);
  margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>📊 Enhanced Market Segments Analysis</h1>
  <p>Comprehensive analysis of car market segments based on K-Means clustering with price, year, mileage, and performance characteristics.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Helper functions ----------------
def load_segment_data():
    """Load all segment-related data files"""
    this_file = Path(__file__).resolve()
    candidates = [
        this_file.parent.parent,   # <repo>/
        this_file.parent,          # <repo>/app/
        Path.cwd(),                # current working dir
    ]
    
    data_files = {}
    file_names = {
        'enhanced_profile': 'reports/results/enhanced_cluster_profile.csv',
        'brand_analysis': 'reports/results/segment_brand_analysis.csv',
        'year_trends': 'reports/results/segment_year_trends.csv',
        'original_profile': 'reports/results/cluster_profile.csv'
    }
    
    for data_key, file_path in file_names.items():
        found_path = None
        for root in candidates:
            p = root / file_path
            if p.exists():
                found_path = p
                break
        
        if found_path and found_path.exists():
            try:
                df = pd.read_csv(found_path)
                if "Unnamed: 0" in df.columns:
                    df = df.drop(columns=["Unnamed: 0"])
                data_files[data_key] = df
            except Exception as e:
                st.error(f"Error loading {file_path}: {e}")
        else:
            data_files[data_key] = None
    
    return data_files

def create_segment_card(segment_data):
    """Create a styled card for segment summary"""
    segment = segment_data['Segment']
    card_class = segment.lower().replace('-', '')
    
    return f"""
    <div class="segment-card {card_class}">
        <h3 style="margin: 0 0 1rem 0; color: #1f2937;">🏷️ {segment} Segment</h3>
        <div class="metric-row">
            <span class="metric-label">Market Share:</span>
            <span class="metric-value">{segment_data['Percentage']} ({segment_data['Count']:,} cars)</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Avg Price:</span>
            <span class="metric-value">${segment_data['Avg_Price($)']:,.0f}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Price Range:</span>
            <span class="metric-value">${segment_data['Min_Price($)']:,.0f} - ${segment_data['Max_Price($)']:,.0f}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Top Brand:</span>
            <span class="metric-value">{segment_data['Top_Brand']} ({segment_data['Top_BodyType']})</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Avg Year:</span>
            <span class="metric-value">{segment_data['Avg_Year']}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Avg Options:</span>
            <span class="metric-value">{segment_data['Avg_Options']} features</span>
        </div>
    </div>
    """

# ---------------- Load data ----------------
data_files = load_segment_data()

if data_files['enhanced_profile'] is not None:
    enhanced_df = data_files['enhanced_profile']
    
    # ---------------- Segment Overview Cards ----------------
    st.markdown("## 📈 Market Segment Overview")
    
    # Create three columns for segment cards
    col1, col2, col3 = st.columns(3)
    
    for idx, (_, segment_data) in enumerate(enhanced_df.iterrows()):
        col = [col1, col2, col3][idx % 3]
        with col:
            st.markdown(create_segment_card(segment_data), unsafe_allow_html=True)

    
    # ---------------- Enhanced Visualizations ----------------
    st.markdown("## 📊 Advanced Analytics & Visualizations")
    
    # Price Distribution Analysis
    st.markdown("### 💰 Price Distribution by Segment")
    col1, col2 = st.columns(2)
    
    with col1:
        # Price comparison bar chart
        price_fig = px.bar(
            enhanced_df,
            x='Segment',
            y='Avg_Price($)',
            color='Segment',
            title='Average Price by Segment',
            color_discrete_map={
                'Budget': '#10b981',
                'Mid-range': '#f59e0b', 
                'Luxury': '#3b82f6'
            }
        )
        price_fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(price_fig, use_container_width=True)
    
    with col2:
        # Market share pie chart
        market_fig = px.pie(
            enhanced_df,
            values='Count',
            names='Segment',
            title='Market Share by Segment',
            color_discrete_map={
                'Budget': '#10b981',
                'Mid-range': '#f59e0b',
                'Luxury': '#3b82f6'
            }
        )
        market_fig.update_layout(height=400)
        st.plotly_chart(market_fig, use_container_width=True)
    
    # Vehicle Characteristics Radar Chart
    st.markdown("### 🎯 Vehicle Characteristics Comparison")
    
    # Prepare data for radar chart
    characteristics = ['Avg_Year', 'Avg_Mileage(km)', 'Avg_Horsepower', 'Avg_EngineSize(L)', 'Avg_FuelEfficiency(L/100km)']
    
    # Normalize data for radar chart (0-1 scale)
    radar_data = enhanced_df.copy()
    for char in characteristics:
        if char in radar_data.columns:
            min_val = radar_data[char].min()
            max_val = radar_data[char].max()
            if max_val > min_val:
                radar_data[f'{char}_normalized'] = (radar_data[char] - min_val) / (max_val - min_val)
            else:
                radar_data[f'{char}_normalized'] = 0.5
    
    # Create radar chart
    fig_radar = go.Figure()
    
    colors = {'Budget': '#10b981', 'Mid-range': '#f59e0b', 'Luxury': '#3b82f6'}
    
    for _, row in radar_data.iterrows():
        segment = row['Segment']
        values = [row[f'{char}_normalized'] for char in characteristics if f'{char}_normalized' in row]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the radar chart
            theta=[char.replace('Avg_', '').replace('(km)', '').replace('(L)', '').replace('(L/100km)', '') 
                   for char in characteristics] + [characteristics[0].replace('Avg_', '').replace('(km)', '')],
            fill='toself',
            name=segment,
            line_color=colors[segment]
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Normalized Vehicle Characteristics by Segment",
        height=500
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Brand Analysis
    if data_files['brand_analysis'] is not None:
        st.markdown("### 🏢 Top Brands by Segment")
        brand_df = data_files['brand_analysis']
        
        # Create tabs for each segment
        budget_brands = brand_df[brand_df['Segment'] == 'Budget'].head(5)
        midrange_brands = brand_df[brand_df['Segment'] == 'Mid-range'].head(5)
        luxury_brands = brand_df[brand_df['Segment'] == 'Luxury'].head(5)
        
        tab1, tab2, tab3 = st.tabs(["🟢 Budget", "🟡 Mid-range", "🔵 Luxury"])
        
        with tab1:
            if not budget_brands.empty:
                brand_fig = px.bar(
                    budget_brands,
                    x='Brand',
                    y='Count',
                    title='Top Budget Segment Brands',
                    color_discrete_sequence=['#10b981']
                )
                st.plotly_chart(brand_fig, use_container_width=True)
                st.dataframe(budget_brands[['Brand', 'Count', 'Percentage', 'Avg_Price($)']], 
                           use_container_width=True, hide_index=True)
        
        with tab2:
            if not midrange_brands.empty:
                brand_fig = px.bar(
                    midrange_brands,
                    x='Brand',
                    y='Count',
                    title='Top Mid-range Segment Brands',
                    color_discrete_sequence=['#f59e0b']
                )
                st.plotly_chart(brand_fig, use_container_width=True)
                st.dataframe(midrange_brands[['Brand', 'Count', 'Percentage', 'Avg_Price($)']], 
                           use_container_width=True, hide_index=True)
        
        with tab3:
            if not luxury_brands.empty:
                brand_fig = px.bar(
                    luxury_brands,
                    x='Brand',
                    y='Count',
                    title='Top Luxury Segment Brands',
                    color_discrete_sequence=['#3b82f6']
                )
                st.plotly_chart(brand_fig, use_container_width=True)
                st.dataframe(luxury_brands[['Brand', 'Count', 'Percentage', 'Avg_Price($)']], 
                           use_container_width=True, hide_index=True)
    
    # Detailed Statistics Table
    st.markdown("### 📋 Comprehensive Segment Statistics")
    
    # Select key columns for display
    display_columns = ['Segment', 'Count', 'Percentage', 'Avg_Price($)', 'Min_Price($)', 'Max_Price($)',
                      'Avg_Year', 'Top_Brand', 'Top_BodyType', 'New_Percentage', 'Avg_Options']
    
    available_columns = [col for col in display_columns if col in enhanced_df.columns]
    
    st.dataframe(
        enhanced_df[available_columns],
        use_container_width=True,
        hide_index=True
    )
    
    # Download enhanced data
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            "📥 Download Enhanced Profile",
            enhanced_df.to_csv(index=False).encode("utf-8"),
            file_name="enhanced_cluster_profile.csv",
            use_container_width=True
        )
    
    with col2:
        if data_files['brand_analysis'] is not None:
            st.download_button(
                "📥 Download Brand Analysis",
                data_files['brand_analysis'].to_csv(index=False).encode("utf-8"),
                file_name="segment_brand_analysis.csv",
                use_container_width=True
            )
    
    with col3:
        if data_files['year_trends'] is not None:
            st.download_button(
                "📥 Download Year Trends",
                data_files['year_trends'].to_csv(index=False).encode("utf-8"),
                file_name="segment_year_trends.csv",
                use_container_width=True
            )
    
    # Key Insights
    st.markdown("### 💡 Key Market Insights")
    
    insights = []
    insight_types = []
    budget_data = enhanced_df[enhanced_df['Segment'] == 'Budget'].iloc[0] if len(enhanced_df[enhanced_df['Segment'] == 'Budget']) > 0 else None
    midrange_data = enhanced_df[enhanced_df['Segment'] == 'Mid-range'].iloc[0] if len(enhanced_df[enhanced_df['Segment'] == 'Mid-range']) > 0 else None
    luxury_data = enhanced_df[enhanced_df['Segment'] == 'Luxury'].iloc[0] if len(enhanced_df[enhanced_df['Segment'] == 'Luxury']) > 0 else None
    
    if budget_data is not None:
        insights.append(f"🟢 **Budget Segment** dominates with {budget_data['Percentage']} market share, averaging ${budget_data['Avg_Price($)']:,.0f}")
        insight_types.append("success")
    
    if midrange_data is not None and luxury_data is not None:
        price_diff = luxury_data['Avg_Price($)'] - midrange_data['Avg_Price($)']
        insights.append(f"💰 **Luxury premium** averages ${price_diff:,.0f} more than mid-range vehicles")
        insight_types.append("warning")
    
    if budget_data is not None and luxury_data is not None:
        year_diff = luxury_data['Avg_Year'] - budget_data['Avg_Year']
        insights.append(f"📅 **Luxury cars** are on average {year_diff:.1f} years newer than budget vehicles")
        insight_types.append("info")
    
    for insight, insight_type in zip(insights, insight_types):
        st.markdown(f"""
        <div class="insight-box {insight_type}">
            <p style="margin: 0; font-size: 1.1rem; font-weight: 600; color: #1f2937; line-height: 1.4;">{insight}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ---------------- Interactive Segment Explorer ----------------
    st.markdown("## 🔍 Interactive Segment Explorer")
    
    # Try to load the actual car data for exploration
    try:
        this_file = Path(__file__).resolve()
        candidates = [
            this_file.parent.parent,   # <repo>/
            this_file.parent,          # <repo>/app/
            Path.cwd(),                # current working dir
        ]
        
        car_data = None
        for root in candidates:
            train_path = root / "data" / "processed" / "train.csv"
            if train_path.exists():
                car_data = pd.read_csv(train_path)
                break
        
        if car_data is not None:
            # Add segment predictions to car data
            import pickle
            import json
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            
            # Recreate clustering (since we can't load the pickle file)
            features = ['Mileage(km)', 'Year', 'Horsepower', 'EngineSize(L)']
            X = car_data[features]
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Create price-based labels
            car_data['Cluster'] = clusters
            cluster_price_medians = car_data.groupby('Cluster')['Price($)'].median().sort_values()
            price_rank = cluster_price_medians.rank().astype(int)
            
            label_map = {}
            for cluster, rank in price_rank.items():
                if rank == 1:
                    label_map[cluster] = "Budget"
                elif rank == 2:
                    label_map[cluster] = "Mid-range"
                else:
                    label_map[cluster] = "Luxury"
            
            car_data['Segment'] = car_data['Cluster'].map(label_map)
            
            # Segment selector
            col1, col2 = st.columns([1, 3])
            
            with col1:
                selected_segment = st.selectbox(
                    "🎯 Select Market Segment",
                    options=['All Segments'] + list(enhanced_df['Segment'].unique()),
                    help="Choose a specific segment to explore detailed car listings"
                )
            
            with col2:
                if selected_segment != 'All Segments':
                    segment_info = enhanced_df[enhanced_df['Segment'] == selected_segment].iloc[0]
                    st.info(f"**{selected_segment}** • {segment_info['Count']:,} cars • Avg: ${segment_info['Avg_Price($)']:,.0f} • Range: ${segment_info['Min_Price($)']:,.0f} - ${segment_info['Max_Price($)']:,.0f}")
            
            # Filter data based on selection
            if selected_segment == 'All Segments':
                filtered_data = car_data
            else:
                filtered_data = car_data[car_data['Segment'] == selected_segment]
            
            # Additional filters
            st.markdown("#### 🔧 Advanced Filters")
            filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
            
            with filter_col1:
                price_range = st.slider(
                    "Price Range ($)",
                    min_value=int(filtered_data['Price($)'].min()),
                    max_value=int(filtered_data['Price($)'].max()),
                    value=(int(filtered_data['Price($)'].min()), int(filtered_data['Price($)'].max())),
                    step=1000
                )
            
            with filter_col2:
                year_range = st.slider(
                    "Year Range",
                    min_value=int(filtered_data['Year'].min()),
                    max_value=int(filtered_data['Year'].max()),
                    value=(int(filtered_data['Year'].min()), int(filtered_data['Year'].max()))
                )
            
            with filter_col3:
                selected_brands = st.multiselect(
                    "Brands",
                    options=sorted(filtered_data['Brand'].unique()),
                    default=[],
                    help="Leave empty to show all brands"
                )
            
            with filter_col4:
                selected_body_types = st.multiselect(
                    "Body Types",
                    options=sorted(filtered_data['BodyType'].unique()),
                    default=[],
                    help="Leave empty to show all body types"
                )
            
            # Apply filters
            display_data = filtered_data[
                (filtered_data['Price($)'] >= price_range[0]) &
                (filtered_data['Price($)'] <= price_range[1]) &
                (filtered_data['Year'] >= year_range[0]) &
                (filtered_data['Year'] <= year_range[1])
            ]
            
            if selected_brands:
                display_data = display_data[display_data['Brand'].isin(selected_brands)]
            
            if selected_body_types:
                display_data = display_data[display_data['BodyType'].isin(selected_body_types)]
            
            # Display filtered results
            st.markdown(f"#### 📋 Results: {len(display_data):,} cars found")
            
            if len(display_data) > 0:
                # Summary stats for filtered data
                summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                
                with summary_col1:
                    st.metric("Average Price", f"${display_data['Price($)'].mean():,.0f}")
                
                with summary_col2:
                    st.metric("Average Year", f"{display_data['Year'].mean():.1f}")
                
                with summary_col3:
                    st.metric("Most Common Brand", display_data['Brand'].mode().iloc[0] if not display_data['Brand'].mode().empty else "N/A")
                
                with summary_col4:
                    st.metric("Most Common Body Type", display_data['BodyType'].mode().iloc[0] if not display_data['BodyType'].mode().empty else "N/A")
                
                # Display options for the table
                show_cols = st.multiselect(
                    "Select columns to display:",
                    options=['Brand', 'Model', 'Year', 'Price($)', 'Mileage(km)', 'EngineSize(L)', 
                            'Horsepower', 'FuelType', 'BodyType', 'Transmission', 'Color', 'City', 'Segment'],
                    default=['Brand', 'Model', 'Year', 'Price($)', 'BodyType', 'Segment']
                )
                
                if show_cols:
                    # Sample the data if too large
                    if len(display_data) > 1000:
                        sample_size = st.slider("Sample size (for performance)", 100, 1000, 500)
                        display_sample = display_data.sample(n=sample_size, random_state=42)
                        st.warning(f"Showing {sample_size} random samples from {len(display_data):,} total results")
                    else:
                        display_sample = display_data
                    
                    st.dataframe(
                        display_sample[show_cols].sort_values('Price($)', ascending=False),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Download filtered results
                    st.download_button(
                        "📥 Download Filtered Results",
                        display_sample.to_csv(index=False).encode("utf-8"),
                        file_name=f"{selected_segment.lower().replace(' ', '_')}_segment_cars.csv",
                        use_container_width=True
                    )
            else:
                st.warning("No cars found with the selected filters. Try adjusting your criteria.")
        
        else:
            st.warning("Car data not available for interactive exploration. The enhanced analysis shows segment summaries only.")
            
    except Exception as e:
        st.error(f"Error loading car data for exploration: {e}")
        st.info("Interactive exploration requires access to the processed training data.")
    
    # ---------------- Market Trends & Insights ----------------
    if data_files['year_trends'] is not None:
        st.markdown("## 📈 Market Trends & Evolution")
        
        year_trends_df = data_files['year_trends']
        
        # Clean and prepare year trend data
        if not year_trends_df.empty:
            # Extract year information for better plotting
            year_trends_df['Year_Start'] = year_trends_df['Year_Range'].str.extract(r'\(([\d.]+),').astype(float)
            year_trends_df['Year_End'] = year_trends_df['Year_Range'].str.extract(r', ([\d.]+)\]').astype(float)
            year_trends_df['Year_Mid'] = (year_trends_df['Year_Start'] + year_trends_df['Year_End']) / 2
            
            # Price evolution over time
            price_trend_fig = px.line(
                year_trends_df,
                x='Year_Mid',
                y='Avg_Price($)',
                color='Segment',
                title='Average Price Evolution by Segment Over Time',
                markers=True,
                color_discrete_map={
                    'Budget': '#10b981',
                    'Mid-range': '#f59e0b',
                    'Luxury': '#3b82f6'
                }
            )
            price_trend_fig.update_layout(height=400)
            st.plotly_chart(price_trend_fig, use_container_width=True)
            
            # Market share evolution
            market_evolution_fig = px.bar(
                year_trends_df,
                x='Year_Mid',
                y='Count',
                color='Segment',
                title='Market Volume by Segment Over Time',
                color_discrete_map={
                    'Budget': '#10b981',
                    'Mid-range': '#f59e0b',
                    'Luxury': '#3b82f6'
                }
            )
            market_evolution_fig.update_layout(height=400)
            st.plotly_chart(market_evolution_fig, use_container_width=True)
            
            # Trend insights
            st.markdown("### 📊 Trend Analysis")
            
            trend_insights = []
            
            # Calculate price growth rates by segment
            for segment in ['Budget', 'Mid-range', 'Luxury']:
                segment_data = year_trends_df[year_trends_df['Segment'] == segment].sort_values('Year_Mid')
                if len(segment_data) >= 2:
                    earliest_price = segment_data.iloc[0]['Avg_Price($)']
                    latest_price = segment_data.iloc[-1]['Avg_Price($)']
                    growth_rate = ((latest_price - earliest_price) / earliest_price) * 100
                    
                    if growth_rate > 0:
                        trend_insights.append(f"📈 **{segment}** segment prices increased by {growth_rate:.1f}% over time")
                    else:
                        trend_insights.append(f"📉 **{segment}** segment prices decreased by {abs(growth_rate):.1f}% over time")
            
            # Volume trends
            total_by_year = year_trends_df.groupby('Year_Mid')['Count'].sum().reset_index()
            if len(total_by_year) >= 2:
                earliest_vol = total_by_year.iloc[0]['Count']
                latest_vol = total_by_year.iloc[-1]['Count']
                vol_change = ((latest_vol - earliest_vol) / earliest_vol) * 100
                
                if vol_change > 0:
                    trend_insights.append(f"📊 Overall market volume grew by {vol_change:.1f}% in recent years")
                else:
                    trend_insights.append(f"📊 Overall market volume declined by {abs(vol_change):.1f}% in recent years")
            
            for insight in trend_insights:
                st.markdown(f"""
                <div class="insight-box info">
                    <p style="margin: 0; font-size: 1.1rem; font-weight: 600; color: #1f2937; line-height: 1.4;">{insight}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # ---------------- Predictive Insights & Recommendations ----------------
    st.markdown("## 🔮 Market Predictions & Recommendations")
    
    prediction_col1, prediction_col2 = st.columns(2)
    
    with prediction_col1:
        st.markdown("""
        <div class="insight-box success">
            <h4 style="margin: 0 0 1rem 0; color: #1f2937; font-weight: 700;">🎯 Investment Opportunities</h4>
            <ul style="margin: 0; padding-left: 1.2rem; color: #374151; font-weight: 500;">
                <li><strong>Budget Segment:</strong> High volume, consistent demand - ideal for volume sales</li>
                <li><strong>Mid-range:</strong> Balanced market with growth potential</li>
                <li><strong>Luxury:</strong> Higher margins but smaller market - premium positioning</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with prediction_col2:
        st.markdown("""
        <div class="insight-box warning">
            <h4 style="margin: 0 0 1rem 0; color: #1f2937; font-weight: 700;">🚀 Market Opportunities</h4>
            <ul style="margin: 0; padding-left: 1.2rem; color: #374151; font-weight: 500;">
                <li><strong>Electric Vehicles:</strong> Growing trend across all segments</li>
                <li><strong>SUV Popularity:</strong> Increasing demand in mid-range and luxury</li>
                <li><strong>Feature-rich Budget:</strong> More options in affordable vehicles</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Segment comparison matrix
    if enhanced_df is not None and len(enhanced_df) >= 3:
        st.markdown("### 🔄 Segment Comparison Matrix")
        
        comparison_metrics = ['Avg_Price($)', 'Avg_Year', 'Avg_Horsepower', 'Avg_EngineSize(L)']
        available_metrics = [m for m in comparison_metrics if m in enhanced_df.columns]
        
        if available_metrics:
            comparison_data = enhanced_df[['Segment'] + available_metrics].set_index('Segment')
            
            # Create a heatmap-style comparison
            fig_comparison = px.imshow(
                comparison_data.T,
                labels=dict(x="Segment", y="Metrics", color="Value"),
                x=comparison_data.index,
                y=comparison_data.columns,
                color_continuous_scale="RdYlBu_r",
                title="Segment Characteristics Heatmap"
            )
            fig_comparison.update_layout(height=400)
            st.plotly_chart(fig_comparison, use_container_width=True)

else:
    # Fallback to original data if enhanced profile not available
    if data_files['original_profile'] is not None:
        df = data_files['original_profile']
        st.warning("Enhanced profile not found. Showing basic segment data. Run the enhanced analysis script to get full features.")
        
        st.subheader("Basic Segment Profile")
        st.dataframe(df, use_container_width=True)
        
        # Basic visualization if possible
        if 'Segment' in df.columns:
            numeric_cols = [c for c in df.columns if c != 'Segment' and pd.api.types.is_numeric_dtype(df[c])]
            if numeric_cols:
                st.subheader("Basic Comparison")
                chart = alt.Chart(df.melt(id_vars='Segment', value_vars=numeric_cols)).mark_bar().encode(
                    x='variable:N',
                    y='value:Q',
                    color='Segment:N',
                    column='Segment:N'
                )
                st.altair_chart(chart, use_container_width=True)
    else:
        st.error("""
        📊 **No segment data found!**
        
        To see market segment analysis, please:
        1. Run the enhanced segment analysis script: `python enhanced_segment_analysis.py`
        2. Or check that clustering analysis has been completed in your modeling notebook
        3. Expected files:
           - `reports/results/enhanced_cluster_profile.csv`
           - `reports/results/segment_brand_analysis.csv`
           - `reports/results/segment_year_trends.csv`
        """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.9rem; margin-top: 2rem;">
    <p><strong>Enhanced Market Segments Analysis</strong></p>
    <p>Segments are determined using K-Means clustering on vehicle characteristics and labeled by price tiers.<br>
    Analysis includes comprehensive statistics, brand preferences, and market trends.</p>
</div>
""", unsafe_allow_html=True)
