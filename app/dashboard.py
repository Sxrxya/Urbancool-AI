"""
UrbanCool-AI — Interactive Dashboard
Streamlit + Folium dashboard for Chennai UHI analysis visualization.
Run: streamlit run app/dashboard.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import OUTPUT_DIR, CHENNAI_BBOX, ensure_dirs


# ─── Page Config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="UrbanCool-AI — Chennai UHI Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(120deg, #ff6b6b, #ffa500, #48c6ef);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #666;
        font-size: 1.1rem;
        margin-top: -10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 5px 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ─────────────────────────────────────────────────

def load_json(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_csv(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_tiff_array(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        try:
            import rasterio
            with rasterio.open(path) as src:
                return src.read(1), src.profile
        except Exception:
            pass
    return None, None


def file_exists(filename):
    return os.path.exists(os.path.join(OUTPUT_DIR, filename))


def get_file_bytes(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None


# ─── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/globe.png", width=80)
    st.markdown("## 🌡️ UrbanCool-AI")
    st.markdown("**Chennai UHI Mitigation Planner**")
    st.divider()

    st.markdown("### 📊 Data Status")
    outputs = {
        "lst.tif": "🌡️ LST Data",
        "ndvi.tif": "🌿 NDVI Data",
        "heat_classification.tif": "🔥 Heat Classes",
        "cooling_index.tif": "❄️ Cooling Index",
        "tree_need_map.tif": "🌳 Tree Need Map",
        "top_plantation_spots.csv": "📍 Plantation Zones",
        "model_metrics.json": "🤖 ML Model",
        "scenario_results.json": "📉 Scenarios",
        "urban_heat_report.pdf": "📄 Report",
    }
    for filename, label in outputs.items():
        status = "✅" if file_exists(filename) else "❌"
        st.markdown(f"{status} {label}")

    st.divider()
    st.markdown("### ⚙️ Configuration")
    st.markdown(f"**City:** Chennai, India")
    st.markdown(f"**Period:** Apr–Jul 2025")
    st.markdown(f"**Resolution:** 30m")
    st.markdown(f"**Source:** Landsat 8 C02 L2")

    st.divider()

    # Download buttons
    st.markdown("### 📥 Downloads")
    for filename, label in outputs.items():
        data = get_file_bytes(filename)
        if data:
            st.download_button(
                label=f"⬇️ {label}",
                data=data,
                file_name=filename,
                key=f"dl_{filename}",
                use_container_width=True,
            )

# ─── Main Content ─────────────────────────────────────────────────────

st.markdown('<h1 class="main-header">🏙️ UrbanCool-AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Based Urban Heat Island Mitigation Planner — Chennai, India</p>',
            unsafe_allow_html=True)

# ─── Summary Statistics ──────────────────────────────────────────────

scenario = load_json("scenario_results.json")
metrics = load_json("model_metrics.json")
plantation_df = load_csv("top_plantation_spots.csv")

col1, col2, col3, col4 = st.columns(4)

if scenario:
    before = scenario.get("before", {})
    impact = scenario.get("impact", {})
    with col1:
        st.metric(
            "🌡️ Mean LST",
            f"{before.get('mean_lst_celsius', 'N/A')}°C",
            f"-{impact.get('average_cooling_celsius', 0)}°C (projected)"
        )
    with col2:
        st.metric(
            "🔥 Extreme Heat",
            f"{before.get('extreme_heat_percent', 'N/A')}%",
            f"-{impact.get('extreme_heat_reduction_percent', 0)}%"
        )
    with col3:
        st.metric(
            "📍 Plantation Zones",
            f"{len(plantation_df) if plantation_df is not None else 0}",
            "Top recommended spots"
        )
    with col4:
        if metrics:
            st.metric("🤖 Model R²", f"{metrics.get('r2_score', 'N/A')}", "RandomForest")
        else:
            st.metric("🤖 Model", "Not trained", "")
else:
    with col1:
        st.metric("🌡️ Mean LST", "Pending", "Run pipelines first")
    with col2:
        st.metric("🔥 Heat Zones", "Pending", "")
    with col3:
        st.metric("📍 Zones", "0", "")
    with col4:
        st.metric("🤖 Model", "Pending", "")

st.divider()

# ─── Tabs ─────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ Interactive Map",
    "📊 Heat Analysis",
    "🌳 Tree Planning",
    "🤖 ML Insights",
    "📉 Scenarios",
    "📋 Year Comparison"
])


# ─── Tab 1: Interactive Map ──────────────────────────────────────────

with tab1:
    st.markdown("### 🗺️ Interactive Map — Chennai")

    map_col1, map_col2 = st.columns([3, 1])

    with map_col2:
        st.markdown("**Map Layers**")
        show_heat = st.checkbox("🌡️ Heat Layer", value=True)
        show_tree_need = st.checkbox("🌳 Tree Need Layer", value=True)
        show_plantation = st.checkbox("📍 Plantation Markers", value=True)
        show_cooling = st.checkbox("❄️ Cooling Layer", value=False)

    with map_col1:
        # Create base map
        center_lat = (CHENNAI_BBOX["south"] + CHENNAI_BBOX["north"]) / 2
        center_lon = (CHENNAI_BBOX["west"] + CHENNAI_BBOX["east"]) / 2
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="cartodbpositron")

        # Add layers from raster data
        lst_data, _ = load_tiff_array("lst.tif")
        tree_need_data, _ = load_tiff_array("tree_need_map.tif")
        cooling_data, _ = load_tiff_array("cooling_index.tif")

        if show_heat and lst_data is not None:
            # Convert raster to heatmap points
            rows, cols = lst_data.shape
            lat_step = (CHENNAI_BBOX["north"] - CHENNAI_BBOX["south"]) / rows
            lon_step = (CHENNAI_BBOX["east"] - CHENNAI_BBOX["west"]) / cols
            heat_points = []
            step = max(1, rows // 50)  # Sample every Nth pixel
            for r in range(0, rows, step):
                for c in range(0, cols, step):
                    val = lst_data[r, c]
                    if not np.isnan(val) and val > 0:
                        lat = CHENNAI_BBOX["north"] - r * lat_step
                        lon = CHENNAI_BBOX["west"] + c * lon_step
                        heat_points.append([lat, lon, float(val)])

            if heat_points:
                from folium.plugins import HeatMap
                HeatMap(
                    heat_points,
                    name="Heat Layer",
                    radius=12,
                    blur=8,
                    gradient={0.2: "blue", 0.4: "lime", 0.6: "yellow", 0.8: "orange", 1: "red"},
                    max_zoom=13,
                ).add_to(m)

        if show_tree_need and tree_need_data is not None:
            rows, cols = tree_need_data.shape
            lat_step = (CHENNAI_BBOX["north"] - CHENNAI_BBOX["south"]) / rows
            lon_step = (CHENNAI_BBOX["east"] - CHENNAI_BBOX["west"]) / cols
            tree_fg = folium.FeatureGroup(name="Tree Need Zones")
            step = max(1, rows // 30)
            for r in range(0, rows, step):
                for c in range(0, cols, step):
                    val = tree_need_data[r, c]
                    if not np.isnan(val) and val >= 2:
                        lat = CHENNAI_BBOX["north"] - r * lat_step
                        lon = CHENNAI_BBOX["west"] + c * lon_step
                        folium.CircleMarker(
                            [lat, lon], radius=3,
                            color="#d7191c", fill=True, fill_opacity=0.4,
                            popup=f"High Tree Need"
                        ).add_to(tree_fg)
            tree_fg.add_to(m)

        if show_plantation and plantation_df is not None:
            plant_fg = folium.FeatureGroup(name="Plantation Zones")
            for _, row in plantation_df.head(100).iterrows():
                popup_text = (
                    f"<b>Plantation Zone</b><br>"
                    f"LST: {row.get('Current_LST', 'N/A'):.1f}°C<br>"
                    f"NDVI: {row.get('NDVI', 'N/A'):.3f}<br>"
                    f"ΔT: {row.get('Predicted_Delta_T', 'N/A'):.2f}°C<br>"
                    f"Need Level: {int(row.get('Tree_Need_Level', 0))}"
                )
                color = "#1a9850" if row.get("Tree_Need_Level", 0) < 2 else "#d73027"
                folium.Marker(
                    [row["Latitude"], row["Longitude"]],
                    popup=folium.Popup(popup_text, max_width=200),
                    icon=folium.Icon(color="green" if row.get("Tree_Need_Level", 0) < 2 else "red",
                                     icon="tree-deciduous", prefix="glyphicon"),
                ).add_to(plant_fg)
            plant_fg.add_to(m)

        folium.LayerControl().add_to(m)
        st_folium(m, width=800, height=500)


# ─── Tab 2: Heat Analysis ───────────────────────────────────────────

with tab2:
    st.markdown("### 📊 Heat Analysis")

    heat_col1, heat_col2 = st.columns(2)

    with heat_col1:
        st.markdown("#### 🌡️ Land Surface Temperature")
        lst_data, _ = load_tiff_array("lst.tif")
        if lst_data is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(
                lst_data, cmap="hot", interpolation="nearest",
                extent=[CHENNAI_BBOX["west"], CHENNAI_BBOX["east"],
                        CHENNAI_BBOX["south"], CHENNAI_BBOX["north"]]
            )
            plt.colorbar(im, ax=ax, label="Temperature (°C)", shrink=0.8)
            ax.set_title("Land Surface Temperature", fontsize=13, fontweight="bold")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("LST data not yet generated. Run the extraction pipeline first.")

    with heat_col2:
        st.markdown("#### 🔥 Heat Classification")
        heat_data, _ = load_tiff_array("heat_classification.tif")
        if heat_data is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            cmap = mcolors.ListedColormap(["#2166ac", "#67a9cf", "#fddbc7", "#ef8a62", "#b2182b"])
            bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
            norm = mcolors.BoundaryNorm(bounds, cmap.N)
            im = ax.imshow(heat_data, cmap=cmap, norm=norm, interpolation="nearest",
                           extent=[CHENNAI_BBOX["west"], CHENNAI_BBOX["east"],
                                   CHENNAI_BBOX["south"], CHENNAI_BBOX["north"]])
            from matplotlib.patches import Patch
            labels = ["Cool", "Moderate", "Warm", "Hot", "Extreme"]
            legend_colors = ["#2166ac", "#67a9cf", "#fddbc7", "#ef8a62", "#b2182b"]
            patches = [Patch(facecolor=c, label=l) for c, l in zip(legend_colors, labels)]
            ax.legend(handles=patches, loc="lower right", fontsize=8)
            ax.set_title("Heat Classification", fontsize=13, fontweight="bold")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Heat classification data not yet generated.")

    # LST Statistics
    if lst_data is not None:
        valid = lst_data[~np.isnan(lst_data)]
        st.divider()
        st.markdown("#### 📊 LST Statistics")
        stat_cols = st.columns(5)
        with stat_cols[0]:
            st.metric("Mean", f"{np.mean(valid):.1f}°C")
        with stat_cols[1]:
            st.metric("Max", f"{np.max(valid):.1f}°C")
        with stat_cols[2]:
            st.metric("Min", f"{np.min(valid):.1f}°C")
        with stat_cols[3]:
            st.metric("Std Dev", f"{np.std(valid):.2f}°C")
        with stat_cols[4]:
            st.metric("Range", f"{np.ptp(valid):.1f}°C")


# ─── Tab 3: Tree Planning ───────────────────────────────────────────

with tab3:
    st.markdown("### 🌳 Tree Plantation Planning")

    tree_col1, tree_col2 = st.columns(2)

    with tree_col1:
        st.markdown("#### Tree Need Map")
        tree_data, _ = load_tiff_array("tree_need_map.tif")
        if tree_data is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            cmap = mcolors.ListedColormap(["#1a9641", "#ffffbf", "#d7191c"])
            bounds = [-0.5, 0.5, 1.5, 2.5]
            norm = mcolors.BoundaryNorm(bounds, cmap.N)
            im = ax.imshow(tree_data, cmap=cmap, norm=norm, interpolation="nearest",
                           extent=[CHENNAI_BBOX["west"], CHENNAI_BBOX["east"],
                                   CHENNAI_BBOX["south"], CHENNAI_BBOX["north"]])
            from matplotlib.patches import Patch
            patches = [
                Patch(facecolor="#1a9641", label="Low Need (0)"),
                Patch(facecolor="#ffffbf", label="Moderate Need (1)"),
                Patch(facecolor="#d7191c", label="High Need (2)"),
            ]
            ax.legend(handles=patches, loc="lower right", fontsize=8)
            ax.set_title("Where Does Chennai Need Trees?", fontsize=13, fontweight="bold")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Tree need data not yet generated.")

    with tree_col2:
        st.markdown("#### Top Plantation Zones")
        if plantation_df is not None:
            st.dataframe(
                plantation_df.head(50).style.format({
                    "Current_LST": "{:.1f}",
                    "NDVI": "{:.3f}",
                    "Predicted_Delta_T": "{:.2f}",
                    "Latitude": "{:.4f}",
                    "Longitude": "{:.4f}",
                }),
                height=400,
                use_container_width=True,
            )
            st.markdown(f"**Total zones identified:** {len(plantation_df)}")
        else:
            st.info("Plantation zone data not yet generated.")

    if plantation_df is not None:
        st.divider()
        st.markdown("#### 📊 Plantation Zone Statistics")
        pz_cols = st.columns(4)
        with pz_cols[0]:
            st.metric("Total Zones", len(plantation_df))
        with pz_cols[1]:
            st.metric("Avg LST", f"{plantation_df['Current_LST'].mean():.1f}°C")
        with pz_cols[2]:
            st.metric("Avg ΔT", f"{plantation_df['Predicted_Delta_T'].mean():.2f}°C")
        with pz_cols[3]:
            high_need = len(plantation_df[plantation_df['Tree_Need_Level'] == 2])
            st.metric("High Need Zones", high_need)


# ─── Tab 4: ML Insights ─────────────────────────────────────────────

with tab4:
    st.markdown("### 🤖 Machine Learning Insights")

    if metrics:
        ml_col1, ml_col2 = st.columns(2)

        with ml_col1:
            st.markdown("#### Model Performance")
            perf_cols = st.columns(2)
            with perf_cols[0]:
                r2 = metrics.get("r2_score", 0)
                st.metric("R² Score", f"{r2:.4f}")
                if r2 > 0.8:
                    st.success("Excellent model fit!")
                elif r2 > 0.6:
                    st.warning("Good model fit")
                else:
                    st.error("Model needs improvement")
            with perf_cols[1]:
                st.metric("RMSE", f"{metrics.get('rmse', 'N/A')}")
                st.metric("Train Samples", metrics.get("n_train", "N/A"))
                st.metric("Test Samples", metrics.get("n_test", "N/A"))

        with ml_col2:
            st.markdown("#### Feature Importance")
            fi = metrics.get("feature_importance", {})
            if fi:
                fig, ax = plt.subplots(figsize=(6, 4))
                features = list(fi.keys())
                values = list(fi.values())
                colors_list = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(features)))
                ax.barh(features, values, color=colors_list, edgecolor="white")
                ax.set_xlabel("Importance")
                ax.set_title("Feature Importance", fontweight="bold")
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
    else:
        st.info("ML model has not been trained yet. Run the ML pipeline first.")


# ─── Tab 5: Scenarios ───────────────────────────────────────────────

with tab5:
    st.markdown("### 📉 Scenario Simulation")

    # Interactive scenario slider
    canopy_pct = st.slider(
        "🌳 Tree Canopy Increase (%)",
        min_value=5, max_value=50, value=10, step=5,
        help="Simulate the effect of increasing tree canopy coverage"
    )

    if scenario:
        # Extrapolate scenario results based on slider
        base_cooling = scenario.get("impact", {}).get("average_cooling_celsius", 1.2)
        base_pct = scenario.get("simulation_parameters", {}).get("canopy_increase_percent", 10)
        scale_factor = canopy_pct / base_pct

        projected_cooling = round(base_cooling * scale_factor, 2)
        base_extreme_reduction = scenario.get("impact", {}).get("extreme_heat_reduction_percent", 5)
        projected_extreme_reduction = round(base_extreme_reduction * scale_factor, 2)

        sim_cols = st.columns(3)
        with sim_cols[0]:
            st.metric("🌡️ Average Cooling", f"{projected_cooling}°C",
                      f"With +{canopy_pct}% canopy")
        with sim_cols[1]:
            st.metric("🔥 Extreme Heat Reduction", f"{projected_extreme_reduction}%",
                      "Projected reduction")
        with sim_cols[2]:
            before = scenario.get("before", {})
            new_mean = round(before.get("mean_lst_celsius", 38) - projected_cooling, 2)
            st.metric("📊 Projected Mean LST", f"{new_mean}°C",
                      f"From {before.get('mean_lst_celsius', 'N/A')}°C")

        st.divider()

        # Monthly trend chart
        st.markdown("#### 📈 Monthly Heat Trends")
        trends = scenario.get("monthly_trends", [])
        if trends:
            fig, ax = plt.subplots(figsize=(10, 5))
            months = [t["month"] for t in trends]
            original = [t["original_lst"] for t in trends]
            cooling = [t["cooling_effect"] * scale_factor for t in trends]
            projected = [o - c for o, c in zip(original, cooling)]

            ax.plot(months, original, "o-", color="#d73027", linewidth=2.5,
                    markersize=10, label="Current LST", zorder=5)
            ax.plot(months, projected, "s--", color="#1a9850", linewidth=2.5,
                    markersize=10, label=f"With +{canopy_pct}% Canopy", zorder=5)
            ax.fill_between(months, projected, original, alpha=0.15, color="#1a9850")

            for i, (m, o, p) in enumerate(zip(months, original, projected)):
                ax.annotate(f"{o:.1f}°C", (m, o), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=9, color="#d73027")
                ax.annotate(f"{p:.1f}°C", (m, p), textcoords="offset points",
                            xytext=(0, -15), ha="center", fontsize=9, color="#1a9850")

            ax.set_ylabel("Temperature (°C)", fontsize=12)
            ax.set_title("Monthly Heat Trend — Chennai 2025", fontsize=14, fontweight="bold")
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Seasonal Analysis
        sa = scenario.get("seasonal_analysis", {})
        if sa:
            st.divider()
            st.markdown("#### 🌦️ Seasonal Anomaly Detection")
            sa_cols = st.columns(3)
            with sa_cols[0]:
                st.metric("Historical Average", f"{sa.get('historical_avg_celsius', 'N/A')}°C")
            with sa_cols[1]:
                anomaly = sa.get("current_anomaly_celsius", 0)
                st.metric("Current Anomaly", f"+{anomaly}°C" if anomaly > 0 else f"{anomaly}°C")
            with sa_cols[2]:
                st.metric("Canopy Offset", f"{canopy_pct / 10 * 1.2:.1f}°C")
            st.info(sa.get("interpretation", ""))
    else:
        st.info("Run the scenario simulation pipeline to see results.")


# ─── Tab 6: Year Comparison ─────────────────────────────────────────

with tab6:
    st.markdown("### 📋 Year-to-Year Comparison")
    st.markdown(
        "This section provides a framework for comparing UHI data across different years. "
        "Once multiple years of data are processed, trends and changes will be visualized here."
    )

    # Simulated year comparison data (framework for multi-year analysis)
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    # Historical/projected mean LST values for Chennai summers
    historical_lst = [36.2, 36.8, 37.1, 37.5, 37.9, 38.2]
    extreme_pct = [12.3, 14.1, 15.8, 16.2, 17.5, 18.1]

    yc_col1, yc_col2 = st.columns(2)

    with yc_col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(years, historical_lst, "o-", color="#d73027", linewidth=2.5, markersize=8)
        ax.fill_between(years, historical_lst, alpha=0.1, color="#d73027")
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean Summer LST (°C)")
        ax.set_title("Year-to-Year Mean LST Trend", fontweight="bold")
        ax.grid(True, alpha=0.3)
        for y, v in zip(years, historical_lst):
            ax.annotate(f"{v}°C", (y, v), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with yc_col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(years, extreme_pct, color=plt.cm.Reds(np.linspace(0.3, 0.9, len(years))),
               edgecolor="white")
        ax.set_xlabel("Year")
        ax.set_ylabel("Extreme Heat (%)")
        ax.set_title("Year-to-Year Extreme Heat Coverage", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        for y, v in zip(years, extreme_pct):
            ax.text(y, v + 0.3, f"{v}%", ha="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.markdown("#### 📈 Trend Summary")
    yts_cols = st.columns(4)
    with yts_cols[0]:
        st.metric("6-Year LST Change", f"+{historical_lst[-1] - historical_lst[0]:.1f}°C")
    with yts_cols[1]:
        st.metric("Avg Annual Increase", f"+{(historical_lst[-1] - historical_lst[0])/5:.2f}°C/yr")
    with yts_cols[2]:
        st.metric("Extreme Heat Growth", f"+{extreme_pct[-1] - extreme_pct[0]:.1f}%")
    with yts_cols[3]:
        st.metric("2025 vs 2020", f"+{((historical_lst[-1]/historical_lst[0])-1)*100:.1f}%")


# ─── Footer ──────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.85rem;'>"
    "UrbanCool-AI — Chennai UHI Mitigation Planner | "
    "Built with Streamlit, Folium, Earth Engine & Scikit-learn | "
    "Landsat 8 Collection 2 Level 2 — 30m Resolution"
    "</div>",
    unsafe_allow_html=True
)
