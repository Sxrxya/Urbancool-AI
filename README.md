# 🏙️ UrbanCool-AI — Chennai UHI Mitigation Planner

**AI-Based Urban Heat Island Mitigation Planner for Chennai, India**

Uses Landsat 8 satellite data via Google Earth Engine to detect urban heat islands, identify tree-planting priorities, simulate cooling impacts, and rank optimal plantation zones.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.9+
- Google Earth Engine service account ([setup guide](https://developers.google.com/earth-engine/guides/service_account))
- Service account JSON key file

### 2. Setup
```bash
# Clone / navigate to project
cd UrbanCool-AI

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file in the project root:
```env
GEE_PROJECT_ID=your-project-id
GEE_SERVICE_ACCOUNT=your-service-account@your-project.iam.gserviceaccount.com
GEE_KEY_PATH=service-account.json
```

### 4. Test Connection
```bash
python test_connection.py
```

### 5. Run Analysis Pipeline
```bash
# Step-by-step execution
python -m src.gee_extraction        # Extract LST
python -m src.ndvi                   # Compute NDVI
python -m src.heat_classification    # Classify heat zones
python -m src.cooling_index          # Compute cooling potential
python -m src.tree_need_analysis     # Identify tree planting needs
python -m src.ml_model               # Train ML model & rank zones
python -m src.scenario_simulation    # Run cooling scenarios
python -m src.reporting              # Generate PDF report
```

### 6. Launch Dashboard
```bash
streamlit run app/dashboard.py
```

---

## 📂 Project Structure

```
UrbanCool-AI/
├── app/
│   └── dashboard.py          # Streamlit interactive dashboard
├── src/
│   ├── auth.py               # GEE authentication
│   ├── gee_extraction.py     # Satellite data extraction
│   ├── ndvi.py               # NDVI computation
│   ├── heat_classification.py # Heat zone classification
│   ├── cooling_index.py      # Cooling potential index
│   ├── tree_need_analysis.py # Tree planting urgency analysis
│   ├── ml_model.py           # ML model & plantation ranking
│   ├── scenario_simulation.py # Canopy increase simulation
│   ├── reporting.py          # PDF report generation
│   └── utils.py              # Shared utilities
├── data/raw/                 # Raw data storage
├── outputs/                  # Generated outputs
├── models/                   # Trained models
├── test_connection.py        # GEE connection test
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (not tracked)
└── README.md                 # This file
```

---

## 📊 Outputs

| Output | Description |
|--------|-------------|
| `outputs/lst.tif` | Land Surface Temperature map |
| `outputs/ndvi.tif` | NDVI vegetation index map |
| `outputs/heat_classification.tif` | Heat zone classification (5 classes) |
| `outputs/cooling_index.tif` | Cooling potential index (0–1) |
| `outputs/tree_need_map.tif` | Tree planting priority map (3 levels) |
| `outputs/top_plantation_spots.csv` | Top 200 plantation zones |
| `outputs/top_plantation_spots.geojson` | Plantation zones (GeoJSON) |
| `outputs/model_metrics.json` | ML model performance |
| `outputs/scenario_results.json` | Simulation results |
| `outputs/urban_heat_report.pdf` | Complete analysis report |

---

## 🔬 Methodology

- **Satellite Source:** Landsat 8 Collection 2 Level 2
- **Resolution:** 30 meters
- **Period:** April–July 2025
- **LST:** Thermal Band ST_B10 → Celsius conversion
- **NDVI:** (NIR - Red) / (NIR + Red)
- **Heat Classification:** Relative to mean LST (Cool/Moderate/Warm/Hot/Extreme)
- **Cooling Index:** 50% Tree Canopy + 25% NDVI + 15% Albedo + 10% Water Proximity
- **ML Model:** RandomForestRegressor for ΔT prediction
- **Scenario:** Differential cooling model per heat zone class

---

## 📜 License

Research use only. Not for commercial deployment.
