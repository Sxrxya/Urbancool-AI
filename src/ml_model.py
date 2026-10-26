"""
UrbanCool-AI — Machine Learning Model Module
Trains a RandomForestRegressor to predict cooling potential (ΔT) from pixel features.
Also handles plantation zone ranking.
"""

import ee
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from src.auth import initialize_gee
from src.utils import (
    get_chennai_region, MODELS_DIR, OUTPUT_DIR, ensure_dirs,
    LANDSAT_COLLECTION, DATE_START, DATE_END, CHENNAI_BBOX
)


def prepare_dataset():
    """
    Prepare pixel-level dataset by sampling features from Earth Engine.
    
    Features:
        - LST (Land Surface Temperature)
        - NDVI (Vegetation Index)
        - Distance to coast (proxy)
        - Distance to road (proxy from NDBI)
        - Impervious proxy (NDBI)
    
    Target:
        - Simulated ΔT from cooling index
    
    Returns:
        pandas DataFrame with sampled features and target
    """
    ee_inst = initialize_gee()
    region = get_chennai_region()

    print("📊 Preparing pixel dataset for ML training...")

    # Get Landsat composite
    from src.gee_extraction import get_landsat_composite
    composite, _ = get_landsat_composite()

    # LST — use cloud-masked collection (same as gee_extraction)
    from src.utils import cloud_mask_landsat8
    lst = (
        ee.ImageCollection(LANDSAT_COLLECTION)
        .filterDate(DATE_START, DATE_END)
        .filterBounds(region)
        .map(cloud_mask_landsat8)
        .select("ST_B10")
        .mean()
        .multiply(0.00341802).add(149.0).subtract(273.15)
        .rename("LST")
    ).clip(region)

    # Mask unrealistic LST values (Chennai range: 15–60°C)
    lst = lst.updateMask(lst.gt(15).And(lst.lt(60)))

    # NDVI
    ndvi = composite.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")

    # NDBI (Built-up Index — impervious proxy)
    ndbi = composite.normalizedDifference(["SR_B6", "SR_B5"]).rename("NDBI")

    # Distance to coast proxy (using distance from water)
    try:
        water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
        water_mask = water.gt(50).selfMask()
        dist_coast = (
            water_mask.fastDistanceTransform().sqrt().multiply(30)
            .divide(10000).min(1)
            .rename("DIST_COAST")
        )
    except Exception:
        dist_coast = ee.Image(0.5).rename("DIST_COAST")

    # Road proximity proxy (high NDBI = likely near roads)
    road_proxy = ndbi.max(0).min(1).rename("ROAD_PROXY")

    # Impervious surface proxy
    impervious = ndbi.max(0).rename("IMPERVIOUS")

    # Cooling Index (simplified for target generation)
    tree_canopy = (
        ee.Image(0)
        .where(ndvi.gte(0.1).And(ndvi.lt(0.3)), 0.2)
        .where(ndvi.gte(0.3).And(ndvi.lt(0.6)), 0.6)
        .where(ndvi.gte(0.6), 1.0)
    )

    # Simulated ΔT target: higher cooling index = larger temperature drop potential
    # ΔT = cooling_index * max_cooling_effect (approx 5°C max)
    delta_t = tree_canopy.multiply(0.5).add(ndvi.multiply(0.3)).add(
        ee.Image(1).subtract(dist_coast).multiply(0.2)
    ).multiply(5).rename("DELTA_T")

    # Stack all bands
    stacked = (
        lst.addBands(ndvi)
        .addBands(ndbi)
        .addBands(dist_coast)
        .addBands(road_proxy)
        .addBands(impervious)
        .addBands(delta_t)
    ).clip(region)

    # Sample pixels
    print("📍 Sampling pixels from Earth Engine...")
    sample = stacked.sample(
        region=region,
        scale=30,
        numPixels=5000,
        seed=42,
        geometries=True,
    )

    features = sample.getInfo()["features"]
    print(f"✅ Sampled {len(features)} pixels")

    records = []
    for f in features:
        props = f["properties"]
        coords = f["geometry"]["coordinates"]
        records.append({
            "Longitude": coords[0],
            "Latitude": coords[1],
            "LST": props.get("LST", np.nan),
            "NDVI": props.get("NDVI", np.nan),
            "NDBI": props.get("NDBI", np.nan),
            "DIST_COAST": props.get("DIST_COAST", np.nan),
            "ROAD_PROXY": props.get("ROAD_PROXY", np.nan),
            "IMPERVIOUS": props.get("IMPERVIOUS", np.nan),
            "DELTA_T": props.get("DELTA_T", np.nan),
        })

    df = pd.DataFrame(records).dropna()
    print(f"📋 Clean dataset: {len(df)} samples")
    return df


def train_model(df=None):
    """
    Train a RandomForestRegressor to predict ΔT.
    
    Returns:
        tuple: (model, metrics_dict, feature_importances)
    """
    if df is None:
        df = prepare_dataset()

    ensure_dirs()

    feature_cols = ["LST", "NDVI", "NDBI", "DIST_COAST", "ROAD_PROXY", "IMPERVIOUS"]
    target_col = "DELTA_T"

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🤖 Training RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    importance = dict(zip(feature_cols, model.feature_importances_.tolist()))

    metrics = {
        "r2_score": round(r2, 4),
        "rmse": round(rmse, 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_importance": importance,
    }

    print(f"📊 Model Performance:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   RMSE:     {rmse:.4f}")
    print(f"   Feature Importance: {importance}")

    # Save model
    model_path = os.path.join(MODELS_DIR, "rf_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"💾 Model saved: {model_path}")

    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"📊 Metrics saved: {metrics_path}")

    return model, metrics, importance


def rank_plantation_zones(df=None, model=None):
    """
    Identify and rank top 200 plantation zones.
    
    Filters: Hot/Extreme heat (LST > mean+2) AND NDVI < 0.3
    Ranks by predicted ΔT (highest cooling potential first).
    
    Exports:
        outputs/top_plantation_spots.csv
        outputs/top_plantation_spots.geojson
    """
    if df is None:
        df = prepare_dataset()

    if model is None:
        model_path = os.path.join(MODELS_DIR, "rf_model.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        else:
            model, _, _ = train_model(df)

    ensure_dirs()

    feature_cols = ["LST", "NDVI", "NDBI", "DIST_COAST", "ROAD_PROXY", "IMPERVIOUS"]

    # Determine heat thresholds
    mean_lst = df["LST"].mean()
    hot_threshold = mean_lst + 2  # Hot or Extreme

    # Filter candidate zones
    candidates = df[
        (df["LST"] >= hot_threshold) &
        (df["NDVI"] < 0.3)
    ].copy()

    print(f"🔍 Found {len(candidates)} candidate plantation zones (Hot+ & low NDVI)")

    if len(candidates) == 0:
        print("⚠️ No candidate zones found. Relaxing criteria...")
        candidates = df[df["NDVI"] < 0.4].copy()
        print(f"   Found {len(candidates)} zones with relaxed criteria")

    # Predict ΔT
    candidates["Predicted_Delta_T"] = model.predict(candidates[feature_cols].values)

    # Assign tree need level
    candidates["Tree_Need_Level"] = 1  # Default: Moderate
    candidates.loc[
        (candidates["LST"] >= mean_lst + 4) & (candidates["NDVI"] < 0.2),
        "Tree_Need_Level"
    ] = 2  # High
    candidates.loc[
        candidates["LST"] < mean_lst + 2,
        "Tree_Need_Level"
    ] = 0  # Low

    # Rename for output
    candidates = candidates.rename(columns={
        "LST": "Current_LST",
    })

    # Select top 200 by predicted cooling
    top_zones = (
        candidates
        .nlargest(min(200, len(candidates)), "Predicted_Delta_T")
        .reset_index(drop=True)
    )

    output_cols = [
        "Latitude", "Longitude", "Current_LST", "NDVI",
        "Predicted_Delta_T", "Tree_Need_Level"
    ]

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "top_plantation_spots.csv")
    top_zones[output_cols].to_csv(csv_path, index=False)
    print(f"📍 CSV saved: {csv_path}")

    # Save GeoJSON
    try:
        import geopandas as gpd
        from shapely.geometry import Point

        geometry = [Point(xy) for xy in zip(top_zones["Longitude"], top_zones["Latitude"])]
        gdf = gpd.GeoDataFrame(top_zones[output_cols], geometry=geometry, crs="EPSG:4326")
        geojson_path = os.path.join(OUTPUT_DIR, "top_plantation_spots.geojson")
        gdf.to_file(geojson_path, driver="GeoJSON")
        print(f"🗺️ GeoJSON saved: {geojson_path}")
    except Exception as e:
        print(f"⚠️ GeoJSON export failed: {e}")

    return top_zones


if __name__ == "__main__":
    print("=" * 60)
    print("UrbanCool-AI — ML Model Training & Plantation Ranking")
    print("=" * 60)

    df = prepare_dataset()
    model, metrics, importance = train_model(df)
    top_zones = rank_plantation_zones(df, model)
    print(f"\n✅ Top {len(top_zones)} plantation zones identified!")
    print(top_zones[["Latitude", "Longitude", "Current_LST", "NDVI", "Predicted_Delta_T"]].head(10))
