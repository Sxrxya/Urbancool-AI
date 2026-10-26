"""
UrbanCool-AI — Cooling Index Module
Computes a composite cooling potential index for Chennai.
"""

import ee
import numpy as np
from src.auth import initialize_gee
from src.utils import get_chennai_region, download_ee_image
from src.gee_extraction import get_landsat_composite


def compute_cooling_index():
    """
    Compute Cooling Index:
        CI = 0.50 * Tree_Canopy_Proxy
           + 0.25 * NDVI
           + 0.15 * Albedo
           + 0.10 * Distance_to_Water (normalized, inverted)
    
    All components normalized to [0, 1].
    
    Returns:
        tuple: (cooling_index_image, region)
    """
    composite, region = get_landsat_composite()

    # --- NDVI ---
    ndvi = composite.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
    # Clamp to [0, 1]
    ndvi_norm = ndvi.max(0).min(1)

    # --- Tree Canopy Proxy (from NDVI thresholds) ---
    # High NDVI = high canopy
    tree_canopy = (
        ee.Image(0)
        .where(ndvi.gte(0.1).And(ndvi.lt(0.3)), 0.2)   # Sparse
        .where(ndvi.gte(0.3).And(ndvi.lt(0.6)), 0.6)   # Moderate
        .where(ndvi.gte(0.6), 1.0)                       # Dense
    ).rename("TREE_CANOPY").clip(region)

    # --- Albedo (approximate from visible bands) ---
    # Simple broadband albedo approximation
    albedo = (
        composite.select("SR_B2").multiply(0.356)  # Blue
        .add(composite.select("SR_B4").multiply(0.130))  # Red
        .add(composite.select("SR_B5").multiply(0.373))  # NIR
        .add(composite.select("SR_B6").multiply(0.085))  # SWIR1
        .add(composite.select("SR_B7").multiply(0.072))  # SWIR2
        .subtract(0.018)
    ).rename("ALBEDO").clip(region)
    # Normalize albedo to [0, 1]
    albedo_norm = albedo.max(0).min(1)

    # --- Distance to Water ---
    try:
        # JRC Global Surface Water
        water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
        water_mask = water.gt(50).selfMask()
        # Distance in meters, then normalize (max ~10km)
        dist_water = water_mask.fastDistanceTransform().sqrt().multiply(30)  # approx meters
        dist_water_norm = dist_water.divide(10000).min(1)  # Cap at 10km
        # Invert: closer to water = higher cooling
        dist_water_inv = ee.Image(1).subtract(dist_water_norm).rename("DIST_WATER").clip(region)
    except Exception:
        print("⚠️ Water distance computation failed, using 0.5 default")
        dist_water_inv = ee.Image(0.5).rename("DIST_WATER").clip(region)

    # --- Composite Cooling Index ---
    cooling_index = (
        tree_canopy.multiply(0.50)
        .add(ndvi_norm.multiply(0.25))
        .add(albedo_norm.multiply(0.15))
        .add(dist_water_inv.multiply(0.10))
    ).rename("COOLING_INDEX").clip(region)

    # Final normalization to [0, 1]
    ci_min = cooling_index.reduceRegion(
        reducer=ee.Reducer.min(), geometry=region, scale=30, maxPixels=1e9
    )
    ci_max = cooling_index.reduceRegion(
        reducer=ee.Reducer.max(), geometry=region, scale=30, maxPixels=1e9
    )

    ci_min_val = ee.Number(ci_min.get("COOLING_INDEX"))
    ci_max_val = ee.Number(ci_max.get("COOLING_INDEX"))
    ci_range = ci_max_val.subtract(ci_min_val).max(0.001)

    cooling_normalized = (
        cooling_index
        .subtract(ci_min_val)
        .divide(ci_range)
    ).rename("COOLING_INDEX").clip(region)

    print("❄️ Cooling index computation complete.")
    return cooling_normalized, region


def export_cooling_index():
    """Compute cooling index and save to outputs/cooling_index.tif."""
    ci, region = compute_cooling_index()
    filepath = download_ee_image(ci, region, "cooling_index.tif", bands=["COOLING_INDEX"])
    return filepath


if __name__ == "__main__":
    filepath = export_cooling_index()
    if filepath:
        print(f"✅ Cooling index exported to: {filepath}")
