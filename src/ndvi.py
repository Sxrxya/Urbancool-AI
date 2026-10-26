"""
UrbanCool-AI — NDVI Module
Computes Normalized Difference Vegetation Index from Landsat 8.
"""

import ee
from src.auth import initialize_gee
from src.utils import get_chennai_region, download_ee_image
from src.gee_extraction import get_landsat_composite


def compute_ndvi():
    """
    Compute NDVI from Landsat 8 surface reflectance bands.
    NDVI = (NIR - Red) / (NIR + Red) = (SR_B5 - SR_B4) / (SR_B5 + SR_B4)
    
    Returns:
        tuple: (ndvi_image, region)
    """
    composite, region = get_landsat_composite()

    ndvi = composite.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")

    print("🌿 NDVI computation complete.")
    return ndvi, region


def classify_vegetation(ndvi):
    """
    Classify vegetation based on NDVI thresholds.
    
    Classes:
        4 = Dense (NDVI > 0.6)
        3 = Moderate (0.3 - 0.6)
        2 = Sparse (0.1 - 0.3)
        1 = Built-up (< 0.1)
    
    Returns:
        ee.Image with vegetation classes
    """
    classified = (
        ndvi.where(ndvi.lt(0.1), 1)   # Built-up
        .where(ndvi.gte(0.1).And(ndvi.lt(0.3)), 2)   # Sparse
        .where(ndvi.gte(0.3).And(ndvi.lt(0.6)), 3)   # Moderate
        .where(ndvi.gte(0.6), 4)   # Dense
    ).rename("VEG_CLASS")

    print("🌳 Vegetation classification complete.")
    return classified


def export_ndvi():
    """Compute NDVI and save to outputs/ndvi.tif."""
    ndvi, region = compute_ndvi()
    filepath = download_ee_image(ndvi, region, "ndvi.tif", bands=["NDVI"])
    return filepath


if __name__ == "__main__":
    filepath = export_ndvi()
    if filepath:
        print(f"✅ NDVI exported to: {filepath}")
