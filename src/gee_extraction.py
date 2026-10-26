"""
UrbanCool-AI — GEE Extraction Module
Extracts Land Surface Temperature (LST) from Landsat 8 for Chennai.
"""

import ee
from src.auth import initialize_gee
from src.utils import (
    get_chennai_region, get_chennai_feature_collection,
    cloud_mask_landsat8, download_ee_image,
    LANDSAT_COLLECTION, DATE_START, DATE_END
)


def extract_lst():
    """
    Extract Land Surface Temperature (LST) from Landsat 8 Collection 2 Level 2.
    
    ST_B10 conversion to Celsius:
      Celsius = ST_B10 * 0.00341802 + 149.0 - 273.15
    
    Returns:
        tuple: (lst_image, region) - EE image and region geometry
    """
    ee_instance = initialize_gee()

    region = get_chennai_region()

    print(f"📡 Filtering Landsat 8 for Chennai ({DATE_START} to {DATE_END})...")

    collection = (
        ee.ImageCollection(LANDSAT_COLLECTION)
        .filterDate(DATE_START, DATE_END)
        .filterBounds(region)
        .map(cloud_mask_landsat8)
    )

    count = collection.size().getInfo()
    print(f"📊 Found {count} images after cloud masking")

    if count == 0:
        print("⚠️ No images found! Check date range and region.")
        return None, region

    # Compute mean LST in Celsius
    lst = (
        collection
        .select("ST_B10")
        .mean()
        .multiply(0.00341802)
        .add(149.0)
        .subtract(273.15)
        .rename("LST")
    )

    # Clip to region
    lst = lst.clip(region)

    print("🌡️ LST computation complete.")
    return lst, region


def export_lst():
    """Extract LST and save to outputs/lst.tif."""
    lst, region = extract_lst()
    if lst is None:
        print("❌ No LST data to export.")
        return None

    filepath = download_ee_image(lst, region, "lst.tif", bands=["LST"])
    return filepath


def get_landsat_composite():
    """
    Get a full Landsat 8 composite with all surface reflectance bands.
    Used by other modules (NDVI, Cooling Index, etc.)
    
    Returns:
        tuple: (composite_image, region)
    """
    ee_instance = initialize_gee()
    region = get_chennai_region()

    collection = (
        ee.ImageCollection(LANDSAT_COLLECTION)
        .filterDate(DATE_START, DATE_END)
        .filterBounds(region)
        .map(cloud_mask_landsat8)
    )

    # Apply surface reflectance scale factors
    def apply_scale_factors(image):
        optical = image.select("SR_B.*").multiply(0.0000275).add(-0.2)
        thermal = image.select("ST_B.*").multiply(0.00341802).add(149.0)
        return image.addBands(optical, overwrite=True).addBands(thermal, overwrite=True)

    collection = collection.map(apply_scale_factors)
    composite = collection.mean().clip(region)

    return composite, region


if __name__ == "__main__":
    filepath = export_lst()
    if filepath:
        print(f"✅ LST exported to: {filepath}")
