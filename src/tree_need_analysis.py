"""
UrbanCool-AI — Tree Need Analysis Module
Identifies areas that urgently need tree planting based on heat, vegetation, and impervious surface data.
"""

import ee
import numpy as np
from src.auth import initialize_gee
from src.utils import get_chennai_region, download_ee_image, load_tiff, save_array_as_tiff
from src.gee_extraction import extract_lst, get_landsat_composite
from src.ndvi import compute_ndvi
from src.heat_classification import classify_heat_gee


def compute_tree_need():
    """
    Compute Tree Need Index combining heat class, NDVI, and impervious surface.
    
    Tree Need Levels:
        0 = Low:       Not hot, or well-vegetated
        1 = Moderate:  Warm areas with sparse vegetation
        2 = High:      Hot/Extreme heat AND NDVI < 0.3 AND high impervious surface
    
    Returns:
        tuple: (tree_need_image, region)
    """
    # Get LST and classify heat
    lst, region = extract_lst()
    if lst is None:
        print("❌ Cannot compute tree need without LST data.")
        return None, None

    heat_class, _, mean_val = classify_heat_gee(lst, region)

    # Get NDVI
    ndvi, _ = compute_ndvi()

    # Get impervious surface proxy (low NDVI = high impervious)
    composite, _ = get_landsat_composite()
    # NDBI (Normalized Difference Built-up Index) as impervious proxy
    # NDBI = (SWIR - NIR) / (SWIR + NIR)
    ndbi = composite.normalizedDifference(["SR_B6", "SR_B5"]).rename("NDBI")
    impervious = ndbi.gt(0).rename("IMPERVIOUS")  # Positive NDBI = built-up

    # --- Tree Need Classification ---
    # Start with Low (0)
    tree_need = ee.Image(0).rename("TREE_NEED").clip(region)

    # Moderate (1): Warm areas with sparse vegetation
    moderate_cond = (
        heat_class.gte(3)  # Warm or higher
        .And(ndvi.lt(0.3))  # Sparse or built-up vegetation
    )

    # High (2): Hot/Extreme AND low NDVI AND impervious
    high_cond = (
        heat_class.gte(4)  # Hot or Extreme
        .And(ndvi.lt(0.3))  # Poor vegetation
        .And(impervious.eq(1))  # Built-up/impervious surface
    )

    tree_need = tree_need.where(moderate_cond, 1).where(high_cond, 2)
    tree_need = tree_need.rename("TREE_NEED").clip(region)

    print("🌳 Tree need analysis complete.")
    return tree_need, region


def compute_tree_need_local(lst_array, ndvi_array, heat_class_array):
    """
    Compute tree need from local numpy arrays.
    
    Returns:
        numpy array with tree need levels (0, 1, 2)
    """
    tree_need = np.zeros_like(lst_array, dtype=np.float32)

    # Moderate: Warm+ (>=3) and NDVI < 0.3
    moderate = (heat_class_array >= 3) & (ndvi_array < 0.3)
    tree_need[moderate] = 1

    # High: Hot/Extreme (>=4) and NDVI < 0.3
    high = (heat_class_array >= 4) & (ndvi_array < 0.3)
    tree_need[high] = 2

    tree_need[np.isnan(lst_array)] = np.nan
    return tree_need


def export_tree_need():
    """Compute tree need and save to outputs/tree_need_map.tif."""
    tree_need, region = compute_tree_need()
    if tree_need is None:
        print("❌ No tree need data to export.")
        return None

    filepath = download_ee_image(tree_need, region, "tree_need_map.tif", bands=["TREE_NEED"])
    return filepath


if __name__ == "__main__":
    filepath = export_tree_need()
    if filepath:
        print(f"✅ Tree need map exported to: {filepath}")
