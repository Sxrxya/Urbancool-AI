"""
UrbanCool-AI — Heat Classification Module
Classifies land surface temperature into heat zones relative to mean LST.
"""

import ee
import numpy as np
from src.auth import initialize_gee
from src.utils import get_chennai_region, download_ee_image, load_tiff, save_array_as_tiff
from src.gee_extraction import extract_lst


def classify_heat_gee(lst_image, region):
    """
    Classify heat zones from LST using relative thresholds (GEE-based).
    
    Classes:
        1 = Cool:     LST < Mean - 2°C
        2 = Moderate:  Mean - 2°C to Mean
        3 = Warm:      Mean to Mean + 2°C
        4 = Hot:       Mean + 2°C to Mean + 4°C
        5 = Extreme:   > Mean + 4°C
    
    Returns:
        tuple: (classified_image, region, mean_lst_value)
    """
    # Compute mean LST
    mean_dict = lst_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=30,
        maxPixels=1e9
    )
    mean_lst = ee.Number(mean_dict.get("LST"))
    mean_val = mean_lst.getInfo()
    print(f"🌡️ Mean LST: {mean_val:.2f}°C")

    # Classification thresholds
    cool_thresh = mean_lst.subtract(2)
    warm_thresh = mean_lst.add(2)
    hot_thresh = mean_lst.add(4)

    classified = (
        ee.Image(1)  # Default: Cool
        .where(lst_image.gte(cool_thresh).And(lst_image.lt(mean_lst)), 2)   # Moderate
        .where(lst_image.gte(mean_lst).And(lst_image.lt(warm_thresh)), 3)   # Warm
        .where(lst_image.gte(warm_thresh).And(lst_image.lt(hot_thresh)), 4) # Hot
        .where(lst_image.gte(hot_thresh), 5)  # Extreme
    ).rename("HEAT_CLASS").clip(region)

    print("🔥 Heat classification complete.")
    return classified, region, mean_val


def classify_heat_local(lst_array):
    """
    Classify heat zones from a local numpy LST array.
    
    Returns:
        numpy array with heat classes (1-5)
    """
    valid = lst_array[~np.isnan(lst_array)]
    if len(valid) == 0:
        return np.ones_like(lst_array)

    mean_lst = np.nanmean(lst_array)
    print(f"🌡️ Mean LST (local): {mean_lst:.2f}°C")

    classified = np.ones_like(lst_array, dtype=np.float32)  # Default: Cool (1)
    classified[(lst_array >= mean_lst - 2) & (lst_array < mean_lst)] = 2  # Moderate
    classified[(lst_array >= mean_lst) & (lst_array < mean_lst + 2)] = 3  # Warm
    classified[(lst_array >= mean_lst + 2) & (lst_array < mean_lst + 4)] = 4  # Hot
    classified[lst_array >= mean_lst + 4] = 5  # Extreme
    classified[np.isnan(lst_array)] = np.nan

    return classified


def export_heat_classification():
    """Extract LST, classify heat zones, and save to outputs/heat_classification.tif."""
    lst, region = extract_lst()
    if lst is None:
        print("❌ No LST data for heat classification.")
        return None

    classified, region, mean_val = classify_heat_gee(lst, region)
    filepath = download_ee_image(classified, region, "heat_classification.tif", bands=["HEAT_CLASS"])
    return filepath


if __name__ == "__main__":
    filepath = export_heat_classification()
    if filepath:
        print(f"✅ Heat classification exported to: {filepath}")
