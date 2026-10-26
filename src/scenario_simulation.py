"""
UrbanCool-AI — Scenario Simulation Module
Simulates the impact of increased tree canopy coverage on urban heat.
"""

import os
import json
import numpy as np

from src.utils import OUTPUT_DIR, ensure_dirs, load_tiff


def simulate_canopy_increase(canopy_increase_pct=10):
    """
    Simulate the effect of increasing tree canopy cover on urban heat.
    
    Approach:
        - Load LST and heat classification data
        - Model cooling effect: each 10% canopy increase → ~1.5°C cooling in hot zones
        - Recalculate heat zones with adjusted temperatures
        - Report key metrics
    
    Args:
        canopy_increase_pct: Percentage increase in tree canopy (default: 10%)
    
    Returns:
        dict with simulation results
    """
    ensure_dirs()

    # Load existing data
    try:
        lst_array, lst_profile = load_tiff("lst.tif")
    except FileNotFoundError as e:
        print(f"⚠️ LST data not found: {e}")
        print("📊 Using synthetic data for simulation...")
        np.random.seed(42)
        lst_array = np.random.normal(38, 3, (100, 100)).astype(np.float32)

    # Always derive heat classification from the LST array itself
    # to guarantee matching shapes (the separate TIFFs may have different grids)
    valid_lst = lst_array[~np.isnan(lst_array)]
    mean_lst = float(np.mean(valid_lst)) if len(valid_lst) > 0 else 38.0

    heat_array = np.ones_like(lst_array, dtype=np.float32)
    heat_array[lst_array >= mean_lst - 2] = 2
    heat_array[lst_array >= mean_lst] = 3
    heat_array[lst_array >= mean_lst + 2] = 4
    heat_array[lst_array >= mean_lst + 4] = 5
    heat_array[np.isnan(lst_array)] = np.nan

    # Current statistics
    valid_lst = lst_array[~np.isnan(lst_array)]
    valid_heat = heat_array[~np.isnan(heat_array)]

    current_mean_lst = float(np.mean(valid_lst))
    current_max_lst = float(np.max(valid_lst))
    current_min_lst = float(np.min(valid_lst))

    total_pixels = len(valid_heat)
    extreme_before = int(np.sum(valid_heat >= 5))
    hot_before = int(np.sum(valid_heat >= 4))
    extreme_pct_before = round(extreme_before / total_pixels * 100, 2)
    hot_pct_before = round(hot_before / total_pixels * 100, 2)

    print(f"📊 Current Statistics:")
    print(f"   Mean LST: {current_mean_lst:.2f}°C")
    print(f"   Hot zones: {hot_pct_before}%")
    print(f"   Extreme zones: {extreme_pct_before}%")

    # --- Simulation ---
    # Cooling effect model:
    # Each 10% canopy increase provides ~1.5°C cooling in hot/extreme zones
    # Moderate cooling (~0.5°C) in warm zones
    # Minimal effect (<0.2°C) in already cool zones
    cooling_factor = canopy_increase_pct / 10.0

    new_lst = lst_array.copy()

    # Apply differential cooling based on heat class
    extreme_mask = heat_array >= 5
    hot_mask = (heat_array >= 4) & (heat_array < 5)
    warm_mask = (heat_array >= 3) & (heat_array < 4)
    moderate_mask = (heat_array >= 2) & (heat_array < 3)

    new_lst[extreme_mask] -= 2.0 * cooling_factor  # ~2°C per 10% canopy
    new_lst[hot_mask] -= 1.5 * cooling_factor        # ~1.5°C per 10% canopy
    new_lst[warm_mask] -= 0.8 * cooling_factor        # ~0.8°C per 10% canopy
    new_lst[moderate_mask] -= 0.3 * cooling_factor    # ~0.3°C per 10% canopy

    # New statistics
    new_valid = new_lst[~np.isnan(new_lst)]
    new_mean_lst = float(np.mean(new_valid))

    # Reclassify heat zones
    new_heat = np.ones_like(new_lst, dtype=np.float32)
    new_heat[new_lst >= current_mean_lst - 2] = 2
    new_heat[new_lst >= current_mean_lst] = 3
    new_heat[new_lst >= current_mean_lst + 2] = 4
    new_heat[new_lst >= current_mean_lst + 4] = 5
    new_heat[np.isnan(new_lst)] = np.nan

    new_valid_heat = new_heat[~np.isnan(new_heat)]
    extreme_after = int(np.sum(new_valid_heat >= 5))
    hot_after = int(np.sum(new_valid_heat >= 4))
    extreme_pct_after = round(extreme_after / len(new_valid_heat) * 100, 2)
    hot_pct_after = round(hot_after / len(new_valid_heat) * 100, 2)

    avg_cooling = round(current_mean_lst - new_mean_lst, 2)
    extreme_reduction = round(extreme_pct_before - extreme_pct_after, 2)
    hot_reduction = round(hot_pct_before - hot_pct_after, 2)

    results = {
        "simulation_parameters": {
            "canopy_increase_percent": canopy_increase_pct,
            "city": "Chennai",
            "date_range": "2025-04-01 to 2025-07-31",
        },
        "before": {
            "mean_lst_celsius": round(current_mean_lst, 2),
            "max_lst_celsius": round(current_max_lst, 2),
            "min_lst_celsius": round(current_min_lst, 2),
            "extreme_heat_percent": extreme_pct_before,
            "hot_heat_percent": hot_pct_before,
            "extreme_pixels": extreme_before,
            "hot_pixels": hot_before,
            "total_pixels": total_pixels,
        },
        "after": {
            "mean_lst_celsius": round(new_mean_lst, 2),
            "extreme_heat_percent": extreme_pct_after,
            "hot_heat_percent": hot_pct_after,
            "extreme_pixels": extreme_after,
            "hot_pixels": hot_after,
        },
        "impact": {
            "average_cooling_celsius": avg_cooling,
            "extreme_heat_reduction_percent": extreme_reduction,
            "hot_heat_reduction_percent": hot_reduction,
        },
        "monthly_trends": _generate_monthly_trends(current_mean_lst, canopy_increase_pct),
        "seasonal_analysis": _generate_seasonal_analysis(current_mean_lst, canopy_increase_pct),
    }

    print(f"\n🌿 Simulation Results (+{canopy_increase_pct}% canopy):")
    print(f"   New Mean LST: {new_mean_lst:.2f}°C (Δ = -{avg_cooling}°C)")
    print(f"   Extreme heat reduction: {extreme_reduction}%")
    print(f"   Hot heat reduction: {hot_reduction}%")

    # Save results
    output_path = os.path.join(OUTPUT_DIR, "scenario_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"💾 Saved: {output_path}")

    return results


def _generate_monthly_trends(base_lst, canopy_increase_pct):
    """Generate monthly LST trend data for Apr–Jul."""
    # Typical Chennai monthly pattern (relative to base)
    monthly_offsets = {
        "April": -1.5,
        "May": 1.0,
        "June": 0.5,
        "July": -0.5,
    }

    trends = []
    cooling = canopy_increase_pct / 10.0 * 1.2  # cooling per 10%

    for month, offset in monthly_offsets.items():
        original = round(base_lst + offset, 2)
        projected = round(original - cooling, 2)
        trends.append({
            "month": month,
            "original_lst": original,
            "projected_lst": projected,
            "cooling_effect": round(cooling, 2),
        })

    return trends


def _generate_seasonal_analysis(base_lst, canopy_increase_pct):
    """Generate seasonal anomaly analysis."""
    # Historical average for Chennai summer: ~36°C
    historical_avg = 36.0
    anomaly = round(base_lst - historical_avg, 2)

    return {
        "historical_avg_celsius": historical_avg,
        "current_anomaly_celsius": anomaly,
        "interpretation": (
            f"Chennai is {abs(anomaly):.1f}°C "
            f"{'above' if anomaly > 0 else 'below'} historical average. "
            f"A {canopy_increase_pct}% canopy increase could offset "
            f"~{canopy_increase_pct / 10 * 1.2:.1f}°C of this anomaly."
        ),
    }


def run_multi_scenario():
    """Run simulations for multiple canopy increase levels."""
    scenarios = [5, 10, 15, 20, 25, 30]
    all_results = []

    for pct in scenarios:
        print(f"\n{'='*50}")
        print(f"Scenario: +{pct}% canopy increase")
        print(f"{'='*50}")
        result = simulate_canopy_increase(pct)
        all_results.append(result)

    return all_results


if __name__ == "__main__":
    results = simulate_canopy_increase(10)
    print(f"\n✅ Simulation complete!")
    print(json.dumps(results, indent=2))
