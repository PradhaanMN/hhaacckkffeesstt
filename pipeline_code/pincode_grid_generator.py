# -*- coding: utf-8 -*-
"""
Pincode Grid Generator for Solar Panel Detection Pipeline
=========================================================
Takes a Pincode / Postal Code, fetches its geographic boundary from
OpenStreetMap via osmnx, divides the area into a uniform grid of
10,000 sq ft cells (~30.48 m × 30.48 m), and exports the centroids
as an Excel file ready for the main detection pipeline.

Output format (pipeline-compatible):
    sample_id  |  latitude  |  longitude
    Grid_001   |  12.9716   |  77.5946
    Grid_002   |  12.9719   |  77.5949
    ...

Usage:
    python pincode_grid_generator.py --pincode 560001
    python pincode_grid_generator.py --pincode 560001 --country India
    python pincode_grid_generator.py --place "Koramangala, Bengaluru"
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point
import osmnx as ox


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

# 10,000 sq ft in metric:  1 ft = 0.3048 m
# √10000 ft = 100 ft per side → 100 × 0.3048 = 30.48 m per side
CELL_SIDE_M = 30.48  # meters — each grid cell is 30.48 m × 30.48 m

# CRS codes
WGS84 = "EPSG:4326"

# Output directory (relative to pipeline_code/)
SCRIPT_DIR = Path(__file__).parent.resolve()
INPUTS_DIR = SCRIPT_DIR / "inputs"


# ─────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────

def fetch_boundary_by_pincode(pincode: str, country: str = "India") -> gpd.GeoDataFrame:
    """
    Fetch the boundary polygon for a given pincode from OpenStreetMap.

    Uses osmnx.geocode_to_gdf() with a structured query targeting
    postal code boundaries.

    Args:
        pincode:  The postal / PIN code string (e.g. "560001").
        country:  Country name to narrow down the geocoding result.

    Returns:
        GeoDataFrame with the boundary polygon in WGS84.
    """
    print(f"\n[*] Fetching boundary for pincode: {pincode} ({country})...")

    # Try multiple query strategies for robustness
    queries = [
        # Strategy 1: Structured query with postal_code tag
        {"postalcode": pincode, "country": country},
        # Strategy 2: Free-form text query
        f"{pincode}, {country}",
        # Strategy 3: Just the pincode
        f"pincode {pincode} {country}",
    ]

    last_error = None
    for query in queries:
        try:
            gdf = ox.geocode_to_gdf(query)
            if gdf is not None and not gdf.empty:
                print(f"   [OK] Boundary found! Geometry type: {gdf.geometry.iloc[0].geom_type}")
                print(f"   [>] Display name: {gdf.iloc[0].get('display_name', 'N/A')}")
                return gdf
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(
        f"Could not find boundary for pincode '{pincode}' in '{country}'.\n"
        f"Last error: {last_error}\n"
        f"Tips:\n"
        f"  • Ensure the pincode is valid and exists on OpenStreetMap.\n"
        f"  • Try using --place 'City Name, Country' instead.\n"
    )


def fetch_boundary_by_place(place_name: str) -> gpd.GeoDataFrame:
    """
    Fetch the boundary polygon for a named place (city, neighbourhood, etc.).

    Args:
        place_name: Human-readable place name (e.g. "Koramangala, Bengaluru").

    Returns:
        GeoDataFrame with the boundary polygon in WGS84.
    """
    print(f"\n[*] Fetching boundary for place: '{place_name}'...")
    try:
        gdf = ox.geocode_to_gdf(place_name)
        if gdf is not None and not gdf.empty:
            print(f"   [OK] Boundary found! Geometry type: {gdf.geometry.iloc[0].geom_type}")
            print(f"   [>] Display name: {gdf.iloc[0].get('display_name', 'N/A')}")
            return gdf
    except Exception as e:
        raise RuntimeError(
            f"Could not find boundary for place '{place_name}'.\n"
            f"Error: {e}\n"
            f"Tip: Try a more specific name like 'Locality, City, State, Country'."
        )

    raise RuntimeError(f"Empty result for place '{place_name}'.")


def get_metric_crs(gdf: gpd.GeoDataFrame) -> str:
    """
    Determine a suitable metric (metre-based) CRS for the given geometry.

    Uses the UTM zone that covers the centroid of the boundary.

    Args:
        gdf: GeoDataFrame in WGS84.

    Returns:
        EPSG string for the appropriate UTM zone (e.g. "EPSG:32643").
    """
    centroid = gdf.geometry.union_all().centroid
    lon = centroid.x
    lat = centroid.y

    # Compute UTM zone number
    zone_number = int((lon + 180) / 6) + 1

    # Northern or Southern hemisphere
    if lat >= 0:
        epsg_code = 32600 + zone_number  # UTM North
    else:
        epsg_code = 32700 + zone_number  # UTM South

    crs_str = f"EPSG:{epsg_code}"
    print(f"   [CRS] Using metric CRS: {crs_str} (UTM Zone {zone_number}{'N' if lat >= 0 else 'S'})")
    return crs_str


def generate_grid_centroids(boundary_gdf: gpd.GeoDataFrame, cell_side_m: float = CELL_SIDE_M) -> gpd.GeoDataFrame:
    """
    Generate a uniform grid over the boundary polygon and return the
    centroids of all cells that fall INSIDE the boundary.

    Each cell is exactly cell_side_m × cell_side_m (default 30.48 m = 10,000 sq ft).

    Args:
        boundary_gdf:  GeoDataFrame with boundary in WGS84.
        cell_side_m:   Side length of each square grid cell in metres.

    Returns:
        GeoDataFrame of centroid Points in WGS84 with columns:
            sample_id, latitude, longitude
    """
    print(f"\n[GRID] Generating grid (cell size: {cell_side_m:.2f}m x {cell_side_m:.2f}m = 10,000 sq ft)...")

    # Step 1: Determine metric CRS and reproject
    metric_crs = get_metric_crs(boundary_gdf)
    boundary_metric = boundary_gdf.to_crs(metric_crs)
    boundary_polygon = boundary_metric.geometry.union_all()

    # Step 2: Get bounding box of the boundary in metric coordinates
    minx, miny, maxx, maxy = boundary_polygon.bounds
    print(f"   [BOX] Bounding box (metric): {maxx - minx:.0f}m x {maxy - miny:.0f}m")

    # Step 3: Generate grid cell coordinates
    # Create arrays of x and y coordinates for cell origins
    x_coords = np.arange(minx, maxx, cell_side_m)
    y_coords = np.arange(miny, maxy, cell_side_m)

    total_cells = len(x_coords) * len(y_coords)
    print(f"   [#] Total grid cells in bounding box: {total_cells:,}")

    # Step 4: Build grid cells and filter by boundary intersection
    print(f"   [...] Filtering cells inside boundary polygon...")
    start_time = time.time()

    centroids = []
    for x in x_coords:
        for y in y_coords:
            # Create the grid cell as a square
            cell = box(x, y, x + cell_side_m, y + cell_side_m)

            # Check if the cell centroid falls inside the boundary
            cell_centroid = cell.centroid
            if boundary_polygon.contains(cell_centroid):
                centroids.append(cell_centroid)

    elapsed = time.time() - start_time
    print(f"   [OK] Found {len(centroids):,} grid points inside boundary ({elapsed:.1f}s)")

    if len(centroids) == 0:
        raise RuntimeError(
            "No grid cells fell inside the boundary polygon. "
            "The area might be too small for the 10,000 sq ft grid resolution. "
            "Try a larger pincode or region."
        )

    # Step 5: Create GeoDataFrame with centroids in metric CRS
    centroids_gdf = gpd.GeoDataFrame(
        geometry=centroids,
        crs=metric_crs
    )

    # Step 6: Reproject centroids back to WGS84 (lat/lon)
    centroids_wgs84 = centroids_gdf.to_crs(WGS84)

    # Step 7: Build the final output DataFrame
    # Extract lat/lon from the Point geometries
    lats = centroids_wgs84.geometry.y.values
    lons = centroids_wgs84.geometry.x.values

    # Generate sequential sample IDs: Grid_001, Grid_002, ...
    sample_ids = [f"Grid_{i+1:03d}" for i in range(len(centroids_wgs84))]

    result_df = pd.DataFrame({
        "sample_id": sample_ids,
        "latitude": np.round(lats, 7),
        "longitude": np.round(lons, 7),
    })

    return result_df


def export_to_excel(df: pd.DataFrame, pincode_or_place: str) -> str:
    """
    Export the grid centroids DataFrame to an Excel file in the inputs/ directory.

    Args:
        df:               DataFrame with sample_id, latitude, longitude.
        pincode_or_place: Used for naming the output file.

    Returns:
        Absolute path to the generated Excel file.
    """
    # Ensure inputs directory exists
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Clean the name for use in a filename
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in pincode_or_place)
    filename = f"grid_{safe_name}.xlsx"
    output_path = INPUTS_DIR / filename

    # Write to Excel
    df.to_excel(str(output_path), index=False, engine="openpyxl")

    print(f"\n[SAVED] Excel file saved to: {output_path}")
    print(f"   Total grid points: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")

    return str(output_path)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate grid centroids from a Pincode/Place for the Solar Panel Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pincode_grid_generator.py --pincode 560001
  python pincode_grid_generator.py --pincode 560001 --country India
  python pincode_grid_generator.py --place "Koramangala, Bengaluru, India"
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pincode",
        type=str,
        help="Postal / PIN code to fetch boundary for (e.g. 560001)"
    )
    group.add_argument(
        "--place",
        type=str,
        help="Place name to fetch boundary for (e.g. 'Koramangala, Bengaluru')"
    )

    parser.add_argument(
        "--country",
        type=str,
        default="India",
        help="Country for pincode lookup (default: India)"
    )

    args = parser.parse_args()

    # ── Banner ──
    print("=" * 70)
    print("  SOLAR PANEL DETECTION - Pincode Grid Generator")
    print("  Generates pipeline-compatible input from geographic boundaries")
    print("=" * 70)

    # ── Step 1: Fetch boundary ──
    if args.pincode:
        boundary_gdf = fetch_boundary_by_pincode(args.pincode, args.country)
        label = args.pincode
    else:
        boundary_gdf = fetch_boundary_by_place(args.place)
        label = args.place

    # ── Step 2: Compute boundary area for context ──
    metric_crs = get_metric_crs(boundary_gdf)
    boundary_metric = boundary_gdf.to_crs(metric_crs)
    area_sqm = boundary_metric.geometry.union_all().area
    area_sqkm = area_sqm / 1e6
    area_sqft = area_sqm / 0.092903
    print(f"   [AREA] Boundary area: {area_sqkm:.3f} km2 ({area_sqft:,.0f} sq ft)")

    # ── Step 3: Generate grid centroids ──
    result_df = generate_grid_centroids(boundary_gdf)

    # ── Step 4: Preview ──
    print(f"\n[PREVIEW] Generated grid (first 5 rows):")
    print(result_df.head().to_string(index=False))

    # ── Step 5: Export ──
    output_path = export_to_excel(result_df, label)

    # ── Summary ──
    print("\n" + "=" * 70)
    print(f"  DONE! Generated {len(result_df):,} grid points")
    print(f"  Output: {output_path}")
    print(f"  Next step: Run the detection pipeline with this file:")
    print(f"     python -m pipeline.main \"{output_path}\"")
    print("=" * 70)


if __name__ == "__main__":
    main()

