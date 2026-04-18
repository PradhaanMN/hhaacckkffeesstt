"""
Main pipeline orchestrator for end-to-end inference.
Processes Excel input and generates JSON predictions with overlays.
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import List, Dict, Tuple

import pandas as pd

# Import pipeline modules
from pipeline.config import (
    BUFFER_ZONE_1, BUFFER_ZONE_2, IMAGE_SIZE_PX,
    IMAGERY_FETCH_SIZE_SQFT,
    MODEL_WEIGHTS_PATH, ENSEMBLE_MODELS, OUTPUT_PREDICTIONS_DIR, OUTPUT_OVERLAYS_DIR
)
from pipeline.buffer_geometry import compute_pixel_scale, point_in_polygon, compute_polygon_area, compute_buffer_radius_pixels
from pipeline.imagery_fetcher import fetch_arcgis_world_imagery
from pipeline.qc_logic import determine_qc_status, check_image_quality
from pipeline.overlay_generator import create_overlay_image, encode_polygon_for_json
from pipeline.json_writer import write_prediction_json, create_summary_report
from model.model_inference import SolarPanelDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


def find_panels_in_buffer(
    detections: List[Dict],
    buffer_center_px: Tuple[float, float],
    buffer_radius_px: float
) -> List[Dict]:
    """
    Find which detected panels overlap with the buffer zone.
    Uses shapely for accurate geometric intersection detection.
    
    Args:
        detections: List of detection dictionaries
        buffer_center_px: (x, y) center of buffer in pixels
        buffer_radius_px: Radius of buffer in pixels
        
    Returns:
        List of detections that overlap with buffer
    """
    from shapely.geometry import Polygon, Point
    
    panels_in_buffer = []
    
    # Create buffer circle geometry
    buffer_circle = Point(buffer_center_px[0], buffer_center_px[1]).buffer(buffer_radius_px)
    
    for det in detections:
        polygon = det.get("polygon", [])
        if len(polygon) < 3:
            continue
        
        try:
            # Create shapely polygon from detection
            panel_poly = Polygon(polygon)
            
            # Check if panel intersects buffer (any overlap counts)
            if buffer_circle.intersects(panel_poly):
                panels_in_buffer.append(det)
        except Exception as e:
            # Fallback: check if any point is inside buffer
            for point in polygon:
                dx = point[0] - buffer_center_px[0]
                dy = point[1] - buffer_center_px[1]
                distance = (dx**2 + dy**2) ** 0.5
                
                if distance <= buffer_radius_px:
                    panels_in_buffer.append(det)
                    break
    
    return panels_in_buffer


def calculate_clipped_panel_area(
    panel_polygon: List[List[float]],
    buffer_center_px: Tuple[float, float],
    buffer_radius_px: float
) -> float:
    """
    Calculate the area of the panel that falls INSIDE the buffer zone (clipped area).
    
    Args:
        panel_polygon: List of [x, y] coordinates
        buffer_center_px: (x, y) center of buffer
        buffer_radius_px: Radius of buffer in pixels
        
    Returns:
        Area in pixels of the clipped portion inside buffer
    """
    from shapely.geometry import Polygon, Point
    
    try:
        # Create shapely geometries
        panel_poly = Polygon(panel_polygon)
        buffer_circle = Point(buffer_center_px[0], buffer_center_px[1]).buffer(buffer_radius_px)
        
        # Calculate intersection
        clipped_poly = panel_poly.intersection(buffer_circle)
        
        # Return area
        return clipped_poly.area if not clipped_poly.is_empty else 0.0
    
    except Exception as e:
        logger.warning(f"Failed to calculate clipped area: {e}")
        # Fallback to full panel area if inside
        import numpy as np
        centroid_x = np.mean([p[0] for p in panel_polygon])
        centroid_y = np.mean([p[1] for p in panel_polygon])
        distance = np.sqrt((centroid_x - buffer_center_px[0])**2 + (centroid_y - buffer_center_px[1])**2)
        
        if distance <= buffer_radius_px:
            # Calculate full polygon area
            return compute_polygon_area(panel_polygon)
        else:
            return 0.0


def select_largest_panel_in_buffer(
    detections: List[Dict],
    buffer_center_px: Tuple[float, float],
    buffer_radius_px: float
) -> Dict:
    """
    Select the panel with the largest CLIPPED area inside the buffer zone.
    Only counts the portion of each panel that falls within the buffer.
    
    Args:
        detections: List of detection dictionaries
        buffer_center_px: (x, y) center of buffer
        buffer_radius_px: Radius of buffer
        
    Returns:
        Detection with largest clipped area, or None
    """
    panels_in_buffer = find_panels_in_buffer(detections, buffer_center_px, buffer_radius_px)
    
    if not panels_in_buffer:
        return None
    
    # Calculate clipped area for each panel and select the largest
    best_panel = None
    best_clipped_area = 0
    
    for panel in panels_in_buffer:
        clipped_area = calculate_clipped_panel_area(
            panel["polygon"],
            buffer_center_px,
            buffer_radius_px
        )
        
        if clipped_area > best_clipped_area:
            best_clipped_area = clipped_area
            best_panel = panel
            # Store clipped area in detection for later use
            best_panel["clipped_area_px"] = clipped_area
    
    logger.info(f"Selected panel with clipped area: {best_clipped_area:.1f}px (full area: {best_panel['area_px']:.1f}px)")
    
    return best_panel


def calculate_euclidean_distance(
    panel_polygon: List,
    center_px: tuple,
    meters_per_pixel_x: float,
    meters_per_pixel_y: float
) -> float:
    """
    Calculate Euclidean distance from image center to panel centroid in meters.
    
    Args:
        panel_polygon: List of [x, y] coordinates
        center_px: (x, y) center of image in pixels
        meters_per_pixel_x: Ground resolution in X
        meters_per_pixel_y: Ground resolution in Y
        
    Returns:
        Distance in meters
    """
    import numpy as np
    
    # Calculate panel centroid
    polygon_array = np.array(panel_polygon)
    centroid_x = np.mean(polygon_array[:, 0])
    centroid_y = np.mean(polygon_array[:, 1])
    
    # Distance in pixels
    dx_px = centroid_x - center_px[0]
    dy_px = centroid_y - center_px[1]
    
    # Convert to meters
    dx_m = dx_px * meters_per_pixel_x
    dy_m = dy_px * meters_per_pixel_y
    
    # Euclidean distance
    distance_m = np.sqrt(dx_m**2 + dy_m**2)
    
    return distance_m


def estimate_power_generation(area_sqm: float) -> dict:
    """
    Estimate power generation capacity from solar panel area.

    Args:
        area_sqm: Panel area in square meters

    Returns:
        Dictionary with power estimates, economic savings, and CO2 metrics.
    """
    # ── Solar generation constants (India) ───────────────────────────────
    PANEL_EFFICIENCY  = 0.18   # 18% (modern mono-PERC panels)
    PEAK_SUN_HOURS    = 5.5    # Average India daily peak sun hours
    SYSTEM_EFFICIENCY = 0.80   # Inverter + wiring + dust + temp losses

    # ── Economic constants (India) ────────────────────────────────────────
    ELECTRICITY_RATE_INR = 8.0   # ₹/kWh — average residential tariff (2024)
    PANEL_LIFETIME_YEARS = 25    # standard warranty period

    # ── Environmental constants ───────────────────────────────────────────
    # CEA India grid emission factor (CO₂ basis), FY 2022-23: 0.82 kg CO₂/kWh
    CO2_FACTOR_KG_PER_KWH = 0.82
    # 1 mature tree absorbs ~21 kg CO₂/year (US EPA estimate)
    TREE_ABSORPTION_KG_YR = 21.0

    # ── Calculations ─────────────────────────────────────────────────────
    peak_power_kw      = area_sqm * PANEL_EFFICIENCY
    daily_energy_kwh   = peak_power_kw * PEAK_SUN_HOURS * SYSTEM_EFFICIENCY
    monthly_energy_kwh = daily_energy_kwh * 30
    yearly_energy_kwh  = daily_energy_kwh * 365

    # Economic
    monthly_savings_inr  = monthly_energy_kwh  * ELECTRICITY_RATE_INR
    yearly_savings_inr   = yearly_energy_kwh   * ELECTRICITY_RATE_INR
    lifetime_savings_inr = yearly_savings_inr  * PANEL_LIFETIME_YEARS

    # Environmental
    co2_reduction_kg_yr      = yearly_energy_kwh * CO2_FACTOR_KG_PER_KWH
    co2_reduction_tonnes_25y = co2_reduction_kg_yr * PANEL_LIFETIME_YEARS / 1000
    trees_equivalent_yr      = co2_reduction_kg_yr / TREE_ABSORPTION_KG_YR

    return {
        # Power
        "peak_power_kw":          round(peak_power_kw, 2),
        "daily_energy_kwh":       round(daily_energy_kwh, 2),
        "monthly_energy_kwh":     round(monthly_energy_kwh, 2),
        "yearly_energy_kwh":      round(yearly_energy_kwh, 2),
        # Economic (INR)
        "electricity_rate_inr":   ELECTRICITY_RATE_INR,
        "monthly_savings_inr":    round(monthly_savings_inr, 2),
        "yearly_savings_inr":     round(yearly_savings_inr, 2),
        "lifetime_savings_inr":   round(lifetime_savings_inr, 2),
        # Environmental
        "co2_reduction_kg_yr":    round(co2_reduction_kg_yr, 2),
        "co2_reduction_tonnes_25y": round(co2_reduction_tonnes_25y, 3),
        "trees_equivalent_yr":    round(trees_equivalent_yr, 1),
    }


def convert_pixel_area_to_sqm(
    area_px: float,
    meters_per_pixel_x: float,
    meters_per_pixel_y: float
) -> float:
    """
    Convert pixel area to square meters.
    
    Args:
        area_px: Area in pixels
        meters_per_pixel_x: Ground resolution in X direction
        meters_per_pixel_y: Ground resolution in Y direction
        
    Returns:
        Area in square meters
    """
    # Use average pixel scale
    avg_meters_per_pixel = (meters_per_pixel_x + meters_per_pixel_y) / 2
    area_sqm = area_px * (avg_meters_per_pixel ** 2)
    
    return area_sqm


def process_single_location(
    sample_id: int,
    lat: float,
    lon: float,
    detector: SolarPanelDetector,
    temp_dir: Path,
    use_hybrid: bool = True
) -> Dict:
    """
    Process a single location through the complete pipeline.
    
    UPDATED WORKFLOW:
    1. Fetch LARGE satellite image (100m x 100m) via Google Maps for comprehensive coverage
    2. Run YOLO inference with 5 models (hybrid/standard mode) to detect ALL solar panels
    3. Apply buffer zone logic (1200 sq.ft -> 2400 sq.ft) with clipped area calculation
    4. Generate overlay showing all panels with buffer zones
    
    Args:
        sample_id: Unique sample identifier
        lat: Latitude
        lon: Longitude
        detector: SolarPanelDetector instance
        temp_dir: Directory for temporary files
        
    Returns:
        Prediction dictionary
    """
    logger.info(f"Processing sample {sample_id}: ({lat}, {lon})")
    
    # Fetch LARGER imagery (100m x 100m) via Google Maps for comprehensive coverage
    # We'll apply buffer logic after detection
    temp_image_path = temp_dir / f"{sample_id}_satellite.png"
    
    logger.info(f"Fetching satellite imagery ({IMAGERY_FETCH_SIZE_SQFT} sq.ft coverage)")
    fetch_result = fetch_arcgis_world_imagery(
        lat=lat,
        lon=lon,
        area_sqft=IMAGERY_FETCH_SIZE_SQFT,  # Fetch large area
        size_px=IMAGE_SIZE_PX,
        out_path=str(temp_image_path)
    )
    
    if not fetch_result["success"]:
        logger.warning(f"Failed to fetch imagery: {fetch_result.get('error')}")
        # Clean up any partial temp file
        try:
            if temp_image_path.exists():
                temp_image_path.unlink()
        except Exception:
            pass
        # Return NOT_VERIFIABLE result
        return {
            "sample_id": sample_id,
            "lat": lat,
            "lon": lon,
            "has_solar": False,
            "confidence": 0.0,
            "pv_area_sqm_est": 0.0,
            "buffer_radius_sqft": BUFFER_ZONE_1,
            "qc_status": "NOT_VERIFIABLE",
            "bbox_or_mask": "",
            "image_metadata": {"source": "Google Maps Satellite", "capture_date": "Variable by location (typically 2020-2024)"},
            "notes": f"Image fetch failed: {fetch_result.get('error')}"
        }
    
    # Run inference with ADVANCED AI features for maximum detection
    mode_str = "Hybrid Ensemble" if use_hybrid else "Single Model"
    logger.info(f"Running model inference (confidence=0.08, mode={mode_str}, FAST MODE)")
    detections = detector.run_inference(
        str(temp_image_path), 
        conf_threshold=0.08,
        use_tta=False,  # Disabled for 2x speed boost
        use_multiscale=False,  # Disabled for 3x speed boost
        use_hybrid=use_hybrid
    )
    logger.info(f"Detected {len(detections)} solar panels in image")
    
    # Image center is the target location
    center_px = (IMAGE_SIZE_PX / 2, IMAGE_SIZE_PX / 2)
    
    # Compute buffer radii in pixels for both zones
    buffer_1_radius_px = compute_buffer_radius_pixels(
        BUFFER_ZONE_1,
        IMAGERY_FETCH_SIZE_SQFT,
        IMAGE_SIZE_PX
    )
    buffer_2_radius_px = compute_buffer_radius_pixels(
        BUFFER_ZONE_2,
        IMAGERY_FETCH_SIZE_SQFT,
        IMAGE_SIZE_PX
    )
    
    logger.info(f"Buffer 1 (1200 sq.ft) = {buffer_1_radius_px:.1f}px radius")
    logger.info(f"Buffer 2 (2400 sq.ft) = {buffer_2_radius_px:.1f}px radius")
    
    # Try buffer zone 1 first (1200 sq.ft)
    selected_panel = select_largest_panel_in_buffer(
        detections,
        center_px,
        buffer_1_radius_px
    )
    
    buffer_zone = BUFFER_ZONE_1
    
    # If no panels in buffer zone 1, try buffer zone 2 (2400 sq.ft)
    if selected_panel is None:
        logger.info("No solar in buffer zone 1, checking buffer zone 2")
        selected_panel = select_largest_panel_in_buffer(
            detections,
            center_px,
            buffer_2_radius_px
        )
        buffer_zone = BUFFER_ZONE_2
    
    # Determine results
    has_solar = selected_panel is not None
    confidence = selected_panel["confidence"] if has_solar else 0.0
    
    # Calculate area in square meters and euclidean distance
    # Use clipped_area_px (only the part inside buffer) for accurate area calculation
    if has_solar:
        # Use clipped area if available, otherwise fall back to full area
        panel_area_px = selected_panel.get("clipped_area_px", selected_panel["area_px"])
        area_sqm = convert_pixel_area_to_sqm(
            panel_area_px,
            fetch_result["meters_per_pixel_x"],
            fetch_result["meters_per_pixel_y"]
        )
        bbox_or_mask = encode_polygon_for_json(selected_panel["polygon"])
        euclidean_distance_m = calculate_euclidean_distance(
            selected_panel["polygon"],
            center_px,
            fetch_result["meters_per_pixel_x"],
            fetch_result["meters_per_pixel_y"]
        )
        logger.info(f"Panel area: {panel_area_px:.1f}px (clipped) = {area_sqm:.2f} sq.m")
    else:
        area_sqm = 0.0
        bbox_or_mask = ""
        euclidean_distance_m = 0.0
    
    # Determine QC status
    image_quality = check_image_quality(str(temp_image_path)) if fetch_result["success"] else None
    qc_status = determine_qc_status(
        image_fetch_success=fetch_result["success"],
        detections=detections,
        image_metadata=image_quality,
        notes=None
    )
    
    # Create overlay with ALL detections and buffer zone visualization
    overlay_path = OUTPUT_OVERLAYS_DIR / f"{sample_id}_overlay.png"
    create_overlay_image(
        image_path=str(temp_image_path),
        detections=detections,  # ALL detections from the model
        selected_panel=selected_panel,
        buffer_zone={"type": "circle", "center": center_px, "radius": buffer_1_radius_px},
        output_path=str(overlay_path),
        buffer_sqft=buffer_zone,  # Which buffer was successful (1200 or 2400)
        imagery_sqft=IMAGERY_FETCH_SIZE_SQFT  # Total imagery area
    )
    
    # Estimate power generation if solar detected
    power_estimate = estimate_power_generation(area_sqm) if has_solar else {
        "peak_power_kw": 0.0,
        "daily_energy_kwh": 0.0,
        "monthly_energy_kwh": 0.0,
        "yearly_energy_kwh": 0.0
    }
    
    # Build prediction
    prediction = {
        "sample_id": sample_id,
        "lat": lat,
        "lon": lon,
        "has_solar": has_solar,
        "confidence": confidence,
        "pv_area_sqm_est": area_sqm,
        "euclidean_distance_m_est": euclidean_distance_m,
        "buffer_radius_sqft": buffer_zone,
        "qc_status": qc_status,
        "bbox_or_mask": bbox_or_mask,
        "power_estimate": power_estimate,
        "image_metadata": {
            "source": "Google Maps Satellite",
            "capture_date": "Variable by location (typically 2020-2024, updated regularly)"
        }
    }
    
    logger.info(f"Sample {sample_id}: has_solar={has_solar}, confidence={confidence:.2f}, "
                f"area={area_sqm:.2f} m², QC={qc_status}")
    
    # Clean up temporary image file
    try:
        if temp_image_path.exists():
            temp_image_path.unlink()
            logger.debug(f"Deleted temporary image: {temp_image_path}")
    except Exception as e:
        logger.warning(f"Failed to delete temporary image {temp_image_path}: {e}")
    
    return prediction


def process_location_sweep(
    sample_id: int,
    lat: float,
    lon: float,
    detector: "SolarPanelDetector",
    temp_dir: Path,
    exclusion_geom=None,   # optional shapely geometry (WGS84) to filter detections
) -> Dict:
    """
    Buffer-free sweep variant used by the Area Sweep endpoint.

    Detects ALL solar panels in the satellite tile (no 1200/2400 sq-ft restriction).
    The overlay shows every detected panel in GREEN.

    Args:
        sample_id: Unique identifier
        lat, lon: WGS84 coordinates of the grid centroid
        detector: Loaded SolarPanelDetector
        temp_dir: Directory for temporary imagery

    Returns:
        Prediction dictionary (same schema as process_single_location but
        ``buffer_radius_sqft`` is set to None / 0).
    """
    logger.info(f"[Sweep] Processing sample {sample_id}: ({lat}, {lon})")

    import time as _time
    _t = {}          # timing buckets (seconds)
    _t0_total = _time.perf_counter()

    temp_image_path = temp_dir / f"{sample_id}_satellite.png"

    # ── 1. Fetch satellite imagery ─────────────────────────────────────────
    _t1 = _time.perf_counter()
    fetch_result = fetch_arcgis_world_imagery(
        lat=lat,
        lon=lon,
        area_sqft=IMAGERY_FETCH_SIZE_SQFT,
        size_px=IMAGE_SIZE_PX,
        out_path=str(temp_image_path),
    )
    _t["fetch_s"] = _time.perf_counter() - _t1

    if not fetch_result["success"]:
        logger.warning(f"[Sweep] Imagery fetch failed: {fetch_result.get('error')}")
        try:
            if temp_image_path.exists():
                temp_image_path.unlink()
        except Exception:
            pass
        return {
            "sample_id": sample_id, "lat": lat, "lon": lon,
            "has_solar": False, "confidence": 0.0,
            "pv_area_sqm_est": 0.0, "panel_count": 0,
            "buffer_radius_sqft": 0,
            "qc_status": "NOT_VERIFIABLE", "bbox_or_mask": "",
            "image_metadata": {"source": "ArcGIS World Imagery"},
            "notes": f"Imagery fetch failed: {fetch_result.get('error')}",
            "timing_s": _t,
        }

    # ── 2. YOLO inference ──────────────────────────────────────────────────
    _t1 = _time.perf_counter()
    detections = detector.run_inference(
        str(temp_image_path),
        conf_threshold=0.08,
        use_tta=False,
        use_multiscale=False,
        use_hybrid=True,
    )
    _t["inference_s"] = _time.perf_counter() - _t1
    raw_count = len(detections)
    logger.info(f"[Sweep] {raw_count} panel(s) detected at ({lat:.5f}, {lon:.5f})")

    # ── 3. Exclusion-zone filtering ────────────────────────────────────────
    _t1 = _time.perf_counter()
    if exclusion_geom is not None and not exclusion_geom.is_empty and detections:
        import math
        from shapely.geometry import Point
        mppx_pre = fetch_result["meters_per_pixel_x"]
        mppy_pre = fetch_result["meters_per_pixel_y"]
        img_cx = IMAGE_SIZE_PX / 2
        img_cy = IMAGE_SIZE_PX / 2
        filtered = []
        for det in detections:
            poly_pts = det.get("polygon", [])
            if not poly_pts:
                filtered.append(det)
                continue
            cx_px = sum(p[0] for p in poly_pts) / len(poly_pts)
            cy_px = sum(p[1] for p in poly_pts) / len(poly_pts)
            dx_m = (cx_px - img_cx) * mppx_pre
            dy_m = (img_cy - cy_px) * mppy_pre
            det_lat = lat + dy_m / 111320
            det_lon = lon + dx_m / (111320 * math.cos(math.radians(lat)))
            if exclusion_geom.contains(Point(det_lon, det_lat)):
                continue
            filtered.append(det)
        excluded = raw_count - len(filtered)
        if excluded:
            logger.info(f"[Sweep] Filtered {excluded} detection(s) inside exclusion zones")
        detections = filtered
    _t["exclusion_filter_s"] = _time.perf_counter() - _t1

    # ── 4. Area / confidence calculations ─────────────────────────────────
    _t1 = _time.perf_counter()
    has_solar  = len(detections) > 0
    confidence = max((d.get("confidence", 0) for d in detections), default=0.0)
    mppx = fetch_result["meters_per_pixel_x"]
    mppy = fetch_result["meters_per_pixel_y"]
    total_area_sqm = sum(
        convert_pixel_area_to_sqm(d.get("area_px", 0), mppx, mppy)
        for d in detections
    )
    bbox_or_mask = encode_polygon_for_json(detections[0]["polygon"]) if detections else ""
    _t["calc_s"] = _time.perf_counter() - _t1

    # ── 5. QC / image quality check ───────────────────────────────────────
    _t1 = _time.perf_counter()
    image_quality = check_image_quality(str(temp_image_path))
    qc_status = determine_qc_status(
        image_fetch_success=True,
        detections=detections,
        image_metadata=image_quality,
        notes=None,
    )
    _t["qc_s"] = _time.perf_counter() - _t1

    # ── 6. Overlay generation ─────────────────────────────────────────────
    _t1 = _time.perf_counter()
    overlay_path = OUTPUT_OVERLAYS_DIR / f"{sample_id}_overlay.png"
    create_overlay_image(
        image_path=str(temp_image_path),
        detections=detections,
        selected_panel=None,
        buffer_zone=None,
        output_path=str(overlay_path),
        buffer_sqft=None,
        imagery_sqft=IMAGERY_FETCH_SIZE_SQFT,
        skip_buffer=True,
    )
    _t["overlay_s"] = _time.perf_counter() - _t1

    # ── 7. Power estimate ─────────────────────────────────────────────────
    _t1 = _time.perf_counter()
    power_estimate = estimate_power_generation(total_area_sqm) if has_solar else {
        "peak_power_kw": 0.0, "daily_energy_kwh": 0.0,
        "monthly_energy_kwh": 0.0, "yearly_energy_kwh": 0.0,
    }
    _t["power_calc_s"] = _time.perf_counter() - _t1

    # ── 8. Cleanup temp image ─────────────────────────────────────────────
    _t1 = _time.perf_counter()
    try:
        if temp_image_path.exists():
            temp_image_path.unlink()
    except Exception as e:
        logger.warning(f"[Sweep] Failed to delete temp image: {e}")
    _t["cleanup_s"] = _time.perf_counter() - _t1

    _t["total_s"] = _time.perf_counter() - _t0_total

    return {
        "sample_id": sample_id, "lat": lat, "lon": lon,
        "has_solar": has_solar,
        "confidence": round(confidence, 4),
        "pv_area_sqm_est": round(total_area_sqm, 2),
        "panel_count": len(detections),
        "buffer_radius_sqft": 0,
        "qc_status": qc_status,
        "bbox_or_mask": bbox_or_mask,
        "power_estimate": power_estimate,
        "image_metadata": {"source": "ArcGIS World Imagery"},
        "timing_s": _t,    # passed back for aggregate summary
    }



def process_excel_file(
    excel_path: str,
    model_path: str,
    output_dir: str,
    temp_dir: str = "temp_images"
) -> List[Dict]:
    """
    Process an Excel file containing multiple locations.
    
    Args:
        excel_path: Path to input Excel file
        model_path: Path to model weights
        output_dir: Directory for output files
        temp_dir: Directory for temporary image files
        
    Returns:
        List of all predictions
    """
    # Load Excel file
    logger.info(f"Loading Excel file: {excel_path}")
    df = pd.read_excel(excel_path)
    
    # Validate columns
    required_cols = ["sample_id", "latitude", "longitude"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    logger.info(f"Loaded {len(df)} locations from Excel")
    
    # Create temp directory
    temp_path = Path(temp_dir)
    temp_path.mkdir(parents=True, exist_ok=True)
    
    # Load ensemble models - HYBRID ENSEMBLE/ADVERSARIAL approach
    logger.info(f"Loading primary model from {model_path}")
    
    # Check which ensemble models exist from config
    available_models = [m for m in ENSEMBLE_MODELS if Path(m).exists()]
    
    if available_models:
        logger.info(f"Found {len(available_models)} additional ensemble models:")
        for model in available_models:
            logger.info(f"  - {model}")
    else:
        logger.warning("No ensemble models found - using single model only")
    
    # Initialize detector with HYBRID ensemble/adversarial approach (4 seg + 1 det = 5 models)
    detector = SolarPanelDetector(model_path, ensemble_models=available_models if available_models else None)
    logger.info(f"🤖 HYBRID Ensemble/Adversarial System: {len(available_models) + 1} models")
    logger.info(f"   • 4 Segmentation + 1 Detection models for maximum diversity")
    logger.info(f"   • Ensemble voting for robust detection")
    logger.info(f"   • Adversarial confidence adjustment (consensus-based)")
    logger.info(f"   • High consensus → confidence boost, Low consensus → penalty")
    
    # Process each location
    predictions = []
    
    for idx, row in df.iterrows():
        sample_id = int(row["sample_id"])
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        
        try:
            prediction = process_single_location(
                sample_id=sample_id,
                lat=lat,
                lon=lon,
                detector=detector,
                temp_dir=temp_path
            )
            
            # Write JSON
            write_prediction_json(
                sample_id=prediction["sample_id"],
                lat=prediction["lat"],
                lon=prediction["lon"],
                has_solar=prediction["has_solar"],
                confidence=prediction["confidence"],
                pv_area_sqm_est=prediction["pv_area_sqm_est"],
                buffer_radius_sqft=prediction["buffer_radius_sqft"],
                qc_status=prediction["qc_status"],
                bbox_or_mask=prediction["bbox_or_mask"],
                image_metadata=prediction["image_metadata"],
                output_dir=output_dir
            )
            
            predictions.append(prediction)
            
        except Exception as e:
            logger.exception(f"Error processing sample {sample_id}: {e}")
            continue
    
    logger.info(f"Processed {len(predictions)} out of {len(df)} locations")
    
    # Create summary report
    summary_path = Path(output_dir) / "summary_report.json"
    create_summary_report(predictions, str(summary_path))
    
    return predictions


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="End-to-end rooftop PV detection pipeline"
    )
    parser.add_argument(
        "input_excel",
        type=str,
        help="Path to input Excel file with sample_id, latitude, longitude columns"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_WEIGHTS_PATH,
        help="Path to YOLOv8 model weights (default: trained_model/custommodelonmydataset.pt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_PREDICTIONS_DIR,
        help="Output directory for JSON predictions (default: outputs/predictions)"
    )
    parser.add_argument(
        "--temp",
        type=str,
        default="temp_images",
        help="Temporary directory for downloaded images (default: temp_images)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input_excel).exists():
        logger.error(f"Input file not found: {args.input_excel}")
        sys.exit(1)
    
    # Validate model file
    if not Path(args.model).exists():
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)
    
    # Run pipeline
    logger.info("=" * 80)
    logger.info("Rooftop PV Detection Pipeline")
    logger.info("=" * 80)
    
    predictions = process_excel_file(
        excel_path=args.input_excel,
        model_path=args.model,
        output_dir=args.output,
        temp_dir=args.temp
    )
    
    logger.info("=" * 80)
    logger.info(f"Pipeline complete! Processed {len(predictions)} locations")
    logger.info(f"Results saved to: {args.output}")
    logger.info(f"Overlays saved to: {OUTPUT_OVERLAYS_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

