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
    MODEL_WEIGHTS_PATH, OUTPUT_PREDICTIONS_DIR, OUTPUT_OVERLAYS_DIR
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
    
    Args:
        detections: List of detection dictionaries
        buffer_center_px: (x, y) center of buffer in pixels
        buffer_radius_px: Radius of buffer in pixels
        
    Returns:
        List of detections that overlap with buffer
    """
    panels_in_buffer = []
    
    for det in detections:
        polygon = det["polygon"]
        
        # Check if any point of the polygon is inside the buffer circle
        for point in polygon:
            dx = point[0] - buffer_center_px[0]
            dy = point[1] - buffer_center_px[1]
            distance = (dx**2 + dy**2) ** 0.5
            
            if distance <= buffer_radius_px:
                panels_in_buffer.append(det)
                break
    
    return panels_in_buffer


def select_largest_panel_in_buffer(
    detections: List[Dict],
    buffer_center_px: Tuple[float, float],
    buffer_radius_px: float
) -> Dict:
    """
    Select the panel with the largest overlap with the buffer zone.
    
    Args:
        detections: List of detection dictionaries
        buffer_center_px: (x, y) center of buffer
        buffer_radius_px: Radius of buffer
        
    Returns:
        Detection with largest overlap, or None
    """
    panels_in_buffer = find_panels_in_buffer(detections, buffer_center_px, buffer_radius_px)
    
    if not panels_in_buffer:
        return None
    
    # For simplicity, select the panel with the largest area
    # (More sophisticated: calculate actual overlap area)
    largest_panel = max(panels_in_buffer, key=lambda d: d["area_px"])
    
    return largest_panel


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
    temp_dir: Path
) -> Dict:
    """
    Process a single location through the complete pipeline.
    
    UPDATED WORKFLOW:
    1. Fetch LARGE satellite image (100m x 100m) to satisfy ArcGIS requirements
    2. Run YOLO inference on ENTIRE image to detect ALL solar panels
    3. Apply buffer zone logic (1200 sq.ft -> 2400 sq.ft) to filter detections
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
    
    # Fetch LARGER imagery (100m x 100m) to satisfy ArcGIS API requirements
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
            "image_metadata": {"source": "ArcGIS World_Imagery", "capture_date": "UNKNOWN"},
            "notes": f"Image fetch failed: {fetch_result.get('error')}"
        }
    
    # Run inference on ENTIRE image - detect ALL solar panels
    logger.info("Running model inference on full image")
    detections = detector.run_inference(str(temp_image_path))
    logger.info(f"Detected {len(detections)} total solar panels in image")
    
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
    
    # Calculate area in square meters
    if has_solar:
        area_sqm = convert_pixel_area_to_sqm(
            selected_panel["area_px"],
            fetch_result["meters_per_pixel_x"],
            fetch_result["meters_per_pixel_y"]
        )
        bbox_or_mask = encode_polygon_for_json(selected_panel["polygon"])
    else:
        area_sqm = 0.0
        bbox_or_mask = ""
    
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
    
    # Build prediction
    prediction = {
        "sample_id": sample_id,
        "lat": lat,
        "lon": lon,
        "has_solar": has_solar,
        "confidence": confidence,
        "pv_area_sqm_est": area_sqm,
        "buffer_radius_sqft": buffer_zone,
        "qc_status": qc_status,
        "bbox_or_mask": bbox_or_mask,
        "image_metadata": {
            "source": "ArcGIS World_Imagery",
            "capture_date": "UNKNOWN"
        }
    }
    
    logger.info(f"Sample {sample_id}: has_solar={has_solar}, confidence={confidence:.2f}, "
                f"area={area_sqm:.2f} m², QC={qc_status}")
    
    return prediction


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
    
    # Load ensemble models - combining 3 models with equal weighting
    logger.info(f"Loading primary model from {model_path}")
    
    # Define all ensemble models
    ensemble_models = [
        "model/ensemble_models/solarpanel_seg_v2.pt",
        "model/ensemble_models/solarpanel_seg_v3.pt"
    ]
    
    # Check which models exist
    available_models = [m for m in ensemble_models if Path(m).exists()]
    
    if available_models:
        logger.info(f"Found {len(available_models)} additional ensemble models:")
        for model in available_models:
            logger.info(f"  - {model}")
    
    # Initialize detector with ensemble
    detector = SolarPanelDetector(model_path, ensemble_models=available_models if available_models else None)
    logger.info(f"Ensemble initialized: {len(available_models) + 1} models with equal weighting")
    
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
        description="End-to-end rooftop PV detection pipeline for EcoInnovators Ideathon"
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
        help="Path to YOLOv8 model weights (default: model/model_weights/solarpanel_seg_v1.pt)"
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
    logger.info("EcoInnovators Rooftop PV Detection Pipeline")
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
