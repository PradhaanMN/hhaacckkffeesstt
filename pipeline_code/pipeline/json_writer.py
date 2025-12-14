"""
JSON writer for creating output files in the required EcoInnovators format.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def write_prediction_json(
    sample_id: int,
    lat: float,
    lon: float,
    has_solar: bool,
    confidence: float,
    pv_area_sqm_est: float,
    buffer_radius_sqft: int,
    qc_status: str,
    bbox_or_mask: str,
    image_metadata: Dict,
    output_dir: str
) -> str:
    """
    Write prediction results to JSON file in the required format.
    
    Required JSON schema:
    {
      "sample_id": 1234,
      "lat": 12.9716,
      "lon": 77.5946,
      "has_solar": true,
      "confidence": 0.92,
      "pv_area_sqm_est": 23.5,
      "buffer_radius_sqft": 1200,
      "qc_status": "VERIFIABLE",
      "bbox_or_mask": "<encoded polygon or bbox>",
      "image_metadata": {"source": "ArcGIS World_Imagery", "capture_date": "UNKNOWN"}
    }
    
    Args:
        sample_id: Unique sample identifier
        lat: Latitude
        lon: Longitude
        has_solar: Whether solar panels are present
        confidence: Model confidence score
        pv_area_sqm_est: Estimated PV area in square meters
        buffer_radius_sqft: Buffer radius used (1200 or 2400)
        qc_status: "VERIFIABLE" or "NOT_VERIFIABLE"
        bbox_or_mask: Encoded polygon or bounding box
        image_metadata: Metadata about the source image
        output_dir: Directory to save JSON file
        
    Returns:
        Path to the saved JSON file
    """
    # Ensure image_metadata has required fields
    if not image_metadata:
        image_metadata = {}
    
    if "source" not in image_metadata:
        image_metadata["source"] = "ArcGIS World_Imagery"
    
    if "capture_date" not in image_metadata:
        image_metadata["capture_date"] = "UNKNOWN"
    
    # Create the JSON structure
    result = {
        "sample_id": int(sample_id),
        "lat": round(float(lat), 6),
        "lon": round(float(lon), 6),
        "has_solar": bool(has_solar),
        "confidence": round(float(confidence), 4),
        "pv_area_sqm_est": round(float(pv_area_sqm_est), 2),
        "buffer_radius_sqft": int(buffer_radius_sqft),
        "qc_status": str(qc_status),
        "bbox_or_mask": str(bbox_or_mask),
        "image_metadata": image_metadata
    }
    
    # Create output directory if needed
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write JSON file
    json_file = output_path / f"{sample_id}.json"
    
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Wrote prediction JSON to {json_file}")
        return str(json_file)
        
    except Exception as e:
        logger.exception(f"Error writing JSON file: {e}")
        return None


def create_summary_report(
    predictions: List[Dict],
    output_path: str
) -> str:
    """
    Create a summary report of all predictions.
    
    Args:
        predictions: List of all prediction dictionaries
        output_path: Path to save the summary report
        
    Returns:
        Path to the saved summary file
    """
    if not predictions:
        logger.warning("No predictions to summarize")
        return None
    
    # Calculate statistics
    total_samples = len(predictions)
    with_solar = sum(1 for p in predictions if p.get("has_solar", False))
    without_solar = total_samples - with_solar
    verifiable = sum(1 for p in predictions if p.get("qc_status") == "VERIFIABLE")
    not_verifiable = total_samples - verifiable
    
    # Calculate average PV area (only for sites with solar)
    pv_areas = [p.get("pv_area_sqm_est", 0) for p in predictions if p.get("has_solar", False)]
    avg_pv_area = sum(pv_areas) / len(pv_areas) if pv_areas else 0
    
    # Calculate average confidence
    confidences = [p.get("confidence", 0) for p in predictions]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    # Create summary
    summary = {
        "total_samples": total_samples,
        "with_solar": with_solar,
        "without_solar": without_solar,
        "verifiable": verifiable,
        "not_verifiable": not_verifiable,
        "statistics": {
            "solar_detection_rate": round(with_solar / total_samples * 100, 2) if total_samples > 0 else 0,
            "verifiable_rate": round(verifiable / total_samples * 100, 2) if total_samples > 0 else 0,
            "average_pv_area_sqm": round(avg_pv_area, 2),
            "average_confidence": round(avg_confidence, 4)
        },
        "buffer_zone_usage": {
            "1200_sqft": sum(1 for p in predictions if p.get("buffer_radius_sqft") == 1200),
            "2400_sqft": sum(1 for p in predictions if p.get("buffer_radius_sqft") == 2400)
        }
    }
    
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Wrote summary report to {output_path}")
        return str(output_file)
        
    except Exception as e:
        logger.exception(f"Error writing summary report: {e}")
        return None
