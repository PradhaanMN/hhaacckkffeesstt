"""
FastAPI Backend for EcoInnovators Ideathon 2026
Rooftop Solar Panel Detection - Governance-Ready Digital Verification Pipeline
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import logging
from pathlib import Path
import sys
import uuid
import json
from datetime import datetime

# Add parent directory to path to import pipeline modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import (
    BUFFER_ZONE_1, BUFFER_ZONE_2, IMAGE_SIZE_PX,
    MODEL_WEIGHTS_PATH, OUTPUT_PREDICTIONS_DIR, OUTPUT_OVERLAYS_DIR
)
from pipeline.buffer_geometry import compute_pixel_scale
from pipeline.imagery_fetcher import fetch_arcgis_world_imagery, validate_coordinates  # Backward compatible alias
from pipeline.qc_logic import determine_qc_status, check_image_quality
from pipeline.overlay_generator import create_overlay_image, encode_polygon_for_json
from pipeline.json_writer import write_prediction_json
from model.model_inference import SolarPanelDetector
from pipeline.main import (
    find_panels_in_buffer, 
    select_largest_panel_in_buffer,
    convert_pixel_area_to_sqm,
    process_single_location
)
from backend.pdf_generator import create_pdf_report, create_batch_pdf_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Input validation and security utilities
import re
from typing import Union

def sanitize_string_input(value: str, max_length: int = 100) -> str:
    """
    Sanitize string input to prevent injection attacks.
    Removes special characters, limits length.
    """
    if not isinstance(value, str):
        raise ValueError(f"Expected string, got {type(value).__name__}")
    
    # Remove any special characters that could be used in injection attacks
    # Allow only alphanumeric, spaces, hyphens, underscores
    sanitized = re.sub(r'[^a-zA-Z0-9\s\-_]', '', value)
    
    # Limit length
    sanitized = sanitized[:max_length]
    
    return sanitized.strip()

def validate_coordinate_detailed(lat: float, lon: float) -> tuple[bool, str]:
    """
    Enhanced coordinate validation with detailed error messages.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    # Type validation
    if not isinstance(lat, (int, float)):
        return False, f"Latitude must be numeric, got {type(lat).__name__}"
    
    if not isinstance(lon, (int, float)):
        return False, f"Longitude must be numeric, got {type(lon).__name__}"
    
    # Check for NaN or Infinity
    import math
    if math.isnan(lat) or math.isinf(lat):
        return False, f"Latitude cannot be NaN or Infinity"
    
    if math.isnan(lon) or math.isinf(lon):
        return False, f"Longitude cannot be NaN or Infinity"
    
    # Range validation
    if lat < -90 or lat > 90:
        return False, f"Latitude {lat} out of range (must be -90 to 90)"
    
    if lon < -180 or lon > 180:
        return False, f"Longitude {lon} out of range (must be -180 to 180)"
    
    # Check for suspicious exact values (potential test/injection values)
    if lat == 0 and lon == 0:
        return False, "Null Island (0,0) is not a valid location for rooftop detection"
    
    # Precision check (too many decimals might indicate malicious input)
    lat_str = str(lat)
    lon_str = str(lon)
    if '.' in lat_str and len(lat_str.split('.')[1]) > 10:
        return False, "Latitude precision too high (max 10 decimal places)"
    
    if '.' in lon_str and len(lon_str.split('.')[1]) > 10:
        return False, "Longitude precision too high (max 10 decimal places)"
    
    return True, ""

def validate_sample_id(sample_id: int) -> tuple[bool, str]:
    """
    Validate sample_id to prevent overflow and injection.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(sample_id, int):
        return False, f"sample_id must be integer, got {type(sample_id).__name__}"
    
    # Check reasonable range (timestamp-based IDs should be positive and within reasonable bounds)
    if sample_id <= 0:
        return False, "sample_id must be positive"
    
    # Check for unreasonably large values (could be overflow attempt)
    MAX_SAMPLE_ID = 9999999999999  # Year 2286 in milliseconds
    if sample_id > MAX_SAMPLE_ID:
        return False, f"sample_id {sample_id} is unreasonably large"
    
    return True, ""

# Initialize FastAPI app
app = FastAPI(
    title="NeuralStack Rooftop PV Detection API",
    description="Governance-ready digital verification pipeline for PM Surya Ghar",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (frontend)
frontend_path = Path(__file__).parent / "static"
frontend_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

# Global model instance (load once)
model_instance = None


def get_model():
    """Lazy load the ensemble model (load once, reuse)"""
    global model_instance
    if model_instance is None:
        logger.info("Loading YOLOv8 ensemble model...")
        
        # Get the main project root
        main_root = Path(__file__).parent.parent.parent
        
        # PRIMARY MODEL: Custom-trained model (solarpanel_seg_v1.pt)
        # - Annotated and trained specifically for this project
        # - Capable of both detection and segmentation
        # - Given 2x weight in ensemble voting
        # - Gets priority in confidence calculation
        
        # ENSEMBLE MODELS: Additional models for consensus voting
        ensemble_models = [
            str(main_root / "trained_model" / "solarpanel_seg_v2.pt"),
            str(main_root / "trained_model" / "solarpanel_seg_v3.pt"),
            str(main_root / "trained_model" / "solarpanel_seg_v4.pt"),
            str(main_root / "trained_model" / "solarpanel_det_v4.pt")  # Detection-only model
        ]
        
        # Check which models exist
        available_models = [m for m in ensemble_models if Path(m).exists()]
        
        if available_models:
            logger.info(f"Found {len(available_models)} additional ensemble models")
            for m in available_models:
                logger.info(f"  - {Path(m).name}")
        
        # Initialize detector with ensemble
        model_instance = SolarPanelDetector(
            MODEL_WEIGHTS_PATH,
            ensemble_models=available_models if available_models else None
        )
        logger.info(f"Ensemble loaded: {len(available_models) + 1} models (v1 + ensemble)")
    return model_instance


# Pydantic models for API
class LocationRequest(BaseModel):
    """Single location verification request with enhanced validation"""
    sample_id: int = Field(..., gt=0, lt=9999999999999, description="Unique sample identifier (positive integer)")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")
    use_hybrid: bool = Field(True, description="Use hybrid ensemble algorithm (default: True)")
    
    class Config:
        # Prevent extra fields that could be used for injection
        extra = 'forbid'
    
    @staticmethod
    def validate_request(request: 'LocationRequest') -> tuple[bool, str]:
        """Additional validation beyond Pydantic constraints"""
        # Validate sample_id
        valid_id, id_error = validate_sample_id(request.sample_id)
        if not valid_id:
            return False, id_error
        
        # Detailed coordinate validation
        valid_coords, coord_error = validate_coordinate_detailed(request.latitude, request.longitude)
        if not valid_coords:
            return False, coord_error
        
        return True, ""


class BatchLocationRequest(BaseModel):
    """Batch location verification request"""
    locations: List[LocationRequest] = Field(..., description="List of locations to process")


class PowerEstimate(BaseModel):
    """Power generation estimate"""
    peak_power_kw: float
    daily_energy_kwh: float
    monthly_energy_kwh: float
    yearly_energy_kwh: float


class VerificationResponse(BaseModel):
    """Verification result for a single location"""
    sample_id: int
    lat: float
    lon: float
    has_solar: bool
    confidence: float
    pv_area_sqm_est: float
    buffer_radius_sqft: int
    qc_status: str
    bbox_or_mask: str
    power_estimate: PowerEstimate
    image_metadata: dict
    overlay_url: Optional[str] = None
    processing_time_seconds: Optional[float] = None


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None


class FeedbackRequest(BaseModel):
    """User feedback for reinforcement learning with validation"""
    sample_id: str = Field(..., min_length=1, max_length=50, description="Sample identifier")
    rating: str = Field(..., pattern="^(good|bad)$", description="Rating: 'good' or 'bad'")
    timestamp: str = Field(..., min_length=1, max_length=100, description="ISO timestamp")
    satellite_image_available: Optional[bool] = False
    
    class Config:
        extra = 'forbid'
    
    @staticmethod
    def validate_feedback(feedback: 'FeedbackRequest') -> tuple[bool, str]:
        """Validate feedback request for security"""
        # Sanitize sample_id (prevent path traversal)
        if '../' in feedback.sample_id or '..' in feedback.sample_id:
            return False, "sample_id contains invalid path traversal characters"
        
        # Validate sample_id contains only safe characters
        if not re.match(r'^[a-zA-Z0-9_-]+$', feedback.sample_id):
            return False, "sample_id contains invalid characters (only alphanumeric, _, - allowed)"
        
        # Validate rating
        if feedback.rating not in ['good', 'bad']:
            return False, "rating must be 'good' or 'bad'"
        
        # Validate timestamp format (basic ISO 8601 check)
        if not re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', feedback.timestamp):
            return False, "timestamp must be in ISO 8601 format"
        
        return True, ""


@app.get("/")
async def root():
    """Serve the frontend HTML"""
    html_file = Path(__file__).parent / "static" / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    return {"message": "NeuralStack Rooftop PV Detection API", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_instance is not None,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/verify/single", response_model=VerificationResponse)
async def verify_single_location(request: LocationRequest):
    """
    Verify a single location for rooftop solar panels.
    
    This endpoint:
    1. Fetches imagery from automated retrieval system (no API key required)
    2. Runs ML inference using trained YOLOv8 model
    3. Applies buffer zone logic (1200 sq.ft → 2400 sq.ft)
    4. Calculates panel area and QC status
    5. Returns JSON response with all required fields
    """
    import time
    start_time = time.time()
    
    try:
        # Enhanced validation
        is_valid, error_msg = LocationRequest.validate_request(request)
        if not is_valid:
            logger.warning(f"Validation failed: {error_msg}")
            raise HTTPException(
                status_code=400,
                detail=f"Input validation failed: {error_msg}"
            )
        
        # Additional coordinate validation (backward compatible)
        if not validate_coordinates(request.latitude, request.longitude):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid coordinates: lat={request.latitude}, lon={request.longitude}"
            )
        
        # Create temporary directory for this request
        temp_dir = Path("temp_images")
        temp_dir.mkdir(exist_ok=True)
        
        # Load model
        detector = get_model()
        
        # Process the location using existing pipeline logic
        logger.info(f"Processing location: sample_id={request.sample_id}, lat={request.latitude}, lon={request.longitude}, use_hybrid={request.use_hybrid}")
        
        result = process_single_location(
            sample_id=request.sample_id,
            lat=request.latitude,
            lon=request.longitude,
            detector=detector,
            temp_dir=temp_dir,
            use_hybrid=request.use_hybrid
        )
        
        # Write JSON to output
        json_path = write_prediction_json(
            sample_id=result["sample_id"],
            lat=result["lat"],
            lon=result["lon"],
            has_solar=result["has_solar"],
            confidence=result["confidence"],
            pv_area_sqm_est=result["pv_area_sqm_est"],
            euclidean_distance_m_est=result["euclidean_distance_m_est"],
            buffer_radius_sqft=result["buffer_radius_sqft"],
            qc_status=result["qc_status"],
            bbox_or_mask=result["bbox_or_mask"],
            image_metadata=result["image_metadata"],
            output_dir=str(OUTPUT_PREDICTIONS_DIR)
        )
        
        # Get overlay URL (relative path for frontend)
        overlay_file = OUTPUT_OVERLAYS_DIR / f"{request.sample_id}_overlay.png"
        overlay_url = f"/outputs/overlays/{request.sample_id}_overlay.png" if overlay_file.exists() else None
        
        processing_time = time.time() - start_time
        
        logger.info(f"Successfully processed sample {request.sample_id} in {processing_time:.2f}s")
        
        return VerificationResponse(
            sample_id=result["sample_id"],
            lat=result["lat"],
            lon=result["lon"],
            has_solar=result["has_solar"],
            confidence=result["confidence"],
            pv_area_sqm_est=result["pv_area_sqm_est"],
            buffer_radius_sqft=result["buffer_radius_sqft"],
            qc_status=result["qc_status"],
            bbox_or_mask=result["bbox_or_mask"],
            power_estimate=result.get("power_estimate", {
                "peak_power_kw": 0.0,
                "daily_energy_kwh": 0.0,
                "monthly_energy_kwh": 0.0,
                "yearly_energy_kwh": 0.0
            }),
            image_metadata=result["image_metadata"],
            overlay_url=overlay_url,
            processing_time_seconds=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.exception(f"Error processing location: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/verify/batch")
async def verify_batch_locations(request: BatchLocationRequest):
    """
    Verify multiple locations in batch.
    
    Processes each location sequentially and returns results for all.
    """
    results = []
    errors = []
    
    for loc in request.locations:
        try:
            result = await verify_single_location(loc)
            results.append(result.dict())
        except HTTPException as e:
            errors.append({
                "sample_id": loc.sample_id,
                "error": e.detail
            })
        except Exception as e:
            errors.append({
                "sample_id": loc.sample_id,
                "error": str(e)
            })
    
    return {
        "total_requested": len(request.locations),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors
    }


@app.get("/api/result/{sample_id}")
async def get_result(sample_id: int):
    """
    Retrieve previously computed result for a sample ID.
    """
    json_file = OUTPUT_PREDICTIONS_DIR / f"{sample_id}.json"
    
    if not json_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No result found for sample_id {sample_id}"
        )
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Add overlay URL if exists
    overlay_file = OUTPUT_OVERLAYS_DIR / f"{sample_id}_overlay.png"
    if overlay_file.exists():
        data["overlay_url"] = f"/outputs/overlays/{sample_id}_overlay.png"
    
    return data


@app.get("/api/overlay/{sample_id}")
async def get_overlay(sample_id: int):
    """
    Retrieve overlay image for a sample ID.
    """
    overlay_file = OUTPUT_OVERLAYS_DIR / f"{sample_id}_overlay.png"
    
    if not overlay_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No overlay found for sample_id {sample_id}"
        )
    
    return FileResponse(overlay_file, media_type="image/png")


@app.get("/api/demo/overlay")
async def get_demo_overlay():
    """
    Get demo overlay image showing the visualization system.
    This demonstrates the GREEN/RED box system even when imagery is unavailable.
    """
    # Try test visualization first
    test_overlay = Path(__file__).parent.parent / "outputs" / "test_visualizations" / "overlay_buffer_1200.png"
    
    if test_overlay.exists():
        return FileResponse(test_overlay, media_type="image/png")
    
    # Fallback to any available overlay
    overlays_dir = Path(__file__).parent.parent / "outputs" / "overlays"
    if overlays_dir.exists():
        overlays = list(overlays_dir.glob("*_overlay.png"))
        if overlays:
            return FileResponse(overlays[0], media_type="image/png")
    
    raise HTTPException(
        status_code=404,
        detail="No demo overlay available. Run test_visualization.py first."
    )


# Mount outputs directory for serving overlays
outputs_path = Path(__file__).parent.parent / "outputs"
if outputs_path.exists():
    app.mount("/outputs", StaticFiles(directory=str(outputs_path)), name="outputs")


@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Collect user feedback for reinforcement learning
    
    This endpoint saves user ratings (thumbs up/down) for detected solar panels.
    Only BAD ratings are saved with images for retraining purposes.
    The feedback can be used to:
    - Retrain models with corrected annotations
    - Identify and fix false positives/negatives
    - Improve detection accuracy over time
    """
    try:
        # Validate feedback request
        is_valid, error_msg = FeedbackRequest.validate_feedback(feedback)
        if not is_valid:
            logger.warning(f"Feedback validation failed: {error_msg}")
            raise HTTPException(
                status_code=400,
                detail=f"Feedback validation failed: {error_msg}"
            )
        
        # Create feedback directory structure
        feedback_dir = Path(__file__).parent.parent / "outputs" / "feedback"
        feedback_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectory only for bad images (for retraining)
        images_bad_dir = feedback_dir / "images" / "bad"
        images_bad_dir.mkdir(parents=True, exist_ok=True)
        
        # Save feedback to CSV file
        feedback_file = feedback_dir / "user_feedback.csv"
        
        # Create file with header if it doesn't exist
        if not feedback_file.exists():
            with open(feedback_file, 'w') as f:
                f.write("timestamp,sample_id,rating,overlay_path,satellite_path\n")
        
        # Only save images for BAD ratings (for retraining)
        overlay_saved_path = None
        satellite_saved_path = None
        
        if feedback.rating == "bad":
            import shutil
            timestamp_clean = feedback.timestamp.replace(':', '-').replace('.', '-')
            
            # Save overlay image
            overlay_source = Path(__file__).parent.parent / "outputs" / "overlays" / f"{feedback.sample_id}_overlay.png"
            if overlay_source.exists():
                destination_file = images_bad_dir / f"{feedback.sample_id}_{timestamp_clean}_overlay.png"
                shutil.copy2(overlay_source, destination_file)
                overlay_saved_path = str(destination_file.relative_to(Path(__file__).parent.parent))
                logger.info(f"Saved overlay for retraining: {overlay_saved_path}")
            else:
                logger.warning(f"Overlay image not found: {overlay_source}")
                overlay_saved_path = "overlay_not_found"
            
            # Save raw satellite image (for retraining with original imagery)
            satellite_source = Path("temp_images") / f"{feedback.sample_id}_satellite.png"
            if satellite_source.exists():
                destination_file = images_bad_dir / f"{feedback.sample_id}_{timestamp_clean}_satellite.png"
                shutil.copy2(satellite_source, destination_file)
                satellite_saved_path = str(destination_file.relative_to(Path(__file__).parent.parent))
                logger.info(f"Saved raw satellite image for retraining: {satellite_saved_path}")
            else:
                logger.warning(f"Satellite image not found: {satellite_source}")
                satellite_saved_path = "satellite_not_found"
        else:
            # Good rating - no images saved, just log the feedback
            overlay_saved_path = "not_saved_good_rating"
            satellite_saved_path = "not_saved_good_rating"
        
        # Append feedback with both image paths
        with open(feedback_file, 'a') as f:
            f.write(f"{feedback.timestamp},{feedback.sample_id},{feedback.rating},{overlay_saved_path},{satellite_saved_path}\n")
        
        logger.info(f"Feedback received: {feedback.sample_id} - {feedback.rating}")
        
        message = "Feedback with overlay and satellite images saved for retraining" if feedback.rating == "bad" else "Feedback recorded (images not saved for good ratings)"
        
        return {
            "status": "success",
            "message": message,
            "sample_id": feedback.sample_id,
            "rating": feedback.rating,
            "overlay_saved": overlay_saved_path if feedback.rating == "bad" else None,
            "satellite_saved": satellite_saved_path if feedback.rating == "bad" else None
        }
    
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")


@app.get("/api/export/pdf/{sample_id}")
async def export_pdf_report(sample_id: int):
    """
    Export detection results as PDF report with statistics.
    
    Args:
        sample_id: Sample ID to export
    
    Returns:
        PDF file download
    """
    try:
        # Find the JSON result file
        json_path = Path(OUTPUT_PREDICTIONS_DIR) / f"{sample_id}.json"
        
        if not json_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Results not found for sample_id {sample_id}"
            )
        
        # Load result data
        with open(json_path, 'r') as f:
            result_data = json.load(f)
        
        # Find overlay image
        overlay_path = Path(OUTPUT_OVERLAYS_DIR) / f"{sample_id}_overlay.png"
        if not overlay_path.exists():
            overlay_path = None
        
        # Create PDF output directory
        pdf_dir = Path(__file__).parent.parent / "outputs" / "reports"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate PDF
        pdf_path = pdf_dir / f"report_{sample_id}.pdf"
        create_pdf_report(
            result_data=result_data,
            overlay_path=str(overlay_path) if overlay_path else None,
            output_path=str(pdf_path),
            include_statistics=True
        )
        
        logger.info(f"Generated PDF report: {pdf_path}")
        
        # Return PDF file
        return FileResponse(
            path=str(pdf_path),
            media_type='application/pdf',
            filename=f"solar_detection_report_{sample_id}.pdf"
        )
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Create necessary directories
    Path("temp_images").mkdir(exist_ok=True)
    Path(OUTPUT_PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_OVERLAYS_DIR).mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting NeuralStack Rooftop PV Detection API...")
    logger.info(f"Model path: {MODEL_WEIGHTS_PATH}")
    logger.info(f"Output predictions: {OUTPUT_PREDICTIONS_DIR}")
    logger.info(f"Output overlays: {OUTPUT_OVERLAYS_DIR}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
