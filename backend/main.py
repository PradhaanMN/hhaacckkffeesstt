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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="EcoInnovators Rooftop PV Detection API",
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
        
        # Define all ensemble models (3 models combined)
        ensemble_models = [
            "model/ensemble_models/solarpanel_seg_v2.pt",
            "model/ensemble_models/solarpanel_seg_v3.pt"
        ]
        
        # Check which models exist
        available_models = [m for m in ensemble_models if Path(m).exists()]
        
        if available_models:
            logger.info(f"Found {len(available_models)} additional ensemble models")
        
        # Initialize detector with ensemble
        model_instance = SolarPanelDetector(
            MODEL_WEIGHTS_PATH,
            ensemble_models=available_models if available_models else None
        )
        logger.info(f"Ensemble loaded: {len(available_models) + 1} models with equal weighting")
    return model_instance


# Pydantic models for API
class LocationRequest(BaseModel):
    """Single location verification request"""
    sample_id: int = Field(..., description="Unique sample identifier")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")


class BatchLocationRequest(BaseModel):
    """Batch location verification request"""
    locations: List[LocationRequest] = Field(..., description="List of locations to process")


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
    image_metadata: dict
    overlay_url: Optional[str] = None
    processing_time_seconds: Optional[float] = None


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None


@app.get("/")
async def root():
    """Serve the frontend HTML"""
    html_file = Path(__file__).parent / "static" / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    return {"message": "EcoInnovators Rooftop PV Detection API", "status": "running"}


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
        # Validate coordinates
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
        logger.info(f"Processing location: sample_id={request.sample_id}, lat={request.latitude}, lon={request.longitude}")
        
        result = process_single_location(
            sample_id=request.sample_id,
            lat=request.latitude,
            lon=request.longitude,
            detector=detector,
            temp_dir=temp_dir
        )
        
        # Write JSON to output
        json_path = write_prediction_json(
            sample_id=result["sample_id"],
            lat=result["lat"],
            lon=result["lon"],
            has_solar=result["has_solar"],
            confidence=result["confidence"],
            pv_area_sqm_est=result["pv_area_sqm_est"],
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
    
    import json
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


if __name__ == "__main__":
    import uvicorn
    
    # Create necessary directories
    Path("temp_images").mkdir(exist_ok=True)
    Path(OUTPUT_PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_OVERLAYS_DIR).mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting EcoInnovators Rooftop PV Detection API...")
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
