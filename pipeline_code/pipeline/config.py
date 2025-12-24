"""
Configuration file for the EcoInnovators Ideathon pipeline.
Contains paths, constants, and settings.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
# Go up one more level to reach the main project root
MAIN_PROJECT_ROOT = PROJECT_ROOT.parent

# Input/Output directories
INPUTS_DIR = PROJECT_ROOT / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
OVERLAYS_DIR = OUTPUTS_DIR / "overlays"
OUTPUT_PREDICTIONS_DIR = str(PREDICTIONS_DIR)  # String version for compatibility
OUTPUT_OVERLAYS_DIR = OVERLAYS_DIR  # Path object
LOGS_DIR = PROJECT_ROOT / "logs"

# Model directories - Point to actual trained model files location
MODEL_WEIGHTS_DIR = MAIN_PROJECT_ROOT / "trained_model_files"
MODEL_PATH = MODEL_WEIGHTS_DIR / "solarpanel_seg_v1.pt"
MODEL_WEIGHTS_PATH = str(MODEL_PATH)  # String version for compatibility

# Buffer zone settings (in square feet)
BUFFER_ZONE_1 = 1200  # First buffer zone: 1200 sq.ft
BUFFER_ZONE_2 = 2400  # Second buffer zone: 2400 sq.ft

# Image settings
IMAGE_SIZE_PX = 640  # Image size in pixels (640x640 for YOLOv8)
IMAGE_FORMAT = "png"

# Imagery fetch settings
# Google Maps captures at 12,900 sq ft (max zoom 21 for highest detail)
# Buffer zones (1200/2400 sqft) are filtered in pixel space during detection
IMAGERY_FETCH_SIZE_M = 33.3  # ~12,900 sq ft = ~1,200 sq m = 33.3m x 33.3m
IMAGERY_FETCH_SIZE_SQFT = 12900  # Fixed capture area for max zoom detail

# Google Maps settings
# Automated satellite imagery retrieval system
# Maximum zoom level 21 for highest detail satellite imagery
# Multi-platform backend support for reliability
GOOGLE_MAPS_ZOOM_LEVEL = 21  # Maximum zoom for detailed imagery

# Conversion constants
SQFT_TO_SQM = 0.092903  # 1 sq.ft = 0.092903 sq.m
METERS_PER_DEGREE_LAT = 111320  # Approximate meters per degree latitude (WGS84)

# QC Status thresholds
MIN_CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence for detection
MIN_IMAGE_QUALITY_THRESHOLD = 0.5  # Placeholder for image quality assessment

# Retry settings
MAX_RETRIES = 3  # Increased retries for network issues
RETRY_DELAY = 3  # seconds

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Overlay visualization settings (BGR format for OpenCV)
OVERLAY_PANEL_COLOR = (0, 255, 0)  # Green for detected panels
OVERLAY_BUFFER_COLOR = (255, 165, 0)  # Blue for buffer zone
OVERLAY_SELECTED_COLOR = (255, 0, 0)  # Red for selected panel
OVERLAY_LINE_THICKNESS = 2
OVERLAY_ALPHA = 0.3  # Transparency for filled polygons

# Create directories if they don't exist
for directory in [INPUTS_DIR, PREDICTIONS_DIR, OVERLAYS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
