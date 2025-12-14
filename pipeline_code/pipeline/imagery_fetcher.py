"""
Imagery fetcher for Google Maps Satellite imagery.
Automated retrieval system - no API key required.
"""

import time
import logging
import math
import os
from pathlib import Path
from typing import Dict, Optional
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from .config import (
    IMAGE_SIZE_PX,
    MAX_RETRIES,
    RETRY_DELAY
)
from .buffer_geometry import compute_pixel_scale

logger = logging.getLogger(__name__)


def get_browser_driver():
    """
    Initialize imagery capture driver with multi-backend fallback support.
    Tries Chrome → Edge → Firefox → Brave → Opera until one works.
    
    Returns:
        Driver instance for imagery access
    """
    browsers = [
        ('Chrome', lambda: webdriver.Chrome(options=get_chrome_options())),
        ('Edge', lambda: webdriver.Edge(options=get_edge_options())),
        ('Firefox', lambda: webdriver.Firefox(options=get_firefox_options())),
        ('Brave', lambda: webdriver.Chrome(options=get_brave_options())),
        ('Opera', lambda: webdriver.Chrome(options=get_opera_options())),
    ]
    
    last_error = None
    for browser_name, get_driver in browsers:
        try:
            logger.debug(f"Trying {browser_name} browser...")
            driver = get_driver()
            logger.info(f"✓ Using {browser_name} browser")
            return driver
        except Exception as e:
            last_error = e
            logger.debug(f"{browser_name} not available: {str(e)}")
            continue
    
    raise RuntimeError(
        f"No browser available. Install Chrome, Edge, Firefox, Brave, or Opera.\n"
        f"Last error: {last_error}"
    )


def get_chrome_options():
    """Get Chrome browser options for headless mode."""
    chrome_options = ChromeOptions()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--log-level=3")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
    return chrome_options


def get_edge_options():
    """Get Edge browser options for headless mode."""
    edge_options = EdgeOptions()
    edge_options.add_argument("--headless=new")
    edge_options.add_argument("--window-size=1920,1080")
    edge_options.add_argument("--disable-blink-features=AutomationControlled")
    edge_options.add_argument("--disable-gpu")
    edge_options.add_argument("--no-sandbox")
    return edge_options


def get_firefox_options():
    """Get Firefox browser options for headless mode."""
    firefox_options = FirefoxOptions()
    firefox_options.add_argument("--headless")
    firefox_options.add_argument("--width=1920")
    firefox_options.add_argument("--height=1080")
    return firefox_options


def get_brave_options():
    """Get Brave browser options for headless mode."""
    brave_options = ChromeOptions()
    
    # Common Brave installation paths
    brave_paths = [
        r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
        r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
        os.path.expanduser(r"~\AppData\Local\BraveSoftware\Brave-Browser\Application\brave.exe"),
    ]
    
    for brave_path in brave_paths:
        if os.path.exists(brave_path):
            brave_options.binary_location = brave_path
            break
    
    brave_options.add_argument("--headless=new")
    brave_options.add_argument("--window-size=1920,1080")
    brave_options.add_argument("--disable-blink-features=AutomationControlled")
    brave_options.add_argument("--disable-gpu")
    brave_options.add_argument("--no-sandbox")
    brave_options.add_argument("--disable-dev-shm-usage")
    return brave_options


def get_opera_options():
    """Get Opera browser options for headless mode."""
    opera_options = ChromeOptions()
    
    # Common Opera installation paths
    opera_paths = [
        r"C:\Program Files\Opera\launcher.exe",
        r"C:\Program Files (x86)\Opera\launcher.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Opera\launcher.exe"),
    ]
    
    for opera_path in opera_paths:
        if os.path.exists(opera_path):
            opera_options.binary_location = opera_path
            break
    
    opera_options.add_argument("--headless=new")
    opera_options.add_argument("--window-size=1920,1080")
    opera_options.add_argument("--disable-blink-features=AutomationControlled")
    opera_options.add_argument("--disable-gpu")
    opera_options.add_argument("--no-sandbox")
    return opera_options


def fetch_google_maps_satellite(
    lat: float,
    lon: float,
    area_sqft: float,
    size_px: int = IMAGE_SIZE_PX,
    out_path: str = None
) -> Dict:
    """
    Fetch satellite imagery from Google Maps at maximum zoom level.
    
    Uses automated retrieval system to capture imagery at zoom level 21 for highest detail.
    Captures 12,900 sq ft area, then buffer filtering is applied in the detection 
    pipeline based on the requested area_sqft parameter.
    
    No API key or authentication required. Suitable for academic/ideathon use.
    
    Args:
        lat: Latitude in degrees (WGS84)
        lon: Longitude in degrees (WGS84)
        area_sqft: Buffer area in square feet (used for metadata, actual capture is 12,900 sqft)
        size_px: Output image size in pixels (square image)
        out_path: Path where the image should be saved
        
    Returns:
        Dictionary containing:
            - success: bool indicating if fetch was successful
            - image_path: path to saved image
            - bbox: (xmin, ymin, xmax, ymax) - estimated
            - ground_width_m: width of image in meters (actual 12,900 sqft area)
            - ground_height_m: height of image in meters
            - meters_per_pixel_x: horizontal resolution
            - meters_per_pixel_y: vertical resolution
            - error: error message if failed
    """
    result = {
        "success": False,
        "image_path": None,
        "bbox": None,
        "ground_width_m": None,
        "ground_height_m": None,
        "meters_per_pixel_x": None,
        "meters_per_pixel_y": None,
        "error": None
    }
    
    driver = None
    temp_screenshot = None
    
    try:
        # IMPORTANT: Always capture at 12,900 sq ft for max zoom detail
        # The area_sqft parameter is used for metadata only
        # Buffer zone filtering happens later in the pipeline based on area_sqft
        capture_area_sqft = 12900
        
        logger.info(f"Fetching Google Maps imagery for lat={lat}, lon={lon}")
        logger.debug(f"Capture area: {capture_area_sqft} sq.ft (max zoom 21), requested buffer: {area_sqft} sq.ft")
        
        # Initialize browser driver
        driver = get_browser_driver()
        
        # Construct Google Maps URL with satellite view and max zoom
        # @satellite specifies satellite view, z=21 sets zoom level to maximum
        maps_url = f"https://www.google.com/maps/@{lat},{lon},21z/data=!3m1!1e3"
        
        logger.debug(f"Loading Google Maps at zoom 21...")
        driver.get(maps_url)
        
        # Smart waiting - wait for map to load
        max_wait = 10
        start_time = time.time()
        map_loaded = False
        
        while (time.time() - start_time) < max_wait and not map_loaded:
            try:
                driver.find_element(By.ID, "mapDiv")
                if driver.execute_script("return document.readyState") == "complete":
                    map_loaded = True
                    elapsed = round(time.time() - start_time, 1)
                    logger.debug(f"Map loaded in {elapsed}s")
                    break
            except:
                pass
            time.sleep(0.3)
        
        if not map_loaded:
            logger.warning("Map loading timeout, continuing anyway...")
            time.sleep(1)
        
        # Wait for satellite tiles to load
        logger.debug("Waiting for satellite imagery tiles...")
        time.sleep(0.5)
        
        try:
            pending = driver.execute_script("""
                return window.performance.getEntriesByType('resource')
                    .filter(r => r.duration === 0).length;
            """)
            
            if pending > 5:
                logger.debug(f"Waiting for {pending} resources to load...")
                time.sleep(1.5)
            else:
                time.sleep(0.5)
        except:
            time.sleep(1)
        
        # Take screenshot
        logger.debug("Capturing screenshot...")
        temp_screenshot = f"temp_screenshot_{lat}_{lon}.png"
        driver.save_screenshot(temp_screenshot)
        
        # Crop to 12,900 sq ft area (max zoom detail)
        logger.debug(f"Processing image for {capture_area_sqft} sq.ft area at max zoom...")
        crop_to_area(temp_screenshot, out_path, capture_area_sqft, lat, size_px)
        
        # Compute pixel scale based on ACTUAL captured area (12,900 sqft)
        ground_width_m, ground_height_m, meters_per_pixel_x, meters_per_pixel_y = \
            compute_pixel_scale(capture_area_sqft, size_px)
        
        # Estimate bbox (approximate, based on ACTUAL captured area)
        side_length_m = math.sqrt(capture_area_sqft * 0.09290304)  # sqft to sqm
        lat_delta = (side_length_m / 2) / 111320  # meters to degrees latitude
        lon_delta = (side_length_m / 2) / (111320 * math.cos(math.radians(lat)))
        
        xmin = lon - lon_delta
        ymin = lat - lat_delta
        xmax = lon + lon_delta
        ymax = lat + lat_delta
        
        # Fill in result
        result["success"] = True
        result["image_path"] = str(out_path)
        result["bbox"] = (xmin, ymin, xmax, ymax)
        result["ground_width_m"] = ground_width_m
        result["ground_height_m"] = ground_height_m
        result["meters_per_pixel_x"] = meters_per_pixel_x
        result["meters_per_pixel_y"] = meters_per_pixel_y
        
        logger.info(f"Successfully saved imagery to {out_path}")
        logger.debug(f"Image scale: {meters_per_pixel_x:.2f} m/px")
        
    except Exception as e:
        result["error"] = f"Error fetching Google Maps imagery: {str(e)}"
        logger.exception("Error in fetch_google_maps_satellite")
    
    finally:
        # Cleanup
        if driver:
            try:
                driver.quit()
            except:
                pass
        
        if temp_screenshot and os.path.exists(temp_screenshot):
            try:
                os.remove(temp_screenshot)
            except:
                pass
    
    return result


def crop_to_area(input_file: str, output_file: str, area_sqft: float, latitude: float, output_size_px: int = 640):
    """
    Crop screenshot to cover specific area in square feet and resize to target size.
    Uses precise Google Maps zoom level 21 calculations.
    
    Args:
        input_file: Input screenshot file path
        output_file: Output cropped file path
        area_sqft: Target area in square feet
        latitude: Latitude for accurate meter-to-pixel conversion
        output_size_px: Final output image size in pixels (square)
    """
    # Open the screenshot
    img = Image.open(input_file)
    width, height = img.size
    
    # Google Maps zoom level 21 calculation:
    # At zoom level z, the map width is 256 * 2^z pixels for the full world
    # Earth's circumference at equator = 40,075,017 meters
    # meters_per_pixel = (Earth_circumference * cos(latitude)) / (256 * 2^zoom)
    
    zoom_level = 21
    earth_circumference = 40075017  # meters at equator
    
    # Calculate meters per pixel at this zoom and latitude
    meters_per_pixel = (earth_circumference * math.cos(math.radians(latitude))) / (256 * math.pow(2, zoom_level))
    
    # Convert square feet to square meters (1 sq ft = 0.09290304 sq m)
    area_sqm = area_sqft * 0.09290304
    
    # Calculate side length in meters (assuming square area)
    side_length_meters = math.sqrt(area_sqm)
    
    # Convert to pixels
    side_length_pixels = int(side_length_meters / meters_per_pixel)
    
    # Calculate crop box (centered on middle of screenshot)
    center_x = width // 2
    center_y = height // 2
    
    half_side = side_length_pixels // 2
    
    left = max(0, center_x - half_side)
    top = max(0, center_y - half_side)
    right = min(width, center_x + half_side)
    bottom = min(height, center_y + half_side)
    
    # Crop
    cropped = img.crop((left, top, right, bottom))
    
    # Resize to target output size
    resized = cropped.resize((output_size_px, output_size_px), Image.Resampling.LANCZOS)
    
    # Save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    resized.save(output_file)
    
    logger.debug(f"Cropped from {img.size} to {cropped.size}, resized to {resized.size}")
    logger.debug(f"Area covered: ~{int(area_sqft)} sq.ft ({int(area_sqm)} sq.m)")


# Main fetch function (alias for backward compatibility)
fetch_arcgis_world_imagery = fetch_google_maps_satellite


def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Validate that coordinates are within valid ranges.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        
    Returns:
        True if coordinates are valid, False otherwise
    """
    if not (-90 <= lat <= 90):
        logger.error(f"Invalid latitude: {lat} (must be between -90 and 90)")
        return False
    
    if not (-180 <= lon <= 180):
        logger.error(f"Invalid longitude: {lon} (must be between -180 and 180)")
        return False
    
    return True
