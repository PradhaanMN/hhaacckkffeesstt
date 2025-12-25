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
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
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
    Tries default browser first, then Chrome → Chromium → Edge → Firefox → Brave → Vivaldi → Opera.
    
    Returns:
        Driver instance for imagery access
    """
    import winreg
    import subprocess
    
    # Try to get default browser
    default_browser = None
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Software\Microsoft\Windows\Shell\Associations\UrlAssociations\http\UserChoice') as key:
            prog_id = winreg.QueryValueEx(key, 'ProgId')[0]
            if 'Chrome' in prog_id and 'Chromium' not in prog_id:
                default_browser = 'Chrome'
            elif 'Chromium' in prog_id:
                default_browser = 'Chromium'
            elif 'Edge' in prog_id or 'MSEdge' in prog_id:
                default_browser = 'Microsoft Edge'
            elif 'Firefox' in prog_id:
                default_browser = 'Firefox'
            elif 'Brave' in prog_id:
                default_browser = 'Brave'
            elif 'Vivaldi' in prog_id:
                default_browser = 'Vivaldi'
            elif 'Opera' in prog_id:
                default_browser = 'Opera'
    except:
        pass
    
    # Define all browsers with their initialization functions
    all_browsers = {
        'Chrome': lambda: webdriver.Chrome(options=get_chrome_options()),
        'Chromium': lambda: webdriver.Chrome(options=get_chromium_options()),
        'Microsoft Edge': lambda: webdriver.Edge(options=get_edge_options()),
        'Firefox': lambda: webdriver.Firefox(options=get_firefox_options()),
        'Brave': lambda: webdriver.Chrome(options=get_brave_options()),
        'Vivaldi': lambda: webdriver.Chrome(options=get_vivaldi_options()),
        'Opera': lambda: webdriver.Chrome(options=get_opera_options()),
    }
    
    # Build priority list: default browser first, then others
    browsers = []
    if default_browser and default_browser in all_browsers:
        browsers.append((default_browser, all_browsers[default_browser]))
        logger.info(f"Detected default browser: {default_browser}")
    
    # Add remaining browsers in preferred order
    preferred_order = ['Chrome', 'Chromium', 'Microsoft Edge', 'Firefox', 'Brave', 'Vivaldi', 'Opera']
    for browser_name in preferred_order:
        if browser_name not in [b[0] for b in browsers]:  # Skip if already added as default
            browsers.append((browser_name, all_browsers[browser_name]))
    
    logger.info("Detecting available browsers for satellite imagery capture...")
    last_error = None
    
    for browser_name, get_driver in browsers:
        try:
            logger.debug(f"Attempting to initialize {browser_name}...")
            driver = get_driver()
            logger.info(f"✓ Successfully initialized {browser_name} browser")
            return driver
        except Exception as e:
            last_error = e
            logger.debug(f"✗ {browser_name} not available: {str(e)[:100]}")
            continue
    
    error_msg = (
        "No compatible browser found!\n"
        "\n"
        "The system requires one of the following browsers:\n"
        "  • Google Chrome (recommended)\n"
        "  • Chromium\n"
        "  • Microsoft Edge\n"
        "  • Mozilla Firefox\n"
        "  • Brave Browser\n"
        "  • Vivaldi\n"
        "  • Opera Browser\n"
        "\n"
        "Please install at least one browser and ensure it's accessible.\n"
        f"\n"
        f"Technical details: {str(last_error)[:200]}"
    )
    
    logger.error(error_msg)
    raise RuntimeError(error_msg)



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
    chrome_options.add_argument("--disable-cache")
    chrome_options.add_argument("--disk-cache-size=0")
    chrome_options.add_argument("--incognito")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
    return chrome_options

def get_chromium_options():
    """Get Chromium browser options for headless mode."""
    chromium_options = ChromeOptions()
    
    # Common Chromium installation paths
    chromium_paths = [
        r"C:\Program Files\Chromium\Application\chrome.exe",
        r"C:\Program Files (x86)\Chromium\Application\chrome.exe",
        os.path.expanduser(r"~\AppData\Local\Chromium\Application\chrome.exe"),
    ]
    
    for chromium_path in chromium_paths:
        if os.path.exists(chromium_path):
            chromium_options.binary_location = chromium_path
            break
    
    chromium_options.add_argument("--headless=new")
    chromium_options.add_argument("--window-size=1920,1080")
    chromium_options.add_argument("--disable-blink-features=AutomationControlled")
    chromium_options.add_argument("--disable-gpu")
    chromium_options.add_argument("--no-sandbox")
    chromium_options.add_argument("--disable-dev-shm-usage")
    chromium_options.add_argument("--disable-extensions")
    chromium_options.add_argument("--log-level=3")
    chromium_options.add_argument("--disable-cache")
    chromium_options.add_argument("--disk-cache-size=0")
    chromium_options.add_argument("--incognito")
    chromium_options.add_experimental_option("excludeSwitches", ["enable-logging"])
    return chromium_options

def get_chromium_options():
    """Get Chromium browser options for headless mode."""
    chromium_options = ChromeOptions()
    
    # Common Chromium installation paths
    chromium_paths = [
        r"C:\Program Files\Chromium\Application\chrome.exe",
        r"C:\Program Files (x86)\Chromium\Application\chrome.exe",
        os.path.expanduser(r"~\AppData\Local\Chromium\Application\chrome.exe"),
    ]
    
    for chromium_path in chromium_paths:
        if os.path.exists(chromium_path):
            chromium_options.binary_location = chromium_path
            break
    
    chromium_options.add_argument("--headless=new")
    chromium_options.add_argument("--window-size=1920,1080")
    chromium_options.add_argument("--disable-blink-features=AutomationControlled")
    chromium_options.add_argument("--disable-gpu")
    chromium_options.add_argument("--no-sandbox")
    chromium_options.add_argument("--disable-dev-shm-usage")
    chromium_options.add_argument("--disable-extensions")
    chromium_options.add_argument("--log-level=3")
    chromium_options.add_argument("--disable-cache")
    chromium_options.add_argument("--disk-cache-size=0")
    chromium_options.add_argument("--incognito")
    chromium_options.add_experimental_option("excludeSwitches", ["enable-logging"])
    return chromium_options


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


def get_vivaldi_options():
    """Get Vivaldi browser options for headless mode."""
    vivaldi_options = ChromeOptions()
    
    # Common Vivaldi installation paths
    vivaldi_paths = [
        r"C:\Program Files\Vivaldi\Application\vivaldi.exe",
        r"C:\Program Files (x86)\Vivaldi\Application\vivaldi.exe",
        os.path.expanduser(r"~\AppData\Local\Vivaldi\Application\vivaldi.exe"),
    ]
    
    for vivaldi_path in vivaldi_paths:
        if os.path.exists(vivaldi_path):
            vivaldi_options.binary_location = vivaldi_path
            break
    
    vivaldi_options.add_argument("--headless=new")
    vivaldi_options.add_argument("--window-size=1920,1080")
    vivaldi_options.add_argument("--disable-blink-features=AutomationControlled")
    vivaldi_options.add_argument("--disable-gpu")
    vivaldi_options.add_argument("--no-sandbox")
    vivaldi_options.add_argument("--disable-dev-shm-usage")
    return vivaldi_options


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
        # Note: Labels are embedded in Google's satellite tiles and cannot be easily removed
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
        
        # Wait longer for everything to settle
        logger.debug("Waiting for page to fully render...")
        time.sleep(4)
        
        # Comprehensive label removal attempt
        try:
            logger.debug("Starting label removal process...")
            
            # Step 1: Click Layers button using multiple methods
            layers_opened = False
            
            # Method A: Try XPath
            try:
                layers_btn = driver.find_element(By.XPATH, "//button[contains(@aria-label, 'Layer') or contains(@aria-label, 'layer')]")
                layers_btn.click()
                layers_opened = True
                logger.debug("✓ Layers opened via XPath")
            except:
                pass
            
            # Method B: Try JavaScript with all buttons
            if not layers_opened:
                try:
                    js_result = driver.execute_script("""
                        var buttons = document.getElementsByTagName('button');
                        for (var i = 0; i < buttons.length; i++) {
                            var label = buttons[i].getAttribute('aria-label') || '';
                            if (label.toLowerCase().includes('layer')) {
                                buttons[i].click();
                                return true;
                            }
                        }
                        return false;
                    """)
                    if js_result:
                        layers_opened = True
                        logger.debug("✓ Layers opened via JavaScript")
                except:
                    pass
            
            if layers_opened:
                time.sleep(2)  # Wait for menu animation
                
                # Step 2: Find and click the Labels checkbox
                try:
                    # Try XPath to find text "Labels" and nearby checkbox
                    checkbox_result = driver.execute_script("""
                        // Find all text nodes containing "Label"
                        var walker = document.createTreeWalker(
                            document.body,
                            NodeFilter.SHOW_TEXT,
                            null,
                            false
                        );
                        
                        var textNodes = [];
                        while(walker.nextNode()) {
                            if (walker.currentNode.nodeValue.match(/label/i)) {
                                textNodes.push(walker.currentNode);
                            }
                        }
                        
                        // For each text node, find nearby checkbox
                        for (var textNode of textNodes) {
                            var parent = textNode.parentElement;
                            if (!parent) continue;
                            
                            // Look for checkbox in parent or siblings
                            var container = parent.closest('div');
                            if (!container) continue;
                            
                            var checkbox = container.querySelector('input[type="checkbox"]');
                            if (!checkbox) {
                                checkbox = container.querySelector('[role="checkbox"]');
                            }
                            
                            if (checkbox) {
                                var isChecked = checkbox.checked || 
                                               checkbox.getAttribute('aria-checked') === 'true';
                                               
                                if (isChecked) {
                                    checkbox.click();
                                    return 'clicked';
                                } else {
                                    return 'already_unchecked';
                                }
                            }
                        }
                        
                        return 'not_found';
                    """)
                    
                    logger.debug(f"Checkbox result: {checkbox_result}")
                    
                    if checkbox_result == 'clicked':
                        logger.debug("✓ Successfully unchecked Labels!")
                        time.sleep(3)  # Wait for map to reload without labels
                    elif checkbox_result == 'already_unchecked':
                        logger.debug("Labels already unchecked")
                    else:
                        logger.debug("Could not find Labels checkbox")
                        
                except Exception as e:
                    logger.debug(f"Checkbox interaction failed: {e}")
            else:
                logger.debug("Could not open Layers menu")
                
        except Exception as e:
            logger.debug(f"Label removal failed: {e}")
        
        # Wait for satellite tiles to load
        time.sleep(1)
        logger.debug("Waiting for satellite imagery tiles...")
        time.sleep(0.5)
        
        # Disable labels using Google Maps URL parameter
        try:
            # Add URL parameter to disable labels
            logger.debug("Disabling map labels...")
            driver.execute_script("""
                // Navigate to URL with labels disabled
                var currentUrl = window.location.href;
                if (!currentUrl.includes('!4m2!6m1!1s')) {
                    window.location.href = currentUrl + '!4m2!6m1!1s';
                }
            """)
            time.sleep(1)
        except Exception as e:
            logger.debug(f"URL method failed: {e}")
        
        # Hide map labels using CSS as fallback
        try:
            driver.execute_script("""
                var style = document.createElement('style');
                style.innerHTML = `
                    /* Hide all text/label layers */
                    div[style*="font-"] { display: none !important; }
                    div[style*="text"] { display: none !important; }
                    span[style*="font-"] { display: none !important; }
                    /* Hide POI markers and labels */
                    div[class*="widget"] { display: none !important; }
                    div[class*="label"] { display: none !important; }
                `;
                document.head.appendChild(style);
            """)
            logger.debug("Applied CSS to hide labels")
            time.sleep(0.3)
        except Exception as e:
            logger.warning(f"Could not hide labels: {e}")
        
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
