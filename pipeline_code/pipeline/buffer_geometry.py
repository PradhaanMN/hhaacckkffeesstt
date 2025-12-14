"""
Buffer geometry calculations for the rooftop PV detection pipeline.
Converts area in square feet to bounding boxes in WGS84 coordinates,
and handles pixel-to-meter scale conversions.
"""

import math
from typing import Tuple
from .config import SQFT_TO_SQM, METERS_PER_DEGREE_LAT


def area_sqft_to_side_m(area_sqft: float) -> float:
    """
    Convert area in square feet to the side length of an equivalent square in meters.
    
    Args:
        area_sqft: Area in square feet
        
    Returns:
        Side length in meters
    """
    area_m2 = area_sqft * SQFT_TO_SQM
    side_m = math.sqrt(area_m2)
    return side_m


def compute_bbox(lat: float, lon: float, area_sqft: float) -> Tuple[float, float, float, float]:
    """
    Compute bounding box (xmin, ymin, xmax, ymax) in WGS84 degrees
    for a square buffer zone centered at (lat, lon).
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        area_sqft: Buffer area in square feet (or side length in meters)
        
    Returns:
        Tuple of (xmin, ymin, xmax, ymax) in degrees
    """
    # Convert area to square side length in meters
    side_m = area_sqft_to_side_m(area_sqft)
    half_side_m = side_m / 2.0
    
    # Convert meters to degrees
    # Latitude: 1 degree ≈ 111,320 meters (constant)
    delta_lat = half_side_m / METERS_PER_DEGREE_LAT
    
    # Longitude: depends on latitude (cos correction)
    lat_rad = math.radians(lat)
    meters_per_deg_lon = METERS_PER_DEGREE_LAT * math.cos(lat_rad)
    
    # Handle edge case near poles
    if meters_per_deg_lon < 1:
        meters_per_deg_lon = 1
        
    delta_lon = half_side_m / meters_per_deg_lon
    
    # Compute bounding box
    xmin = lon - delta_lon
    xmax = lon + delta_lon
    ymin = lat - delta_lat
    ymax = lat + delta_lat
    
    return (xmin, ymin, xmax, ymax)


def compute_pixel_scale(
    area_sqft: float,
    size_px: int
) -> Tuple[float, float, float, float]:
    """
    Compute pixel-to-meter scale factors for an image.
    
    Args:
        area_sqft: Ground area in square feet
        size_px: Image size in pixels (assumes square image)
        
    Returns:
        Tuple of (ground_width_m, ground_height_m, meters_per_pixel_x, meters_per_pixel_y)
    """
    # Get the side length of the square buffer in meters
    side_m = area_sqft_to_side_m(area_sqft)
    
    # For a square buffer, width = height
    ground_width_m = side_m
    ground_height_m = side_m
    
    # Compute meters per pixel
    meters_per_pixel_x = ground_width_m / size_px
    meters_per_pixel_y = ground_height_m / size_px
    
    return (ground_width_m, ground_height_m, meters_per_pixel_x, meters_per_pixel_y)


def compute_buffer_radius_pixels(
    buffer_sqft: float,
    image_sqft: float,
    image_size_px: int
) -> float:
    """
    Compute buffer zone radius in pixels.
    
    Args:
        buffer_sqft: Buffer zone area in square feet (1200 or 2400)
        image_sqft: Total image area in square feet
        image_size_px: Image size in pixels
        
    Returns:
        Buffer radius in pixels from center
    """
    # Convert to side lengths in meters
    buffer_side_m = area_sqft_to_side_m(buffer_sqft)
    image_side_m = area_sqft_to_side_m(image_sqft)
    
    # Ratio of buffer to image
    ratio = buffer_side_m / image_side_m
    
    # Buffer radius in pixels (half the buffer side length)
    buffer_radius_px = (ratio * image_size_px) / 2.0
    
    return buffer_radius_px


def compute_buffer_circle_pixels(
    center_x_px: int,
    center_y_px: int,
    area_sqft: float,
    size_px: int
) -> Tuple[int, int, float]:
    """
    Compute buffer zone as a circle in pixel coordinates.
    
    Args:
        center_x_px: Center X coordinate in pixels
        center_y_px: Center Y coordinate in pixels
        area_sqft: Buffer area in square feet
        size_px: Image size in pixels
        
    Returns:
        Tuple of (center_x, center_y, radius_px)
    """
    # Get pixel scale
    _, _, meters_per_pixel_x, meters_per_pixel_y = compute_pixel_scale(area_sqft, size_px)
    
    # Average meters per pixel
    meters_per_pixel = (meters_per_pixel_x + meters_per_pixel_y) / 2.0
    
    # Convert area to radius
    # For a circle: area = π * r²
    area_m2 = area_sqft * SQFT_TO_SQM
    radius_m = math.sqrt(area_m2 / math.pi)
    
    # Convert radius to pixels
    radius_px = radius_m / meters_per_pixel
    
    return (center_x_px, center_y_px, radius_px)


def point_in_polygon(point: Tuple[float, float], polygon: list) -> bool:
    """
    Check if a point is inside a polygon using ray-casting algorithm.
    
    Args:
        point: (x, y) coordinates
        polygon: List of [x, y] coordinates forming the polygon
        
    Returns:
        True if point is inside polygon, False otherwise
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def compute_polygon_area(polygon: list) -> float:
    """
    Compute the area of a polygon using the Shoelace formula.
    
    Args:
        polygon: List of [x, y] coordinates forming the polygon
        
    Returns:
        Area in the same units as the input coordinates
    """
    if len(polygon) < 3:
        return 0.0
    
    area = 0.0
    n = len(polygon)
    
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    
    return abs(area) / 2.0
