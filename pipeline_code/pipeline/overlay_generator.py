"""
Overlay generator for creating annotated images with detection results.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from .config import (
    OVERLAY_PANEL_COLOR,
    OVERLAY_BUFFER_COLOR,
    OVERLAY_SELECTED_COLOR,
    OVERLAY_LINE_THICKNESS,
    OVERLAY_ALPHA
)

logger = logging.getLogger(__name__)


def draw_bbox(
    image: np.ndarray,
    bbox: List[float],
    color: Tuple[int, int, int],
    thickness: int = 2,
    label: str = None
) -> np.ndarray:
    """
    Draw a bounding box rectangle on an image.
    
    Args:
        image: Input image (will be modified)
        bbox: [x1, y1, x2, y2] coordinates
        color: BGR color tuple
        thickness: Line thickness
        label: Optional text label to display
        
    Returns:
        Modified image
    """
    if not bbox or len(bbox) < 4:
        return image
    
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Add label if provided
    if label:
        # Background for text
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        cv2.rectangle(
            image,
            (x1, y1 - text_height - 8),
            (x1 + text_width + 4, y1),
            color,
            -1
        )
        # Text
        cv2.putText(
            image,
            label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    
    return image


def draw_polygon(
    image: np.ndarray,
    polygon: List[List[float]],
    color: Tuple[int, int, int],
    thickness: int = 2,
    filled: bool = False,
    alpha: float = 0.3
) -> np.ndarray:
    """
    Draw a polygon on an image.
    
    Args:
        image: Input image (will be modified)
        polygon: List of [x, y] coordinates
        color: BGR color tuple
        thickness: Line thickness
        filled: Whether to fill the polygon
        alpha: Transparency for filled polygons
        
    Returns:
        Modified image
    """
    if not polygon or len(polygon) < 3:
        return image
    
    # Convert polygon to numpy array of integer coordinates
    pts = np.array([[int(p[0]), int(p[1])] for p in polygon], dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    
    if filled:
        # Create overlay for transparency
        overlay = image.copy()
        cv2.fillPoly(overlay, [pts], color)
        # Blend overlay with original image
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # Draw outline
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)
    
    return image


def draw_split_polygon(
    image: np.ndarray,
    polygon: List[List[float]],
    center: Tuple[int, int],
    radius: float,
    color_inside: Tuple[int, int, int],
    color_outside: Tuple[int, int, int],
    thickness: int = 2,
    filled: bool = False,
    alpha: float = 0.3
) -> np.ndarray:
    """
    Draw a polygon split by a circular buffer zone.
    Part inside buffer = one color, part outside = another color.
    
    Args:
        image: Input image
        polygon: List of [x, y] coordinates
        center: (x, y) center of buffer circle
        radius: Radius of buffer circle
        color_inside: BGR color for part inside buffer (green)
        color_outside: BGR color for part outside buffer (red)
        thickness: Line thickness
        filled: Whether to fill the polygons
        alpha: Transparency for filled polygons
        
    Returns:
        Modified image
    """
    if not polygon or len(polygon) < 3:
        return image
    
    try:
        # Create shapely polygon from detection
        panel_poly = Polygon(polygon)
        
        # Create circular buffer zone
        buffer_circle = Point(center[0], center[1]).buffer(radius)
        
        # Clip polygon: part inside buffer
        inside_part = panel_poly.intersection(buffer_circle)
        
        # Clip polygon: part outside buffer
        outside_part = panel_poly.difference(buffer_circle)
        
        # Draw inside part (GREEN)
        if not inside_part.is_empty:
            if inside_part.geom_type == 'Polygon':
                coords = list(inside_part.exterior.coords)
                draw_polygon(image, coords, color_inside, thickness, filled, alpha)
            elif inside_part.geom_type == 'MultiPolygon':
                for poly in inside_part.geoms:
                    coords = list(poly.exterior.coords)
                    draw_polygon(image, coords, color_inside, thickness, filled, alpha)
        
        # Draw outside part (RED)
        if not outside_part.is_empty:
            if outside_part.geom_type == 'Polygon':
                coords = list(outside_part.exterior.coords)
                draw_polygon(image, coords, color_outside, thickness, filled, alpha)
            elif outside_part.geom_type == 'MultiPolygon':
                for poly in outside_part.geoms:
                    coords = list(poly.exterior.coords)
                    draw_polygon(image, coords, color_outside, thickness, filled, alpha)
    
    except Exception as e:
        logger.warning(f"Failed to split polygon: {e}. Drawing as single color.")
        # Fallback: determine color by centroid
        centroid_x = np.mean([p[0] for p in polygon])
        centroid_y = np.mean([p[1] for p in polygon])
        distance = np.sqrt((centroid_x - center[0])**2 + (centroid_y - center[1])**2)
        color = color_inside if distance <= radius else color_outside
        draw_polygon(image, polygon, color, thickness, filled, alpha)
    
    return image


def draw_circle(
    image: np.ndarray,
    center: Tuple[int, int],
    radius: float,
    color: Tuple[int, int, int],
    thickness: int = 2,
    filled: bool = False,
    alpha: float = 0.3
) -> np.ndarray:
    """
    Draw a circle on an image.
    
    Args:
        image: Input image (will be modified)
        center: (x, y) center coordinates
        radius: Radius in pixels
        color: BGR color tuple
        thickness: Line thickness
        filled: Whether to fill the circle
        alpha: Transparency for filled circles
        
    Returns:
        Modified image
    """
    center_int = (int(center[0]), int(center[1]))
    radius_int = int(radius)
    
    if filled:
        # Create overlay for transparency
        overlay = image.copy()
        cv2.circle(overlay, center_int, radius_int, color, -1)
        # Blend overlay with original image
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # Draw outline
    cv2.circle(image, center_int, radius_int, color, thickness)
    
    return image


def create_overlay_image(
    image_path: str,
    detections: List[Dict],
    selected_panel: Optional[Dict] = None,
    buffer_zone: Optional[Dict] = None,
    output_path: str = None,
    buffer_sqft: int = None,
    imagery_sqft: int = None
) -> str:
    """
    Create an annotated overlay image showing detections and buffer zones.
    
    NEW Workflow:
    1. Load LARGE satellite image from API (e.g., 100m x 100m)
    2. Show ALL detections from model (detected on full image)
    3. Draw both buffer zones (1200 and 2400 sq.ft circles)
    4. Mark panels IN the active buffer zone as GREEN
    5. Mark panels OUTSIDE buffer zone as RED
    
    Args:
        image_path: Path to the original satellite image
        detections: List of ALL detection dictionaries from model
        selected_panel: The panel selected in buffer zone (for verification)
        buffer_zone: Dictionary with buffer zone info
        output_path: Where to save the overlay image
        buffer_sqft: Which buffer was used (1200 or 2400)
        imagery_sqft: Total imagery area fetched (for calculating buffer sizes)
        
    Returns:
        Path to the saved overlay image
    """
    try:
        # Read the satellite image from API
        image = cv2.imread(image_path)
        
        if image is None:
            logger.error(f"Failed to read image: {image_path}")
            return None
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        logger.info(f"Creating overlay for {len(detections)} detection(s)")
        
        # Calculate buffer zone radii in pixels
        from .config import BUFFER_ZONE_1, BUFFER_ZONE_2
        from .buffer_geometry import compute_buffer_radius_pixels
        
        if imagery_sqft:
            radius_1200 = compute_buffer_radius_pixels(BUFFER_ZONE_1, imagery_sqft, w)
            radius_2400 = compute_buffer_radius_pixels(BUFFER_ZONE_2, imagery_sqft, w)
        else:
            # Fallback
            radius_1200 = int(w * 0.35)
            radius_2400 = int(w * 0.5)
        
        logger.info(f"Buffer radii: 1200 sq.ft = {radius_1200}px, 2400 sq.ft = {radius_2400}px")
        
        # Determine which radius is active
        active_radius = radius_1200 if buffer_sqft == 1200 else radius_2400
        
        # Draw BOTH buffer zones on the image
        # Buffer zone 2 (2400 sq.ft) - outer circle
        if buffer_sqft == 2400:
            # Active buffer - YELLOW with highlight
            draw_circle(
                image,
                center,
                int(radius_2400),
                (0, 255, 255),  # Yellow (BGR)
                thickness=4,
                filled=True,
                alpha=0.15
            )
        else:
            # Inactive buffer - GRAY
            draw_circle(
                image,
                center,
                int(radius_2400),
                (128, 128, 128),  # Gray
                thickness=2,
                filled=False
            )
        
        # Buffer zone 1 (1200 sq.ft) - inner circle
        if buffer_sqft == 1200:
            # Active buffer - YELLOW with highlight
            draw_circle(
                image,
                center,
                int(radius_1200),
                (0, 255, 255),  # Yellow (BGR)
                thickness=4,
                filled=True,
                alpha=0.15
            )
        else:
            # Inactive buffer - BLUE (only if 2400 is active)
            draw_circle(
                image,
                center,
                int(radius_1200),
                OVERLAY_BUFFER_COLOR,  # Blue
                thickness=2,
                filled=True,
                alpha=0.1
            )
        
        # Draw ALL detections with split coloring
        # Green for part inside buffer, red for part outside
        for detection in detections:
            polygon = detection.get("polygon", [])
            bbox = detection.get("bbox", [])
            confidence = detection.get("confidence", 0)
            
            if not polygon:
                continue
            
            # Draw polygon split by buffer zone
            # Green inside, Red outside
            draw_split_polygon(
                image,
                polygon,
                center,
                active_radius,
                color_inside=(0, 255, 0),   # Green (BGR)
                color_outside=(0, 0, 255),  # Red (BGR)
                thickness=2,
                filled=True,
                alpha=0.3
            )
            
            # Draw bounding box
            # Determine bbox color by centroid location
            centroid_x = np.mean([p[0] for p in polygon])
            centroid_y = np.mean([p[1] for p in polygon])
            distance = np.sqrt((centroid_x - center[0])**2 + (centroid_y - center[1])**2)
            
            if distance <= active_radius:
                # Mostly inside - green bbox
                bbox_color = (0, 255, 0)  # Green
                label = f"Solar {confidence:.0%}"
                thickness = 3
            else:
                # Mostly outside - red bbox
                bbox_color = (0, 0, 255)  # Red
                label = f"{confidence:.0%}"
                thickness = 2
            
            if bbox and len(bbox) >= 4:
                draw_bbox(image, bbox, bbox_color, thickness=thickness, label=label)
        
        # Add comprehensive legend with background
        legend_height = 160
        cv2.rectangle(image, (5, 5), (520, legend_height), (0, 0, 0), -1)
        cv2.rectangle(image, (5, 5), (520, legend_height), (255, 255, 255), 2)
        
        legend_y = 25
        cv2.putText(image, "GREEN: Panel area INSIDE buffer zone", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        legend_y += 20
        cv2.putText(image, "RED: Panel area OUTSIDE buffer zone", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        legend_y += 20
        cv2.putText(image, "(Panels split by buffer boundary)", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        legend_y += 20
        cv2.putText(image, f"YELLOW CIRCLE: Active buffer ({buffer_sqft} sq.ft)", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        legend_y += 20
        inactive_buffer = "2400" if buffer_sqft == 1200 else "1200"
        inactive_color = (128, 128, 128) if buffer_sqft == 1200 else OVERLAY_BUFFER_COLOR
        inactive_label = "GRAY" if buffer_sqft == 1200 else "BLUE"
        cv2.putText(image, f"{inactive_label} CIRCLE: Inactive buffer ({inactive_buffer} sq.ft)", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, inactive_color, 2)
        
        # Count panels by centroid location for display
        panels_in = sum(1 for d in detections if d.get("polygon") and 
                       np.sqrt((np.mean([p[0] for p in d["polygon"]]) - center[0])**2 + 
                              (np.mean([p[1] for p in d["polygon"]]) - center[1])**2) <= active_radius)
        panels_out = len(detections) - panels_in
        
        # Add detection counts
        legend_y += 20
        count_text = f"In buffer: {panels_in} | Outside: {panels_out}"
        cv2.putText(image, count_text, (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Save the overlay image
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            success = cv2.imwrite(str(output_file), image)
            
            if success:
                logger.info(f"Saved overlay image to {output_path}")
                return str(output_file)
            else:
                logger.error(f"Failed to save overlay image to {output_path}")
                return None
        
        return None
        
    except Exception as e:
        logger.exception(f"Error creating overlay image: {e}")
        return None


def encode_polygon_for_json(polygon: List[List[float]]) -> str:
    """
    Encode a polygon for JSON storage.
    
    Args:
        polygon: List of [x, y] coordinates
        
    Returns:
        String representation of the polygon
    """
    if not polygon:
        return "[]"
    
    # Format as "[[x1,y1],[x2,y2],...]"
    coords = ",".join([f"[{p[0]:.1f},{p[1]:.1f}]" for p in polygon])
    return f"[{coords}]"
