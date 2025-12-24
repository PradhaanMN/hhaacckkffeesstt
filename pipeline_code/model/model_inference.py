"""
Model inference wrapper for YOLOv8 segmentation model with ensemble support.

IMPORTANT NOTE FOR RETRAINING:
- Current model detects entire solar arrays as single bounding boxes
- For accurate individual panel detection, retrain with annotations where:
  1. Each individual solar panel is labeled separately
  2. Do NOT label entire arrays as one box
  3. Each panel should have its own bounding box
- This will significantly improve area calculation accuracy
- Current code has post-processing filters to mitigate this issue, but
  proper training data is the permanent solution
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Try to import YOLOv8
try:
    from ultralytics import YOLO
except ImportError:
    logger.error("ultralytics package not found. Install it with: pip install ultralytics")
    YOLO = None


class SolarPanelDetector:
    """
    Wrapper for YOLOv8 segmentation model for solar panel detection.
    Supports ensemble of multiple models with equal weighting.
    """
    
    def __init__(self, model_path: str, ensemble_models: Optional[List[str]] = None):
        """
        Initialize the detector with a trained YOLOv8 model or ensemble.
        
        Args:
            model_path: Path to the primary .pt model weights file
            ensemble_models: Optional list of additional model paths for ensemble
        """
        if YOLO is None:
            raise ImportError("ultralytics package is required. Install with: pip install ultralytics")
        
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load primary model
        logger.info(f"Loading primary YOLOv8 model from {model_path}")
        self.model = YOLO(str(model_path))
        logger.info("Primary model loaded successfully")
        
        # Load ensemble models if provided
        self.ensemble_models = [self.model]
        if ensemble_models:
            for model_path_str in ensemble_models:
                model_path_obj = Path(model_path_str)
                if model_path_obj.exists():
                    logger.info(f"Loading ensemble model from {model_path_str}")
                    ensemble_model = YOLO(str(model_path_str))
                    self.ensemble_models.append(ensemble_model)
                else:
                    logger.warning(f"Ensemble model not found: {model_path_str}")
        
        logger.info(f"Total models in ensemble: {len(self.ensemble_models)}")
    
    def run_inference(
        self,
        image_path: str,
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.45
    ) -> List[Dict]:
        """
        Run inference on an image and extract segmentation results.
        Uses ensemble averaging if multiple models are loaded.
        
        Args:
            image_path: Path to the input image
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            List of detections, each containing:
            {
                "polygon": [[x1, y1], [x2, y2], ...],  # Segmentation polygon
                "area_px": float,                       # Area in pixels
                "confidence": float,                    # Detection confidence
                "bbox": [x1, y1, x2, y2]               # Bounding box
            }
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return []
        
        try:
            # Run inference on all ensemble models
            all_detections = []
            
            for idx, model in enumerate(self.ensemble_models):
                # Run inference
                results = model.predict(
                    source=str(image_path),
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False
                )
                
                # Extract detections from this model
                model_detections = self._extract_detections(results)
                all_detections.extend(model_detections)
                
                logger.debug(f"Model {idx+1}/{len(self.ensemble_models)}: {len(model_detections)} detections")
            
            # Merge overlapping detections from ensemble (simple NMS)
            if len(self.ensemble_models) > 1:
                detections = self._merge_ensemble_detections(all_detections, iou_threshold)
                logger.info(f"Ensemble merged: {len(all_detections)} → {len(detections)} detections")
            else:
                detections = all_detections
                logger.info(f"Found {len(detections)} solar panel detections")
            
            return detections
            
        except Exception as e:
            logger.exception(f"Error during inference: {e}")
            return []
    
    def _extract_detections(self, results) -> List[Dict]:
        """Extract detections from model results."""
        detections = []
        
        if results and len(results) > 0:
            result = results[0]  # Get first result (single image)
            
            # Get image dimensions
            image_height, image_width = result.orig_shape if hasattr(result, 'orig_shape') else (None, None)
            
            # Check if masks are available (segmentation models)
            if result.masks is not None and len(result.masks) > 0:
                masks = result.masks.xy  # Get polygon coordinates
                boxes = result.boxes  # Get bounding boxes
                
                for i, mask_coords in enumerate(masks):
                    if len(mask_coords) < 3:
                        # Skip invalid polygons (need at least 3 points)
                        continue
                    
                    # Convert to list of [x, y] points
                    polygon = [[float(x), float(y)] for x, y in mask_coords]
                    
                    # Calculate area using Shoelace formula (more accurate than bbox)
                    area_px = self._calculate_polygon_area(polygon)
                    
                    # Get confidence
                    confidence = float(boxes.conf[i])
                    
                    # Recalculate bbox from actual polygon for accuracy
                    # (model's bbox might be inaccurate, but polygon is precise)
                    bbox = self._calculate_polygon_bbox(polygon)
                    
                    detection = {
                        "polygon": polygon,
                        "area_px": area_px,
                        "confidence": confidence,
                        "bbox": bbox
                    }
                    
                    # Apply shape filters to reduce false positives
                    if self._is_valid_solar_panel(detection, image_width, image_height):
                        detections.append(detection)
            
            # Fallback for detection-only models (no masks, only bounding boxes)
            elif result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # Get bbox and convert to polygon (4 corners)
                    bbox_coords = boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]
                    x1, y1, x2, y2 = bbox_coords
                    
                    # Create polygon from bbox corners
                    polygon = [
                        [x1, y1],  # top-left
                        [x2, y1],  # top-right
                        [x2, y2],  # bottom-right
                        [x1, y2]   # bottom-left
                    ]
                    
                    # Calculate area from bbox
                    area_px = (x2 - x1) * (y2 - y1)
                    
                    # Get confidence
                    confidence = float(boxes.conf[i])
                    
                    detection = {
                        "polygon": polygon,
                        "area_px": area_px,
                        "confidence": confidence,
                        "bbox": [float(b) for b in bbox_coords]
                    }
                    
                    # Apply shape filters to reduce false positives
                    if self._is_valid_solar_panel(detection, image_width, image_height):
                        detections.append(detection)
        
        return detections
    
    def _is_valid_solar_panel(self, detection: Dict, image_width: int = None, image_height: int = None) -> bool:
        """
        Filter out false positives based on shape characteristics.
        Solar panels should be:
        - Rectangular/regular shapes
        - Reasonable size (not too small/large)
        - Reasonable aspect ratio
        - Not occupying too much of the image (likely entire roof)
        """
        bbox = detection['bbox']
        area_px = detection['area_px']
        
        # Calculate bounding box dimensions
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        bbox_area = width * height
        
        # Filter 1: Minimum size (too small = noise)
        MIN_AREA_PX = 100  # ~10x10 pixels minimum
        if area_px < MIN_AREA_PX:
            return False
        
        # Filter 2: Maximum size (too large = likely entire roof/array)
        # Individual solar panels are typically 1.6m x 1m (~17 sq.ft each)
        # At typical satellite resolution (0.3-0.6m/pixel), one panel = ~2000-8000 pixels
        MAX_AREA_PX = 15000  # Reduced from 100000 to reject large roof sections
        if area_px > MAX_AREA_PX:
            return False
        
        # Filter 2b: Relative to image size (reject if detection is too large)
        # A single panel should not occupy more than 10% of the satellite image
        if image_width and image_height:
            image_area = image_width * image_height
            area_ratio = area_px / image_area
            MAX_IMAGE_RATIO = 0.10  # Maximum 10% of image
            if area_ratio > MAX_IMAGE_RATIO:
                return False
        
        # Filter 3: Aspect ratio (solar panels are typically 1:1 to 3:1)
        # Street names/text are very elongated, so use stricter ratio
        if width > 0 and height > 0:
            aspect_ratio = max(width, height) / min(width, height)
            MAX_ASPECT_RATIO = 3.0  # Stricter to exclude text labels
            if aspect_ratio > MAX_ASPECT_RATIO:
                return False
        
        # Filter 4: Fill ratio (polygon area vs bbox area)
        # Solar panels should fill most of their bounding box
        if bbox_area > 0:
            fill_ratio = area_px / bbox_area
            MIN_FILL_RATIO = 0.5  # At least 50% filled
            if fill_ratio < MIN_FILL_RATIO:
                return False
        
        # Filter 5: Minimum dimension (avoid thin lines like street names)
        MIN_DIMENSION = 10  # pixels - increased from 5 to reject thin text
        if width < MIN_DIMENSION or height < MIN_DIMENSION:
            return False
        
        # Filter 6: Reject very thin shapes (text labels, road markings)
        # Check thickness relative to length
        if width > 0 and height > 0:
            min_thickness = min(width, height)
            max_length = max(width, height)
            thickness_ratio = min_thickness / max_length
            MIN_THICKNESS_RATIO = 0.15  # At least 15% thick
            if thickness_ratio < MIN_THICKNESS_RATIO:
                return False
        
        return True
    
    def _merge_ensemble_detections(self, detections: List[Dict], iou_threshold: float) -> List[Dict]:
        """
        Merge overlapping detections from ensemble models.
        Uses NMS-style merging with equal confidence weighting.
        """
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        merged = []
        used = set()
        
        for i, det in enumerate(detections):
            if i in used:
                continue
            
            # Find overlapping detections
            overlapping = [det]
            for j in range(i + 1, len(detections)):
                if j in used:
                    continue
                
                iou = self._calculate_bbox_iou(det['bbox'], detections[j]['bbox'])
                if iou > iou_threshold:
                    overlapping.append(detections[j])
                    used.add(j)
            
            # Average the detections with equal weight
            if len(overlapping) > 1:
                merged_det = self._average_detections(overlapping)
                merged.append(merged_det)
            else:
                merged.append(det)
        
        return merged
    
    def _average_detections(self, detections: List[Dict]) -> Dict:
        """Average multiple detections with equal weighting."""
        # Average confidence
        avg_confidence = sum(d['confidence'] for d in detections) / len(detections)
        
        # Average bbox
        avg_bbox = [
            sum(d['bbox'][i] for d in detections) / len(detections)
            for i in range(4)
        ]
        
        # Use the polygon from the highest confidence detection
        best_det = max(detections, key=lambda x: x['confidence'])
        
        return {
            "polygon": best_det['polygon'],
            "area_px": best_det['area_px'],
            "confidence": avg_confidence,
            "bbox": avg_bbox
        }
    
    def _calculate_bbox_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # Calculate union
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _calculate_polygon_area(self, polygon: List[List[float]]) -> float:
        """
        Calculate the area of a polygon using the Shoelace formula.
        
        Args:
            polygon: List of [x, y] coordinates
            
        Returns:
            Area in square pixels
        """
        if len(polygon) < 3:
            return 0.0
        
        # Shoelace formula
        area = 0.0
        n = len(polygon)
        
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
        
        return abs(area) / 2.0
    
    def _calculate_polygon_bbox(self, polygon: List[List[float]]) -> List[float]:
        """
        Calculate accurate bounding box from polygon coordinates.
        This is more reliable than model's bbox predictions.
        
        Args:
            polygon: List of [x, y] coordinates
            
        Returns:
            Bounding box as [x1, y1, x2, y2]
        """
        if not polygon or len(polygon) == 0:
            return [0.0, 0.0, 0.0, 0.0]
        
        x_coords = [p[0] for p in polygon]
        y_coords = [p[1] for p in polygon]
        
        return [
            float(min(x_coords)),  # x1
            float(min(y_coords)),  # y1
            float(max(x_coords)),  # x2
            float(max(y_coords))   # y2
        ]


def run_inference_on_image(
    image_path: str,
    model_path: str = "model/model_weights/solarpanel_seg_v1.pt",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> List[Dict]:
    """
    Convenience function to run inference on a single image.
    
    Args:
        image_path: Path to the input image
        model_path: Path to the model weights
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
        
    Returns:
        List of detections
    """
    detector = SolarPanelDetector(model_path)
    return detector.run_inference(image_path, conf_threshold, iou_threshold)


def get_model_info(model_path: str) -> Dict:
    """
    Get information about the model.
    
    Args:
        model_path: Path to the model weights
        
    Returns:
        Dictionary with model information
    """
    if YOLO is None:
        return {"error": "ultralytics package not installed"}
    
    try:
        model = YOLO(model_path)
        
        info = {
            "model_path": str(model_path),
            "model_type": "YOLOv8s-seg",
            "task": "segmentation",
            "names": getattr(model.names, 'copy', lambda: model.names)() if hasattr(model, 'names') else {},
        }
        
        return info
        
    except Exception as e:
        logger.exception(f"Error getting model info: {e}")
        return {"error": str(e)}
