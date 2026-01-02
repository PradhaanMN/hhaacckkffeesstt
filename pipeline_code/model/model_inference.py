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
    Wrapper for YOLOv8 segmentation model ensemble for solar panel detection.
    
    Features:
    - 5-model ensemble (4 segmentation + 1 detection)
    - Toggleable hybrid ensemble/adversarial approach with consensus voting
    - Standard NMS merging when hybrid disabled
    - Test-Time Augmentation (horizontal flip)
    - Multi-scale inference (90%, 100%, 110%)
    - Shape validation filters for rectangular panels
    - Polygon refinement and clipping
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
        logger.info(f"Hybrid ensemble algorithm available (toggleable via use_hybrid parameter)")
    
    def run_inference(
        self,
        image_path: str,
        conf_threshold: float = 0.08,
        iou_threshold: float = 0.45,
        use_tta: bool = True,
        use_multiscale: bool = True,
        use_hybrid: bool = True
    ) -> List[Dict]:
        """
        Run inference on an image with ADVANCED DETECTION strategies.
        
        Features:
        - Hybrid ensemble/adversarial approach (4 models) [toggleable]
        - Test-Time Augmentation (TTA): flip/rotate variations
        - Multi-scale inference: multiple image sizes
        - Polygon refinement: smooth and optimize masks
        
        Args:
            image_path: Path to the input image
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            use_tta: Enable Test-Time Augmentation (default: True)
            use_multiscale: Enable multi-scale inference (default: True)
            use_hybrid: Enable hybrid ensemble algorithm (default: True)
            
        Returns:
            List of detections, each containing:
            {
                "polygon": [[x1, y1], [x2, y2], ...],  # Segmentation polygon
                "area_px": float,                       # Area in pixels
                "confidence": float,                    # Detection confidence
                "bbox": [x1, y1, x2, y2]               # Bounding box
            }
        """
        logger.info(f"🔧 INFERENCE CONFIG: use_hybrid={use_hybrid}, use_tta={use_tta}, use_multiscale={use_multiscale}")
        
        image_path = Path(image_path)
        
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return []
        
        try:
            # Load image for advanced processing
            from PIL import Image
            import cv2
            base_image = Image.open(image_path)
            
            # Generate augmented versions for TTA (Test-Time Augmentation)
            image_variants = [str(image_path)]  # Original
            
            if use_tta:
                logger.info("Applying Test-Time Augmentation (TTA)...")
                # Horizontal flip
                flipped = base_image.transpose(Image.FLIP_LEFT_RIGHT)
                flip_path = str(image_path).replace('.png', '_flip.png')
                flipped.save(flip_path)
                image_variants.append(flip_path)
            
            if use_multiscale:
                logger.info("Applying Multi-Scale Inference...")
                # Scale variations (90%, 110% of original size)
                for scale, suffix in [(0.9, '_s90'), (1.1, '_s110')]:
                    new_size = (int(base_image.width * scale), int(base_image.height * scale))
                    scaled = base_image.resize(new_size, Image.LANCZOS)
                    scale_path = str(image_path).replace('.png', f'{suffix}.png')
                    scaled.save(scale_path)
                    image_variants.append(scale_path)
            
            logger.info(f"Processing {len(image_variants)} image variants (TTA+MultiScale)")
            
            # Run inference on all variants
            all_variant_detections = []
            
            for variant_idx, variant_path in enumerate(image_variants):
                # Determine transformation for this variant
                is_flipped = '_flip' in variant_path
                scale_factor = 1.0
                if '_s90' in variant_path:
                    scale_factor = 0.9
                elif '_s110' in variant_path:
                    scale_factor = 1.1
                
                # Run inference on all ensemble models
                all_model_detections = []  # Track per-model results
                
                for idx, model in enumerate(self.ensemble_models):
                    # Run inference
                    results = model.predict(
                        source=variant_path,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        verbose=False
                    )
                    
                    # Extract detections from this model
                    model_detections = self._extract_detections(results)
                    
                    # Transform detections back to original image space
                    if is_flipped or scale_factor != 1.0:
                        model_detections = self._transform_detections_back(
                            model_detections, is_flipped, scale_factor, base_image.width, base_image.height
                        )
                    
                    all_model_detections.append({
                        'model_id': idx,
                        'detections': model_detections
                    })
                    
                    logger.debug(f"Variant {variant_idx+1}, Model {idx+1}/{len(self.ensemble_models)}: {len(model_detections)} detections")
                
                # Merge results: HYBRID algorithm vs Standard NMS
                if use_hybrid:
                    # HYBRID MODE: Advanced consensus voting + adversarial filtering
                    variant_detections = self._hybrid_ensemble_adversarial_merge(
                        all_model_detections, iou_threshold
                    )
                    logger.debug(f"Variant {variant_idx+1}: Hybrid merge applied")
                else:
                    # STANDARD MODE: Simple NMS merging (all detections, equal weight)
                    all_dets = []
                    for model_data in all_model_detections:
                        all_dets.extend(model_data['detections'])
                    variant_detections = self._merge_ensemble_detections(all_dets, iou_threshold)
                    logger.debug(f"Variant {variant_idx+1}: Standard NMS merge applied")
                
                all_variant_detections.extend(variant_detections)
            
            # Final merge across all variants (TTA consensus)
            if len(image_variants) > 1:
                logger.info(f"Merging {len(all_variant_detections)} detections from TTA/MultiScale...")
                detections = self._merge_tta_detections(all_variant_detections, iou_threshold)
                logger.info(f"TTA merge: {len(all_variant_detections)} → {len(detections)} final detections")
            else:
                # Run all ensemble models
                all_model_detections = []
                for idx, model in enumerate(self.ensemble_models):
                    results = model.predict(
                        source=str(image_path),
                        conf=conf_threshold,
                        iou=iou_threshold,
                        verbose=False
                    )
                    model_detections = self._extract_detections(results)
                    all_model_detections.append({
                        'model_id': idx,
                        'detections': model_detections
                    })
                    logger.debug(f"Model {idx+1}/{len(self.ensemble_models)}: {len(model_detections)} detections")
                
                # Merge strategy based on use_hybrid flag
                if use_hybrid:
                    # HYBRID MODE: Advanced consensus voting + adversarial filtering
                    detections = self._hybrid_ensemble_adversarial_merge(
                        all_model_detections, iou_threshold
                    )
                    total_raw = sum(len(m['detections']) for m in all_model_detections)
                    logger.info(f"HYBRID merge: {total_raw} raw → {len(detections)} final detections")
                    logger.info(f"Consensus/adversarial filtering applied with confidence adjustment")
                else:
                    # STANDARD MODE: Simple NMS merging without hybrid logic
                    all_dets = []
                    for model_data in all_model_detections:
                        all_dets.extend(model_data['detections'])
                    detections = self._merge_ensemble_detections(all_dets, iou_threshold)
                    total_raw = sum(len(m['detections']) for m in all_model_detections)
                    logger.info(f"STANDARD merge: {total_raw} raw → {len(detections)} final detections")
                    logger.info(f"Simple NMS merging without hybrid voting")
            
            # Apply polygon refinement for better edge quality
            detections = self._refine_polygons(detections)
            
            # Clean up temporary TTA/multiscale files
            if use_tta or use_multiscale:
                for variant_path in image_variants[1:]:  # Skip original
                    try:
                        Path(variant_path).unlink()
                    except:
                        pass
            
            return detections
            
        except Exception as e:
            logger.exception(f"Error during inference: {e}")
            return []
    
    def _extract_detections(self, results) -> List[Dict]:
        """Extract detections from model results."""
        detections = []
        raw_detection_count = 0
        
        if results and len(results) > 0:
            result = results[0]  # Get first result (single image)
            
            # Get image dimensions
            image_height, image_width = result.orig_shape if hasattr(result, 'orig_shape') else (None, None)
            
            # Check if masks are available (segmentation models)
            if result.masks is not None and len(result.masks) > 0:
                masks = result.masks.xy  # Get polygon coordinates
                boxes = result.boxes  # Get bounding boxes
                raw_detection_count = len(masks)
                logger.info(f"RAW MODEL OUTPUT: {raw_detection_count} detections from YOLO before filtering")
                
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
                
                logger.info(f"EXTRACTION RESULT: {len(detections)}/{raw_detection_count} detections passed validation")
            
            # Fallback for detection-only models (no masks, only bounding boxes)
            elif result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                raw_detection_count = len(boxes)
                logger.info(f"RAW MODEL OUTPUT: {raw_detection_count} bbox detections (detection-only model)")
                
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
        MIN_AREA_PX = 100  # ~10x10 pixels minimum - reasonable size
        if area_px < MIN_AREA_PX:
            return False
        
        # Filter 2: Maximum size (too large = likely entire roof/array)
        # Individual solar panels are typically 1.6m x 1m (~17 sq.ft each)
        # At typical satellite resolution (0.3-0.6m/pixel), one panel = ~2000-8000 pixels
        # Allow up to 50000 for commercial installations
        MAX_AREA_PX = 50000
        if area_px > MAX_AREA_PX:
            return False
        
        # Filter 2b: Relative to image size (reject if detection is too large)
        # Solar panel arrays shouldn't occupy more than 30% of the image
        if image_width and image_height:
            image_area = image_width * image_height
            area_ratio = area_px / image_area
            MAX_IMAGE_RATIO = 0.30  # Maximum 30% of image
            if area_ratio > MAX_IMAGE_RATIO:
                return False
        
        # Filter 3: Aspect ratio (solar panels are typically 1:1 to 4:1)
        # Reject very elongated irregular shapes
        if width > 0 and height > 0:
            aspect_ratio = max(width, height) / min(width, height)
            MAX_ASPECT_RATIO = 4.0  # Reasonable for panel arrays
            if aspect_ratio > MAX_ASPECT_RATIO:
                return False
        
        # Filter 4: Fill ratio (polygon area vs bbox area)
        # Solar panels MUST be rectangular - reject irregular blobs
        if bbox_area > 0:
            fill_ratio = area_px / bbox_area
            MIN_FILL_RATIO = 0.45  # At least 45% fill - enforce rectangular shape
            if fill_ratio < MIN_FILL_RATIO:
                return False
        
        # Filter 5: Minimum dimension (avoid thin lines like street names)
        MIN_DIMENSION = 8  # pixels - reasonable minimum
        if width < MIN_DIMENSION or height < MIN_DIMENSION:
            return False
        
        # Filter 6: Reject very thin shapes (text labels, road markings)
        # Solar panels should have reasonable width-to-height ratio
        if width > 0 and height > 0:
            min_thickness = min(width, height)
            max_length = max(width, height)
            thickness_ratio = min_thickness / max_length
            MIN_THICKNESS_RATIO = 0.15  # At least 15% thick - reject thin shapes
            if thickness_ratio < MIN_THICKNESS_RATIO:
                return False
        
        return True
    
    def _hybrid_ensemble_adversarial_merge(
        self, 
        all_model_detections: List[Dict], 
        iou_threshold: float
    ) -> List[Dict]:
        """
        HYBRID ENSEMBLE/ADVERSARIAL approach with CUSTOM MODEL PRIORITY:
        1. Group overlapping detections across models (ensemble voting)
        2. Apply adversarial confidence adjustment based on model agreement
        3. Give HIGHER WEIGHT to model_id=0 (custom-trained model) - 2x confidence weight
        4. Boost confidence for high consensus, penalize low consensus
        5. Filter out low-confidence adversarial challenges
        
        Custom Model Priority:
        - model_id=0 (solarpanel_seg_v1.pt): Your custom-trained model gets 2x weight
        - Capable of both detection and segmentation
        - Takes precedence in confidence calculation and filtering decisions
        """
        # Flatten all detections with model tracking
        all_detections = []
        for model_data in all_model_detections:
            for det in model_data['detections']:
                det_copy = det.copy()
                det_copy['model_id'] = model_data['model_id']
                all_detections.append(det_copy)
        
        if not all_detections:
            return []
        
        # Sort by confidence
        all_detections = sorted(all_detections, key=lambda x: x['confidence'], reverse=True)
        
        merged = []
        used = set()
        
        for i, det in enumerate(all_detections):
            if i in used:
                continue
            
            # Find overlapping detections (consensus group)
            consensus_group = [det]
            model_ids_in_group = {det['model_id']}
            
            for j in range(i + 1, len(all_detections)):
                if j in used:
                    continue
                
                iou = self._calculate_bbox_iou(det['bbox'], all_detections[j]['bbox'])
                if iou > iou_threshold:
                    consensus_group.append(all_detections[j])
                    model_ids_in_group.add(all_detections[j]['model_id'])
                    used.add(j)
            
            # ADVERSARIAL CONFIDENCE ADJUSTMENT
            num_models = len(all_model_detections)
            num_agreeing = len(model_ids_in_group)
            consensus_ratio = num_agreeing / num_models
            
            # CUSTOM MODEL PRIORITY: Give higher weight to model_id=0 (your custom model)
            # Calculate weighted average confidence with 2x weight for custom model
            total_weight = 0
            weighted_confidence_sum = 0
            custom_model_present = False
            
            for d in consensus_group:
                if d['model_id'] == 0:  # Custom model gets 2.5x weight
                    weight = 2.5
                    custom_model_present = True
                else:
                    weight = 1.0
                weighted_confidence_sum += d['confidence'] * weight
                total_weight += weight
            
            base_confidence = weighted_confidence_sum / total_weight
            
            # Extra boost if custom model is present
            if custom_model_present:
                base_confidence = min(base_confidence * 1.15, 1.0)  # +15% bonus, cap at 100%
            
            # Apply confidence adjustment based on consensus (relaxed thresholds)
            if consensus_ratio >= 0.6:
                # HIGH CONSENSUS (60%+ models agree): Boost confidence
                confidence_multiplier = 1.0 + (0.2 * (consensus_ratio - 0.6) / 0.4)  # +0% to +20%
                status = "HIGH_CONSENSUS"
            elif consensus_ratio >= 0.4:
                # MEDIUM CONSENSUS (40-60% models agree): Neutral
                confidence_multiplier = 1.0
                status = "MEDIUM_CONSENSUS"
            else:
                # LOW CONSENSUS (<40% models agree): Light penalize
                confidence_multiplier = 0.7 + (0.3 * consensus_ratio / 0.4)  # 70% to 100%
                status = "ADVERSARIAL_CHALLENGE"
            
            adjusted_confidence = base_confidence * confidence_multiplier
            
            # Filter out weak adversarial challenges
            # BUT: If custom model (id=0) detected it, be more lenient
            if custom_model_present:
                MIN_CONFIDENCE_THRESHOLD = 0.025  # Lower threshold for custom model detections
            else:
                MIN_CONFIDENCE_THRESHOLD = 0.05  # Normal threshold
                
            if adjusted_confidence < MIN_CONFIDENCE_THRESHOLD:
                logger.debug(f"Filtered out {status} detection: {num_agreeing}/{num_models} models, "
                           f"conf {base_confidence:.3f} → {adjusted_confidence:.3f}")
                continue
            
            # Create merged detection with adjusted confidence
            merged_det = self._average_detections(consensus_group)
            merged_det['confidence'] = adjusted_confidence
            merged_det['consensus_ratio'] = consensus_ratio
            merged_det['num_agreeing_models'] = num_agreeing
            merged_det['consensus_status'] = status
            
            if custom_model_present:
                merged_det['custom_model_detection'] = True
                logger.debug(f"{status} [CUSTOM MODEL]: {num_agreeing}/{num_models} models, "
                            f"conf {base_confidence:.3f} → {adjusted_confidence:.3f} (custom model priority applied)")
            else:
                logger.debug(f"{status}: {num_agreeing}/{num_models} models, "
                            f"conf {base_confidence:.3f} → {adjusted_confidence:.3f}")
            
            merged.append(merged_det)
        
        # Log summary of custom model detections
        custom_count = sum(1 for d in merged if d.get('custom_model_detection', False))
        if custom_count > 0:
            logger.info(f"✓ Custom model contributed to {custom_count}/{len(merged)} final detections")
        
        return merged
    
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
    
    def _transform_detections_back(
        self,
        detections: List[Dict],
        is_flipped: bool,
        scale_factor: float,
        orig_width: int,
        orig_height: int
    ) -> List[Dict]:
        """Transform detections from augmented image back to original space."""
        transformed = []
        
        for det in detections:
            polygon = det['polygon']
            bbox = det['bbox']
            
            # Undo scaling
            if scale_factor != 1.0:
                polygon = [[x / scale_factor, y / scale_factor] for x, y in polygon]
                bbox = [coord / scale_factor for coord in bbox]
            
            # Undo flipping
            if is_flipped:
                polygon = [[orig_width - x, y] for x, y in polygon]
                bbox = [orig_width - bbox[2], bbox[1], orig_width - bbox[0], bbox[3]]
            
            # Recalculate area after transformation
            area_px = self._calculate_polygon_area(polygon)
            
            transformed.append({
                'polygon': polygon,
                'bbox': bbox,
                'area_px': area_px,
                'confidence': det['confidence']
            })
        
        return transformed
    
    def _merge_tta_detections(self, detections: List[Dict], iou_threshold: float) -> List[Dict]:
        """
        Merge detections from Test-Time Augmentation variants.
        Detections that appear across multiple augmentations get confidence boost.
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
            
            # Find overlapping detections from TTA variants
            tta_group = [det]
            for j in range(i + 1, len(detections)):
                if j in used:
                    continue
                
                iou = self._calculate_bbox_iou(det['bbox'], detections[j]['bbox'])
                if iou > iou_threshold:
                    tta_group.append(detections[j])
                    used.add(j)
            
            # TTA consensus boost: more variants = higher confidence
            base_confidence = sum(d['confidence'] for d in tta_group) / len(tta_group)
            tta_boost = min(0.15, 0.05 * (len(tta_group) - 1))  # Up to +15% boost
            adjusted_confidence = min(1.0, base_confidence + tta_boost)
            
            # Average the detections
            merged_det = self._average_detections(tta_group)
            merged_det['confidence'] = adjusted_confidence
            merged_det['tta_variants'] = len(tta_group)
            
            merged.append(merged_det)
        
        return merged
    
    def _refine_polygons(self, detections: List[Dict]) -> List[Dict]:
        """
        Refine polygon masks using advanced techniques:
        - Smooth jagged edges (Douglas-Peucker algorithm)
        - Remove tiny artifacts
        - Optimize vertex count
        """
        import cv2
        import numpy as np
        
        refined = []
        
        for det in detections:
            polygon = det['polygon']
            
            if len(polygon) < 3:
                continue
            
            # Convert to numpy array
            poly_array = np.array(polygon, dtype=np.float32)
            
            # Apply Douglas-Peucker simplification for smoother edges
            epsilon = 0.5  # Simplification tolerance
            simplified = cv2.approxPolyDP(poly_array, epsilon, closed=True)
            
            if len(simplified) < 3:
                simplified = poly_array  # Keep original if simplification failed
            
            # Convert back to list
            refined_polygon = [[float(x), float(y)] for x, y in simplified.reshape(-1, 2)]
            
            # Recalculate area and bbox
            refined_area = self._calculate_polygon_area(refined_polygon)
            refined_bbox = self._calculate_polygon_bbox(refined_polygon)
            
            refined.append({
                'polygon': refined_polygon,
                'area_px': refined_area,
                'bbox': refined_bbox,
                'confidence': det['confidence'],
                **{k: v for k, v in det.items() if k not in ['polygon', 'area_px', 'bbox', 'confidence']}
            })
        
        return refined
    
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
