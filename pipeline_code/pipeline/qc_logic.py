"""
Quality Control (QC) logic for determining if detections are verifiable.
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def determine_qc_status(
    image_fetch_success: bool,
    detections: List[Dict],
    image_metadata: Dict = None,
    notes: str = ""
) -> str:
    """
    Determine the QC status for a detection result.
    
    Returns "VERIFIABLE" if there is clear evidence either way (solar present or not present)
    with reasonably clean imagery.
    
    Returns "NOT_VERIFIABLE" if:
        - Imagery download failed
        - Imagery is too low resolution
        - Heavy cloud cover or shadows
        - Roof is fully occluded by trees/water tanks
        - Imagery is missing or corrupted
    
    Args:
        image_fetch_success: Whether the image was successfully fetched
        detections: List of detection dictionaries from the model
        image_metadata: Optional metadata about the image quality
        notes: Additional notes about quality issues
        
    Returns:
        "VERIFIABLE" or "NOT_VERIFIABLE"
    """
    
    # Rule 1: If image fetch failed, not verifiable
    if not image_fetch_success:
        logger.info("QC Status: NOT_VERIFIABLE - Image fetch failed")
        return "NOT_VERIFIABLE"
    
    # Rule 2: If metadata indicates quality issues
    if image_metadata:
        if image_metadata.get("quality_issue"):
            logger.info(f"QC Status: NOT_VERIFIABLE - Quality issue: {image_metadata.get('quality_issue')}")
            return "NOT_VERIFIABLE"
        
        # Check for very low resolution (if available)
        if image_metadata.get("resolution_warning"):
            logger.info("QC Status: NOT_VERIFIABLE - Low resolution")
            return "NOT_VERIFIABLE"
    
    # Rule 3: Check for explicit quality notes
    not_verifiable_keywords = [
        "cloud", "shadow", "occluded", "tree", "tank",
        "missing", "corrupted", "poor quality", "low resolution"
    ]
    
    if notes:
        notes_lower = notes.lower()
        for keyword in not_verifiable_keywords:
            if keyword in notes_lower:
                logger.info(f"QC Status: NOT_VERIFIABLE - Found keyword '{keyword}' in notes")
                return "NOT_VERIFIABLE"
    
    # Rule 4: If we have detections, it's verifiable (we detected something)
    if detections and len(detections) > 0:
        logger.info(f"QC Status: VERIFIABLE - Found {len(detections)} detection(s)")
        return "VERIFIABLE"
    
    # Rule 5: No detections but image is OK - still verifiable (confirmed no solar)
    # This is important: absence of evidence with good imagery is still evidence
    logger.info("QC Status: VERIFIABLE - No detections, but image quality is acceptable")
    return "VERIFIABLE"


def check_image_quality(image_path: str) -> Dict:
    """
    Perform basic image quality checks.
    
    This is a placeholder for more sophisticated quality checks that could include:
    - Blur detection
    - Cloud detection
    - Occlusion detection
    - Contrast/brightness analysis
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with quality assessment:
            - quality_ok: bool
            - quality_issue: str or None
            - resolution_warning: bool
    """
    import cv2
    
    result = {
        "quality_ok": True,
        "quality_issue": None,
        "resolution_warning": False
    }
    
    try:
        # Read image
        img = cv2.imread(image_path)
        
        if img is None:
            result["quality_ok"] = False
            result["quality_issue"] = "Failed to read image"
            return result
        
        # Check resolution
        height, width = img.shape[:2]
        if height < 200 or width < 200:
            result["quality_ok"] = False
            result["quality_issue"] = "Image resolution too low"
            result["resolution_warning"] = True
            return result
        
        # Check if image is too dark or too bright
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = gray.mean()
        
        if mean_brightness < 20:
            result["quality_ok"] = False
            result["quality_issue"] = "Image too dark"
        elif mean_brightness > 235:
            result["quality_ok"] = False
            result["quality_issue"] = "Image too bright (possible cloud cover)"
        
        # Check variance (very low variance might indicate cloud cover or blur)
        variance = gray.var()
        if variance < 100:
            logger.warning(f"Low image variance ({variance:.1f}) - possible cloud cover or blur")
            # Don't mark as not verifiable, but log a warning
        
    except Exception as e:
        logger.warning(f"Error in quality check: {e}")
        result["quality_ok"] = True  # Default to OK if we can't check
    
    return result
