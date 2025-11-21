"""
COCO to YOLO Dataset Converter
Converts COCO JSON format to YOLO format and validates the dataset.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np


class COCOToYOLOConverter:
    """Converts COCO format annotations to YOLO format."""
    
    def __init__(self, coco_json_path: str, images_dir: str, output_dir: str):
        """
        Initialize the converter.
        
        Args:
            coco_json_path: Path to COCO JSON annotation file
            images_dir: Directory containing the images
            output_dir: Output directory for YOLO format dataset
        """
        self.coco_json_path = coco_json_path
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.coco_data = None
        self.category_mapping = {}
        
    def load_and_validate_coco(self) -> bool:
        """Load and validate COCO JSON file."""
        try:
            with open(self.coco_json_path, 'r') as f:
                self.coco_data = json.load(f)
            
            print(f"✓ Loaded COCO JSON: {self.coco_json_path}")
            
            # Validate required fields
            required_fields = ['images', 'annotations', 'categories']
            for field in required_fields:
                if field not in self.coco_data:
                    print(f"✗ Missing required field: {field}")
                    self.coco_data[field] = []
            
            # Auto-fix: Ensure categories exist
            if not self.coco_data['categories']:
                print("⚠ No categories found. Creating default 'solar_panel' category.")
                self.coco_data['categories'] = [
                    {"id": 1, "name": "solar_panel", "supercategory": "object"}
                ]
            
            # Build category mapping
            for cat in self.coco_data['categories']:
                self.category_mapping[cat['id']] = cat
            
            print(f"✓ Categories: {[cat['name'] for cat in self.coco_data['categories']]}")
            print(f"✓ Images: {len(self.coco_data['images'])}")
            print(f"✓ Annotations: {len(self.coco_data['annotations'])}")
            
            return True
            
        except FileNotFoundError:
            print(f"✗ COCO JSON file not found: {self.coco_json_path}")
            return False
        except json.JSONDecodeError as e:
            print(f"✗ Invalid JSON format: {e}")
            return False
    
    def validate_and_fix_annotations(self):
        """Validate and auto-fix common annotation issues."""
        fixed_count = 0
        removed_count = 0
        valid_annotations = []
        
        for ann in self.coco_data['annotations']:
            # Check if bbox exists and is valid
            if 'bbox' not in ann or not ann['bbox']:
                removed_count += 1
                continue
            
            bbox = ann['bbox']
            
            # COCO format: [x, y, width, height]
            # Fix negative or zero dimensions
            if len(bbox) != 4:
                removed_count += 1
                continue
            
            x, y, w, h = bbox
            
            # Fix negative dimensions
            if w <= 0 or h <= 0:
                print(f"⚠ Fixed invalid bbox dimensions: {bbox}")
                w = max(1, abs(w))
                h = max(1, abs(h))
                ann['bbox'] = [x, y, w, h]
                fixed_count += 1
            
            # Ensure category_id exists
            if 'category_id' not in ann:
                ann['category_id'] = 1  # Default to first category
                fixed_count += 1
            
            valid_annotations.append(ann)
        
        self.coco_data['annotations'] = valid_annotations
        
        if fixed_count > 0:
            print(f"⚠ Fixed {fixed_count} annotation issues")
        if removed_count > 0:
            print(f"⚠ Removed {removed_count} invalid annotations")
    
    def convert_bbox_coco_to_yolo(self, bbox: List[float], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        Convert COCO bbox to YOLO format.
        
        COCO: [x, y, width, height] (absolute)
        YOLO: [x_center, y_center, width, height] (normalized 0-1)
        """
        x, y, w, h = bbox
        
        # Convert to center coordinates
        x_center = x + w / 2
        y_center = y + h / 2
        
        # Normalize
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        w_norm = w / img_width
        h_norm = h / img_height
        
        # Clamp values to [0, 1]
        x_center_norm = max(0, min(1, x_center_norm))
        y_center_norm = max(0, min(1, y_center_norm))
        w_norm = max(0, min(1, w_norm))
        h_norm = max(0, min(1, h_norm))
        
        return x_center_norm, y_center_norm, w_norm, h_norm
    
    def convert(self):
        """Convert COCO dataset to YOLO format."""
        if not self.load_and_validate_coco():
            print("✗ Failed to load COCO data. Aborting conversion.")
            return False
        
        self.validate_and_fix_annotations()
        
        # Create output directories
        output_path = Path(self.output_dir)
        images_output = output_path / "images"
        labels_output = output_path / "labels"
        images_output.mkdir(parents=True, exist_ok=True)
        labels_output.mkdir(parents=True, exist_ok=True)
        
        # Build image id to annotations mapping
        img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        # Build image id to image info mapping
        img_id_to_info = {img['id']: img for img in self.coco_data['images']}
        
        converted_count = 0
        skipped_count = 0
        
        for img_info in self.coco_data['images']:
            img_id = img_info['id']
            img_filename = img_info['file_name']
            img_width = img_info.get('width', 0)
            img_height = img_info.get('height', 0)
            
            # Source image path
            src_img_path = Path(self.images_dir) / img_filename
            
            # Check if image exists
            if not src_img_path.exists():
                print(f"⚠ Image not found: {src_img_path}")
                skipped_count += 1
                continue
            
            # If width/height not in JSON, read from image
            if img_width == 0 or img_height == 0:
                img = cv2.imread(str(src_img_path))
                if img is None:
                    skipped_count += 1
                    continue
                img_height, img_width = img.shape[:2]
            
            # Copy image
            dst_img_path = images_output / img_filename
            shutil.copy2(src_img_path, dst_img_path)
            
            # Convert annotations
            label_filename = Path(img_filename).stem + '.txt'
            label_path = labels_output / label_filename
            
            annotations = img_to_anns.get(img_id, [])
            
            with open(label_path, 'w') as f:
                for ann in annotations:
                    category_id = ann['category_id']
                    bbox = ann['bbox']
                    
                    # Convert to YOLO format (class_id is 0-indexed)
                    class_id = category_id - 1  # YOLO uses 0-based indexing
                    x_c, y_c, w, h = self.convert_bbox_coco_to_yolo(bbox, img_width, img_height)
                    
                    # Write to label file
                    f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
            
            converted_count += 1
        
        print(f"\n✓ Conversion complete!")
        print(f"  Converted: {converted_count} images")
        print(f"  Skipped: {skipped_count} images")
        print(f"  Output: {output_path}")
        
        # Save class names
        class_names_path = output_path / "classes.txt"
        with open(class_names_path, 'w') as f:
            for cat in sorted(self.coco_data['categories'], key=lambda x: x['id']):
                f.write(f"{cat['name']}\n")
        
        print(f"  Classes saved to: {class_names_path}")
        
        return True


def convert_coco_dataset(coco_json: str, images_dir: str, output_dir: str):
    """Helper function to convert COCO dataset."""
    converter = COCOToYOLOConverter(coco_json, images_dir, output_dir)
    return converter.convert()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python coco_to_yolo.py <coco_json> <images_dir> <output_dir>")
        sys.exit(1)
    
    coco_json = sys.argv[1]
    images_dir = sys.argv[2]
    output_dir = sys.argv[3]
    
    convert_coco_dataset(coco_json, images_dir, output_dir)
