"""
Example: Create a sample COCO dataset for testing
This script creates a minimal working example for testing the pipeline.
"""

import json
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import random


def create_sample_image_with_bbox(output_path, image_size=(640, 640), num_objects=3):
    """
    Create a sample image with simple rectangles representing solar panels.
    
    Args:
        output_path: Path to save the image
        image_size: Tuple of (width, height)
        num_objects: Number of solar panel rectangles to draw
        
    Returns:
        List of bounding boxes in COCO format [x, y, width, height]
    """
    # Create a blank image (simulating satellite view)
    img = Image.new('RGB', image_size, color=(100, 120, 100))  # Greenish background
    draw = ImageDraw.Draw(img)
    
    bboxes = []
    
    for i in range(num_objects):
        # Random position and size for solar panel
        panel_w = random.randint(30, 80)
        panel_h = random.randint(20, 60)
        x = random.randint(10, image_size[0] - panel_w - 10)
        y = random.randint(10, image_size[1] - panel_h - 10)
        
        # Draw rectangle (solar panel)
        draw.rectangle(
            [x, y, x + panel_w, y + panel_h],
            fill=(50, 70, 120),  # Bluish color
            outline=(30, 50, 100),
            width=2
        )
        
        # Store bbox in COCO format
        bboxes.append([x, y, panel_w, panel_h])
    
    # Save image
    img.save(output_path)
    return bboxes


def create_sample_dataset(num_train=10, num_val=5, num_test=3):
    """
    Create a sample dataset for testing the pipeline.
    
    Args:
        num_train: Number of training images to create
        num_val: Number of validation images to create
        num_test: Number of test images to create
    """
    print("Creating sample dataset for testing...")
    print("This is for DEMONSTRATION purposes only.")
    print("Replace with your real dataset before actual training!\n")
    
    splits = {
        'train': num_train,
        'val': num_val,
        'test': num_test
    }
    
    for split, num_images in splits.items():
        print(f"Creating {split} set ({num_images} images)...")
        
        # Create directories
        split_dir = Path(f"dataset/{split}")
        images_dir = split_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize COCO format
        coco_data = {
            "info": {
                "description": f"Sample Solar Panel Dataset - {split.upper()}",
                "version": "1.0",
                "year": 2024,
                "contributor": "Auto-generated",
                "date_created": "2024-01-01"
            },
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": 1,
                    "name": "solar_panel",
                    "supercategory": "object"
                }
            ]
        }
        
        ann_id = 1
        
        # Create images
        for img_id in range(1, num_images + 1):
            filename = f"{split}_{img_id:04d}.jpg"
            img_path = images_dir / filename
            
            # Create sample image
            num_panels = random.randint(1, 5)
            bboxes = create_sample_image_with_bbox(img_path, num_objects=num_panels)
            
            # Add image info
            coco_data["images"].append({
                "id": img_id,
                "file_name": filename,
                "width": 640,
                "height": 640
            })
            
            # Add annotations
            for bbox in bboxes:
                area = bbox[2] * bbox[3]
                coco_data["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0
                })
                ann_id += 1
        
        # Save annotations
        ann_path = split_dir / "annotations.json"
        with open(ann_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"  ✓ Created {num_images} images")
        print(f"  ✓ Created {len(coco_data['annotations'])} annotations")
        print(f"  ✓ Saved to {split_dir}/\n")
    
    print("="*60)
    print("SAMPLE DATASET CREATED")
    print("="*60)
    print("\nNext steps:")
    print("  1. Visualize: python visualize.py")
    print("  2. Train: python train.py")
    print("  3. Predict: python predict.py --source dataset/test/images/test_0001.jpg")
    print("\n⚠ Remember: This is synthetic data for testing!")
    print("Replace with real satellite imagery for actual deployment.")


if __name__ == "__main__":
    create_sample_dataset(
        num_train=10,
        num_val=5,
        num_test=3
    )
