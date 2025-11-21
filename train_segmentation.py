"""
YOLOv8 Segmentation Model Training with Combined Datasets
Merges detection and segmentation datasets for comprehensive training
"""

import json
import shutil
from pathlib import Path
from ultralytics import YOLO
import yaml

def merge_datasets():
    """Merge detection and segmentation datasets into unified structure"""
    print("\n" + "="*60)
    print("MERGING DATASETS")
    print("="*60)
    
    # Create output directories
    output_dir = Path("dataset_combined")
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Dataset sources
    seg_dataset = Path("solarpv-INDIA.v2i.coco-segmentation")
    det_dataset = Path("dataset")
    
    stats = {'train': 0, 'val': 0, 'test': 0}
    
    # Process segmentation dataset (satellite images with polygons)
    print("\n→ Processing segmentation dataset (satellite/aerial views)...")
    seg_splits = {'train': 'train', 'valid': 'val', 'test': 'test'}
    
    for seg_split, yolo_split in seg_splits.items():
        ann_file = seg_dataset / seg_split / "_annotations.coco.json"
        with open(ann_file) as f:
            coco_data = json.load(f)
        
        # Create image lookup
        img_lookup = {img['id']: img for img in coco_data['images']}
        
        # Group annotations by image
        img_annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_annotations:
                img_annotations[img_id] = []
            img_annotations[img_id].append(ann)
        
        # Process each image
        for img_id, img_info in img_lookup.items():
            img_filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Copy image
            src_img = seg_dataset / seg_split / img_filename
            dst_img = output_dir / yolo_split / 'images' / img_filename
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
                
                # Create YOLO format label with segmentation
                label_file = output_dir / yolo_split / 'labels' / f"{Path(img_filename).stem}.txt"
                with open(label_file, 'w') as f:
                    if img_id in img_annotations:
                        for ann in img_annotations[img_id]:
                            # Class (always 0 for solarpanel)
                            class_id = 0
                            
                            # Convert polygon segmentation to YOLO format
                            if 'segmentation' in ann and ann['segmentation']:
                                seg = ann['segmentation'][0] if isinstance(ann['segmentation'], list) else ann['segmentation']
                                # Normalize coordinates
                                normalized_coords = []
                                for i in range(0, len(seg), 2):
                                    x = seg[i] / img_width
                                    y = seg[i+1] / img_height
                                    normalized_coords.extend([x, y])
                                
                                # Write segmentation line
                                f.write(f"{class_id} " + " ".join([f"{c:.6f}" for c in normalized_coords]) + "\n")
                
                stats[yolo_split] += 1
    
    print(f"  ✓ Segmentation: {stats['train']} train, {stats['val']} val, {stats['test']} test")
    
    # Process detection dataset (ground-level images with bboxes)
    print("\n→ Processing detection dataset (ground-level views)...")
    det_stats = {'train': 0, 'val': 0, 'test': 0}
    det_splits = {'train': 'train', 'val': 'val', 'test': 'test'}
    
    for det_split, yolo_split in det_splits.items():
        ann_file = det_dataset / det_split / "images" / "_annotations.coco.json"
        if not ann_file.exists():
            continue
            
        with open(ann_file) as f:
            coco_data = json.load(f)
        
        # Create image lookup
        img_lookup = {img['id']: img for img in coco_data['images']}
        
        # Group annotations by image
        img_annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_annotations:
                img_annotations[img_id] = []
            img_annotations[img_id].append(ann)
        
        # Process each image
        for img_id, img_info in img_lookup.items():
            img_filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Copy image with unique name to avoid conflicts
            src_img = det_dataset / det_split / "images" / img_filename
            dst_img = output_dir / yolo_split / 'images' / f"det_{img_filename}"
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
                
                # Create YOLO format label (convert bbox to segmentation format - 4 corners)
                label_file = output_dir / yolo_split / 'labels' / f"det_{Path(img_filename).stem}.txt"
                with open(label_file, 'w') as f:
                    if img_id in img_annotations:
                        for ann in img_annotations[img_id]:
                            # Class (always 0 for solarpanel)
                            class_id = 0
                            
                            # Convert bbox to 4-point polygon for segmentation
                            bbox = ann['bbox']  # [x, y, width, height]
                            x, y, w, h = bbox
                            
                            # Create 4 corners in normalized format
                            x1, y1 = x / img_width, y / img_height
                            x2, y2 = (x + w) / img_width, y / img_height
                            x3, y3 = (x + w) / img_width, (y + h) / img_height
                            x4, y4 = x / img_width, (y + h) / img_height
                            
                            # Write as segmentation (clockwise from top-left)
                            f.write(f"{class_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}\n")
                
                det_stats[yolo_split] += 1
    
    print(f"  ✓ Detection: {det_stats['train']} train, {det_stats['val']} val, {det_stats['test']} test")
    
    # Update total stats
    for split in ['train', 'val', 'test']:
        stats[split] += det_stats[split]
    
    # Create data.yaml
    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {0: 'solarpanel'},
        'nc': 1
    }
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\n✓ Dataset merged successfully!")
    print(f"  Total: {stats['train']} train, {stats['val']} val, {stats['test']} test")
    print(f"  Config: {yaml_path}")
    
    return yaml_path

def train_segmentation_model(data_yaml):
    """Train YOLOv8 segmentation model on combined dataset"""
    print("\n" + "="*60)
    print("TRAINING YOLOV8 SEGMENTATION MODEL")
    print("="*60)
    
    # Load YOLOv8 segmentation model
    print("\n→ Loading YOLOv8n-seg model...")
    model = YOLO('yolov8n-seg.pt')  # Segmentation model
    
    # Training configuration
    print("\n→ Starting training...")
    print("  Architecture: YOLOv8n-seg (segmentation)")
    print("  Epochs: 100")
    print("  Image size: 640x640")
    print("  Device: GPU (CUDA)")
    print("  Dataset: Combined detection + segmentation")
    
    # Train the model
    results = model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=-1,  # Auto batch size
        name='solar_panel_segmentation',
        project='results',
        device=0,  # Use first GPU
        patience=20,  # Early stopping
        save=True,
        plots=True,
        verbose=True
    )
    
    print("\n✓ Training complete!")
    
    # Save models
    models_dir = Path("models_segmentation")
    models_dir.mkdir(exist_ok=True)
    
    # Copy best model
    best_model = Path("results/solar_panel_segmentation/weights/best.pt")
    last_model = Path("results/solar_panel_segmentation/weights/last.pt")
    
    if best_model.exists():
        shutil.copy2(best_model, models_dir / "best_seg.pt")
        print(f"  ✓ Best model saved: models_segmentation/best_seg.pt")
    
    if last_model.exists():
        shutil.copy2(last_model, models_dir / "last_seg.pt")
        print(f"  ✓ Last model saved: models_segmentation/last_seg.pt")
    
    return results

def validate_model():
    """Validate the trained segmentation model"""
    print("\n" + "="*60)
    print("VALIDATING MODEL")
    print("="*60)
    
    model_path = Path("models_segmentation/best_seg.pt")
    if not model_path.exists():
        print("⚠ Model not found!")
        return
    
    model = YOLO(str(model_path))
    
    # Run validation
    results = model.val(
        data='dataset_combined/data.yaml',
        imgsz=640,
        batch=8,
        verbose=True
    )
    
    print("\n✓ Validation complete!")
    print(f"  Precision: {results.box.p.mean():.4f}")
    print(f"  Recall: {results.box.r.mean():.4f}")
    print(f"  mAP@0.5: {results.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {results.box.map:.4f}")

if __name__ == "__main__":
    try:
        # Step 1: Merge datasets
        data_yaml_path = merge_datasets()
        
        # Step 2: Train segmentation model
        train_segmentation_model(data_yaml_path)
        
        # Step 3: Validate model
        validate_model()
        
        print("\n" + "="*60)
        print("ALL STEPS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Test model: python predict_segmentation.py --source <image>")
        print("  2. Check results: results/solar_panel_segmentation/")
        print("  3. Use model: models_segmentation/best_seg.pt")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
