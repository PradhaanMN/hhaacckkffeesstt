"""
YOLOv8 Solar Panel Detection Training Script
Trains YOLOv8 model on custom COCO dataset for solar panel detection.
"""

import os
import sys
import yaml
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from utils.coco_to_yolo import convert_coco_dataset


def create_dataset_yaml(dataset_root: str, class_names: list, output_path: str):
    """Create YOLO dataset configuration file."""
    dataset_config = {
        'path': str(Path(dataset_root).absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {i: name for i, name in enumerate(class_names)},
        'nc': len(class_names)
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"✓ Created dataset config: {output_path}")
    return output_path


def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = ['models', 'results', 'results/samples', 'dataset', 'dataset/yolo_format']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓ Directories initialized")


def convert_datasets():
    """Convert COCO datasets to YOLO format."""
    print("\n" + "="*60)
    print("DATASET CONVERSION")
    print("="*60)
    
    dataset_splits = ['train', 'val', 'test']
    conversion_successful = False
    
    for split in dataset_splits:
        coco_json = f"dataset/{split}/annotations.json"
        images_dir = f"dataset/{split}/images"
        output_dir = f"dataset/yolo_format/{split}"
        
        # Check if COCO JSON exists
        if not Path(coco_json).exists():
            print(f"⚠ {split.upper()} dataset not found: {coco_json}")
            continue
        
        # Check if images directory exists
        if not Path(images_dir).exists():
            # Try alternative structure
            images_dir = f"dataset/{split}"
            if not Path(images_dir).exists():
                print(f"⚠ {split.upper()} images not found: {images_dir}")
                continue
        
        print(f"\n→ Converting {split.upper()} dataset...")
        success = convert_coco_dataset(coco_json, images_dir, output_dir)
        
        if success:
            conversion_successful = True
    
    if not conversion_successful:
        print("\n✗ No datasets were converted. Please ensure your dataset is in the correct format:")
        print("  dataset/")
        print("    train/")
        print("      annotations.json")
        print("      images/")
        print("    val/")
        print("      annotations.json")
        print("      images/")
        print("    test/")
        print("      annotations.json")
        print("      images/")
        return False
    
    return True


def load_class_names():
    """Load class names from converted dataset."""
    class_file = Path("dataset/yolo_format/train/classes.txt")
    
    if not class_file.exists():
        # Try val or test
        class_file = Path("dataset/yolo_format/val/classes.txt")
        if not class_file.exists():
            class_file = Path("dataset/yolo_format/test/classes.txt")
    
    if class_file.exists():
        with open(class_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        print(f"✓ Loaded classes: {classes}")
        return classes
    else:
        # Default for solar panel detection
        print("⚠ Using default class: solar_panel")
        return ['solar_panel']


def train_yolo():
    """Train YOLOv8 model."""
    print("\n" + "="*60)
    print("YOLO TRAINING")
    print("="*60)
    
    # Setup directories
    setup_directories()
    
    # Convert datasets
    if not convert_datasets():
        print("\n⚠ Skipping training - no valid datasets found.")
        print("Please place your COCO dataset in the dataset/ folder and try again.")
        return None
    
    # Load class names
    class_names = load_class_names()
    
    # Create dataset YAML
    dataset_yaml = create_dataset_yaml(
        dataset_root="dataset/yolo_format",
        class_names=class_names,
        output_path="dataset/yolo_format/data.yaml"
    )
    
    # Initialize YOLOv8 model
    print("\n→ Initializing YOLOv8n model...")
    model = YOLO('yolov8n.pt')  # Load pretrained YOLOv8 nano model
    
    # Training configuration
    training_config = {
        'data': dataset_yaml,
        'epochs': 100,
        'imgsz': 640,
        'batch': -1,  # Auto batch size
        'optimizer': 'SGD',
        'project': 'results',
        'name': 'solar_panel_detection',
        'exist_ok': True,
        'patience': 50,
        'save': True,
        'save_period': 10,
        'plots': True,
        'verbose': True,
    }
    
    print("\n→ Training Configuration:")
    for key, value in training_config.items():
        print(f"  {key}: {value}")
    
    # Create training log file
    log_file = Path("results/training_log.txt")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, 'w') as f:
        f.write(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: YOLOv8n\n")
        f.write(f"Classes: {class_names}\n")
        f.write(f"Configuration: {training_config}\n")
        f.write("="*60 + "\n\n")
    
    print(f"\n✓ Training log: {log_file}")
    
    # Start training
    print("\n" + "="*60)
    print("TRAINING STARTED")
    print("="*60 + "\n")
    
    try:
        results = model.train(**training_config)
        
        # Log results
        with open(log_file, 'a') as f:
            f.write(f"\nTraining completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Results saved to: {results.save_dir}\n")
        
        # Copy best model to models directory
        best_model_src = Path(results.save_dir) / "weights" / "best.pt"
        last_model_src = Path(results.save_dir) / "weights" / "last.pt"
        
        if best_model_src.exists():
            import shutil
            shutil.copy2(best_model_src, "models/best.pt")
            print(f"\n✓ Best model saved to: models/best.pt")
        
        if last_model_src.exists():
            import shutil
            shutil.copy2(last_model_src, "models/last.pt")
            print(f"✓ Last model saved to: models/last.pt")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        
        return results
        
    except Exception as e:
        error_msg = f"\n✗ Training failed: {str(e)}\n"
        print(error_msg)
        with open(log_file, 'a') as f:
            f.write(error_msg)
        return None


def run_validation_test(model_path: str = "models/best.pt"):
    """Run inference on validation samples."""
    print("\n" + "="*60)
    print("VALIDATION TEST")
    print("="*60)
    
    if not Path(model_path).exists():
        print(f"✗ Model not found: {model_path}")
        return
    
    # Find validation images
    val_images_dir = Path("dataset/yolo_format/val/images")
    
    if not val_images_dir.exists():
        print(f"⚠ Validation images not found: {val_images_dir}")
        return
    
    # Get first 5 validation images
    image_files = list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png"))
    test_images = image_files[:5]
    
    if not test_images:
        print("⚠ No validation images found")
        return
    
    print(f"\n→ Running inference on {len(test_images)} validation samples...")
    
    # Load model
    model = YOLO(model_path)
    
    # Run predictions
    output_dir = Path("results/samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in test_images:
        results = model.predict(
            source=str(img_path),
            save=True,
            project=str(output_dir),
            name="",
            exist_ok=True,
            conf=0.25
        )
        print(f"  ✓ Processed: {img_path.name}")
    
    print(f"\n✓ Sample predictions saved to: {output_dir}")


def main():
    """Main training pipeline."""
    print("\n" + "="*60)
    print("SOLAR PANEL DETECTION - YOLOV8 TRAINING")
    print("="*60 + "\n")
    
    # Train model
    results = train_yolo()
    
    if results is not None:
        # Run validation test
        run_validation_test()
        
        print("\n" + "="*60)
        print("ALL TASKS COMPLETE")
        print("="*60)
        print("\nNext steps:")
        print("  1. Review training results in results/solar_panel_detection/")
        print("  2. Check validation samples in results/samples/")
        print("  3. Run predictions: python predict.py --source <image_path>")
        print("  4. Visualize annotations: python visualize.py")
    else:
        print("\n⚠ Training was not completed. Please check the errors above.")


if __name__ == "__main__":
    main()
