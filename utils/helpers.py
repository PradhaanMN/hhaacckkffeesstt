"""
Utility functions for the solar panel detection project.
"""

from pathlib import Path
import json


def ensure_directories():
    """Create necessary project directories if they don't exist."""
    dirs = [
        'models',
        'results',
        'results/samples',
        'dataset',
        'dataset/train',
        'dataset/val',
        'dataset/test',
        'dataset/yolo_format'
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def get_latest_model(models_dir: str = "models") -> str:
    """
    Get the most recently modified model file.
    
    Args:
        models_dir: Directory containing model files
        
    Returns:
        Path to the latest model file
    """
    models_path = Path(models_dir)
    
    # Try best.pt first
    best_model = models_path / "best.pt"
    if best_model.exists():
        return str(best_model)
    
    # Otherwise find the most recent .pt file
    pt_files = list(models_path.glob("*.pt"))
    
    if not pt_files:
        raise FileNotFoundError(f"No model files found in {models_dir}")
    
    latest = max(pt_files, key=lambda p: p.stat().st_mtime)
    return str(latest)


def load_coco_json(json_path: str) -> dict:
    """
    Load and return COCO JSON file.
    
    Args:
        json_path: Path to COCO JSON file
        
    Returns:
        Dictionary containing COCO data
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def save_coco_json(data: dict, output_path: str):
    """
    Save COCO format data to JSON file.
    
    Args:
        data: COCO format dictionary
        output_path: Output JSON file path
    """
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def validate_dataset_structure():
    """
    Validate that the dataset has the expected structure.
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check dataset directory exists
    if not Path("dataset").exists():
        errors.append("dataset/ directory not found")
        return False, errors
    
    # Check for at least one split
    splits = ['train', 'val', 'test']
    found_splits = []
    
    for split in splits:
        split_dir = Path(f"dataset/{split}")
        if split_dir.exists():
            found_splits.append(split)
            
            # Check for annotations.json
            ann_path = split_dir / "annotations.json"
            if not ann_path.exists():
                errors.append(f"{split}/annotations.json not found")
            
            # Check for images directory or images in split dir
            images_dir = split_dir / "images"
            if not images_dir.exists():
                # Check if images are directly in split dir
                image_files = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
                if not image_files:
                    errors.append(f"{split}/images/ directory not found and no images in {split}/")
    
    if not found_splits:
        errors.append("No dataset splits (train/val/test) found")
        return False, errors
    
    return len(errors) == 0, errors


if __name__ == "__main__":
    # Test utilities
    print("Testing dataset structure...")
    is_valid, errors = validate_dataset_structure()
    
    if is_valid:
        print("✓ Dataset structure is valid")
    else:
        print("✗ Dataset structure issues:")
        for error in errors:
            print(f"  - {error}")
