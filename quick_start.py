"""
Quick Start Script for Solar Panel Detection
Runs a complete check of the environment and guides you through next steps.
"""

import sys
from pathlib import Path


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*60)
    print(text)
    print("="*60 + "\n")


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 10:
        print("✓ Python version is compatible")
        return True
    else:
        print("⚠ Python 3.10+ recommended (you have {}.{})".format(
            version.major, version.minor))
        return False


def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'torch',
        'torchvision', 
        'ultralytics',
        'cv2',
        'numpy',
        'matplotlib',
        'yaml',
        'PIL',
        'pycocotools'
    ]
    
    missing = []
    installed = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                __import__('cv2')
            elif package == 'PIL':
                __import__('PIL')
            elif package == 'yaml':
                __import__('yaml')
            else:
                __import__(package)
            installed.append(package)
        except ImportError:
            missing.append(package)
    
    if installed:
        print(f"✓ Installed packages: {', '.join(installed)}")
    
    if missing:
        print(f"✗ Missing packages: {', '.join(missing)}")
        print("\nInstall missing packages with:")
        print("  pip install -r requirements.txt")
        return False
    
    return True


def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {gpu_name}")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠ No GPU detected. Training will use CPU (slower)")
            return False
    except Exception as e:
        print(f"⚠ Could not check GPU: {e}")
        return False


def check_directories():
    """Check project directory structure."""
    required_dirs = ['models', 'results', 'dataset', 'utils']
    
    all_exist = True
    for d in required_dirs:
        if Path(d).exists():
            print(f"✓ {d}/ directory exists")
        else:
            print(f"✗ {d}/ directory missing")
            all_exist = False
    
    return all_exist


def check_dataset():
    """Check if dataset is available."""
    dataset_path = Path("dataset")
    
    if not dataset_path.exists():
        print("✗ dataset/ directory not found")
        return False
    
    # Check for splits
    splits = ['train', 'val', 'test']
    found_splits = []
    
    for split in splits:
        split_path = dataset_path / split / "annotations.json"
        if split_path.exists():
            found_splits.append(split)
            print(f"✓ {split} annotations found")
    
    if not found_splits:
        print("⚠ No dataset found. Place your COCO dataset in dataset/")
        print("\nExpected structure:")
        print("  dataset/")
        print("    train/")
        print("      annotations.json")
        print("      images/")
        print("    val/")
        print("      annotations.json")
        print("      images/")
        return False
    
    return True


def check_trained_model():
    """Check if a trained model exists."""
    model_path = Path("models/best.pt")
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ Trained model found: models/best.pt ({size_mb:.1f} MB)")
        return True
    else:
        print("⚠ No trained model found. Run training first:")
        print("  python train.py")
        return False


def show_next_steps(has_dataset, has_model):
    """Show recommended next steps."""
    print_header("RECOMMENDED NEXT STEPS")
    
    if not has_dataset:
        print("📁 STEP 1: Prepare your dataset")
        print("  1. Create dataset/train/, dataset/val/, dataset/test/ folders")
        print("  2. Place your COCO JSON files (annotations.json) in each folder")
        print("  3. Place corresponding images in images/ subfolder")
        print("  4. See dataset/README.md for format details")
        print("\n  Then run: python visualize.py")
    
    elif not has_model:
        print("🎓 STEP 1: Visualize your dataset")
        print("  python visualize.py")
        print()
        print("🚀 STEP 2: Train the model")
        print("  python train.py")
        print()
        print("  Training will:")
        print("  - Convert COCO to YOLO format")
        print("  - Validate and fix annotations")
        print("  - Train YOLOv8 model")
        print("  - Save best model to models/best.pt")
    
    else:
        print("🎯 Your system is ready! You can:")
        print()
        print("1. Run predictions on new images:")
        print("   python predict.py --source test.jpg")
        print()
        print("2. Run predictions on a folder:")
        print("   python predict.py --source dataset/test/images/")
        print()
        print("3. Run predictions on video:")
        print("   python predict.py --source video.mp4 --video")
        print()
        print("4. Visualize dataset annotations:")
        print("   python visualize.py --samples 10")
        print()
        print("5. Retrain with more epochs:")
        print("   python train.py")


def main():
    """Main function."""
    print_header("SOLAR PANEL DETECTION - SYSTEM CHECK")
    
    # Python version
    print_header("Python Version")
    check_python_version()
    
    # Dependencies
    print_header("Dependencies")
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n❌ Please install dependencies first:")
        print("   pip install -r requirements.txt")
        return
    
    # GPU
    print_header("GPU Check")
    check_gpu()
    
    # Directories
    print_header("Project Structure")
    check_directories()
    
    # Dataset
    print_header("Dataset")
    has_dataset = check_dataset()
    
    # Model
    print_header("Trained Model")
    has_model = check_trained_model()
    
    # Next steps
    show_next_steps(has_dataset, has_model)
    
    print("\n" + "="*60)
    print("For detailed instructions, see README.md")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
