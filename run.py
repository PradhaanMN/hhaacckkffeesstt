"""
All-in-One Runner Script
Quick access to all major functions of the solar panel detection system.
"""

import sys
import subprocess
from pathlib import Path


def print_banner():
    """Print welcome banner."""
    print("\n" + "="*60)
    print("   SOLAR PANEL DETECTION - YOLOv8")
    print("="*60 + "\n")


def print_menu():
    """Print main menu."""
    print("Select an option:\n")
    print("  1. System Check          - Verify installation")
    print("  2. Create Sample Dataset - Generate test data")
    print("  3. Visualize Dataset     - View annotations")
    print("  4. Train Model           - Train YOLOv8")
    print("  5. Run Prediction        - Predict on images")
    print("  6. Help & Docs           - View documentation")
    print("  7. Exit")
    print()


def run_command(script_name, args=""):
    """Run a Python script."""
    # Use the virtual environment Python
    venv_python = Path(__file__).parent / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        python_exe = str(venv_python)
    else:
        python_exe = sys.executable
    
    cmd = f'"{python_exe}" {script_name} {args}'
    print(f"\n→ Running: {script_name} {args}\n")
    result = subprocess.run(cmd, shell=True)
    return result.returncode


def system_check():
    """Run system check."""
    return run_command("quick_start.py")


def create_sample():
    """Create sample dataset."""
    print("\nThis will create a synthetic dataset for testing.")
    print("⚠ Warning: This will REPLACE any existing data in dataset/\n")
    confirm = input("Continue? (y/n): ").lower()
    
    if confirm == 'y':
        return run_command("create_sample_dataset.py")
    else:
        print("Cancelled.")
        return 0


def visualize_dataset():
    """Visualize dataset."""
    print("\nVisualization Options:")
    print("  1. Train set (5 samples)")
    print("  2. Validation set (5 samples)")
    print("  3. Test set (5 samples)")
    print("  4. Custom")
    print()
    
    choice = input("Select option (1-4): ").strip()
    
    if choice == "1":
        return run_command("visualize.py")
    elif choice == "2":
        return run_command("visualize.py", "--split val")
    elif choice == "3":
        return run_command("visualize.py", "--split test")
    elif choice == "4":
        split = input("Enter split (train/val/test): ").strip()
        samples = input("Number of samples (default: 5): ").strip() or "5"
        return run_command("visualize.py", f"--split {split} --samples {samples}")
    else:
        print("Invalid choice.")
        return 0


def train_model():
    """Train YOLOv8 model."""
    print("\nStarting training...")
    print("This will:")
    print("  - Convert COCO to YOLO format")
    print("  - Validate annotations")
    print("  - Train YOLOv8 for 100 epochs")
    print("  - Save best model to models/best.pt")
    print("\n⏱ This may take 20 minutes to several hours depending on:")
    print("   - Dataset size")
    print("   - GPU availability")
    print("   - Model size\n")
    
    confirm = input("Start training? (y/n): ").lower()
    
    if confirm == 'y':
        return run_command("train.py")
    else:
        print("Training cancelled.")
        return 0


def run_prediction():
    """Run predictions."""
    print("\nPrediction Options:")
    print("  1. Single image")
    print("  2. Folder of images")
    print("  3. Video")
    print("  4. Custom command")
    print()
    
    choice = input("Select option (1-4): ").strip()
    
    if choice == "1":
        source = input("Enter image path: ").strip()
        conf = input("Confidence threshold (default: 0.25): ").strip() or "0.25"
        return run_command("predict.py", f"--source {source} --conf {conf}")
    
    elif choice == "2":
        source = input("Enter folder path: ").strip()
        conf = input("Confidence threshold (default: 0.25): ").strip() or "0.25"
        return run_command("predict.py", f"--source {source} --conf {conf}")
    
    elif choice == "3":
        source = input("Enter video path: ").strip()
        conf = input("Confidence threshold (default: 0.25): ").strip() or "0.25"
        return run_command("predict.py", f"--source {source} --conf {conf} --video")
    
    elif choice == "4":
        args = input("Enter custom arguments: ").strip()
        return run_command("predict.py", args)
    
    else:
        print("Invalid choice.")
        return 0


def show_help():
    """Show help and documentation."""
    print("\n" + "="*60)
    print("DOCUMENTATION")
    print("="*60 + "\n")
    
    print("Available documentation files:\n")
    print("  1. README.md         - Complete project guide")
    print("  2. QUICKSTART.md     - Quick reference")
    print("  3. SETUP_COMPLETE.md - Setup summary")
    print("  4. dataset/README.md - Dataset format guide")
    print()
    
    docs = {
        "1": "README.md",
        "2": "QUICKSTART.md",
        "3": "SETUP_COMPLETE.md",
        "4": "dataset/README.md"
    }
    
    choice = input("View which file? (1-4, or Enter to skip): ").strip()
    
    if choice in docs:
        file_path = Path(docs[choice])
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print("\n" + "="*60)
            print(f" {file_path}")
            print("="*60 + "\n")
            print(content)
            print("\n" + "="*60 + "\n")
            input("Press Enter to continue...")
        else:
            print(f"File not found: {file_path}")
    
    print("\n📚 Quick Command Reference:\n")
    print("  System check:     python quick_start.py")
    print("  Create sample:    python create_sample_dataset.py")
    print("  Visualize:        python visualize.py")
    print("  Train:            python train.py")
    print("  Predict:          python predict.py --source <image>")
    print()


def main():
    """Main menu loop."""
    print_banner()
    
    while True:
        print_menu()
        choice = input("Enter your choice (1-7): ").strip()
        
        if choice == "1":
            system_check()
        elif choice == "2":
            create_sample()
        elif choice == "3":
            visualize_dataset()
        elif choice == "4":
            train_model()
        elif choice == "5":
            run_prediction()
        elif choice == "6":
            show_help()
        elif choice == "7":
            print("\nExiting. Good luck with your solar panel detection! 🌞")
            break
        else:
            print("\n❌ Invalid choice. Please enter 1-7.\n")
        
        if choice in ["1", "2", "3", "4", "5"]:
            input("\n\nPress Enter to return to main menu...")
            print("\n" * 2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
