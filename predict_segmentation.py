"""
YOLOv8 Segmentation Model - Inference Script
Runs predictions with pixel-level solar panel segmentation
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2

def predict_segmentation(source, model_path="models_segmentation/best_seg.pt", conf=0.25, save=True):
    """
    Run segmentation prediction on images/videos
    
    Args:
        source: Path to image, directory, or video
        model_path: Path to trained segmentation model
        conf: Confidence threshold
        save: Whether to save results
    """
    print("\n" + "="*60)
    print("YOLOV8 SOLAR PANEL SEGMENTATION - INFERENCE")
    print("="*60)
    
    # Check model exists
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"\n❌ Model not found: {model_path}")
        print("   Please train the model first: python train_segmentation.py")
        return
    
    print(f"\nModel: {model_path}")
    print(f"Source: {source}")
    print(f"Confidence threshold: {conf}")
    
    # Load model
    print("\n→ Loading segmentation model...")
    model = YOLO(str(model_path))
    
    # Run inference
    print("→ Running segmentation inference...")
    results = model.predict(
        source=source,
        conf=conf,
        save=save,
        project="results/predictions",
        name="predict_seg",
        exist_ok=True,
        show_labels=True,
        show_conf=True,
        line_width=2,
        verbose=True
    )
    
    # Print results summary
    print("\n" + "="*60)
    print("SEGMENTATION INFERENCE COMPLETE")
    print("="*60)
    
    if save:
        print(f"\n✓ Results saved to: results/predictions/predict_seg")
    
    # Count detections
    total_detections = 0
    for r in results:
        if r.masks is not None:
            total_detections += len(r.masks)
    
    print(f"\n📊 Detection Summary:")
    print(f"  Images processed: {len(results)}")
    print(f"  Total solar panels detected: {total_detections}")
    if total_detections > 0:
        print(f"  Average per image: {total_detections/len(results):.2f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Solar Panel Segmentation Inference")
    parser.add_argument("--source", type=str, required=True, 
                       help="Path to image, directory, or video")
    parser.add_argument("--model", type=str, default="models_segmentation/best_seg.pt",
                       help="Path to trained segmentation model")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold (0-1)")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save prediction images")
    
    args = parser.parse_args()
    
    predict_segmentation(
        source=args.source,
        model_path=args.model,
        conf=args.conf,
        save=not args.no_save
    )

if __name__ == "__main__":
    main()
