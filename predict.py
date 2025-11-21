"""
YOLOv8 Solar Panel Detection - Inference Script
Runs predictions on images or videos using trained model.
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO
import cv2


def check_model_exists(model_path: str) -> bool:
    """Check if the model file exists."""
    if not Path(model_path).exists():
        print(f"\n✗ Model not found: {model_path}")
        print("\nPlease train a model first by running:")
        print("  python train.py")
        print("\nOr specify a custom model path with --model")
        return False
    return True


def predict_image(model_path: str, source: str, conf_threshold: float = 0.25, 
                 save: bool = True, show: bool = False, output_dir: str = "results/predictions"):
    """
    Run inference on image(s).
    
    Args:
        model_path: Path to trained model
        source: Path to image file, folder, or pattern
        conf_threshold: Confidence threshold for detections
        save: Whether to save results
        show: Whether to display results
        output_dir: Directory to save predictions
    """
    if not check_model_exists(model_path):
        return
    
    print("\n" + "="*60)
    print("YOLOV8 SOLAR PANEL DETECTION - INFERENCE")
    print("="*60)
    print(f"\nModel: {model_path}")
    print(f"Source: {source}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"Output directory: {output_dir}")
    
    # Load model
    print("\n→ Loading model...")
    model = YOLO(model_path)
    
    # Run predictions
    print("→ Running inference...\n")
    
    results = model.predict(
        source=source,
        conf=conf_threshold,
        save=save,
        project=output_dir,
        name="",
        exist_ok=True,
        show=show,
        verbose=True,
        line_width=2,
        boxes=True
    )
    
    # Print results summary
    print("\n" + "="*60)
    print("INFERENCE COMPLETE")
    print("="*60)
    
    if save:
        print(f"\n✓ Results saved to: {output_dir}")
    
    # Print detection statistics
    total_detections = 0
    for result in results:
        if result.boxes is not None:
            total_detections += len(result.boxes)
    
    print(f"\n📊 Detection Summary:")
    print(f"  Images processed: {len(results)}")
    print(f"  Total detections: {total_detections}")
    
    if total_detections > 0:
        print(f"  Average per image: {total_detections / len(results):.2f}")
    
    return results


def predict_video(model_path: str, source: str, conf_threshold: float = 0.25,
                 save: bool = True, show: bool = False, output_dir: str = "results/predictions"):
    """
    Run inference on video.
    
    Args:
        model_path: Path to trained model
        source: Path to video file or camera index (0 for webcam)
        conf_threshold: Confidence threshold for detections
        save: Whether to save results
        show: Whether to display results
        output_dir: Directory to save predictions
    """
    if not check_model_exists(model_path):
        return
    
    print("\n" + "="*60)
    print("YOLOV8 SOLAR PANEL DETECTION - VIDEO INFERENCE")
    print("="*60)
    print(f"\nModel: {model_path}")
    print(f"Source: {source}")
    print(f"Confidence threshold: {conf_threshold}")
    
    # Load model
    print("\n→ Loading model...")
    model = YOLO(model_path)
    
    # Run predictions on video
    print("→ Processing video...\n")
    
    results = model.predict(
        source=source,
        conf=conf_threshold,
        save=save,
        project=output_dir,
        name="",
        exist_ok=True,
        show=show,
        verbose=True,
        stream=True  # Use streaming for videos
    )
    
    # Process results
    frame_count = 0
    total_detections = 0
    
    for result in results:
        frame_count += 1
        if result.boxes is not None:
            total_detections += len(result.boxes)
        
        if frame_count % 30 == 0:  # Print every 30 frames
            print(f"  Processed {frame_count} frames, {total_detections} detections")
    
    print("\n" + "="*60)
    print("VIDEO INFERENCE COMPLETE")
    print("="*60)
    
    if save:
        print(f"\n✓ Results saved to: {output_dir}")
    
    print(f"\n📊 Detection Summary:")
    print(f"  Frames processed: {frame_count}")
    print(f"  Total detections: {total_detections}")
    
    if frame_count > 0:
        print(f"  Average per frame: {total_detections / frame_count:.2f}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="YOLOv8 Solar Panel Detection - Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on single image
  python predict.py --source test.jpg
  
  # Predict on multiple images in a folder
  python predict.py --source dataset/test/images/
  
  # Predict on video
  python predict.py --source video.mp4 --video
  
  # Use custom model and confidence threshold
  python predict.py --source test.jpg --model models/best.pt --conf 0.5
  
  # Display results while processing
  python predict.py --source test.jpg --show
        """
    )
    
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Path to image/video file, folder, or camera index (0 for webcam)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/best.pt',
        help='Path to trained model (default: models/best.pt)'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold for detections (default: 0.25)'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        default=True,
        help='Save prediction results (default: True)'
    )
    
    parser.add_argument(
        '--no-save',
        dest='save',
        action='store_false',
        help='Do not save prediction results'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='Display prediction results in real-time'
    )
    
    parser.add_argument(
        '--video',
        action='store_true',
        default=False,
        help='Treat source as video file or camera'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/predictions',
        help='Output directory for predictions (default: results/predictions)'
    )
    
    args = parser.parse_args()
    
    # Run inference
    try:
        if args.video:
            predict_video(
                model_path=args.model,
                source=args.source,
                conf_threshold=args.conf,
                save=args.save,
                show=args.show,
                output_dir=args.output
            )
        else:
            predict_image(
                model_path=args.model,
                source=args.source,
                conf_threshold=args.conf,
                save=args.save,
                show=args.show,
                output_dir=args.output
            )
    
    except KeyboardInterrupt:
        print("\n\n⚠ Inference interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error during inference: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
