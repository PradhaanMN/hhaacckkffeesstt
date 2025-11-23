"""
Test the best available model on the provided satellite image
"""
from ultralytics import YOLO
import cv2
import os

# Find best available model
model_candidates = [
    'results/final_best_model3/weights/best.pt',
    'models_segmentation/best_ultimate.pt',
    'models_segmentation/best_final.pt',
]

model_path = None
for candidate in model_candidates:
    if os.path.exists(candidate):
        model_path = candidate
        print(f"✓ Using model: {model_path}")
        break

if not model_path:
    print("❌ No trained model found!")
    exit(1)

# Load model
model = YOLO(model_path)

# Run prediction on the image
print("\nRunning inference on satellite image...")
results = model.predict(
    source='Screenshot 2025-11-21 222246.png',
    save=True,
    conf=0.25,
    iou=0.45,
    show_labels=True,
    show_conf=True,
    show_boxes=True,
    device=0  # Use GPU
)

# Print results
print("\n" + "="*60)
print("DETECTION RESULTS")
print("="*60)

if results and len(results) > 0:
    result = results[0]
    
    # Get number of detections
    num_detections = len(result.boxes)
    print(f"\n✓ Found {num_detections} solar panels")
    
    if num_detections > 0:
        print(f"\nConfidence scores:")
        for i, box in enumerate(result.boxes):
            conf = box.conf[0].item()
            print(f"  Panel {i+1}: {conf*100:.1f}%")
        
        print(f"\n✓ Results saved to: {result.save_dir}")
        print(f"✓ Annotated image saved with bounding boxes and segmentation masks")
    else:
        print("\n⚠ No solar panels detected in the image")
        print("This might be due to:")
        print("  - Image scale/resolution mismatch")
        print("  - Confidence threshold too high")
        print("  - Solar panels too small or not visible")
else:
    print("\n❌ No results returned from model")

print("\n" + "="*60)
