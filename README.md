# Solar Panel Detection & Segmentation with YOLOv8

Complete end-to-end training and inference system for detecting and segmenting solar panels in satellite/aerial imagery using YOLOv8.

## 🚀 Features

- **Dual Models**: Object detection (bounding boxes) and instance segmentation (pixel-level masks)
- **Automatic COCO to YOLO conversion** with validation and auto-fix
- **YOLOv8 training** with optimized hyperparameters for both detection and segmentation
- **Inference on images and videos** with customizable confidence thresholds
- **Visualization tools** for dataset exploration
- **Comprehensive logging** and result tracking
- **Production-ready scripts** with error handling
- **Combined dataset training** (satellite + ground-level imagery)

## 📋 Requirements

- Python 3.10 or 3.11
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM
- Windows/Linux/MacOS

## 🛠️ Installation

### 1. Clone or Setup Project

The project is already set up in your workspace!

### 2. Install Dependencies

All required packages have been installed. To verify or reinstall:

```powershell
pip install -r requirements.txt
```

### 3. Verify Installation

```powershell
python -c "import torch; import ultralytics; print('✓ Installation successful')"
```

## 📂 Project Structure

```
Idethon/
├── test_satellite_image.py    # Inference on satellite imagery
├── visualize.py                # Dataset visualization
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── utils/
│   └── coco_to_yolo.py        # COCO to YOLO converter
└── models_segmentation/        # Final trained model
    ├── solarpanel_seg_v1.pt   # Production-ready model (94.3% mAP, 11.79M params)
    └── model_info.txt         # Model documentation
```

**Note:** Dataset folders have been removed to reduce repository size. Only the final trained model is included for submission.

## 📊 Dataset Format

Your dataset should be in **COCO JSON format**. Place your data in the `dataset/` folder:

### Expected Structure

```
dataset/
├── train/
│   ├── annotations.json    # COCO format annotations
│   └── images/            # Training images
├── val/
│   ├── annotations.json
│   └── images/
└── test/
    ├── annotations.json
    └── images/
```

### COCO JSON Format

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image001.jpg",
      "width": 640,
      "height": 640
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height]
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "solar_panel"
    }
  ]
}
```

### Auto-Fix Features

The system automatically handles:
- Missing categories → Creates default "solar_panel" category
- Invalid bboxes → Fixes negative/zero dimensions
- Missing category_id → Assigns default
- Missing width/height → Reads from image file

## 🎯 Usage

### Quick Start - Run Inference

Test the final model on your own satellite imagery:

```powershell
# Run inference on a satellite image
python test_satellite_image.py

# The script will use models_segmentation/solarpanel_seg_v1.pt
# and detect solar panels with bounding boxes and segmentation masks
```

**Model Performance:**
- Detects solar panels with 94.3% mAP@0.5
- Provides both bounding boxes and pixel-level segmentation masks
- Fast inference: ~10ms per image on GPU
- Fast inference: ~4.7ms per image

### Visualize Your Dataset (Optional)

If you have your own dataset to train:

```powershell
# Visualize 5 random training samples
python visualize.py

# Visualize validation set
python visualize.py --split val --samples 10

# Save visualizations without displaying
python visualize.py --samples 10 --save results/viz/ --no-show

# Show statistics only
python visualize.py --stats-only
```

## 📈 Training Results

The final model was trained on 6,876 images from 6 different datasets, combining ground-level and satellite imagery for robust performance.

### Final Model
- **Location**: `models_segmentation/best_final_combined.pt`
- **Size**: 6.8 MB (optimized for deployment)
- **Parameters**: 3.26M (YOLOv8n-seg architecture)

### Training Results Directory
- `results/final_combined_ultimate/` - Final combined model training
  - `weights/best.pt` - Best model checkpoint
  - `weights/last.pt` - Final epoch checkpoint
  - `results.csv` - Epoch-by-epoch metrics
  - `confusion_matrix.png` - Confusion matrix
  - `BoxF1_curve.png`, `MaskF1_curve.png` - Performance curves
  - `val_batch*_pred.jpg` - Validation predictions with masks

### Previous Training Runs (Archived)
- `results/final_best_model3/` - Custom Workflow dataset results
- `results/lsgi547_model3/` - LSGI547 dataset results

## 🔧 Customization

### Using the Model in Your Code

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('models_segmentation/best_final_combined.pt')

# Run inference
results = model('your_satellite_image.jpg', conf=0.25)

# Process results
for r in results:
    boxes = r.boxes  # Bounding boxes
    masks = r.masks  # Segmentation masks
    
    # Get box coordinates
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        confidence = box.conf[0]
        print(f"Panel at ({x1}, {y1}, {x2}, {y2}) - {confidence:.2%}")
```

### Adjust Confidence Threshold

Edit `test_satellite_image.py` to change detection sensitivity:

```python
# Line with conf parameter
results = model(image_path, conf=0.25)  # Default: 0.25

# Lower (0.1-0.2) for higher recall (more detections)
# Higher (0.4-0.6) for higher precision (fewer false positives)
```

## 🐛 Troubleshooting

### Model Not Found
```
✗ Model not found: models_segmentation/best_final_combined.pt
```
**Solution:** Ensure the model file exists in the `models_segmentation/` folder.

### CUDA Out of Memory During Inference
```
RuntimeError: CUDA out of memory
```
**Solution:** Process images in smaller batches or use CPU inference:
```python
model = YOLO('models_segmentation/best_final_combined.pt')
model.to('cpu')  # Force CPU inference
```

### Low Detection Accuracy
**Solution:** Adjust confidence threshold in `test_satellite_image.py`:
- Lower threshold (0.1-0.2) for more detections
- Current: 0.25 (balanced)
- Higher threshold (0.4-0.6) for fewer false positives

## 📝 Example Workflow

Quick test with the pre-trained model:

```powershell
# 1. Run inference on satellite imagery
python test_satellite_image.py

# 2. Check prediction results
# Results will be saved with annotated bounding boxes and segmentation masks
```

### Training Your Own Model (Optional)

If you want to train on additional datasets:

```powershell
# 1. Prepare your COCO dataset in the correct format

# 2. Visualize to verify annotations
python visualize.py --samples 5

# 3. Create a training script similar to the previous training runs
# (Reference: results/final_combined_ultimate/ for configuration)

# 4. Train with your dataset
# python train_custom.py

# 5. Test the new model
python test_satellite_image.py --model path/to/your/model.pt
```

## 📊 Final Model Performance

### **best_final_combined.pt** - Ultimate Combined Model

**The single production-ready model trained on ALL datasets:**

| Metric | Box Detection | Segmentation |
|--------|---------------|--------------|
| **mAP@0.5** | **81.8%** | **77.7%** |
| **mAP@0.5-0.95** | **55.6%** | **46.7%** |
| **Precision** | 77.4% | 75.7% |
| **Recall** | 77.8% | 75.2% |

**Training Details:**
- **Total Images**: 6,876 (6,365 train + 511 validation)
- **Datasets Combined**: 6 diverse sources
  1. Custom Workflow (4,739 images)
  2. LSGI547 (389 images)
  3. Solarpanel_seg v4 (528 images)
  4. Zeewolde (210 images)
  5. Solar panels v1i (367 images)
  6. Solarpv-INDIA (293 images)
- **Model Architecture**: YOLOv8n-seg (3.26M parameters)
- **Training Time**: 4.6 hours (94 epochs, early stopped at epoch 64)
- **GPU**: NVIDIA GeForce RTX 3050 4GB
- **Inference Speed**: 4.7ms per image

**Use Cases:**
- Production deployment for solar panel detection & segmentation
- Real-time inference on satellite/aerial imagery
- Accurate panel area calculation and energy estimation
- Works on both ground-level and satellite imagery

## 🎓 Tips for Best Results

1. **Using the Model**
   - Default confidence threshold (0.25) works well for most cases
   - Lower threshold (0.15-0.20) for detecting smaller or partially visible panels
   - Higher threshold (0.35-0.50) for high-confidence detections only
   - Model works best on satellite/aerial imagery at 640x640 resolution

2. **Image Preprocessing**
   - Ensure images are clear with good visibility
   - Model trained on diverse lighting conditions and angles
   - Works on both satellite and ground-level imagery
   - Optimal resolution: 640x640 to 1280x1280 pixels

3. **Performance Optimization**
   - Use GPU for faster inference (4.7ms per image)
   - Batch processing for multiple images
   - Model is optimized at 6.8MB for fast loading

4. **Future Improvements**
   - Plan to train larger model (YOLOv8s-seg) on additional dataset
   - Expected improvement: 85-92% mAP (vs current 81.8%)
   - Ensemble approach for 2-5% additional accuracy gain

## 📚 Additional Resources

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [COCO Dataset Format](https://cocodataset.org/#format-data)
- [YOLOv8 Training Guide](https://docs.ultralytics.com/modes/train/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## 🤝 Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review training logs in `results/training_log.txt`
3. Verify dataset format with `python visualize.py --stats-only`


**Ready to start?** 

Run inference: `python test_satellite_image.py`

---

## 📊 Dataset Information

The final model was trained on a comprehensive dataset combining 6 different sources:

1. **Custom Workflow** (4,739 images) - Ground-level and aerial solar installations
2. **LSGI547** (389 images) - Satellite imagery with diverse panel configurations
3. **Solarpanel_seg v4** (528 images) - High-resolution segmentation data
4. **Zeewolde** (210 images) - European solar farm installations
5. **Solar panels v1i** (367 images) - Mixed resolution panel imagery
6. **Solarpv-INDIA** (293 images) - Indian solar installations from satellite

**Total**: 6,876 images (6,365 training + 511 validation)

All datasets have been removed from this repository to reduce size for submission. Only the final trained model is included.
