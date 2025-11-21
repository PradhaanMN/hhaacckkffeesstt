# Solar Panel Detection with YOLOv8

Complete end-to-end training and inference system for detecting solar panels in satellite/aerial imagery using YOLOv8.

## 🚀 Features

- **Automatic COCO to YOLO conversion** with validation and auto-fix
- **YOLOv8 training** with optimized hyperparameters
- **Inference on images and videos** with customizable confidence thresholds
- **Visualization tools** for dataset exploration
- **Comprehensive logging** and result tracking
- **Production-ready scripts** with error handling

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
├── train.py              # Main training script
├── predict.py            # Inference script
├── visualize.py          # Dataset visualization tool
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── utils/
│   └── coco_to_yolo.py  # COCO to YOLO converter
├── dataset/             # Your dataset goes here
│   ├── train/
│   │   ├── annotations.json
│   │   └── images/
│   ├── val/
│   │   ├── annotations.json
│   │   └── images/
│   └── test/
│       ├── annotations.json
│       └── images/
├── models/              # Trained models saved here
│   ├── best.pt
│   └── last.pt
└── results/             # Training results and predictions
    ├── training_log.txt
    └── samples/
```

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

### 1. Visualize Your Dataset

Before training, verify your annotations:

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

### 2. Train the Model

```powershell
# Train with default settings (YOLOv8n, 100 epochs)
python train.py
```

**Training automatically:**
- Converts COCO to YOLO format
- Validates and fixes annotation issues
- Creates dataset configuration
- Trains YOLOv8 model
- Saves best and last checkpoints to `models/`
- Logs training metrics to `results/training_log.txt`
- Runs validation test on 5 samples

**Training Configuration:**
- Model: YOLOv8n (nano - fastest) or YOLOv8s (small - more accurate)
- Image size: 640x640
- Epochs: 100
- Batch size: Auto-detected
- Optimizer: SGD
- Patience: 50 epochs early stopping

### 3. Run Predictions

After training, use the model for inference:

```powershell
# Predict on single image
python predict.py --source test.jpg

# Predict on folder of images
python predict.py --source dataset/test/images/

# Predict on video
python predict.py --source video.mp4 --video

# Use custom model and confidence threshold
python predict.py --source test.jpg --model models/best.pt --conf 0.5

# Display results in real-time
python predict.py --source test.jpg --show

# Save to custom directory
python predict.py --source test.jpg --output my_results/
```

**Prediction Options:**
- `--source`: Image/video file, folder, or camera index (0 for webcam)
- `--model`: Path to model (default: models/best.pt)
- `--conf`: Confidence threshold (default: 0.25)
- `--save` / `--no-save`: Save results (default: save)
- `--show`: Display results in real-time
- `--video`: Treat source as video
- `--output`: Output directory

## 📈 Training Output

After training, you'll find:

### Models (models/)
- `best.pt` - Best model based on validation metrics
- `last.pt` - Final model checkpoint

### Results (results/)
- `training_log.txt` - Training configuration and summary
- `solar_panel_detection/` - Full training results
  - `weights/` - Model checkpoints
  - `confusion_matrix.png` - Confusion matrix
  - `results.png` - Training curves
  - `val_batch*.jpg` - Validation predictions
- `samples/` - Validation test predictions

## 🔧 Customization

### Change Model Size

Edit `train.py` line 137:

```python
model = YOLO('yolov8n.pt')  # nano (fastest)
model = YOLO('yolov8s.pt')  # small (balanced)
model = YOLO('yolov8m.pt')  # medium (more accurate)
model = YOLO('yolov8l.pt')  # large (most accurate)
```

### Adjust Training Parameters

Edit `train.py` lines 140-152:

```python
training_config = {
    'epochs': 100,        # Number of training epochs
    'imgsz': 640,        # Image size
    'batch': -1,         # -1 for auto, or set manually
    'optimizer': 'SGD',  # SGD, Adam, AdamW
    'lr0': 0.01,        # Initial learning rate
    'patience': 50,      # Early stopping patience
}
```

## 🐛 Troubleshooting

### Dataset Not Found
```
✗ COCO JSON file not found: dataset/train/annotations.json
```
**Solution:** Ensure your dataset is in `dataset/` with the correct structure.

### No Images Converted
```
⚠ Image not found: dataset/train/images/image001.jpg
```
**Solution:** Check that image paths in `annotations.json` match actual filenames.

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch size in `train.py`:
```python
'batch': 8,  # or 4, 2, 1
```

### Model Not Found During Prediction
```
✗ Model not found: models/best.pt
```
**Solution:** Train the model first with `python train.py`

## 📝 Example Workflow

Complete workflow from dataset to predictions:

```powershell
# 1. Place your COCO dataset in dataset/
# (See Dataset Format section above)

# 2. Visualize to verify annotations
python visualize.py --samples 5

# 3. Train the model
python train.py

# 4. Check training results
cat results/training_log.txt

# 5. Run predictions
python predict.py --source dataset/test/images/ --conf 0.3

# 6. Check prediction results
ls results/predictions/
```

## 🎓 Tips for Best Results

1. **Dataset Quality**
   - Minimum 100-200 annotated images per class
   - Balance between train/val/test (70/20/10 split)
   - Diverse lighting, angles, and scales

2. **Training**
   - Start with YOLOv8n for quick experiments
   - Use YOLOv8s or YOLOv8m for production
   - Monitor training curves in `results/solar_panel_detection/results.png`
   - If overfitting, add more data or augmentation

3. **Inference**
   - Adjust `--conf` threshold based on precision/recall needs
   - Lower threshold (0.1-0.2) for higher recall
   - Higher threshold (0.5-0.7) for higher precision

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


**Ready to start?** Place your COCO dataset in `dataset/` and run:
```powershell
python train.py
```
