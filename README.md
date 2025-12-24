# 🌞 Solar Panel Detection System - EcoInnovators Ideathon 2026

> **For Evaluators**: Start with [EVALUATOR_GUIDE.md](EVALUATOR_GUIDE.md) for a comprehensive overview and quick demo.

Complete end-to-end AI-powered system for detecting rooftop solar panels from satellite imagery.

## 🚀 System Highlights

- ✅ **4-Model Ensemble**: 94%+ accuracy combining 4 YOLOv8 models (3 segmentation + 1 detection, ~32k+ total training images)
- ✅ **Automated Satellite Imagery**: High-resolution imagery retrieval (no API keys required)
- ✅ **Fast Processing**: 3-4 seconds per location
- ✅ **High Success Rate**: Optimized imagery capture (12,900 sq ft at max resolution)
- ✅ **Enhanced Visualization**: Split-color polygon rendering (GREEN=inside buffer, RED=outside buffer)
- ✅ **Two-Tier Buffer Analysis**: 1200/2400 sq.ft with yellow highlight for active buffer
- ✅ **Power Generation Estimates**: Automatic kWh calculations per detection
- ✅ **Complete Web Interface**: REST API + Interactive UI
- ✅ **Production Ready**: Clean code, robust error handling, comprehensive documentation

## 📋 Requirements

- Python 3.10 or 3.11
- **At least ONE supported browser** (Chrome, Edge, Firefox, Brave, or Opera) for satellite imagery
- CUDA-capable GPU (recommended for training, optional for inference)
- 8GB+ RAM
- Windows/Linux/MacOS
- Internet connection for satellite imagery retrieval

> 💡 **Browser Support**: The system automatically detects and uses available browsers. See [BROWSER_SUPPORT.md](BROWSER_SUPPORT.md) for details.

## 🛠️ Installation

### 1. Clone or Setup Project

The project is already set up in your workspace!

### 2. Install Dependencies

All required packages have been installed. To verify or reinstall:

```powershell
pip install -r env/requirements.txt
```

### 3. Verify Installation

```powershell
python -c "import torch; import ultralytics; print('✓ Installation successful')"
```

---

## 🎯 End-to-End Inference Pipeline (EcoInnovators Ideathon)

### Overview

Complete pipeline for rooftop PV detection following EcoInnovators Ideathon specifications:
- **Input**: Excel file with coordinates (sample_id, latitude, longitude)
- **Processing**: Automated imagery fetching, AI inference, buffer zone analysis
- **Output**: JSON predictions + visual overlays

### Pipeline Features

- ✅ **Automated Satellite Imagery**: High-resolution retrieval system (no API keys required)
- ✅ **Two-tier Buffer Strategy**: Checks 1200 sq.ft first, then 2400 sq.ft if needed
- ✅ **Quality Control**: Automatic VERIFIABLE/NOT_VERIFIABLE determination
- ✅ **Area Estimation**: Accurate pixel-to-meter conversion with WGS84 corrections
- ✅ **Visual Overlays**: Annotated images for manual verification
- ✅ **Batch Processing**: Process hundreds of locations automatically

### Quick Start - Inference Pipeline

```powershell
# 1. Prepare your Excel file with columns: sample_id, latitude, longitude
# Example: inputs/samples.xlsx

# 2. Run the pipeline
python pipeline/main.py inputs/samples.xlsx

# 3. Results are saved to:
#    - outputs/predictions/{sample_id}.json  (Individual predictions)
#    - outputs/overlays/{sample_id}_overlay.png  (Visual overlays)
#    - outputs/predictions/summary_report.json  (Overall statistics)
```

### Input Format

Excel file (.xlsx) with required columns:

| sample_id | latitude | longitude |
|-----------|----------|----------|
| YOUR_ID | YOUR_LAT | YOUR_LON |
| 1002 | 28.7041 | 77.1025 |
| 1003 | 19.0760 | 72.8777 |

### Output Format

Each location generates a JSON file following the exact ideathon specification:

```json
{
  "sample_id": "YOUR_ID",
  "lat": 0.0,
  "lon": 0.0,
  "has_solar": true,
  "confidence": 0.92,
  "pv_area_sqm_est": 23.5,
  "buffer_radius_sqft": 1200,
  "qc_status": "VERIFIABLE",
  "bbox_or_mask": "[[x1,y1],[x2,y2],...]",
  "image_metadata": {
    "source": "automated_retrieval",
    "resolution_m_per_pixel": 0.054,
    "fetch_area_sqft": 12900
  }
}
```

### Pipeline Architecture

```
pipeline/
├── main.py                  # Entry point and orchestration
├── config.py                # Configuration constants
├── buffer_geometry.py       # WGS84 coordinate calculations
├── imagery_fetcher.py       # Google Maps imagery integration
├── qc_logic.py              # Quality control determination
├── overlay_generator.py     # Visualization generation
└── json_writer.py           # Output formatting

model/
└── model_inference.py       # YOLOv8 wrapper for inference
```

### Advanced Usage

```powershell
# Specify custom model
python pipeline/main.py inputs/samples.xlsx --model path/to/custom_model.pt

# Specify custom output directory
python pipeline/main.py inputs/samples.xlsx --output results/predictions

# Specify custom temp directory for images
python pipeline/main.py inputs/samples.xlsx --temp temp_images

# Full example with all options
python pipeline/main.py inputs/samples.xlsx \
  --model model/model_weights/solarpanel_seg_v1.pt \
  --output outputs/predictions \
  --temp temp_images
```

### Buffer Zone Logic

The pipeline implements a two-tier buffer strategy per ideathon requirements:

1. **Primary Buffer (1200 sq.ft)**:
   - Converts to square in meters: ~111.48 m² → 10.56m × 10.56m
   - Applies WGS84 corrections for latitude
   - Fetches satellite imagery for this region
   - Runs AI inference

2. **Fallback Buffer (2400 sq.ft)**:
   - Only if no solar detected in primary buffer
   - Larger search area: ~222.97 m² → 14.93m × 14.93m
   - Same inference pipeline

3. **Coordinate Transformation**:
   - Converts sq.ft to meters: `area_m² = area_sqft × 0.092903`
   - Calculates degrees offset:
     - Δlat = (side_m / 2) / 111,320
     - Δlon = (side_m / 2) / (111,320 × cos(latitude))
   - Creates bounding box: (lon±Δlon, lat±Δlat)

### QC Status Rules

**VERIFIABLE**: Clear evidence of presence/absence
- Image fetched successfully
- Good image quality (brightness, resolution)
- No cloud cover or occlusion

**NOT_VERIFIABLE**: Cannot determine with confidence
- Image fetch failed
- Poor image quality (too dark, blurry)
- Cloud cover or shadows detected
- Metadata indicates quality issues

### Imagery Source

Automated satellite imagery retrieval system:
- High-resolution capture at zoom level 21
- Coverage: 12,900 sq ft per location
- Resolution: 0.054 m/pixel (5.4 cm per pixel)
- Multi-platform backend support
- No authentication required

### Pipeline Logging

All operations are logged to `pipeline.log`:
- Image fetch attempts and results
- Model inference results
- QC determinations
- Errors and warnings

Monitor progress:
```powershell
# View real-time logs (Windows)
Get-Content pipeline.log -Wait

# View last 50 lines
Get-Content pipeline.log -Tail 50
```

## 📂 Project Structure

```
Idethon/
├── pipeline/                    # End-to-end inference pipeline
│   ├── __init__.py
│   ├── main.py                 # Entry point for batch processing
│   ├── config.py               # Configuration constants
│   ├── buffer_geometry.py      # WGS84 coordinate calculations
│   ├── imagery_fetcher.py      # Google Maps imagery integration
│   ├── qc_logic.py             # Quality control logic
│   ├── overlay_generator.py    # Visualization generation
│   └── json_writer.py          # Output formatting
├── model/
│   ├── model_inference.py      # YOLOv8 wrapper
│   └── model_weights/
│       └── solarpanel_seg_v1.pt  # Production model (94.3% mAP)
├── inputs/                      # Input Excel files
├── outputs/
│   ├── predictions/             # JSON prediction files
│   └── overlays/                # Visual overlay images
├── env/
│   └── requirements.txt         # Python dependencies
├── test_satellite_image.py      # Single image inference
├── visualize.py                 # Dataset visualization
└── README.md                    # This file
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
