# 🚀 QUICK START GUIDE

## Complete Setup in 3 Steps

### Step 1: Verify Installation ✅

Run the system check:
```powershell
python quick_start.py
```

This will verify:
- Python version (3.10+ recommended)
- All dependencies installed
- GPU availability
- Project structure
- Dataset presence

### Step 2: Prepare Your Dataset 📁

#### Option A: Use Your Real Dataset

Place your COCO format dataset in the `dataset/` folder:

```
dataset/
├── train/
│   ├── annotations.json  ← Your COCO annotations
│   └── images/           ← Your training images
├── val/
│   ├── annotations.json
│   └── images/
└── test/
    ├── annotations.json
    └── images/
```

#### Option B: Create Sample Dataset for Testing

For quick testing without real data:
```powershell
python create_sample_dataset.py
```

This creates synthetic images with simple rectangles representing solar panels.

**⚠ Warning:** Sample data is for testing the pipeline only. Use real satellite imagery for actual deployment!

### Step 3: Run the Pipeline 🎯

#### 1. Visualize Your Dataset
```powershell
# View 5 random training samples
python visualize.py

# View validation samples
python visualize.py --split val --samples 10

# Save visualizations
python visualize.py --save results/viz/
```

#### 2. Train the Model
```powershell
python train.py
```

Training automatically:
- Converts COCO → YOLO format
- Validates and fixes annotations
- Trains YOLOv8 model (100 epochs)
- Saves best model to `models/best.pt`
- Runs validation test

**⏱ Expected time:**
- CPU: 2-4 hours (10 images) to 8+ hours (100+ images)
- GPU: 20-40 minutes (10 images) to 1-2 hours (100+ images)

#### 3. Run Predictions
```powershell
# Single image
python predict.py --source test.jpg

# Folder of images
python predict.py --source dataset/test/images/

# Video
python predict.py --source video.mp4 --video

# With custom confidence
python predict.py --source test.jpg --conf 0.5
```

---

## 📋 Common Commands

### Training
```powershell
# Standard training
python train.py

# Monitor training (results are saved automatically)
# Check: results/solar_panel_detection/
```

### Prediction
```powershell
# Basic prediction
python predict.py --source image.jpg

# Higher confidence threshold
python predict.py --source image.jpg --conf 0.5

# Show results in real-time
python predict.py --source image.jpg --show

# Don't save results
python predict.py --source image.jpg --no-save
```

### Visualization
```powershell
# Quick view
python visualize.py

# Different split
python visualize.py --split val

# Statistics only
python visualize.py --stats-only

# Save to folder
python visualize.py --save output_folder/
```

---

## 🔧 Customization

### Change Model Size

Edit `train.py` (line 137):
```python
model = YOLO('yolov8n.pt')  # nano - fastest
model = YOLO('yolov8s.pt')  # small - balanced ✓ recommended
model = YOLO('yolov8m.pt')  # medium - more accurate
model = YOLO('yolov8l.pt')  # large - most accurate
```

### Adjust Training Epochs

Edit `train.py` (line 140):
```python
'epochs': 100,  # Change to 50, 150, 200, etc.
```

### Change Image Size

Edit `train.py` (line 141):
```python
'imgsz': 640,  # Can use 320, 480, 640, 800, 1280
```

---

## 🐛 Troubleshooting

### "No GPU detected"
- Training will work on CPU, just slower
- For GPU: Install CUDA toolkit + PyTorch with CUDA
- Check: `python -c "import torch; print(torch.cuda.is_available())"`

### "Dataset not found"
- Ensure annotations.json is in dataset/train/, dataset/val/, dataset/test/
- Check file names match the expected structure
- Run: `python visualize.py --stats-only` to debug

### "CUDA out of memory" during training
Edit `train.py` (line 142):
```python
'batch': 8,  # Reduce to 4, 2, or 1
```

### "Model not found" during prediction
- Train a model first: `python train.py`
- Or specify model path: `python predict.py --source test.jpg --model path/to/model.pt`

### Import errors
```powershell
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

---

## 📊 Understanding Results

### Training Results Location
```
results/solar_panel_detection/
├── weights/
│   ├── best.pt          ← Use this for predictions
│   └── last.pt          ← Last checkpoint
├── results.png          ← Training curves
├── confusion_matrix.png ← Performance matrix
└── val_batch*.jpg       ← Validation predictions
```

### Training Metrics

**Good signs:**
- mAP@0.5 > 0.7 (70%+)
- Precision and Recall increasing
- Loss decreasing steadily

**Bad signs:**
- Loss not decreasing → Need more data or different model
- Validation loss increasing → Overfitting, need more data
- Very low mAP → Check annotations quality

### Prediction Results

Results saved to: `results/predictions/`
- Original images with bounding boxes
- Confidence scores on each detection
- Class labels

---

## 📈 Performance Tips

1. **Dataset Quality** (Most Important!)
   - Minimum 100-200 images per class
   - Diverse examples (different angles, lighting, scales)
   - Accurate annotations
   - Balanced train/val/test split (70/20/10)

2. **Training**
   - Start with yolov8n for quick experiments
   - Use yolov8s or yolov8m for production
   - More epochs ≠ better (watch for overfitting)
   - Use GPU if available

3. **Inference**
   - Lower --conf (0.1-0.3) for higher recall (find more panels)
   - Higher --conf (0.5-0.7) for higher precision (fewer false positives)
   - Adjust based on your use case

---

## 🎯 Next Steps After Training

1. **Evaluate Results**
   ```powershell
   # Check training curves
   start results/solar_panel_detection/results.png
   
   # View confusion matrix
   start results/solar_panel_detection/confusion_matrix.png
   ```

2. **Test on New Images**
   ```powershell
   python predict.py --source path/to/new/images/
   ```

3. **Improve Performance**
   - Add more training data
   - Use larger model (yolov8s → yolov8m)
   - Increase training epochs
   - Improve annotation quality

4. **Deploy**
   - Use `models/best.pt` for production
   - Integrate with your application
   - Set appropriate confidence threshold

---

## 📚 Additional Resources

- **YOLOv8 Docs:** https://docs.ultralytics.com/
- **COCO Format:** https://cocodataset.org/#format-data
- **Project README:** README.md
- **Dataset Guide:** dataset/README.md

---

## ✅ Checklist

- [ ] Run `python quick_start.py` - all checks pass
- [ ] Dataset in `dataset/` folder with correct structure
- [ ] Run `python visualize.py` - annotations look correct
- [ ] Run `python train.py` - training completes
- [ ] Check `models/best.pt` exists
- [ ] Run `python predict.py --source test.jpg` - predictions work
- [ ] Review results in `results/` folder
- [ ] Adjust confidence threshold for your use case

---

**Need Help?** Re-run `python quick_start.py` to diagnose issues.
