# GitHub Repository Structure Guide
## EcoInnovators Ideathon 2026 - Submission Checklist

This guide helps you organize your project for GitHub submission according to the Ideathon requirements.

---

## 📋 Required Folder Structure

Your GitHub repository must have the following structure:

```
your-repo-name/
│
├── 📁 pipeline_code/              [REQUIRED] ⭐
│   ├── backend/
│   │   ├── main.py
│   │   ├── static/
│   │   │   └── index.html
│   │   └── README.md
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── buffer_geometry.py
│   │   ├── imagery_fetcher.py
│   │   ├── qc_logic.py
│   │   ├── overlay_generator.py
│   │   └── json_writer.py
│   └── model/
│       └── model_inference.py
│
├── 📁 environment_details/         [REQUIRED] ⭐
│   ├── requirements.txt
│   ├── environment.yml
│   └── python_version.txt
│
├── 📁 trained_model_files/         [REQUIRED] ⭐
│   ├── solarpanel_seg_v1.pt       (Primary model - 22.76 MB)
│   ├── solarpanel_seg_v2.pt       (Ensemble model - 22.52 MB)
│   └── solarpanel_seg_v3.pt       (Ensemble model - 23.86 MB)
│
├── 📁 model_card/                  [REQUIRED] ⭐
│   └── MODEL_CARD.pdf             (Convert MODEL_CARD.md to PDF)
│
├── 📁 prediction_files/            [REQUIRED] ⭐
│   └── *.json                     (Your prediction JSON files)
│
├── 📁 artefacts/                   [REQUIRED] ⭐
│   └── *.png                      (Overlay visualization images)
│
├── 📁 training_logs/               [REQUIRED] ⭐
│   └── training_metrics.csv       (or training_logs.txt)
│
└── 📄 README.md                    [REQUIRED] ⭐
    (Clear instructions to run the code)
```

---

## 🗂️ How to Organize Your Files

### Step 1: Create the Repository Structure

Run these commands in your project directory (Git Bash or PowerShell):

```powershell
# Navigate to your project
cd D:\Idethon

# Initialize git repository (if not already done)
git init

# Create the required folders
mkdir pipeline_code
mkdir environment_details
mkdir trained_model_files
mkdir model_card
mkdir prediction_files
mkdir artefacts
mkdir training_logs

```

### Step 2: Copy Files to Required Locations

#### 2.1 Pipeline Code
```powershell
# Copy backend
xcopy /E /I backend pipeline_code\backend

# Copy pipeline
xcopy /E /I pipeline pipeline_code\pipeline

# Copy model inference
mkdir pipeline_code\model
copy model\model_inference.py pipeline_code\model\
```

#### 2.2 Environment Details
```powershell
copy requirements.txt environment_details\
copy environment.yml environment_details\
copy python_version.txt environment_details\
```

#### 2.3 Trained Model Files
```powershell
copy model\model_weights\solarpanel_seg_v1.pt trained_model_files\
copy model\ensemble_models\solarpanel_seg_v2.pt trained_model_files\
copy model\ensemble_models\solarpanel_seg_v3.pt trained_model_files\
```

#### 2.4 Model Card
```powershell
# Convert MODEL_CARD.md to PDF (see instructions below)
# Then copy the PDF
copy MODEL_CARD.pdf model_card\
```

#### 2.5 Prediction Files
```powershell
# Copy all JSON predictions
xcopy /E outputs\predictions\*.json prediction_files\
```

#### 2.6 Artefacts (Overlay Images)
```powershell
# Copy overlay images
xcopy /E outputs\overlays\*.png artefacts\
```

#### 2.7 Training Logs
```powershell
copy training_logs.txt training_logs\
```

---

## 📝 Converting MODEL_CARD.md to PDF

### Option 1: Using Microsoft Word
1. Open `MODEL_CARD.md` in VS Code
2. Copy all content
3. Open Microsoft Word
4. Paste content
5. Format as needed (use Calibri or Arial, 11pt)
6. Save as PDF: File → Save As → MODEL_CARD.pdf

### Option 2: Using Online Converter
1. Go to: https://www.markdowntopdf.com/
2. Upload `MODEL_CARD.md`
3. Download the generated PDF
4. Save as `MODEL_CARD.pdf`

### Option 3: Using Pandoc (if installed)
```powershell
pandoc MODEL_CARD.md -o MODEL_CARD.pdf
```

### Option 4: Print to PDF
1. Open `MODEL_CARD.md` in VS Code
2. Press `Ctrl+Shift+V` for preview
3. Right-click → Print
4. Select "Microsoft Print to PDF"
5. Save as `MODEL_CARD.pdf`

---

## 📄 Update README.md for Submission

Your README.md should have clear run instructions. Here's what to include:

### Essential Sections:
1. **Project Overview** - What the system does
2. **Installation Instructions** - How to set up
3. **Running the Code** - Step-by-step commands
4. **File Structure** - Brief explanation
5. **Dependencies** - Required packages
6. **Expected Output** - What results to expect

Your current README.md already has all of this! ✓

---

## 🚀 Creating the GitHub Repository

### Step 1: Create Repository on GitHub

1. Go to: https://github.com/new
2. Repository name: `solar-panel-detection-ecoinnovators`
3. Description: "AI-powered rooftop solar panel detection system for PM Surya Ghar - EcoInnovators Ideathon 2026"
4. Choose: **Public**
5. Do NOT initialize with README (you already have one)
6. Click "Create repository"

### Step 2: Create .gitignore File

Create a file named `.gitignore` in your project root:

```gitignore
# Python
__pycache__/
*.py[cod]
*.so
*.egg-info/

# Virtual Environment
.venv/
env/
venv/

# IDE
.vscode/
.idea/
*.swp

# Logs
logs/
*.log

# Temporary files
temp_images/
*.tmp

# OS
.DS_Store
Thumbs.db

# Large files (already in submission folders)
/model/
/models_segmentation/
/outputs/

# Keep submission folders
!/trained_model_files/
!/prediction_files/
!/artefacts/
```

### Step 3: Create .gitattributes for Large Files

GitHub has file size limits. Create `.gitattributes`:

```gitattributes
# Track large model files with Git LFS
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text

# Images
*.png binary
*.jpg binary
*.jpeg binary
```

### Step 4: Install Git LFS (for large model files)

```powershell
# Install Git LFS (if not already installed)
# Download from: https://git-lfs.github.com/

# Initialize Git LFS
git lfs install

# Track model files
git lfs track "*.pt"
git lfs track "*.pth"
```

### Step 5: Commit and Push

```powershell
# Add all files
git add .

# Commit
git commit -m "Initial commit: Solar Panel Detection System for EcoInnovators Ideathon 2026"

# Add remote (replace with your repository URL)
git remote add origin https://github.com/YOUR_USERNAME/solar-panel-detection-ecoinnovators.git

# Push to GitHub
git push -u origin main
```

If you get a branch error, try:
```powershell
git branch -M main
git push -u origin main
```

---

## ✅ Pre-Submission Checklist

Before submitting, verify all requirements:

### Required Folders (8 items):
- [ ] `pipeline_code/` - All Python inference code
- [ ] `environment_details/` - requirements.txt, environment.yml, python_version.txt
- [ ] `trained_model_files/` - All 3 .pt model files
- [ ] `model_card/` - MODEL_CARD.pdf (2-3 pages)
- [ ] `prediction_files/` - .json prediction files
- [ ] `artefacts/` - .png/.jpg overlay images
- [ ] `training_logs/` - training metrics CSV or TXT
- [ ] `README.md` - Clear run instructions

### File Verification:
- [ ] All .py files are in `pipeline_code/`
- [ ] requirements.txt lists all dependencies
- [ ] environment.yml is conda-compatible
- [ ] python_version.txt specifies Python version
- [ ] All 3 model files (69 MB total) are included
- [ ] MODEL_CARD.pdf is 2-3 pages with all sections
- [ ] Prediction JSON files follow correct schema
- [ ] Overlay PNG files are present
- [ ] Training logs show metrics (Loss, F1, mAP)
- [ ] README.md has installation and run instructions

### Git Repository:
- [ ] Repository is public
- [ ] .gitignore is configured
- [ ] .gitattributes tracks large files
- [ ] Git LFS is set up for .pt files
- [ ] All commits pushed successfully
- [ ] Repository URL is accessible

---

## 📤 Final Submission

Once your GitHub repository is ready:

1. **Copy Repository URL:**
   ```
   https://github.com/YOUR_USERNAME/solar-panel-detection-ecoinnovators
   ```

2. **Test Access:**
   - Open the URL in incognito/private browser
   - Verify all folders are visible
   - Check that files are accessible
   - Test download of at least one file

3. **Submit:**
   - Paste the GitHub repository URL in the submission form
   - Include any additional notes if required

---

## 🔧 Troubleshooting

### Issue: Model files too large for GitHub
**Solution:** Use Git LFS (see Step 4 above)

### Issue: Push rejected due to file size
**Solution:**
```powershell
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add trained_model_files/
git commit -m "Track model files with LFS"
git push
```

### Issue: Some files not showing on GitHub
**Solution:** Check .gitignore - ensure required folders aren't excluded

### Issue: Can't convert Markdown to PDF
**Solution:** Use online tool: https://www.markdowntopdf.com/

---

## 📞 Need Help?

If you encounter issues:
1. Check Git status: `git status`
2. Check remote: `git remote -v`
3. Check LFS: `git lfs ls-files`
4. Review GitHub documentation: https://docs.github.com/

---

## 🎯 Quick Command Summary

```powershell
# Full workflow
cd D:\Idethon
git init
git lfs install
git lfs track "*.pt"

# Copy files to structure (see Step 2 above)

# Commit and push
git add .
git commit -m "Initial commit: EcoInnovators Ideathon 2026 submission"
git remote add origin YOUR_REPO_URL
git push -u origin main
```

---

**Good luck with your submission!** 🌞

**Date:** December 14, 2025  
**Version:** 1.0
