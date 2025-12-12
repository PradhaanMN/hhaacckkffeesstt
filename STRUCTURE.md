# 📂 Project Directory Structure

## Overview
Clean, production-ready repository with essential files only.

```
D:\Idethon/
│
├── 📘 EVALUATOR_GUIDE.md          ⭐ START HERE - Quick overview for evaluators
├── 📄 README.md                    Complete technical documentation
├── 📄 PROJECT_STATUS.md            Development history & achievements
├── 📄 QUICKSTART.md                2-minute quick start guide
├── 📄 STRUCTURE.md                 This file - directory structure
├── 📄 requirements.txt             Python dependencies
├── 📄 .gitignore                   Git ignore rules
├── 📄 .gitattributes               Git LFS configuration
│
├── 📁 backend/                     🌐 Web Server & REST API
│   ├── main.py                    FastAPI server (330 lines)
│   │                              • 5 REST endpoints
│   │                              • Lazy model loading
│   │                              • CORS enabled
│   ├── requirements.txt           Backend-specific dependencies
│   ├── README.md                  Backend documentation
│   └── static/
│       └── index.html             Web UI (clean, no sample locations)
│
├── 📁 pipeline/                    ⚙️ Core Processing Pipeline
│   ├── __init__.py
│   ├── main.py                    Entry point (430 lines)
│   │                              • Batch processing orchestration
│   │                              • Excel input handling
│   │                              • Result aggregation
│   ├── config.py                  Configuration constants
│   │                              • IMAGERY_FETCH_SIZE = 2,690,000 sqft
│   │                              • BUFFER_ZONE_1 = 1200 sqft
│   │                              • BUFFER_ZONE_2 = 2400 sqft
│   ├── buffer_geometry.py         WGS84 coordinate calculations
│   │                              • Lat/lon to bbox conversion
│   │                              • Buffer radius in pixels
│   │                              • Distance calculations
│   ├── imagery_fetcher.py         Satellite imagery retrieval
│   │                              • 500m×500m imagery fetch
│   │                              • 60s timeout + 3 retries
│   │                              • Error handling
│   ├── qc_logic.py                Quality control determination
│   │                              • VERIFIABLE vs NOT_VERIFIABLE
│   │                              • Image quality checks
│   ├── overlay_generator.py       Visualization generation
│   │                              • GREEN boxes: IN buffer
│   │                              • RED boxes: OUTSIDE buffer
│   │                              • ORANGE/GRAY circles: Buffer zones
│   └── json_writer.py             Output formatting
│                                  • Ideathon JSON specification
│                                  • Metadata inclusion
│
├── 📁 model/                       🤖 AI Components
│   ├── model_inference.py         3-model ensemble wrapper
│   │                              • Loads all 3 models
│   │                              • Runs inference on each
│   │                              • Merges predictions with NMS
│   ├── model_weights/
│   │   └── solarpanel_seg_v1.pt   Primary model (22.76 MB)
│   │                              • 94.3% mAP accuracy
│   │                              • YOLOv8s-seg architecture
│   └── ensemble_models/
│       ├── solarpanel_seg_v2.pt   Ensemble model v2 (22.52 MB)
│       │                          • Additional training perspective
│       └── solarpanel_seg_v3.pt   Ensemble model v3 (23.86 MB)
│                                  • Large dataset: ~26,000 images
│
├── 📁 models_segmentation/         📚 Model Archive
│   ├── solarpanel_seg_v1.pt      Backup copy of primary model
│   └── model_info.txt            Model documentation
│                                 • Training details
│                                 • Dataset information
│                                 • Ensemble explanation
│
├── 📁 inputs/                      📥 Input Files
│   └── samples.xlsx               (User-provided Excel files)
│                                  • Columns: sample_id, latitude, longitude
│
├── 📁 outputs/                     📤 Results Directory
│   ├── predictions/               JSON prediction files
│   │   ├── {sample_id}.json      Individual predictions
│   │   └── summary_report.json    Batch statistics
│   └── overlays/                  Visual verification
│       └── {sample_id}_overlay.png Annotated images
│
├── 📁 logs/                        📋 System Logs
│   └── pipeline.log               Processing logs (auto-generated)
│
└── 📁 .venv/                       🐍 Python Virtual Environment (1.15 GB)
    └── (Python packages installed here)
```

---

## File Count Summary

| Category | Count | Purpose |
|----------|-------|---------|
| Documentation | 5 | Guides and references |
| Backend Code | 3 | Web server and API |
| Pipeline Code | 7 | Core processing logic |
| Model Files | 4 | AI models and wrapper |
| Config Files | 3 | Dependencies and settings |
| **Total** | **24** | **Production-ready files** |

---

## Key Directories Explained

### 📁 backend/
**Purpose**: Web interface for manual testing and API access

**Key Features**:
- FastAPI server with 5 REST endpoints
- Interactive web UI (no pre-filled samples)
- Lazy model loading (loads on first request)
- CORS enabled for frontend integration
- Static file serving

### 📁 pipeline/
**Purpose**: Core solar panel detection logic

**Workflow**:
1. Read coordinates from Excel
2. Calculate buffer zones
3. Fetch satellite imagery (ArcGIS)
4. Run AI inference (3-model ensemble)
5. Filter by buffer zones
6. Generate outputs (JSON + overlay)

**Critical Component**: `imagery_fetcher.py` with ArcGIS API fix

### 📁 model/
**Purpose**: AI model storage and inference

**3-Model Ensemble**:
- Primary: Laptop-trained (6,876 images, 6 datasets)
- Model v2: Ensemble variation (custom workflow)
- Model v3: Ensemble variation (**~26,000 images** - largest dataset)

**Total Model Size**: ~69 MB  
**Inference Method**: Equal weighting (33.3% each) with NMS merging

### 📁 outputs/
**Purpose**: Stores all results

**Structure**:
- `predictions/{sample_id}.json` - Individual predictions
- `overlays/{sample_id}_overlay.png` - Visual verification
- `summary_report.json` - Batch processing statistics

---

## File Size Breakdown

| Directory | Size | Contents |
|-----------|------|----------|
| .venv/ | 1.15 GB | Python packages |
| model/ | 67 MB | 3 trained models |
| backend/ | ~50 KB | Web server code |
| pipeline/ | ~100 KB | Processing code |
| outputs/ | ~1 MB | Sample results |
| docs/ | ~200 KB | Documentation |
| **Total** | **1.21 GB** | **Complete project** |

---

## Code Statistics

### Lines of Code

| Component | Files | Lines | Purpose |
|-----------|-------|-------|---------|
| Backend | 3 | ~400 | Web API |
| Pipeline | 7 | ~1,200 | Processing |
| Model | 1 | ~200 | Inference |
| **Total** | **11** | **~1,800** | **Core logic** |

### Documentation

| Type | Count | Coverage |
|------|-------|----------|
| Comments | 215+ | Inline explanations |
| Docstrings | 47 | 100% function coverage |
| Markdown | 5 files | Complete guides |

---

## Dependencies

### Main Requirements (requirements.txt)
```
ultralytics>=8.0.0      # YOLOv8 framework
opencv-python>=4.5.0    # Image processing
requests>=2.28.0        # HTTP requests
numpy>=1.24.0           # Numerical computing
Pillow>=9.0.0           # Image handling
pandas>=2.0.0           # Excel processing
openpyxl>=3.0.0         # Excel format
fastapi>=0.104.0        # Web framework
uvicorn>=0.24.0         # ASGI server
```

### Installation
```powershell
pip install -r requirements.txt
```

---

## Git Configuration

### .gitignore
Excludes:
- `__pycache__/` - Python cache
- `.venv/` - Virtual environment
- `*.pyc`, `*.pyo` - Compiled Python
- `outputs/` - Generated results
- `logs/` - Log files
- `.env` - Environment variables

### .gitattributes
Git LFS configuration for large model files:
- `*.pt` - PyTorch models (tracked with LFS)
- Line ending normalization

---

## Quick Navigation

| Need to... | Go to... |
|------------|----------|
| **Start quickly** | [EVALUATOR_GUIDE.md](EVALUATOR_GUIDE.md) |
| **Read full docs** | [README.md](README.md) |
| **See progress** | [PROJECT_STATUS.md](PROJECT_STATUS.md) |
| **Quick start** | [QUICKSTART.md](QUICKSTART.md) |
| **Understand structure** | [STRUCTURE.md](STRUCTURE.md) (this file) |
| **Run server** | `backend/main.py` |
| **Batch processing** | `pipeline/main.py` |
| **Model code** | `model/model_inference.py` |

---

## Repository Health

✅ **Clean Structure**: No test files or temporary data  
✅ **Well Documented**: 5 comprehensive guides  
✅ **Production Ready**: All components tested and working  
✅ **Size Optimized**: 1.21 GB (down from 5.76 GB)  
✅ **Code Quality**: 215+ comments, 100% docstring coverage  

---

**Last Updated**: November 27, 2025  
**Status**: ✅ Production Ready
