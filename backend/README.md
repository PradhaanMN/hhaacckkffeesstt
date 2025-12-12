# FastAPI Backend - EcoInnovators Ideathon 2026

## Overview

Governance-ready FastAPI backend for the rooftop solar panel detection system.

## Features

- **REST API** for single and batch location verification
- **Web Frontend** with clean, minimal UI
- **Real-time Processing** with live status updates
- **JSON Output** following exact ideathon specification
- **Overlay Generation** for visual verification
- **Error Handling** for invalid coordinates, missing imagery, etc.

## Installation

```powershell
# Install backend dependencies
pip install -r requirements.txt

# Install main pipeline dependencies (if not already installed)
pip install -r ../env/requirements.txt
```

## Running the Server

### Development Mode

```powershell
# From backend directory
python main.py

# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

The server will start at: **http://localhost:8000**

## API Endpoints

### 1. Frontend UI
- **GET /** - Web interface for manual testing
- Access at: http://localhost:8000

### 2. Health Check
- **GET /health** - Server health status
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-11-24T10:30:00"
}
```

### 3. Single Location Verification
- **POST /api/verify/single**

**Request:**
```json
{
  "sample_id": "YOUR_ID",
  "latitude": YOUR_LATITUDE,
  "longitude": YOUR_LONGITUDE
}
```

**Response:**
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
    "resolution_m_per_pixel": 0.054
  },
  "overlay_url": "/outputs/overlays/1001_overlay.png",
  "processing_time_seconds": 25.3
}
```

### 4. Batch Verification
- **POST /api/verify/batch**

**Request:**
```json
{
  "locations": [
    {"sample_id": "ID1", "latitude": LAT1, "longitude": LON1},
    {"sample_id": 1002, "latitude": 28.7041, "longitude": 77.1025}
  ]
}
```

**Response:**
```json
{
  "total_requested": 2,
  "successful": 2,
  "failed": 0,
  "results": [...],
  "errors": []
}
```

### 5. Get Saved Result
- **GET /api/result/{sample_id}**
- Retrieves previously computed result

### 6. Get Overlay Image
- **GET /api/overlay/{sample_id}**
- Returns PNG overlay image

## Testing with cURL

```powershell
# Health check
curl http://localhost:8000/health

# Single verification
curl -X POST http://localhost:8000/api/verify/single `
  -H "Content-Type: application/json" `
  -d '{\"sample_id\": \"YOUR_ID\", \"latitude\": YOUR_LAT, \"longitude\": YOUR_LON}'

# Get result
curl http://localhost:8000/api/result/1001
```

## Architecture

```
backend/
├── main.py              # FastAPI app
├── static/
│   └── index.html      # Frontend UI
├── requirements.txt    # Backend dependencies
└── README.md           # This file
```

The backend integrates with the main pipeline:
- Uses existing `pipeline/` modules
- Loads YOLOv8 model once (lazy loading)
- Reuses `process_single_location()` logic
- Saves outputs to standard directories

## Frontend Features

The web UI (`index.html`) provides:
- ✅ Clean, minimal, competition-friendly design
- ✅ Input fields for sample_id, latitude, longitude
- ✅ One-click sample locations (India cities)
- ✅ Real-time processing status
- ✅ Visual results display with badges
- ✅ Overlay image preview
- ✅ Full JSON output
- ✅ Error handling and validation

## Configuration

Edit `main.py` to customize:
- Port number (default: 8000)
- CORS settings
- Number of workers
- Log levels

## Performance

- **Latency:** ~20-60 seconds per location (network + inference)
- **Model Loading:** One-time at first request (~2-3 seconds)
- **Concurrent Requests:** Handled sequentially (model not thread-safe)
- **Scalability:** Use multiple workers for parallel batch processing

## Error Handling

The API handles:
- ❌ Invalid coordinates (validation)
- ❌ Image fetch failures (network issues)
- ❌ Model inference errors
- ❌ Missing files
- ❌ Malformed requests

All errors return appropriate HTTP status codes with detailed messages.

## Compliance

✅ **EcoInnovators Ideathon Requirements:**
- Uses automated satellite imagery (no API keys)
- Implements exact buffer logic (1200→2400 sq.ft)
- Produces required JSON format
- Generates explainability overlays
- Includes QC status determination
- Saves all outputs as required

## Next Steps

1. **Deploy to Production:**
   - Use Gunicorn/Uvicorn workers
   - Add HTTPS with SSL certificate
   - Configure reverse proxy (Nginx)
   - Set up monitoring and logging

2. **Scale for Competition:**
   - Add request queuing for batch jobs
   - Implement progress tracking
   - Add authentication if needed
   - Set up database for result storage

3. **Enhance UI:**
   - Add file upload for Excel batch processing
   - Display processing queue
   - Add download buttons for results
   - Implement result history

## Support

For issues:
1. Check `pipeline.log` for detailed errors
2. Verify model file exists at `model/model_weights/solarpanel_seg_v1.pt`
3. Ensure all dependencies installed
4. Test with sample locations first
