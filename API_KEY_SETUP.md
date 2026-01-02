# 🗝️ Google Maps API Key Setup Guide

## Quick Setup (2 Minutes)

### Step 1: Get Your API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select existing)
3. Enable **"Maps Static API"**
4. Go to **Credentials** → **Create Credentials** → **API Key**
5. Copy your API key

### Step 2: Paste Your API Key

**Choose ONE method:**

#### ✅ Method 1: Config File (Easiest)

Open `pipeline_code/pipeline/config.py` and paste your key on **line 51**:

```python
GOOGLE_MAPS_API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual key
```

#### ✅ Method 2: Environment Variable

**Windows PowerShell:**
```powershell
$env:GOOGLE_MAPS_API_KEY="YOUR_API_KEY_HERE"
.\start_server.bat
```

**Windows CMD:**
```cmd
set GOOGLE_MAPS_API_KEY=YOUR_API_KEY_HERE
start_server.bat
```

**Linux/Mac:**
```bash
export GOOGLE_MAPS_API_KEY="YOUR_API_KEY_HERE"
./start_server.bat
```

### Step 3: Verify It's Working

Start the server and check the logs:

```bash
.\start_server.bat
```

**With API:**
```
INFO: Attempting to fetch via Google Maps Static API...
INFO: ✓ Successfully fetched imagery via API (zoom=21)
```

**Without API (fallback):**
```
INFO: Using browser automation for imagery capture...
```

---

## Cost & Free Tier

- **Free Tier**: $200/month credit = ~40,000 free requests
- **After Free Tier**: $2 per 1,000 requests ($0.002 per image)
- **For 1000 locations**: ~$2 (or free if within $200 credit)

---

## API vs Browser Comparison

| Feature | API Mode | Browser Mode |
|---------|----------|--------------|
| **Speed** | 0.5-1 sec | 3-5 sec |
| **Reliability** | 99.9% | 95%+ |
| **Cost** | $2/1000 images | Free |
| **Setup** | API key required | Browser required |
| **Best For** | Production, large scale | Testing, small scale |

---

## Troubleshooting

### "API returned status 403"
- **Issue**: API key invalid or billing not enabled
- **Fix**: Check API key, enable billing in Google Cloud Console

### "API returned status 400"
- **Issue**: Invalid coordinates
- **Fix**: Check latitude (-90 to 90) and longitude (-180 to 180)

### Still using browser automation?
- **Check**: Is your API key set correctly?
- **Verify**: Look for this in logs: `"Attempting to fetch via Google Maps Static API..."`
- **If not showing**: Your key isn't being loaded - check spelling and location

---

## No API Key? No Problem!

The system **automatically falls back** to browser automation if:
- No API key is provided
- API request fails
- API quota exceeded

Just make sure you have Chrome, Edge, Firefox, Brave, or Opera installed!

---

## Where to Paste (Quick Reference)

📁 **File**: `pipeline_code/pipeline/config.py`  
📍 **Line**: 51  
🔧 **Code**: `GOOGLE_MAPS_API_KEY = "YOUR_KEY_HERE"`

**Or set environment variable before running server**
