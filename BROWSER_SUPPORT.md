# Browser Support - Solar Panel Detection System

## Overview

The Solar Panel Detection System uses browser automation to capture high-resolution satellite imagery from Google Maps. The system includes comprehensive multi-browser support with automatic fallback.

## Supported Browsers

The system tries browsers in the following order:

1. **Google Chrome** (Recommended)
2. **Microsoft Edge**
3. **Mozilla Firefox**
4. **Brave Browser**
5. **Opera Browser**

If one browser isn't available, the system automatically tries the next one until a working browser is found.

## Installation Requirements

You need **at least ONE** of the supported browsers installed on your system. The system will automatically detect and use the first available browser.

### Browser Download Links

- **Chrome**: https://www.google.com/chrome/
- **Edge**: Pre-installed on Windows 10/11, or https://www.microsoft.com/edge
- **Firefox**: https://www.mozilla.org/firefox/
- **Brave**: https://brave.com/download/
- **Opera**: https://www.opera.com/download

## Setup Process

When you run `setup.bat`, the system will:

1. Check Python version
2. Create virtual environment
3. Install dependencies
4. **Check for available browsers**
5. Display which browsers are detected

### Example Output

```
[5/5] Checking browser availability...

[OK] Chrome detected
[OK] Microsoft Edge detected
[INFO] Firefox not found in PATH
[INFO] Opera not found in PATH
[INFO] Brave not found

NOTE: The system will automatically use any available browser.
     (Chrome, Edge, Firefox, Brave, or Opera in that order)
```

## Runtime Behavior

When you run `start_server.bat` or when the system needs to fetch imagery:

1. The system logs: "Detecting available browsers for satellite imagery capture..."
2. It attempts each browser in order
3. For each browser:
   - **Success**: "✓ Successfully initialized [Browser Name] browser"
   - **Failure**: "✗ [Browser Name] not available: [error details]"
4. Uses the first working browser for all imagery operations

### Example Runtime Log

```
INFO:pipeline.imagery_fetcher:Detecting available browsers for satellite imagery capture...
DEBUG:pipeline.imagery_fetcher:Attempting to initialize Chrome...
INFO:pipeline.imagery_fetcher:✓ Successfully initialized Chrome browser
INFO:pipeline.imagery_fetcher:Fetching Google Maps imagery for lat=12.9716, lon=77.5946
```

## What Happens If No Browser Is Available?

If no supported browser is found, you'll see:

```
ERROR: No compatible browser found!

The system requires one of the following browsers:
  • Google Chrome (recommended)
  • Microsoft Edge
  • Mozilla Firefox
  • Brave Browser
  • Opera Browser

Please install at least one browser and ensure it's accessible.
```

The server will still start, but imagery fetching will fail until a browser is installed.

## Technical Details

### Headless Mode

All browsers run in **headless mode**, meaning:
- No visible browser window opens
- Operations run in the background
- Better performance and resource usage
- Suitable for server/automation environments

### Browser Configuration

Each browser is configured with:
- Headless mode enabled
- Window size: 1920x1080
- Cache disabled (for fresh imagery)
- Incognito/Private mode
- Anti-automation detection disabled
- GPU acceleration disabled (for server compatibility)

### Why Multiple Browser Support?

1. **Compatibility**: Not all systems have the same browsers
2. **Reliability**: If one browser has issues, others can work
3. **Flexibility**: Users can choose their preferred browser
4. **Enterprise**: Some organizations restrict certain browsers

## Troubleshooting

### Browser Not Detected

**Problem**: Setup shows "[INFO] Chrome not found in PATH"

**Solutions**:
1. Install the browser if not present
2. Ensure browser is in system PATH
3. Try restarting your terminal/command prompt
4. Install another supported browser as fallback

### Selenium Driver Issues

**Problem**: "WebDriver not found" or similar errors

**Solutions**:
1. The system uses Selenium WebDriver Manager which auto-downloads drivers
2. Ensure you have internet connection during first run
3. Check your antivirus isn't blocking driver downloads
4. Try running as Administrator if permission issues occur

### Imagery Capture Fails

**Problem**: Server starts but imagery capture fails

**Solutions**:
1. Check the server logs for browser detection messages
2. Verify at least one browser is installed and detected
3. Ensure you have internet connection for Google Maps access
4. Check firewall isn't blocking browser/internet access

## System Requirements

- Windows 10 or higher (recommended)
- Python 3.10 or higher
- At least ONE supported browser
- Internet connection for imagery capture
- 4GB RAM minimum (8GB recommended)

## Advanced Configuration

The browser selection is automatic, but if you need to force a specific browser for debugging:

1. Edit `pipeline_code/pipeline/imagery_fetcher.py`
2. Modify the `browsers` list in `get_browser_driver()` function
3. Reorder or remove browsers as needed

**Example**: To force Firefox first:
```python
browsers = [
    ('Firefox', lambda: webdriver.Firefox(options=get_firefox_options())),
    ('Chrome', lambda: webdriver.Chrome(options=get_chrome_options())),
    # ... others
]
```

## Performance Notes

- **Chrome**: Fastest, most reliable, recommended
- **Edge**: Similar to Chrome (Chromium-based), good alternative
- **Firefox**: Slightly slower but very reliable
- **Brave**: Chrome-based, good privacy features
- **Opera**: Chrome-based, good for certain regions

All browsers perform similarly for this use case. Choose based on what's already installed on your system.

---

**Last Updated**: December 24, 2025  
**System Version**: EcoInnovators Ideathon 2026
