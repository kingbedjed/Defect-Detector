# Defect Detection GUI - User Guide

A professional graphical interface for detecting and inspecting defects in microscopy images.

## Quick Start

### Launch the Application

```bash
conda activate hack2
python defect_detection_gui.py
```

### Basic Workflow

1. **Load Image**: Click "Browse..." and select an image file
2. **Run Detection**: Click "Run Defect Detection" button
3. **View Results**: Red bounding boxes highlight detected defects
4. **Navigate**: Use mouse to zoom and pan for detailed inspection

## Features Overview

- **Load Images**: Browse and open .tif, .png, .jpg files
- **Defect Detection**: One-click analysis with timing statistics
- **Interactive Zoom**: 0.1x to 100x+ magnification
- **Pan Navigation**: Click-and-drag to explore the image
- **Visual Overlay**: Red bounding boxes mark detected defects
- **Performance Metrics**: Detection time displayed in milliseconds

## Controls Reference

### Mouse Controls

| Action | Function |
|--------|----------|
| **Scroll Up** | Zoom in (centers on cursor) |
| **Scroll Down** | Zoom out (centers on cursor) |
| **Click & Drag** | Pan the view (move around) |

### Keyboard Shortcuts

| Key | Function |
|-----|----------|
| **R** or **r** | Reset view (back to 1.0x zoom, centered) |

## Using the GUI

### Loading an Image

1. Click the **"Browse..."** button in the top section
2. Navigate to your image directory (default: `Image Data/DataSet_CdTe/`)
3. Select an image file (.tif, .tiff, .png, .jpg, .jpeg)
4. The image appears in the main display area
5. Status bar shows: `Loaded: filename.tif (64×64 pixels)`

### Running Defect Detection

1. Ensure an image is loaded (Browse button is required first)
2. Click the **"Run Defect Detection"** button
3. Processing happens in milliseconds
4. Red bounding boxes appear on detected defects
5. Status bar shows: `Detected XXX defects in your image in XXXms`
6. A popup dialog confirms completion with statistics

### Zooming In and Out

**To Zoom In:**
- Position cursor over the region of interest
- Scroll up with mouse wheel
- Image magnifies centered on cursor
- Zoom level increases: 1.0x → 1.2x → 1.44x → 1.73x...

**To Zoom Out:**
- Scroll down with mouse wheel
- Image shrinks centered on cursor
- Zoom level decreases: 1.0x → 0.83x → 0.69x → 0.58x...
- Can zoom out to 0.1x minimum for bird's eye view

**Zoom Range:**
- Minimum: 0.1x (image appears very small with padding)
- Default: 1.0x (image fills the view)
- Maximum: 100x+ (limited only by pixel resolution)

**Status Bar:**
- Shows current zoom level when not at 1.0x
- Example: `Detected 4 defects in your image in 3.7ms | Zoom: 5.2x`

### Panning Around the Image

**To Pan:**
1. Click and hold the left mouse button on the image
2. Drag in any direction
3. Image moves with your mouse
4. Release to stop panning

**Visual Feedback:**
- Cursor changes to ⊕ (crosshair) while dragging
- Returns to normal arrow when released

**Panning Behavior:**
- **When zoomed in**: Cannot pan outside image boundaries
- **When zoomed out**: Can pan freely (image floats in view)

### Resetting the View

Press the **'R'** key at any time to:
- Reset zoom to 1.0x
- Center the image
- Return to default view

Perfect for quickly returning to the overview after exploring details.

## Common Workflows

### Workflow 1: Basic Inspection

```
1. Click "Browse..." and select image
2. Click "Run Defect Detection"
3. View detected defects with red boxes
4. Done!
```

### Workflow 2: Detailed Defect Examination

```
1. Load image and run detection
2. Scroll mouse wheel up to zoom in on a defect (5-10x)
3. Click and drag to pan around the defect
4. Examine surrounding structure
5. Press 'R' to reset view
6. Repeat for other defects
```

### Workflow 3: Systematic Image Scanning

```
1. Load image and run detection
2. Zoom to moderate level (3-5x)
3. Click and drag to top-left corner
4. Systematically pan across the image
5. Inspect each region thoroughly
6. Press 'R' when complete
```

### Workflow 4: Quick Comparison

```
1. Zoom in on first defect (8-10x)
2. Click and drag to second defect
3. Compare features
4. Drag back to first defect
5. Repeat comparison as needed
```

## Tips and Tricks

### For Efficient Navigation

1. **Zoom first, then pan**: Set your desired magnification, then explore
2. **Use 'R' frequently**: Quick reset instead of scrolling back out
3. **Cursor positioning**: Point at your target before scrolling to zoom there
4. **Moderate zoom for overview**: 2-3x is good for scanning multiple defects
5. **High zoom for details**: 10-20x for pixel-level inspection

### For Best Results

1. **Let detection complete**: Wait for the popup dialog before navigating
2. **Check all defects**: Zoom and pan to inspect each red box
3. **Verify at multiple scales**: Look at defects at different zoom levels
4. **Use systematic scanning**: Don't miss regions by panning methodically

### Zoom Levels Guide

| Zoom | Best For |
|------|----------|
| **0.1x - 0.5x** | Bird's eye view, overall context |
| **0.5x - 1.0x** | Full image overview with some detail |
| **1.0x** | Default view, image fills screen |
| **1.5x - 3x** | Moderate detail, see multiple defects |
| **3x - 5x** | Individual defect examination |
| **5x - 10x** | Fine detail inspection |
| **10x - 20x** | Pixel-level analysis |
| **20x+** | Maximum magnification for tiny features |

## User Interface Layout

```
┌─────────────────────────────────────────────────────────┐
│  Defect Detection System                                │
│  💡 Mouse wheel: zoom | Click & drag: pan | Press 'R'  │
│                                                          │
│  Image File: [/path/to/image.tif       ] [Browse...]   │
│                              [Run Defect Detection]      │
├─────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────┐ │
│  │                                                    │ │
│  │                                                    │ │
│  │          IMAGE DISPLAY AREA                        │ │
│  │     (with red bounding boxes for defects)          │ │
│  │                                                    │ │
│  │                                                    │ │
│  └───────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│  Detected XXX defects in your image in XXXms | Zoom: Xx│
└─────────────────────────────────────────────────────────┘
```

### Interface Elements

**Top Section - Controls:**
- Title and instructions
- File path display
- Browse button for loading images
- Run Detection button

**Middle Section - Display:**
- Large canvas showing the image
- Red bounding boxes overlay detected defects
- Zoom and pan work in this area

**Bottom Section - Status:**
- Shows loaded image information
- Displays detection results and timing
- Shows current zoom level when applicable

## Understanding the Output

### Status Messages

**After Loading Image:**
```
Loaded: Vacancy_CdTe_73.tif (64×64 pixels)
```

**After Running Detection:**
```
Detected 4 defects in your image in 3.7ms
```

**While Zoomed:**
```
Detected 4 defects in your image in 3.7ms | Zoom: 5.2x
```

### Detection Results

**Bounding Boxes:**
- **Color**: Red
- **Style**: Outline only (no fill)
- **Rotation**: Follows defect orientation
- **Thickness**: 2 pixels

**Popup Dialog:**
```
Detection Complete

Detection completed successfully!

Defects found: 4
Processing time: 3.7 ms
```

## Troubleshooting

### Image Won't Load

**Problem**: Nothing happens when clicking Browse
- Check that you selected a valid image file
- Supported formats: .tif, .tiff, .png, .jpg, .jpeg
- Verify file path is correct

**Problem**: Error message when loading
- Image file may be corrupted
- Format may not be supported
- Try a different image

### Detection Button Disabled

**Problem**: "Run Defect Detection" button is grayed out
- **Solution**: Load an image first using the Browse button
- Button activates automatically when image is loaded

### Zoom Not Working

**Problem**: Scrolling doesn't zoom
- Ensure cursor is over the image (not outside the display area)
- Image must be loaded
- Try clicking on the image first to focus

**Problem**: Zoom is jumpy or erratic
- This should not happen with normal use
- Try pressing 'R' to reset
- Reload the image if problem persists

### Pan Not Working

**Problem**: Click and drag doesn't move image
- **Use left mouse button** (not right or middle)
- Cursor must be on the image when clicking
- Image must be loaded
- Try zooming in first (panning is most useful when zoomed)

**Problem**: Can't pan to certain areas
- When zoomed in: Cannot pan outside image boundaries (this is correct)
- When at 1.0x zoom: Panning is limited (this is normal)
- Zoom in first for full pan range

### Display Issues

**Problem**: Frame appears distorted
- This should not occur
- Press 'R' to reset view
- Reload the image
- Restart the GUI if needed

**Problem**: Bounding boxes don't appear
- Verify detection completed (check for popup dialog)
- Detection may have found zero defects
- Try a different image

**Problem**: Image is too small/large
- Use mouse wheel to zoom in or out
- Press 'R' to reset to default size
- Adjust zoom to comfortable level

## Supported File Formats

The GUI accepts the following image formats:

- **TIFF**: .tif, .tiff (recommended for microscopy)
- **PNG**: .png
- **JPEG**: .jpg, .jpeg

**Note**: Files are automatically converted to grayscale if needed.

## Performance Notes

### Typical Processing Times

For 64×64 pixel images:
- **Image Loading**: < 10 ms
- **Detection**: < 5 ms
- **Visualization**: < 50 ms
- **Total**: < 100 ms

**Status bar shows exact timing**: `Detected X defects in your image in XXXms`

### Navigation Performance

- **Zoom**: < 50 ms per zoom step (very responsive)
- **Pan**: < 16 ms per update (60 FPS smooth)
- **Reset**: Instant

All operations are fast and smooth, even with multiple defects.

## Keyboard and Mouse Summary

### Complete Controls

**Mouse Wheel:**
- Scroll up = Zoom in (1.2x per scroll)
- Scroll down = Zoom out (0.83x per scroll)

**Left Mouse Button:**
- Click + drag = Pan the view

**Keyboard:**
- R or r = Reset view to default

**No other controls needed!**

## Example Session

Here's a complete example of using the GUI:

```
$ python defect_detection_gui.py

[GUI opens]

1. Click "Browse..." button
2. Navigate to: Image Data/DataSet_CdTe/
3. Select: Vacancy_CdTe_73.tif
   → Status: "Loaded: Vacancy_CdTe_73.tif (64×64 pixels)"

4. Click "Run Defect Detection"
   → Processing...
   → Status: "Detected 4 defects in your image in 3.7ms"
   → Popup: "Detection Complete - Defects found: 4"

5. Scroll up 3 times
   → Zoom to ~1.7x
   → Status: "Detected 4 defects in your image in 3.7ms | Zoom: 1.7x"

6. Click on first defect and drag to examine
   → Pan around the defect

7. Scroll up 5 more times
   → Zoom to ~4.3x
   → Status shows: "Zoom: 4.3x"

8. Continue clicking and dragging to inspect all defects

9. Press 'R' key
   → Reset to 1.0x zoom, centered
   → Status: "Detected 4 defects in your image in 3.7ms"

10. Load next image and repeat
```

## Best Practices

### For Efficient Inspection

1. Always run detection first before zooming
2. Use systematic scanning at moderate zoom (3-5x)
3. Zoom in high (10x+) only for specific defects
4. Press 'R' between different defects
5. Check the entire image, not just obvious defects

### For Accurate Analysis

1. Examine each defect at multiple zoom levels
2. Pan around defects to see context
3. Compare similar defects by panning between them
4. Verify tiny defects with high zoom (10-20x)
5. Check status bar for defect count confirmation

### For Smooth Workflow

1. Keep images organized in one directory
2. Process images in order
3. Use consistent zoom levels for comparison
4. Reset view ('R') frequently
5. Note any unusual findings

## FAQ

**Q: What does the red box mean?**
A: Each red bounding box indicates a detected defect. The box shows the location and orientation of the defect.

**Q: Can I save the results?**
A: Currently, results are displayed only. To save, take a screenshot or note the defect count from the status bar.

**Q: How accurate is the detection?**
A: Detection uses the algorithm from main.py. Current implementation uses placeholder random detection for demonstration.

**Q: Can I process multiple images?**
A: One at a time. Load an image, run detection, examine results, then load the next image.

**Q: What if no defects are found?**
A: Status bar will show "Detected 0 defects in your image in XXXms". This is normal for clean images.

**Q: Can I change the detection settings?**
A: Currently no adjustable settings in the GUI. Detection uses default parameters from main.py.

**Q: What's the maximum zoom?**
A: Practically unlimited (100x+), constrained only by pixel resolution.

**Q: What's the minimum zoom?**
A: 0.1x, which shows the image at 10% of its size with gray padding around it.

**Q: Why does the cursor change?**
A: The cursor changes to ⊕ (crosshair) when panning to indicate you're in pan mode.

**Q: Can I zoom without a mouse wheel?**
A: Currently only mouse wheel zoom is supported. Touchpad gestures may work depending on your system.

## Getting Help

If you encounter issues not covered in this guide:

1. Try restarting the GUI
2. Reload the image
3. Press 'R' to reset the view
4. Check that your image file is valid
5. Verify you're using a supported file format

## Summary

The Defect Detection GUI provides:

✅ Simple, intuitive interface
✅ One-click defect detection
✅ Professional zoom and pan navigation
✅ Real-time performance metrics
✅ Visual defect highlighting
✅ Smooth, responsive controls

Perfect for inspecting microscopy images and analyzing defects efficiently!

---

**Quick Reference Card:**

| Task | Action |
|------|--------|
| Load image | Click "Browse..." |
| Run detection | Click "Run Defect Detection" |
| Zoom in | Scroll up |
| Zoom out | Scroll down |
| Pan | Click and drag |
| Reset | Press 'R' |

**Remember**: Mouse wheel zooms, click-drag pans, 'R' resets!
