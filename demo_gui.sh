#!/bin/bash
# Quick demo launcher for the Defect Detection GUI

echo "========================================"
echo "Defect Detection GUI - Quick Demo"
echo "========================================"
echo ""
echo "Activating hack2 environment..."
echo "Launching GUI..."
echo ""
echo "Instructions:"
echo "  1. Click 'Browse...' to select an image"
echo "  2. Or use the default CdTe images in:"
echo "     Image Data/DataSet_CdTe/"
echo "  3. Adjust chunk size if needed (default: 64)"
echo "  4. Click 'Run Defect Detection'"
echo "  5. View results with timing info!"
echo ""
echo "Press Ctrl+C to exit the GUI"
echo "========================================"
echo ""

conda activate hack2
python defect_detection_gui.py
