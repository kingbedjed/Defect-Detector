# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 10:28:40 2025

@author: proks
"""

import os
from PIL import Image

# Path to the folder containing the .tif images
folder_path = r"C:/Users/proks/OneDrive/Documents/GitHub/2025_Hackathon/Image Data/DataSet_CdTe"  # <-- change this
folder_path = r"C:/Users/proks/OneDrive/Documents/GitHub/2025_Hackathon/Image Data/DataSet_CdTe"
folder_path = r"C:/Users/proks/OneDrive/Documents/GitHub/2025_Hackathon"

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    
    # Check for .tif or .tiff files (case-insensitive)
    if filename.lower().endswith(('.tif', '.tiff')):
        
        tif_path = os.path.join(folder_path, filename)
        
        # Open the TIFF image
        with Image.open(tif_path) as img:
            
            # Convert to RGB (JPEG does not support some TIFF formats)
            img = img.convert("RGB")
            
            # Create output filename
            jpeg_filename = os.path.splitext(filename)[0] + ".jpeg"
            jpeg_path = os.path.join(folder_path, jpeg_filename)
            
            # Save as JPEG
            img.save(jpeg_path, "JPEG", quality=95)
            
            print(f"Converted: {filename} → {jpeg_filename}")
