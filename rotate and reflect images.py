# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 16:53:20 2025

@author: proks
"""

from PIL import Image
import os

# Upload/load image
image_path = r"C:\Users\proks\OneDrive\Documents\GitHub\2025_Hackathon\generated artificial data/".replace("\\", "/")
image_name = "STEM_60.jpeg"
# image_path = input("Enter the path to your image: ")
image = Image.open(image_path + image_name)

# Get the folder and base name for saving
folder, filename = os.path.split(image_path)
name = image_name[:-5]
ext=image_name[-5:]

# Define transformations ---
rotations = [0, 90, 180, 270]  # degrees
reflections = ['none', 'horizontal', 'vertical', 'both']

# --- Step 3: Apply transformations and save ---
for rot in rotations:
    rotated_img = image.rotate(rot, expand=True)  # rotate image
    
    for refl in reflections:
        # Apply reflections
        if refl == 'horizontal':
            transformed_img = rotated_img.transpose(Image.FLIP_LEFT_RIGHT)
        elif refl == 'vertical':
            transformed_img = rotated_img.transpose(Image.FLIP_TOP_BOTTOM)
        elif refl == 'both':
            transformed_img = rotated_img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
        else:  # 'none'
            transformed_img = rotated_img
        
        # print(f"{name}_rot{rot}_refl{refl}{ext}")
        # print(f"{name}_rot{rot}_refl{refl}{ext}")
        
        # Save the transformed image
        save_name = f"{name}_rot{rot}_refl{refl}{ext}"
        save_path = os.path.join(folder, save_name)
        transformed_img.save(save_path)
        print(f"Saved: {save_path}")
