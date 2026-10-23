# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 17:16:17 2025

@author: miche
"""
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from scipy.ndimage import median_filter
import seaborn as sns

sns.set(rc={"figure.dpi":350, 'savefig.dpi':350})
sns.set_style("ticks")
sns.set_context("talk", font_scale=1)

# ==============================================================================
# NEW: BINNING CONFIGURATION & FUNCTIONS
# ==============================================================================
BIN_FACTOR = 8  # Change to 1 (no binning), 2 (2x2), 4 (4x4), 8 (8x8), etc.

def bin_image(image, bin_factor):
    """Bins a 2D image by summing pixels in bin_factor x bin_factor blocks."""
    if bin_factor <= 1:
        return image
        
    # Ensure image is strictly 2D
    if image.ndim == 3:
        image = image[0]
        
    h, w = image.shape
    new_h = h - (h % bin_factor)
    new_w = w - (w % bin_factor)
    
    # Trim edges to make dimensions perfectly divisible by the bin factor
    trimmed = image[:new_h, :new_w]
    
    # Reshape and SUM the pixels (cast to float32 first to prevent 16-bit overflow)
    binned = trimmed.astype(np.float32).reshape(
        new_h // bin_factor, bin_factor, 
        new_w // bin_factor, bin_factor
    ).sum(axis=(1, 3))
    
    return binned

def scale_roi(x1, x2, y1, y2, bin_factor):
    """Automatically scales your hardcoded coordinates to match the binned image size."""
    return x1 // bin_factor, x2 // bin_factor, y1 // bin_factor, y2 // bin_factor

# ==============================================================================

def remove_outliers(image, radius, threshold):
    """Removes isolated outlier pixels (salt-and-pepper noise) from an image."""
    box_size = (2 * radius) + 1
    median_img = median_filter(image, size=box_size)
    
    img_float = image.astype(np.float32)
    median_float = median_img.astype(np.float32)
    difference = np.abs(img_float - median_float)
    
    outlier_mask = difference > threshold
    cleaned_image = np.where(outlier_mask, median_img, image)
    
    return cleaned_image.astype(image.dtype)

def add_rect(x1, x2, y1, y2):
    width = x2 - x1
    height = y2 - y1
    rect = patches.Rectangle(
        (x1, y1), width, height,
        linewidth=1, edgecolor='r', facecolor='none'
    )
    return rect
    
# ==============================================================================
# DATA LOADING & BINNING
# ==============================================================================
# 1. Dark Current 
list_img = []
for i in range(2):
    hdul = fits.open("DC/dc_150s_000"+str(i)+".fits")
    image_data = hdul[0].data
    datt = image_data[0] if image_data.ndim == 3 else image_data
    list_img.append(datt)

# Median stack, then bin
image_dc = bin_image(np.median(list_img, axis=0), BIN_FACTOR)

# 2. Open Beam 
list_img = []
for i in range(2):
    hdul = fits.open("OB/ob_000"+str(i)+".fits")
    image_data = hdul[0].data
    datt = image_data[0] if image_data.ndim == 3 else image_data
    list_img.append(datt)

image_ob = bin_image(np.median(list_img, axis=0), BIN_FACTOR)

# 3. Object FITS file
list_img = []
for i in range(2):
    hdul = fits.open("mat2/mat2_000"+str(i)+".fits")
    image_data = hdul[0].data
    datt = image_data[0] if image_data.ndim == 3 else image_data
    list_img.append(datt)
    
image_object = bin_image(np.median(list_img, axis=0), BIN_FACTOR)

# ==============================================================================
# PROCESSING
# ==============================================================================
OB_image = image_ob - image_dc
OB_image[OB_image <= 0] = 1e-6  # Prevent division by zero errors

Object_image = image_object - image_dc
Object_image[Object_image <= 0] = 0

image = Object_image / OB_image

# Outlier filter (image is now guaranteed to be 2D)
data_show = remove_outliers(image, 3, threshold=0.0001)

# ==============================================================================
# PLOTTING AND ROI EXTRACTION
# ==============================================================================

fig, ax = plt.subplots(figsize=(8, 8))

plt.imshow(data_show, cmap='gray', origin='lower', vmin=0.1, vmax=1.1)  
cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.set_label("Transmittance (-)")

# ==============================================================================
# EXTRACT ROIs (Targeting the 4 middle materials: In, Dy, Cd, Gd)
# ==============================================================================

# 1. Indium (In) - Top Middle
x1, x2, y1, y2 = scale_roi(870, 970, 1160, 1230, BIN_FACTOR)
ax.add_patch(add_rect(x1, x2, y1, y2))
data_In = data_show[y1:y2, x1:x2]

# 2. Dysprosium (Dy) - Second from Top
x1, x2, y1, y2 = scale_roi(830, 920, 900, 970, BIN_FACTOR)
ax.add_patch(add_rect(x1, x2, y1, y2))
data_Dy = data_show[y1:y2, x1:x2]

# 3. Cadmium (Cd) - Third from Top (Dark)
x1, x2, y1, y2 = scale_roi(780, 900, 630, 690, BIN_FACTOR)
ax.add_patch(add_rect(x1, x2, y1, y2))
data_Cd = data_show[y1:y2, x1:x2]

# 4. Gadolinium (Gd) - Bottom (Dark)
x1, x2, y1, y2 = scale_roi(750, 850, 390, 440, BIN_FACTOR)
ax.add_patch(add_rect(x1, x2, y1, y2))
data_Gd = data_show[y1:y2, x1:x2]

# 5. Void / Background (Moved to a safe, empty region on the left)
x1, x2, y1, y2 = scale_roi(500, 1500, 1350, 1570, BIN_FACTOR)
ax.add_patch(add_rect(x1, x2, y1, y2))
data_void = data_show[y1:y2, x1:x2]

plt.title(f'Gray color scale (Binned {BIN_FACTOR}x{BIN_FACTOR})')
plt.xlabel("X / pixels")
plt.ylabel("Y / pixels")
plt.show()

# ==============================================================================
# PRINT RESULTS
# ==============================================================================
print(f"\n=== TRANSMITTANCE RESULTS === Binned {BIN_FACTOR}x{BIN_FACTOR} ")
print(f"Indium (In)     NIR mean = {data_In.mean():.4f} and SD = {data_In.std():.4f}")
print(f"Dysprosium (Dy) NIR mean = {data_Dy.mean():.4f} and SD = {data_Dy.std():.4f}")
print(f"Cadmium (Cd)    NIR mean = {data_Cd.mean():.4f} and SD = {data_Cd.std():.4f}")
print(f"Gadolinium (Gd) NIR mean = {data_Gd.mean():.4f} and SD = {data_Gd.std():.4f}")
print("-" * 30)
print(f"Void            NIR mean = {data_void.mean():.4f} and SD = {data_void.std():.4f}")