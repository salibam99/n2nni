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
    hdul = fits.open("mat1/mat1_000"+str(i)+".fits")
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

# Extract ROIs using the dynamic scaler
# id_101_Pb (First Box)
x1, x2, y1, y2 = scale_roi(510, 700, 1230, 1490, BIN_FACTOR)
ax.add_patch(add_rect(x1, x2, y1, y2))
data_101 = data_show[y1:y2, x1:x2]

# id_101_Pb (Second Box)
x1, x2, y1, y2 = scale_roi(820, 890, 1230, 1490, BIN_FACTOR)
ax.add_patch(add_rect(x1, x2, y1, y2))
data_101_2 = data_show[y1:y2, x1:x2]

# Nickel 102
x1, x2, y1, y2 = scale_roi(1090, 1270, 1290, 1490, BIN_FACTOR)
ax.add_patch(add_rect(x1, x2, y1, y2))
data_102 = data_show[y1:y2, x1:x2]

# In 103
x1, x2, y1, y2 = scale_roi(1700, 1720, 1180, 1230, BIN_FACTOR)
ax.add_patch(add_rect(x1, x2, y1, y2))
data_103 = data_show[y1:y2, x1:x2]

# Alu 104
x1, x2, y1, y2 = scale_roi(450, 780, 470, 820, BIN_FACTOR)
ax.add_patch(add_rect(x1, x2, y1, y2))
data_104 = data_show[y1:y2, x1:x2]

# Zr 105
x1, x2, y1, y2 = scale_roi(1100, 1250, 530, 750, BIN_FACTOR)
ax.add_patch(add_rect(x1, x2, y1, y2))
data_105 = data_show[y1:y2, x1:x2]

# Ti 106
x1, x2, y1, y2 = scale_roi(1620, 1700, 580, 700, BIN_FACTOR)
ax.add_patch(add_rect(x1, x2, y1, y2))
data_106 = data_show[y1:y2, x1:x2]

# Au 113 (Void/Background)
x1, x2, y1, y2 = scale_roi(400, 1500, 1000, 1100, BIN_FACTOR)
ax.add_patch(add_rect(x1, x2, y1, y2))
data_void = data_show[y1:y2, x1:x2]

plt.title(f'Gray color scale (Binned {BIN_FACTOR}x{BIN_FACTOR})')
plt.xlabel("X / pixels")
plt.ylabel("Y / pixels")
plt.show()

print(f"\n=== TRANSMITTANCE RESULTS === Binned {BIN_FACTOR}x{BIN_FACTOR} ")
print(f"Lead_6   NIR mean = {data_101.mean():.4f} and SD = {data_101.std():.4f}")
print(f"Lead_3   NIR mean = {data_101_2.mean():.4f} and SD = {data_101_2.std():.4f}")
print(f"Nickel   NIR mean = {data_102.mean():.4f} and SD = {data_102.std():.4f}")
print(f"Indium   NIR mean = {data_103.mean():.4f} and SD = {data_103.std():.4f}")
print(f"Aluminum NIR mean = {data_104.mean():.4f} and SD = {data_104.std():.4f}")
print(f"Zircon   NIR mean = {data_105.mean():.4f} and SD = {data_105.std():.4f}")
print(f"Titan    NIR mean = {data_106.mean():.4f} and SD = {data_106.std():.4f}")
print("-" * 30)
print(f"Void     NIR mean = {data_void.mean():.4f} and SD = {data_void.std():.4f}")