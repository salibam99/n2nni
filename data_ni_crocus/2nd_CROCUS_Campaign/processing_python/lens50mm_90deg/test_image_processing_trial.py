# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 17:16:17 2025

@author: miche
"""
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from matplotlib.colors import LogNorm
import imagej
import seaborn as sns
sns.set(rc={"figure.dpi": 350, 'savefig.dpi': 350})
sns.set_style("ticks")
sns.set_context("talk", font_scale=1.0)


def remove_outliers(image, radius=1.5, threshold=0.05):
    # Apply a median filter with neighborhood ~ radius
    neighborhood_size = int(np.ceil(radius * 2)) + 1  # e.g. radius=1.5 -> size=4 -> 5x5 window
    local_median = median_filter(image, size=neighborhood_size)

    # Compute difference from local median
    diff = np.abs(image - local_median)

    # Mark pixels that are "too different"
    mask = diff > threshold

    # Replace outliers with local median
    result = image.copy()
    result[mask] = local_median[mask]

    return result

replace_neg_values = 1

list_img = []
# dark current 
for i in range(3):
    hdul = fits.open("../DC/DC_5min_other_000"+str(i)+".fits")
    # Inspect file structure
    hdul.info()
    # Access image data (usually in the primary HDU)
    image_data = hdul[0].data
    datt = image_data[0,:,:] 
    list_img.append(datt)

median_image_dc = np.median(list_img, axis=0).astype(np.uint16) 
dc_imgs = np.array(list_img)

list_img = []
# Open Beam 
for i in range(3):
    hdul = fits.open("../OB_Li6_1/OB5min_000"+str(i)+".fits")
    # Inspect file structure
    hdul.info()
    # Access image data (usually in the primary HDU)
    image_data = hdul[0].data
    datt = image_data[0,:,:] 
    list_img.append(datt)

median_image_ob = np.median(list_img, axis=0).astype(np.uint16)
ob_imgs = np.array(list_img)

list_img = []
# Open Object FITS file
for i in range(3):
    hdul = fits.open("Lens-50mm-90deg_000"+str(i)+".fits")
    
    # Inspect file structure
    hdul.info()
    
    # Access image data (usually in the primary HDU)
    image_data = hdul[0].data
    datt = image_data[0,:,:] 
    list_img.append(datt)
    
median_image_object = np.median(list_img, axis=0).astype(np.uint16)
obj_imgs =np.array(list_img)



OB_image = median_image_ob - median_image_dc
OB_image = np.median(ob_imgs - dc_imgs, axis=0).astype(np.uint16)
OB_image[OB_image <= 0] = replace_neg_values
Object_image = median_image_object - median_image_dc
Object_image = np.median(obj_imgs - dc_imgs, axis=0).astype(np.uint16)
Object_image[Object_image <= 0] = replace_neg_values

image = Object_image / OB_image


data_show = remove_outliers(image, radius=1.5, threshold=0.25)

ny, nx = OB_image.shape
mid_y, mid_x = ny // 2, nx // 2
# Display with matplotlib
OB_image = remove_outliers(OB_image, radius=1.5, threshold=0.25)


#___________________________________
pixel_min = 0
pixel_max = 2048
pixel_range = pixel_max - pixel_min
mm_range = 150             # 15 mm
cm_range = mm_range / 10  # 1.5 cm

# 2. Define Conversion Functions (Pixels <-> cm)
def px_to_cm(x):
    return (x - pixel_min) * (cm_range / pixel_range)

def cm_to_px(x):
    return (x * (pixel_range / cm_range)) + pixel_min

# 3. Create Plot
fig, ax = plt.subplots(figsize=(7, 6))

plt.imshow(OB_image  , cmap='gray', origin='lower', vmin=500, vmax=800)  
cbar = plt.colorbar(ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Intensity / Counts")
plt.xlabel("X / pixels")
plt.ylabel("Y / pixels")
plt.axhline(y = mid_y, color="blue", linestyle = "--", alpha = 0.2)
plt.axvline(x = mid_x, color="red", linestyle = "--", alpha = 0.2)
# plt.xlim(150,895)
# plt.ylim(150,895)
sec_x = ax.secondary_xaxis('top', functions=(px_to_cm, cm_to_px))
sec_x.set_xlabel("Scale / cm")

plt.show()

ny, nx = OB_image.shape
mid_y = ny // 2  # horizontal centerline
mid_x = nx // 2  # vertical centerline

dy, dx = 50, 50  # half-width for averaging

# Average over vertical band around center (horizontal profile)
horizontal_profile = np.mean(OB_image[mid_y - dy : mid_y + dy + 1, :], axis=0)

# Average over horizontal band around center (vertical profile)
vertical_profile = np.mean(OB_image[:, mid_x - dx : mid_x + dx + 1], axis=1)

# --- Plot horizontal profile ---
plt.figure()
plt.plot(np.arange(nx)[140:865], horizontal_profile[140:865], color='tab:blue')
plt.xlabel("X / pixels")
plt.ylabel("Intensity / Counts")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Plot vertical profile ---
plt.figure()
plt.plot( vertical_profile[140:865],np.arange(ny)[140:865], color='tab:red')
plt.ylabel("Y / pixels")
plt.xlabel("Intensity / Counts")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

vmin = 0.3
vmax = 1.4

# Display with matplotlib
plt.imshow(np.flip(data_show,0), cmap='gray', origin='lower', vmin=vmin, vmax=vmax)  
plt.colorbar()


plt.xlabel("X / pixels")
plt.ylabel("Y / pixels")
plt.show()


plt.show()