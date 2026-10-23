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

def remove_outliers(image, radius, threshold):
    """
    Removes isolated outlier pixels (salt-and-pepper noise) from an image.
    """
    # 1. Calculate the neighborhood size based on the radius
    box_size = (2 * radius) + 1
    
    # 2. Calculate the median for every pixel's neighborhood all at once
    median_img = median_filter(image, size=box_size)
    
    # 3. Calculate the absolute difference
    img_float = image.astype(np.float32)
    median_float = median_img.astype(np.float32)
    difference = np.abs(img_float - median_float)
    
    # 4. Flag the outliers
    outlier_mask = difference > threshold
    
    # 5. Replace outliers with the median
    cleaned_image = np.where(outlier_mask, median_img, image)
    
    return cleaned_image.astype(image.dtype)

def add_rect(x1,x2,y1,y2):
    width = x2 - x1
    height = y2 - y1

    rect = patches.Rectangle(
        (x1, y1),          # (x, y) coordinates of the anchor point
        width,             # width
        height,            # height
        linewidth=1,       # Thickness of the line
        edgecolor='r',     # 'r' = red
        facecolor='none'   # 'none' = transparent (so you can see the image inside)
    )
    return rect
    

list_img = []
# dark current 
for i in range(2):
    hdul = fits.open("DC/dc_150s_000"+str(i)+".fits")
    # Inspect file structure
    hdul.info()
    # Access image data (usually in the primary HDU)
    image_data = hdul[0].data
    datt = image_data
    list_img.append(datt)

image_dc = np.median(list_img, axis=0).astype(np.uint16) 

list_img = []
# Open Beam 
for i in range(2):
    hdul = fits.open("OB/ob_000"+str(i)+".fits")
    # Inspect file structure
    hdul.info()
    # Access image data (usually in the primary HDU)
    image_data = hdul[0].data
    datt = image_data#[0,:,:] 
    list_img.append(datt)

image_ob = np.median(list_img, axis=0).astype(np.uint16)

list_img = []
# Open Object FITS file
for i in range(2):
    hdul = fits.open("mat1/mat1_000"+str(i)+".fits")
    
    # Inspect file structure
    hdul.info()
    
    # Access image data (usually in the primary HDU)
    image_data = hdul[0].data
    datt = image_data#[0,:,:] 
    list_img.append(datt)
    
image_object = np.median(list_img, axis=0).astype(np.uint16)
 

OB_image = image_ob - image_dc
OB_image[OB_image <= 0] = 0
# OB_image = remove_outliers(OB_image, 3, threshold= 100)
Object_image = image_object - image_dc
# image_object = remove_outliers(image_object, 3, threshold= 100)
Object_image[Object_image <= 0] = 0

image = Object_image / OB_image

data_show =  remove_outliers(image[0,:,:], 3, threshold = 0.05)
# Display with matplotlib
plt.imshow(data_show, cmap='cividis', origin='lower', vmin=0.5, vmax=1.1)  
plt.colorbar()
# plt.xlim(0, data_show.shape[1])   # x-axis: width
# plt.ylim(0, data_show.shape[0])   # y-axis: height
# plt.xlim(1000,1500)
# plt.ylim(1200,1500)

x1, x2 = 450, 700  # Start and End X (Columns)
y1, y2 = 1230, 1450  # Start and End Y (Rows)

plt.gca().add_patch(add_rect(x1, x2, y1, y2)) # id_101_Pb

data_101 = data_show[y1:y2,x1:x2]

x1, x2 = 810, 870  # Start and End X (Columns)
y1, y2 = 1230, 1450  # Start and End Y (Rows)

plt.gca().add_patch(add_rect(x1, x2, y1, y2)) # id_101_Pb

data_101_2 = data_show[y1:y2,x1:x2]


x1, x2 = 1020, 1220  # Start and End X (Columns)
y1, y2 = 1270, 1470  # Start and End Y (Rows)

plt.gca().add_patch(add_rect(x1, x2, y1, y2))

data_102 = data_show[y1:y2,x1:x2] # Nickel 102

x1, x2 = 1450, 1670  # Start and End X (Columns)
y1, y2 = 1270, 1450  # Start and End Y (Rows)

plt.gca().add_patch(add_rect(x1, x2, y1, y2))

data_103 = data_show[y1:y2,x1:x2] # copper 103

x1, x2 = 350, 750   # Start and End X (Columns)
y1, y2 = 470, 700  # Start and End Y (Rows)

plt.gca().add_patch(add_rect(x1, x2, y1, y2))

data_104 = data_show[y1:y2,x1:x2] # Alu 104

x1, x2 = 990, 1350    # Start and End X (Columns)
y1, y2 = 530, 750  # Start and End Y (Rows)

plt.gca().add_patch(add_rect(x1, x2, y1, y2))

data_105 = data_show[y1:y2,x1:x2] # Zr 105


x1, x2 = 1580, 1710    # Start and End X (Columns)
y1, y2 = 560, 720  # Start and End Y (Rows)

plt.gca().add_patch(add_rect(x1, x2, y1, y2))

data_106 = data_show[y1:y2,x1:x2] # Ti 106

x1, x2 = 800, 1000    # Start and End X (Columns)
y1, y2 = 900, 1000   # Start and End Y (Rows)

plt.gca().add_patch(add_rect(x1, x2, y1, y2))

data_void = data_show[y1:y2,x1:x2] # Au 113



plt.title('Gray color scale')
plt.xlabel("X / pixels")
plt.ylabel("Y / pixels")
plt.show()

print("Lead_6   NIR mean = ",data_101.mean()," and SD = ",data_101.std())
print("Lead_3   NIR mean = ",data_101_2.mean()," and SD = ",data_101_2.std())
print("Nickel NIR mean = ",data_102.mean()," and SD = ",data_102.std())
print("copper-absent NIR mean = ",data_103.mean()," and SD = ",data_103.std())
print("alumin NIR mean = ",data_104.mean()," and SD = ",data_104.std())
print("Zircon NIR mean = ",data_105.mean()," and SD = ",data_105.std())
print("Titan  NIR mean = ",data_106.mean()," and SD = ",data_105.std())
print("**************************************************************")
print("void__ NIR mean = ",data_void.mean()," and SD = ",data_void.std())
#___________________________________
# pixel_min = 150
# pixel_max = 900
# pixel_range = pixel_max - pixel_min
# mm_range = 150             # 15 mm
# cm_range = mm_range / 10  # 1.5 cm

# # 2. Define Conversion Functions (Pixels <-> cm)
# def px_to_cm(x):
#     return (x - pixel_min) * (cm_range / pixel_range)

# def cm_to_px(x):
#     return (x * (pixel_range / cm_range)) + pixel_min

# # 3. Create Plot
# fig, ax = plt.subplots(figsize=(7, 6))

# # Display with matplotlib
# plt.imshow(data_show, cmap='gray_r', origin='lower', vmin=0.3, vmax=1.4)  
# cbar = plt.colorbar(ax=ax, fraction=0.046, pad=0.04)
# cbar.set_label("Intensity / a.u.")
# # plt.xlim(0, data_show.shape[1])   # x-axis: width
# # plt.ylim(0, data_show.shape[0])   # y-axis: height
# plt.xlim(150,900)
# plt.ylim(150,900)
# # plt.title('Inverse Gray color scale')
# plt.xlabel("X / pixels")
# plt.ylabel("Y / pixels")

# sec_x = ax.secondary_xaxis('top', functions=(px_to_cm, cm_to_px))
# sec_x.set_xlabel("Scale / cm")
# plt.show()



# plt.imshow(np.exp(OB_image/100), cmap='gray_r', origin='lower', vmin=500, vmax=3000)  
# plt.colorbar()
# plt.xlim(0, data_show.shape[1])   # x-axis: width
# plt.ylim(0, data_show.shape[0])   # y-axis: height
# plt.title('Beam expo')
# plt.xlabel("X / pixels")
# plt.ylabel("Y / pixels")
# plt.show()


# plt.imshow(Object_image, cmap='gray_r', origin='lower', vmin=500, vmax=3000)  
# plt.colorbar()
# plt.xlim(0, data_show.shape[1])   # x-axis: width
# plt.ylim(0, data_show.shape[0])   # y-axis: height
# plt.title('Object')
# plt.xlabel("X / pixels")
# plt.ylabel("Y / pixels")
# plt.show()



# plt.imshow(Object_image, cmap='gray_r', origin='lower', vmin=0.2, vmax=1.4)  
# plt.colorbar()
# plt.xlim(200, 400)   # x-axis: width
# plt.ylim(500, 600)   # y-axis: height
# plt.title('Zoomed')
# plt.show()

# Sub-image
# sub = data_show[200:400, 530:570]

# # Average over rows (axis=0 → collapse y, keep x)
# profile = sub.mean(axis=1)

# # x-axis: pixel indices or distances
# x = np.arange(150, 300)

# plt.plot(x, data_show[150:300, 550], marker='.')
# plt.xlabel("X position (pixels)")
# plt.ylabel("Average intensity")
# plt.title("Average profile across Y=200–400")
# plt.show()



