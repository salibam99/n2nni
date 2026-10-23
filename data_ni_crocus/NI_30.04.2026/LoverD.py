# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:01:46 2026

@author: miche
"""

from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit, fsolve
import warnings
from scipy.ndimage import gaussian_filter1d

class ExperimentCatalog:
    def __init__(self, master_directory):
        """Scans the master directory and remembers folder locations (no heavy loading yet)."""
        self.master_path = Path(master_directory)
        # Create a dictionary mapping the folder string name to its actual path object
        self.folders = {folder.name: folder for folder in self.master_path.iterdir() if folder.is_dir()}
        print(f"Catalog initialized! Found {len(self.folders)} folders.")
        print(f"Available folders: {list(self.folders.keys())}\n")

    def _load_to_tensor(self, file_paths):
        """Internal helper: Opens FITS files and stacks them into a 3D NumPy array."""
        if not file_paths:
            return None
            
        list_img = []
        for path in file_paths:
            with fits.open(path) as hdul:
                image_data = hdul[0].data
                # Extract first slice if 3D, otherwise take as-is
                datt = image_data[0,:,:] if image_data.ndim == 3 else image_data
                list_img.append(datt)
        return np.array(list_img)

    def get_data(self, folder_name_string):
        """
        You pass the folder name as a string. 
        It returns a single NumPy tensor containing all .fits files in that folder.
        """
        if folder_name_string not in self.folders:
            raise ValueError(f"Folder '{folder_name_string}' not found in catalog!")
            
        print(f"--- Loading Data for '{folder_name_string}' ---")
        folder_path = self.folders[folder_name_string]
        
        # Grab all .fits files in this specific folder
        all_files = list(folder_path.rglob("*.fits"))
        
        if len(all_files) == 0:
            print(f"Warning: No .fits files found in {folder_name_string}")
            return None
            
        # Load all found files directly into one tensor and return it
        return self._load_to_tensor(all_files)
        
    def remove_outliers(self, image, radius, threshold):
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



## MAIN FUNCTION ###

if __name__ == "__main__":
    # 1. Initialize the catalog ONCE
    master_dir = r"C:\Users\miche\Desktop\PhD\Neutron_Imaging_project\NI_30.04.2026"
    catalog = ExperimentCatalog(master_dir)
    
    sample_data = catalog.get_data("B4C-Cd")
    dc = catalog.get_data("DC")
    sample_corrected = sample_data - dc 
    sample_median = np.sum(sample_corrected, axis=0)
    
    # sample_median = np.median(sample_data, axis = 0)
    sample_median = catalog.remove_outliers(sample_median, 2, threshold = 20)
    
    plt.imshow(sample_median, origin='lower', vmin = 0, vmax = 300)
    plt.show()
    
    y_start, y_end = 750, 1000  # Vertical crop range
    x_start, x_end = 800, 1700  # Horizontal crop range
    
    # Assuming sample_median is already defined in your environment
    cropped_cd = sample_median[y_start:y_end, x_start:x_end]
    
    # --- 2. 3D Surface Plot of the Edge ---
    # This will help visualize the "blurriness" as you move away from the sheet
    fig_3d = plt.figure(figsize=(10, 7))
    ax = fig_3d.add_subplot(111, projection='3d')
    
    # Create X and Y grids matching the cropped image dimensions
    X, Y = np.meshgrid(np.arange(cropped_cd.shape[1]), np.arange(cropped_cd.shape[0]))
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, cropped_cd, cmap='inferno', edgecolor='none')
    ax.set_title("3D Surface Plot of the Cadmium Edge")
    ax.set_xlabel("X / pixels")
    ax.set_ylabel("Y / pixels")
    ax.set_zlabel("Intensity / Transmittance")
    fig_3d.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.show()
        
    
    
# --- Plot a single vertical line profile ---
    y_start, y_end = 800, 900  # Your vertical range
    x_start, x_end = 850, 1660  # Your horizontal scanning range            # The specific X pixel column you want to look at
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Grab the 'jet' colormap
    cmap = plt.cm.jet
    
    # 2. Loop through all the X columns
    for target_x in range(x_start, x_end):
        
        # Extract the data
        y_intensities = sample_median[y_start:y_end, target_x]
        y_pixels = range(y_start, y_end)
        
        # Calculate a fraction from 0.0 to 1.0 to pick the right color from the gradient
        color_fraction = (target_x - x_start) / (x_end - x_start)
        line_color = cmap(color_fraction)
        
        # Plot it!
        ax.plot(y_pixels, y_intensities, color=line_color, linewidth=1.5, alpha=0.1)
    
    # --- Add a colorbar legend to the side ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=x_start, vmax=x_end))
    sm.set_array([]) 
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("X-Pixel Position")
    
    # 3. Add titles and show the final combined plot (OUTSIDE the loop)
    plt.title(f"Vertical Intensity Profiles from X={x_start} to {x_end}")
    plt.xlabel("Y / pixels")
    plt.ylabel("Intensity / Counts")
    plt.grid(True, alpha=0.3)
    
    plt.show()
    
# =========================================================================
    # PLOT LSF (DERIVATIVE OF ESF)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.jet
    
    # 2. Loop through all the X columns
    for target_x in range(x_start, x_end+1,50):
        
        # 1. EXTRACT THE DATA (Bug fixed: uncommented and cast to float)
        y_intensities = sample_median[y_start:y_end, target_x].astype(float)
        y_pixels = np.arange(y_start, y_end)
        
        # 2. SMOOTH THE NOISE (Crucial for derivatives!)
        # Adjust sigma (1 to 3) if the curves are still too noisy or too flat
        smoothed_esf = gaussian_filter1d(y_intensities, sigma=2.0)
        
        # 3. CALCULATE DERIVATIVE (LSF)
        # np.gradient keeps the array length the same. 
        # We use np.abs() so the peak points up regardless of edge direction.
        lsf = np.abs(np.gradient(smoothed_esf))
        
        # 4. COLOR MAPPING
        color_fraction = (target_x - x_start) / (x_end - x_start)
        line_color = cmap(color_fraction)
        
        # 5. PLOT IT
        ax.plot(y_pixels, lsf, color=line_color, linewidth=1.5, alpha=0.1)
    
    # --- Add a colorbar legend to the side ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=x_start, vmax=x_end))
    sm.set_array([]) 
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("X-Pixel Position")
    
    # 3. Add titles and show the final combined plot (OUTSIDE the loop)
    plt.title(f"Line Spread Functions (LSF) from X={x_start} to {x_end}")
    plt.xlabel("Y / pixels")
    plt.ylabel("LSF Amplitude (Derivative of Transmittance)")
    plt.grid(True, alpha=0.3)
    
    plt.show()
        
# ==============================================================================
# 4. MEASURE BLURRINESS ACROSS THE ANGLED PLATE
# ==============================================================================

def esf_model(y, I_min, I_max, mu, lam):
    exponent = np.clip((y - mu) / lam, -100, 100)
    return I_min + (I_max - I_min) / (np.exp(exponent) + 1.0)
    
def mtf_normalized(s, lam):
    u = np.pi * s * abs(lam)
    u = np.maximum(u, 1e-10) 
    return u / np.sinh(u)

def calculate_blurriness(lam):
    s_guess = 4.5 / (np.pi * abs(lam))
    s_01 = fsolve(lambda s: mtf_normalized(s, lam) - 0.1, s_guess)[0]
    return 1.0 / s_01

warnings.filterwarnings('ignore')

# Use the safe, clean range avoiding the extreme physical corners
x_start, x_end = 850, 1600
y_start, y_end = 750, 1000

valid_x_pixels = []
blurriness_values = []

print(f"Scanning across the angled edge from X={x_start} to {x_end}...")

for target_x in range(x_start, x_end):
    y_intensities = sample_median[y_start:y_end, target_x]
    y_pixels = np.arange(y_start, y_end)
    
    # Generate initial guesses
    I_min_guess = np.min(y_intensities)
    I_max_guess = np.max(y_intensities)
    mid_val = (I_max_guess + I_min_guess) / 2
    mu_guess = y_pixels[np.argmin(np.abs(y_intensities - mid_val))]
    lam_guess = -2.0  # Better starting guess based on your previous good fits
    
    p0 = [I_min_guess, I_max_guess, mu_guess, lam_guess]
    
    try:
        # Fit the curve
        popt, _ = curve_fit(esf_model, y_pixels, y_intensities, p0=p0, maxfev=1000)
        
        # Calculate blurriness
        fitted_lam = popt[3]
        u_tot = calculate_blurriness(fitted_lam)
        
        # We filter out massive mathematical failures (e.g. u_tot > 10 pixels) 
        # to prevent single bad pixels from ruining the graph
        if u_tot < 10.0:
            valid_x_pixels.append(target_x)
            blurriness_values.append(u_tot)
            
    except RuntimeError:
        continue

# Convert to arrays
x_data = np.array(valid_x_pixels)
u_data = np.array(blurriness_values)

# ==============================================================================
# 5. PLOT THE GRADIENT
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# 1. Plot the raw, individual column fits (will look like a noisy cloud)
ax.plot(x_data, u_data, 'o', color='purple', markersize=3, alpha=0.3, label='Individual Columns')

# 2. Add a strong Rolling Average (e.g., 50 pixels wide) to reveal the physical curve
window = 100
rolling_avg = np.convolve(u_data, np.ones(window)/window, mode='valid')
# Adjust X-axis for the rolling average shift
rolling_x = x_data[window//2 : -window//2 + 1]

# Make sure dimensions match before plotting the rolling average
min_len = min(len(rolling_x), len(rolling_avg))
ax.plot(rolling_x[:min_len], rolling_avg[:min_len], color='red', linewidth=3, label=f'Trend ({window}-px Rolling Avg)')

# 3. Add a Linear Regression Line to explicitly measure the gradient
z = np.polyfit(x_data[:len(x_data)//3], u_data[:len(x_data)//3], 1)
p = np.poly1d(z)
ax.plot(x_data, p(x_data), "k--", linewidth=2, label=f"Linear Fit (Slope: {z[0]:.7f})")

# Determine which side is closer based on your prompt (1650 is close, 850 is far)
# The slope will tell us exactly how the distance changes blurriness
if z[0] > 0:
    trend_text = "Blurriness INCREASES toward X=1600 (Closer to Camera)"
else:
    trend_text = "Blurriness DECREASES toward X=1600 (Closer to Camera)"

plt.title(f"Evolution of Edge Blurriness Across the Angled Plate\n{trend_text}")
plt.xlabel("X-Pixel Position (Proxy for Sample-Scintillator Distance)")
plt.ylabel("Total Blurriness ($u_{tot}$) in pixels")

# Zoom the Y-axis in so we can actually see the gradient curve
# (ignoring extreme mathematical outliers)
plt.ylim(0, np.percentile(u_data, 95) * 1.2) 

plt.legend()
plt.grid(True, alpha=0.4)
plt.show()


print("NEED LONGER MEASUREMENT AND SHARPER EDGE. did 5cm make it tilt more")