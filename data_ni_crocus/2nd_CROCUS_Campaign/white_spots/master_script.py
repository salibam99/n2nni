from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

class ExperimentCatalog:
    def __init__(self, master_directory):
        self.master_path = Path(master_directory)
        self.folders = {folder.name: folder for folder in self.master_path.iterdir() if folder.is_dir()}
        print(f"Catalog initialized! Found {len(self.folders)} folders.")
        
    def _load_to_tensor(self, file_paths):
        if not file_paths:
            return None
        list_img = []
        for path in file_paths:
            with fits.open(path) as hdul:
                image_data = hdul[0].data
                datt = image_data[0,:,:] if image_data.ndim == 3 else image_data
                list_img.append(datt)
        return np.array(list_img)

    def get_data(self, folder_name_string):
        if folder_name_string not in self.folders:
            raise ValueError(f"Folder '{folder_name_string}' not found in catalog!")
        folder_path = self.folders[folder_name_string]
        all_files = list(folder_path.rglob("*.fits"))
        if len(all_files) == 0:
            return None
        return self._load_to_tensor(all_files)

# ==============================================================================
# 1. RAW ARRAY VERIFICATION (SPYDER OPTIMIZED)
# ==============================================================================
def inspect_raw_array(catalog, folder_name, frame_index=0, threshold=15000):
    """
    Pulls a single raw image and plots it. 
    RETURNS the array so you can double-click it in Spyder's Variable Explorer.
    """
    print(f"\n--- Inspecting Raw Array: {folder_name} (Frame {frame_index}) ---")
    images = catalog.get_data(folder_name)
    
    if images is None or frame_index >= len(images):
        print("Invalid folder or frame index.")
        return None, None
        
    img = images[frame_index]
    
    # 1. Print stats to console
    print(f"Array Shape: {img.shape}")
    print(f"Min Value: {np.min(img)}, Max Value: {np.max(img)}")
    
    spots_mask = img > threshold
    actual_count = np.sum(spots_mask)
    print(f">>> Pixels above {threshold}: {actual_count} <<<")

    # 2. Visual Verification Plot
    fig, ax = plt.subplots(figsize=(8, 7))
    # 'vmax' is clamped so the background stays dark and the spots glow brightly
    im = ax.imshow(img, cmap='magma', origin='lower', vmin=np.percentile(img, 5), vmax=np.percentile(img, 98))
    
    # Overlay red dots where the algorithm detected a spot
    y_coords, x_coords = np.where(spots_mask)
    ax.scatter(x_coords, y_coords, color='red', s=10, alpha=0.7, label=f'Detected > {threshold}')
    
    plt.colorbar(im, ax=ax, label="Intensity")
    plt.title(f"Raw Array Verification: {folder_name} (Frame {frame_index})")
    plt.legend(loc='upper right')
    plt.show()
    
    # Returning these allows Spyder to catch them in the Variable Explorer
    return img, spots_mask

# ==============================================================================
# 2. SPATIAL GRADIENT ANALYSIS
# ==============================================================================
def plot_spatial_gradients(catalog, folder_name, threshold=15000):
    """
    Sums the white spot hits across all frames and projects them onto
    the X and Y axes to check for spatial gradients/hotspots.
    """
    print(f"\n--- Checking Spatial Gradients for '{folder_name}' ---")
    images = catalog.get_data(folder_name)
    if images is None: return

    # Master boolean mask for ALL frames
    master_mask = images > threshold
    
    # Sum across all frames to get a 2D map of hit frequencies
    hit_map_2d = np.sum(master_mask, axis=0) 
    
    # Project (sum) the 2D hits onto the 1D axes
    x_profile = np.sum(hit_map_2d, axis=0) # Sum vertically
    y_profile = np.sum(hit_map_2d, axis=1) # Sum horizontally
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Apply a slight smooth to see the trend through the noise
    smooth_x = np.convolve(x_profile, np.ones(50)/50, mode='same')
    smooth_y = np.convolve(y_profile, np.ones(50)/50, mode='same')
    
    ax1.plot(x_profile, alpha=0.3, color='blue', label='Raw Hits')
    ax1.plot(smooth_x, color='darkblue', linewidth=2, label='Smoothed Trend')
    ax1.set_title(f"X-Axis Gradient (Horizontal)\n{folder_name}")
    ax1.set_xlabel("X Pixel Coordinate")
    ax1.set_ylabel("Total Gammas Detected")
    ax1.legend()
    ax1.grid(True, alpha=0.5)
    
    ax2.plot(y_profile, alpha=0.3, color='green', label='Raw Hits')
    ax2.plot(smooth_y, color='darkgreen', linewidth=2, label='Smoothed Trend')
    ax2.set_title(f"Y-Axis Gradient (Vertical)\n{folder_name}")
    ax2.set_xlabel("Y Pixel Coordinate")
    ax2.set_ylabel("Total Gammas Detected")
    ax2.legend()
    ax2.grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.show()
def analyze_gamma_spots(catalog, folder_name, threshold=64000):
    """
    Loads all images in a folder, counts pixels above the threshold for each,
    and calculates the cumulative sum to check for linearity.
    """
    print(f"\n--- Loading Data for '{folder_name}' ---")
    images = catalog.get_data(folder_name)
    
    if images is None:
        return None, None
    
    num_images = images.shape[0]
    counts_per_image = []
    
    print(f"Analyzing {num_images} images for pixels > {threshold}...")
    
    for i in range(num_images):
        # Create a boolean mask where True represents a pixel above threshold
        white_spots_mask = images[i] > threshold
        
        # np.sum on a boolean array counts the True values
        spot_count = np.sum(white_spots_mask)
        counts_per_image.append(spot_count)
        print(f"  Frame {i+1}: {spot_count} spots")
        
    # Calculate the cumulative sum to check for linearity
    cumulative_counts = np.cumsum(counts_per_image)
    print(f"  Cumulative Total: {list(cumulative_counts)}")
    
    return counts_per_image, cumulative_counts

def plot_linearity(folder_names, all_cumulative_counts):
    """
    Plots the cumulative white spot counts to visualize if they scale linearly.
    """
    plt.figure(figsize=(9, 6))
    
    for folder, cum_counts in zip(folder_names, all_cumulative_counts):
        # X-axis represents the frame sequence (proxy for time)
        frames = np.arange(1, len(cum_counts) + 1)
        plt.errorbar(frames, cum_counts, yerr= 2*np.sqrt(cum_counts), marker='o', linestyle='-', linewidth=2, label=folder)
        
    plt.title("Linearity of Gamma Ray Hits Over Time")
    plt.xlabel("Acquisition Frame Number")
    plt.ylabel("Cumulative White Spots Detected")
    # Force X-axis to only show whole numbers for frames
    plt.xticks(np.arange(1, max(len(c) for c in all_cumulative_counts) + 1))
    plt.legend(fontsize = "10", loc = "best")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    master_dir = r"C:\Users\miche\Desktop\PhD\Neutron_Imaging_project\2nd_CROCUS_Campaign\white_spots"
    catalog = ExperimentCatalog(master_dir)
    SPOT_THRESHOLD = 2000
    
    # 1. VERIFY A RAW ARRAY
    # Assign the output to variables so they show up in Spyder's Variable Explorer!
    raw_img, mask_img = inspect_raw_array(catalog, "BoratedPlastic_scint", frame_index=0, threshold=SPOT_THRESHOLD)
    
    # 2. CHECK FOR GRADIENTS
    # This will generate the X and Y plots for the selected folder
    plot_spatial_gradients(catalog, "BoratedPlastic_scint", threshold=SPOT_THRESHOLD)
    
    
    # List the folders from your screenshot you want to compare
    folders_to_test = [
        # "Cawo", 
        # "Cawo_borated_plastic_20cm", 
        # "Gd-Gamma", 
        # "Gd-Gamma_borated_plastic_15cm",
        
        "OB_Li6_1",
        "BoratedPlastic_scint",

        "BoratedPlastic_BlockingBeam5cm",
        "BoratedPlastic_flightube",
        # "mat1"
    ]
    
    all_cum_counts = []
    valid_folders = []
    
    print("\n=======================================")
    print("STARTING WHITE SPOT ANALYSIS")
    print("=======================================")
    
    for folder in folders_to_test:
        if folder in catalog.folders:
            counts, cum_counts = analyze_gamma_spots(catalog, folder, SPOT_THRESHOLD)
            if cum_counts is not None:
                valid_folders.append(folder)
                all_cum_counts.append(cum_counts)
        else:
            print(f"\nSkipping '{folder}': Not found in directory.")
            
    if valid_folders:
        plot_linearity(valid_folders, all_cum_counts)
        
        