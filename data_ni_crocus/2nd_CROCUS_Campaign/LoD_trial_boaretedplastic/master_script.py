from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter1d

class ExperimentCatalog:
    def __init__(self, master_directory):
        """Scans the master directory and remembers folder locations."""
        self.master_path = Path(master_directory)
        self.folders = {folder.name: folder for folder in self.master_path.iterdir() if folder.is_dir()}
        print(f"Catalog initialized! Found {len(self.folders)} folders.")
        print(f"Available folders: {list(self.folders.keys())}\n")

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
            
        print(f"--- Loading Data for '{folder_name_string}' ---")
        folder_path = self.folders[folder_name_string]
        all_files = list(folder_path.rglob("*.fits"))
        
        if len(all_files) == 0:
            print(f"Warning: No .fits files found in {folder_name_string}")
            return None
            
        return self._load_to_tensor(all_files)
        
    def remove_outliers(self, image, radius, threshold):
        box_size = (2 * radius) + 1
        median_img = median_filter(image, size=box_size)
        img_float = image.astype(np.float32)
        median_float = median_img.astype(np.float32)
        difference = np.abs(img_float - median_float)
        outlier_mask = difference > threshold
        cleaned_image = np.where(outlier_mask, median_img, image)
        return cleaned_image.astype(image.dtype)

# ==============================================================================
# HELPER PLOTTING & BLUR FUNCTIONS
# ==============================================================================
def plot_image(image, title, vmin, vmax, cbar_label="Intensity / Counts"):
    fig, ax = plt.subplots(figsize=(7, 6))
    img_plot = plt.imshow(image, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)  
    cbar = plt.colorbar(img_plot, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    plt.title(title)
    plt.xlabel("X / pixels")
    plt.ylabel("Y / pixels")
    plt.xlim(350,1700)
    plt.ylim(350,1700)
    plt.show()

def extract_and_plot_edge(image, title, y_center, x_start, x_end, line_width=10):
    """
    Extracts a horizontal line profile across an edge (averaging over line_width rows for smoothness)
    and plots the image with a red dashed line indicating the extraction region.
    """
    # 1. Plot the image with the red dashed line
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    img_plot = ax1.imshow(image, cmap='gray', origin='lower', vmin=0.2, vmax=1.2)
    ax1.set_title(f"{title} - Edge Selection")
    ax1.set_xlabel("X / pixels")
    ax1.set_ylabel("Y / pixels")
    ax1.set_xlim(350, 1700)
    ax1.set_ylim(350, 1700)
    
    # Draw the red dashed line
    ax1.plot([x_start, x_end], [y_center, y_center], color='red', linestyle='--', linewidth=2)
    # Highlight the averaging thickness slightly
    ax1.axhspan(y_center - line_width//2, y_center + line_width//2, xmin=(x_start-350)/(1700-350), xmax=(x_end-350)/(1700-350), color='red', alpha=0.2)

    # 2. Extract the profile (average across a few rows to reduce noise)
    y_start = y_center - line_width // 2
    y_end = y_center + line_width // 2
    profile = np.mean(image[y_start:y_end, x_start:x_end], axis=0)
    
    # 3. Plot the 1D Profile
    x_axis = np.arange(x_start, x_end)
    ax2.plot(x_axis, profile, color='blue')
    ax2.set_title("1D Edge Spread Function (ESF)")
    ax2.set_xlabel("X position (pixels)")
    ax2.set_ylabel("Transmittance")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return profile

def calculate_blur(profile, pixel_size_mm):
    """
    Calculates blur by smoothing the profile, dynamically defining the
    0% and 100% plateaus from the edges of the extraction window,
    and finding the span where the transmittance is between 10% and 90%.
    """
    # 1. Smooth the profile to handle high-frequency noise spikes
    smoothed = gaussian_filter1d(profile, sigma=3)

    # 2. Define plateaus based on the boundaries of the extracted window.
    # Averaging the first and last 20 pixels avoids single-pixel noise spikes.
    left_plateau = np.mean(smoothed[:20])
    right_plateau = np.mean(smoothed[-20:])

    p_max = max(left_plateau, right_plateau)
    p_min = min(left_plateau, right_plateau)

    # 3. Normalize the smoothed profile between 0 and 1
    norm_profile = (smoothed - p_min) / (p_max - p_min)

    # 4. Find the contiguous region between 10% and 90%
    # This is direction-agnostic (works for both rising and falling edges)
    edge_region = np.where((norm_profile > 0.1) & (norm_profile < 0.9))[0]

    if len(edge_region) < 2:
        pixel_distance = 1 # Fallback for an absolute vertical step
    else:
        # Distance is the span between the 10% and 90% crossings
        pixel_distance = edge_region[-1] - edge_region[0]

    blur_mm = pixel_distance * pixel_size_mm
    return blur_mm

def process_transmittance(catalog, sample_folder, ob_folder, dc_folder, threshold=100, radius=4, show_plots=0):
    print(f"\n>>> Processing Transmittance for: {sample_folder} <<<")
    
    dc = catalog.get_data(dc_folder)
    sample_data = catalog.get_data(sample_folder)
    sample_corrected = sample_data - dc 
    sample_median = np.median(sample_corrected, axis=0)
    sample_clean = catalog.remove_outliers(sample_median, radius=radius, threshold=threshold)
    
    ob_data = catalog.get_data(ob_folder)
    ob_corrected = ob_data - dc 
    ob_median = np.median(ob_corrected, axis=0)
    ob_clean = catalog.remove_outliers(ob_median, radius=radius, threshold=threshold)
    
    transmittance = sample_clean / (ob_clean + 1e-6) 

    if show_plots == 1:
        plot_image(transmittance, title=f"Transmittance: {sample_folder}", vmin=0.2, vmax=1.3, cbar_label="Transmittance (-)")
        
    return transmittance



def calculate_ld_from_two_points(l1_mm, b_tot1_mm, l2_mm, b_tot2_mm):
    """
    Calculates L/D ratio and Intrinsic Blur using measurements from two distinct distances,
    solving the system: B_tot^2 = B_int^2 + (l * (D/L))^2
    """
    print("\n=======================================")
    print("CALCULATING L/D (TWO-POINT METHOD)")
    print("=======================================")
    
    # Square the inputs for the system of equations
    b1_sq = b_tot1_mm**2
    b2_sq = b_tot2_mm**2
    l1_sq = l1_mm**2
    l2_sq = l2_mm**2
    
    # Calculate R squared (where R = D/L)
    # Subtract equation 1 from equation 2 to eliminate B_int:
    # b2_sq - b1_sq = (l2_sq - l1_sq) * R^2
    r_sq = (b2_sq - b1_sq) / (l2_sq - l1_sq)
    
    # Calculate R (D/L) and invert for L/D
    R = np.sqrt(r_sq)
    ld_ratio = 1.0 / R
    
    # Substitute R back into the first equation to find B_int
    b_int_sq = b1_sq - (l1_sq * r_sq)
    
    # Safety check in case noise causes a negative square root
    if b_int_sq < 0:
        print("ERROR: Calculation resulted in negative intrinsic blur squared. Check data.")
        return None, None
        
    b_int = np.sqrt(b_int_sq)
    
    print(f"Data Point 1: Distance = {l1_mm} mm, Total Blur = {b_tot1_mm:.3f} mm")
    print(f"Data Point 2: Distance = {l2_mm} mm, Total Blur = {b_tot2_mm:.3f} mm")
    print("-" * 39)
    print(f"Calculated Intrinsic Blur (B_int): {b_int:.3f} mm")
    print(f">>> ESTIMATED L/D RATIO: {ld_ratio:.1f} <<<")
    
    return ld_ratio, b_int

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # --- USER DEFINED PARAMETERS FOR L/D ---
    PIXEL_SIZE_MM = 0.01059089605734767e1 # 1.025 microns
    

    
    master_dir = r"C:\Users\miche\Desktop\PhD\Neutron_Imaging_project\2nd_CROCUS_Campaign\LoD_trial_boaretedplastic"
    catalog = ExperimentCatalog(master_dir)
    
    experiment_list = [
        ("BoratedPlastic_scint", "OB_Li6_1", "DC"),       # Intrinsic Blur (Touching)
        ("BoratedPlastic_flightube", "OB_Li6_1", "DC")    # Total Blur (2.7m away)
    ]
    
    transmittance_results = {}
    
    for sample, ob, dc in experiment_list:
        print(f"\n=======================================")
        print(f"STARTING BATCH RUN FOR: {sample}")
        print(f"=======================================")
        
        result = process_transmittance(
            catalog=catalog, 
            sample_folder=sample, 
            ob_folder=ob, 
            dc_folder=dc, 
            threshold=70, 
            radius=4,
            show_plots=0
        )
        transmittance_results[sample] = result
        
    # --- L/D CALCULATION BLOCK ---
    print("\n=======================================")
    print("CALCULATING L/D RATIO")
    print("=======================================")
    
    # 1. Calculate Intrinsic Blur (from scintillator image)
    Y_CENTER = 1000       
    X_START = 1000         
    X_END = 1500          
    
    img_scint = transmittance_results["BoratedPlastic_scint"]
    prof_scint = extract_and_plot_edge(img_scint, "Intrinsic Blur (Scint)", Y_CENTER, X_START, X_END)
    Btot1 = calculate_blur(prof_scint, PIXEL_SIZE_MM)
    print(f"Intrinsic Blur (Btot1): {Btot1:.3f} mm")
    
    # 2. Calculate Total Blur (from flight tube image)
    Y_CENTER = 1200       
    X_START = 875         
    X_END = 1500 

    img_flight = transmittance_results["BoratedPlastic_flightube"]
    prof_flight = extract_and_plot_edge(img_flight, "Total Blur (Flight Tube)", Y_CENTER, X_START, X_END)
    Btot2 = calculate_blur(prof_flight, PIXEL_SIZE_MM)
    print(f"Total Blur (Btot2): {Btot2:.3f} mm")
    
    
    l1 = 50.0 #mm

    l2 = 2800.0 #mm

    
    calculate_ld_from_two_points(l1, Btot1, l2, Btot2)
