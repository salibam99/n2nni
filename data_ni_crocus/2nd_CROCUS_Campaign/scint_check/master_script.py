from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.ndimage import median_filter

# ==============================================================================
# CATALOG CLASS (Kept exactly as you wrote it!)
# ==============================================================================
class ExperimentCatalog:
    def __init__(self, master_directory):
        self.master_path = Path(master_directory)
        self.folders = {folder.name: folder for folder in self.master_path.iterdir() if folder.is_dir()}
        print(f"Catalog initialized! Found {len(self.folders)} folders.")

    def _load_to_tensor(self, file_paths):
        if not file_paths: return None
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
# HELPER FUNCTIONS
# ==============================================================================
def get_roi_average(image, box_size=100):
    """Returns the average pixel value of a square in the exact center."""
    h, w = image.shape
    cy, cx = h // 2, w // 2
    half_box = box_size // 2
    roi = image[cy - half_box : cy + half_box, cx - half_box : cx + half_box]
    return np.mean(roi)

# ==============================================================================
# MAIN PROCESSING & PLOTTING
# ==============================================================================
if __name__ == "__main__":
    # 1. Setup Master Directory
    # UPDATE THIS to the path where the folders in your screenshot are located
    master_dir = r"C:\Users\miche\Desktop\PhD\Neutron_Imaging_project\2nd_CROCUS_Campaign\scint_check" 
    catalog = ExperimentCatalog(master_dir)
    
    # 2. Define the Scintillators and their Exposure Times
    # FORMAT: "Folder_Name": {"time": exposure_in_seconds, "type": "neutron" or "gamma"}
    scintillator_config = {
    "Li6_rctritec":    {"time": 300, "type": "neutron", "label": "LiF/ZnS(Cu) (300um) (RCTRITEC)"}, 
    "Li6_Eljen": {"time": 300, "type": "neutron", "label": "LiF/ZnS(Ag) (320um) (ELJEN)"},
    "Gadox":        {"time": 300, "type": "neutron", "label": "Gd2S2:Tb (30um)"},
    "LuO2":         {"time": 300, "type": "gamma",   "label": "Lu2SiO5:Ce (300um)"},
    "Cawo":         {"time": 300, "type": "gamma",   "label": "CaWO (200um)"},    
    "Gd-Gamma":     {"time": 300, "type": "gamma",   "label": "Gd2O2S:Pr,Ce,F (200um)"}
    }
    
    # Process Dark Current ONCE
    print("\n--- Processing Master Dark Current ---")
    dc_data = catalog.get_data("DC") # Assuming "DC" is your master dark current folder
    dc_median = np.median(dc_data, axis=0)
    
    # 3. Process each scintillator
    labels = []
    rates = []
    types = []
    
    for folder, config in scintillator_config.items():
        print(f"Processing {folder}...")
        
        # Load and stack
        raw_data = catalog.get_data(folder)
        img_median = np.median(raw_data, axis=0)
        
        # Dark subtract & clean
        corrected = img_median - dc_median
        corrected[corrected <= 0] = 1 # Remove negatives
        cleaned = catalog.remove_outliers(corrected, radius=1, threshold=150) # Tweak threshold if needed
        
        # Get count rate
        avg_counts = get_roi_average(cleaned, box_size=150)
        count_rate = avg_counts / config["time"]
        
        # === CHANGED THIS LINE ===
        labels.append(config["label"]) 
        # =========================
        
        rates.append(count_rate)
        types.append(config["type"])

    # ==========================================
    # PLOTTING
    # ==========================================
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # The color variable you can freely choose
    bar_color = '#00ffff' # Cyan
    
    y_positions = np.arange(len(labels))
    
    # Create the bars
    bars = ax.barh(y_positions, rates, color=bar_color, height=0.6, edgecolor='white', linewidth=1)
    
    # Apply hatching based on type
    for i, bar in enumerate(bars):
            # 1. Apply hatching for gamma
            if types[i] == "gamma":
                bar.set_hatch('///') 
                
            # 2. Format the number to 3 decimal places (e.g., 0.456)
            # (If your numbers get REALLY tiny, you can change this to :.3e for scientific notation)
            value_text = f"{rates[i]:.3f}" 
            
            # 3. Place the text! 
            # MULTIPLY by ~1.15 or 1.2 to create a clean, uniform gap on a log scale
            ax.text(bar.get_width() * 1.05, 
                    bar.get_y() + bar.get_height() / 2, 
                    value_text, 
                    va='center', ha='left', color='white', fontsize=11, fontweight='bold')
    
    # Styling Axes
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, color='#cccccc', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    
    # You asked for a log scale possibility previously. 
    # Uncomment the next line if you want the X axis to be logarithmic:
    ax.set_xscale('log') 
    
    ax.set_xlabel('Counts per pixel per second', color='#ffffff', fontsize=12)
    ax.tick_params(axis='x', colors='#cccccc')
    
    # Grid and Spines
    ax.grid(True, axis='x', which='both', linestyle='--', color='#444444', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#ffffff')
    
    # Custom Legend
    neutron_patch = mpatches.Patch(facecolor=bar_color, edgecolor='white', label='Neutron Sensitive')
    gamma_patch = mpatches.Patch(facecolor=bar_color, edgecolor='white', hatch='///', label='Gamma Sensitive')
    # ax.legend(handles=[neutron_patch, gamma_patch], frameon=False, labelcolor='white', loc='best')
    
    plt.tight_layout()
    plt.savefig("Scintillator_Comparison_Yield.png", transparent=True, dpi=350)
    plt.show()