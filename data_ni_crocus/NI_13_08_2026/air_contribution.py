# -*- coding: utf-8 -*-
from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

# ==============================================================================
# CATALOG CLASS
# ==============================================================================
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
            print(f"Warning: No .fits files found in {folder_name_string}")
            return None
        return self._load_to_tensor(all_files)
        
    def remove_outliers(self, image, radius=1, threshold=150):
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
def get_roi_median(image, box_size=60):
    """Calculates the median pixel value in a central square zone."""
    h, w = image.shape
    cy, cx = h // 2, w // 2
    half_box = box_size // 2
    roi = image[cy - half_box : cy + half_box, cx - half_box : cx + half_box]
    return np.median(roi)

# ==============================================================================
# MAIN EXECUTABLE
# ==============================================================================
if __name__ == "__main__":
    master_dir = r"C:\Users\miche\Desktop\PhD\Neutron_Imaging_project\NI_13_08_2026" 
    catalog = ExperimentCatalog(master_dir)
    
    ROI_BOX_SIZE = 60 

    # 1. Load Dark Current
    print("\n--- 1. Loading Dark Current ---")
    dc_stack = catalog.get_data("DC_no_power")
    if dc_stack is None:
        raise ValueError("Critical Error: 'DC_no_power' folder not found!")
    
    dc_median = np.median(dc_stack, axis=0)

    # 2. Process old_OB & OB_BVSV
    print("\n--- 2. Processing old_OB and OB_BVSV ---")
    results = {}
    
    for folder in ["old_OB", "OB_BVSV"]:
        raw_stack = catalog.get_data(folder)
        
        # Z-axis median across measurements
        z_median = np.median(raw_stack, axis=0)
        
        # Dark Current Subtraction
        dc_subtracted = z_median - dc_median
        
        # Outlier Removal
        cleaned = catalog.remove_outliers(dc_subtracted, radius=1, threshold=150)
        
        # Median of central square ROI
        roi_val = get_roi_median(cleaned, box_size=ROI_BOX_SIZE)
        results[folder] = roi_val

    # --------------------------------------------------------------------------
    # 3. PRINT THE COMPARISON DIRECTLY IN TERMINAL
    # --------------------------------------------------------------------------
    old_ob = results["old_OB"]
    bvsv = results["OB_BVSV"]
    diff = bvsv - old_ob
    pct_change = (diff / old_ob) * 100
    transmission = (bvsv / old_ob) * 100

    print("\n" + "="*50)
    print("           COMPARISON: old_OB vs OB_BVSV           ")
    print("="*50)
    print(f" -> old_OB  (No Tubes, DC Subtracted) : {old_ob:.2f} counts")
    print(f" -> OB_BVSV (Both Vacuum, DC Subtracted) : {bvsv:.2f} counts")
    print("-" * 50)
    print(f" -> Net Count Difference (BVSV - old_OB) : {diff:+.2f} counts")
    print(f" -> Percentage Change                    : {pct_change:+.2f} %")
    print(f" -> Hardware Transmission Ratio          : {transmission:.2f} %")
    print("="*50 + "\n")

    # --------------------------------------------------------------------------
    # 4. PLOT COMPARISON
    # --------------------------------------------------------------------------
    categories = ['old_OB\n(No Tubes)', 'OB_BVSV\n(Both Vacuum)']
    values = [old_ob, bvsv]
    colors = ['#ff9e80', '#00e5ff']

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    bars = ax.bar(categories, values, color=colors, edgecolor='white', width=0.4, linewidth=1.2)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + (max(values) * 0.02),
                f"{val:.1f} cts", ha='center', va='bottom', color='white', fontweight='bold', fontsize=10)

    ax.set_ylabel(f'Dark-Subtracted Median Counts (ROI {ROI_BOX_SIZE}x{ROI_BOX_SIZE} px)', color='white', fontsize=11)
    ax.set_title(f'OB Comparison\n(Net Transmission: {transmission:.1f}% | Difference: {pct_change:+.1f}%)',
                 color='white', fontsize=12, pad=15)
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(True, axis='y', linestyle='--', color='#444444', alpha=0.5)
    ax.set_axisbelow(True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')

    plt.tight_layout()
    plt.savefig("OB_vs_BVSV_Comparison.png", dpi=300, transparent=True)
    plt.show()