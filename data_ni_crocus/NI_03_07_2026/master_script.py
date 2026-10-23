from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
import bm3d


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



COMBINE_METHOD = "median"
 
 
def combine(stack, method=COMBINE_METHOD, nsigma=4.0, iters=2, trim=3):
    """Combine a (n_frames, H, W) stack into a single (H, W) image.
 
    mean               : lowest noise, but gamma spots leak through
    median             : robust, ~25% noisier than mean
    sigma_clipped_mean : rejects gamma spots per pixel, then means the rest
                         (recommended: median-level robustness, mean-level noise)
    trimmed_mean       : drops `trim` highest/lowest values per pixel, means rest
                         (simple, fast, nearly as good)
    """
    stack = np.asarray(stack, dtype=np.float32)
 
    if method == "mean":
        return stack.mean(axis=0)
 
    if method == "median":
        return np.median(stack, axis=0)
 
    if method == "trimmed_mean":
        s = np.sort(stack, axis=0)
        return s[trim:-trim].mean(axis=0)
 
    if method == "sigma_clipped_mean":
        med = np.median(stack, axis=0)          # robust center, iter 0
        mad = np.median(np.abs(stack - med), axis=0)
        sigma = 1.4826 * mad
        # floor sigma so near-constant/discrete pixels don't over-clip
        sigma = np.maximum(sigma, 1e-3 * np.abs(med) + 1e-6)
        mask = np.abs(stack - med) < nsigma * sigma
        for _ in range(iters - 1):              # refine center on survivors
            med = np.nanmedian(np.where(mask, stack, np.nan), axis=0)
            mask = np.abs(stack - med) < nsigma * sigma
        rej = (~mask).sum()
        print(f"    sigma-clip: rejected {rej} samples "
              f"({100 * rej / mask.size:.4f}%)")
        n_keep = mask.sum(axis=0)
        out = np.where(mask, stack, 0).sum(axis=0) / np.maximum(n_keep, 1)
        return np.where(n_keep > 0, out, med)   # fallback: median
 
    raise ValueError(f"unknown combine method: {method}")
 
 
# =====================================================================
# Your block, with the combiner swapped in
# =====================================================================


# ==============================================================================
# HELPER PLOTTING FUNCTION
# ==============================================================================
def plot_image(image, title, vmin, vmax, cbar_label="Intensity / Counts"):
    """A helper function to cleanly plot images without copy-pasting code."""
    fig, ax = plt.subplots(figsize=(7, 6))
    
    img_plot = plt.imshow(image, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)  
    cbar = plt.colorbar(img_plot, ax=ax, fraction=0.046, pad=0.04)
    
    cbar.set_label(cbar_label)
    plt.title(title)
    plt.xlabel("X / pixels")
    plt.ylabel("Y / pixels")
    
    # Uncomment these if you need them later:
    # plt.axhline(y = mid_y, color="blue", linestyle = "--", alpha = 0.2)
    # plt.axvline(x = mid_x, color="red", linestyle = "--", alpha = 0.2)
    plt.xlim(350,1700)
    plt.ylim(350,1700)
    plt.grid(visible=False)
    plt.show()
    
def plot_image_save(image, title, vmin, vmax, cbar_label="Intensity / Counts",folder_name="name1"):
    """A helper function to cleanly plot images without copy-pasting code."""
    fig, ax = plt.subplots(figsize=(7, 6))
    
    img_plot = plt.imshow(image, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)  
    cbar = plt.colorbar(img_plot, ax=ax, fraction=0.046, pad=0.04)
    
    cbar.set_label(cbar_label)
    plt.title(title)
    plt.xlabel("X / pixels")
    plt.ylabel("Y / pixels")
    
    # Uncomment these if you need them later:
    # plt.axhline(y = mid_y, color="blue", linestyle = "--", alpha = 0.2)
    # plt.axvline(x = mid_x, color="red", linestyle = "--", alpha = 0.2)
    plt.xlim(350,1700)
    plt.ylim(350,1700)
    plt.grid(visible=False)
    
    plt.savefig(f"processed_images/{folder_name}.png", dpi=350)

def process_transmittance(catalog, sample_folder, ob_folder, dc_folder, threshold=100, radius=4, show_plots=0, option = 0):
    """
    Automates the entire Flat-Field Correction process.
    Takes the folder names, applies dark field correction, median stacking, 
    outlier removal, and computes transmittance.
    """
    print(f"\n>>> Processing Transmittance for: {sample_folder} <<<")
    
    dc = catalog.get_data(dc_folder)
    binn = np.arange(500,1000,1)
    counts, edges = np.histogram(dc.ravel(), bins=binn)
    plt.stairs(counts, edges)
    # plt.yscale("log")            # <- essential
    plt.xlabel("dark counts"); plt.ylabel("pixels")
    plt.show()
    
    dc_median = combine(dc, method=COMBINE_METHOD)
    counts, edges = np.histogram(dc_median.ravel(), bins=binn)
    plt.title("dc_median")
    plt.stairs(counts, edges)
    # plt.yscale("log")            # <- essential
    plt.xlabel("dark counts"); plt.ylabel("pixels")
    plt.show()
    
     
    # 2. Process Sample
    sample_data = catalog.get_data(sample_folder)
    sample_corrected = sample_data - dc
    print(sample_corrected.shape)
    sample_clean = combine(sample_corrected, method=COMBINE_METHOD)
    # sample_clean = catalog.remove_outliers(sample_clean, radius=radius, threshold=threshold)
     
    # 3. Process Open Beam (OB)
    ob_data = catalog.get_data(ob_folder)
    ob_corrected = ob_data - dc
    print(ob_corrected.shape)
    ob_clean = combine(ob_corrected, method=COMBINE_METHOD)
    # ob_clean = catalog.remove_outliers(ob_clean, radius=radius, threshold=threshold)
 

    
    # if show_plots == 1:
    #     plot_image(ob_clean, title=f"Open Beam: {ob_folder} (Dark Corrected)", vmin=10, vmax=300)

    # 4. Calculate Transmittance
    # Adding 1e-6 to prevent division by zero errors
    transmittance = sample_clean / (ob_clean + 1e-6) 

    if show_plots == 1:
        plot_image(transmittance, title=f"Transmittance: {sample_folder} + {COMBINE_METHOD}", vmin=0.2, vmax=1.3, cbar_label="Transmittance (-)")

        # plot_image_save(transmittance, title=f"Transmittance: {sample_folder}", vmin=0.2, vmax=1.3, cbar_label="Transmittance (-)")
    if show_plots == 2:
        # plot_image(transmittance, title=f"Transmittance: {sample_folder}", vmin=0.2, vmax=1.3, cbar_label="Transmittance (-)")
        plot_image_save(transmittance, title=f"Transmittance: {sample_folder}", vmin=0.2, vmax=1.3, cbar_label="Transmittance (-)", folder_name = sample_folder)
        
         

    return transmittance

# ----------------------------------------------------------------------

## MAIN FUNCTION ###

if __name__ == "__main__":
    # 1. Initialize the catalog ONCE
    master_dir = r"C:\Users\miche\Desktop\PhD\Neutron_Imaging_project\NI_03_07_2026"
    catalog = ExperimentCatalog(master_dir)
    
    experiment_list = [
        ("water_samples", "OB", "closed_shutter_DC_with_beam_dupplicated"),
        ("PE_Steps", "OB", "closed_shutter_DC_with_beam_dupplicated")
    ]
    
    transmittance_results = {}
    
# 3. Loop through the list!
    for sample, ob, dc in experiment_list:
        print(f"\n=======================================")
        print(f"STARTING BATCH RUN FOR: {sample}")
        print(f"=======================================")
        
        # Run the function using the variables from the loop
        # Note: We set show_plots=False so the code doesn't pause and wait 
        # for you to close the window for every single image!
        result = process_transmittance(
            catalog=catalog, 
            sample_folder=sample, 
            ob_folder=ob, 
            dc_folder=dc, 
            threshold=70, 
            radius=2,
            show_plots=1
        )
    
    import pickle
    with open("result_pe_steps.pkl", "wb") as f:
        pickle.dump({"result": result}, f)
    print("saved", result.shape, result.dtype)
