# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:38:31 2026

@author: miche
"""

from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
import warnings
import pandas as pd

# FIXED: Initialized as a list [] to store dictionary records
extracted_sigmas = []

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


# ==============================================================================
# HELPER PLOTTING FUNCTION
# ==============================================================================
def plot_image(image, title, vmin, vmax, cbar_label="Intensity / Counts"):
    """A helper function to cleanly plot images without copy-pasting code."""
    fig, ax = plt.subplots(figsize=(7, 6))
    
    img_plot = plt.imshow(image, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)  
    cbar = plt.colorbar(img_plot, ax=ax, fraction=0.046, pad=0.04)
    
    cbar.set_label(cbar_label)
    plt.xlabel("X / pixels")
    plt.ylabel("Y / pixels")
    
    plt.xlim(350,1700)
    plt.ylim(350,1700)
    
    plt.show()
    
def plot_image_save(image, title, vmin, vmax, cbar_label="Intensity / Counts",folder_name="name1"):
    """A helper function to cleanly plot images without copy-pasting code."""
    fig, ax = plt.subplots(figsize=(7, 6))
    
    img_plot = plt.imshow(image, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)  
    cbar = plt.colorbar(img_plot, ax=ax, fraction=0.046, pad=0.04)
    
    cbar.set_label(cbar_label)
    plt.xlabel("X / pixels")
    plt.ylabel("Y / pixels")
    
    plt.xlim(350,1700)
    plt.ylim(350,1700)
    
    plt.savefig(f"processed_images/{folder_name}.png", dpi=350)

def process_transmittance(catalog, sample_folder, ob_folder, dc_folder, threshold=100, radius=4, show_plots=0):
    """
    Automates the entire Flat-Field Correction process.
    Takes the folder names, applies dark field correction, median stacking, 
    outlier removal, and computes transmittance.
    """
    print(f"\n>>> Processing Transmittance for: {sample_folder} <<<")
    
    # 1. Get Dark Current (DC)
    dc = catalog.get_data(dc_folder)

    # 2. Process Sample
    sample_data = catalog.get_data(sample_folder)
    sample_corrected = sample_data - dc 
    sample_median = np.median(sample_corrected, axis=0)
    sample_clean = catalog.remove_outliers(sample_median, radius=radius, threshold=threshold)
    
    if show_plots == 1:
        plot_image(sample_clean, title=f"Sample: {sample_folder} (Dark Corrected)", vmin=10, vmax=300)

    # 3. Process Open Beam (OB)
    ob_data = catalog.get_data(ob_folder)
    ob_corrected = ob_data - dc 
    ob_median = np.median(ob_corrected, axis=0)
    ob_clean = catalog.remove_outliers(ob_median, radius=radius, threshold=threshold)
    
    if show_plots == 1:
        plot_image(ob_clean, title=f"Open Beam: {ob_folder} (Dark Corrected)", vmin=10, vmax=300)

    # 4. Calculate Transmittance
    # Adding 1e-6 to prevent division by zero errors
    transmittance = sample_clean / (ob_clean + 1e-6) 

    if show_plots == 1:
        plot_image(transmittance, title=f"Transmittance: {sample_folder}", vmin=0.2, vmax=1.3, cbar_label="Transmittance (-)")
    if show_plots == 2:
        plot_image_save(transmittance, title=f"Transmittance: {sample_folder}", vmin=0.2, vmax=1.3, cbar_label="Transmittance (-)", folder_name = sample_folder)
         

    return transmittance

## MAIN FUNCTION ###

if __name__ == "__main__":
    # 1. Initialize the catalog ONCE
    master_dir = r"C:\Users\miche\Desktop\PhD\Neutron_Imaging_project\2nd_CROCUS_Campaign\blackbody"
    catalog = ExperimentCatalog(master_dir)
    
    experiment_list = [
        ("SampleSet1_0cm", "OB", "DC2"),
        ("SampleSet2_0cm", "OB", "DC2")
    ]
    
    transmittance_results = {}
    ii = 0
    # Loop through the list!
    for sample, ob, dc in experiment_list:
        print(f"\n=======================================")
        print(f"STARTING BATCH RUN FOR: {sample}")
        print(f"=======================================")
        
        result = process_transmittance(
            catalog=catalog, 
            sample_folder=sample, 
            ob_folder=ob, 
            dc_folder=dc, 
            threshold=100, 
            radius=3,
            show_plots=1
        )
        
        transmittance_results[sample] = result
    
        if (ii == 0):
            transmittance_img = result
        else:
            transmittance_img2 = result
            
        ii = ii + 1

    # ==============================================================================
    # 1. DEFINE THE CYLINDER TRANSMISSION MODEL
    # ==============================================================================
    def cylinder_transmittance(x, T0, Sigma, R, x0):
        """Theoretical transmittance through a solid cylinder."""
        inside_sqrt = R**2 - (x - x0)**2
        inside_sqrt = np.where(inside_sqrt > 0, inside_sqrt, 0)
        path_length = 2 * np.sqrt(inside_sqrt)
        return T0 * np.exp(-Sigma * path_length)

    # ==============================================================================
    # 2a. EXTRACT AND FIT PROFILES (SET 1)
    # ==============================================================================
    y_start, y_end = 1150, 1600
    cylinders_set1 = {
        "SS316L": (460, 600),
        "Zn": (650, 780),
        "Al": (870, 1090),
        "Ta": (1120, 1300),
        "C": (1360, 1500)
    }
    
    warnings.filterwarnings('ignore') 
    
    fig1, axes1 = plt.subplots(1, len(cylinders_set1), figsize=(18, 4), sharey=True)
    print("Extracting profiles and fitting Set 1 cylinders...\n")
    
    for idx, (name, (x_min, x_max)) in enumerate(cylinders_set1.items()):
        ax = axes1[idx]
        roi = transmittance_img[y_start:y_end, x_min:x_max]
        median_profile = np.median(roi, axis=0)
        x_pixels = np.arange(x_min, x_max)
        
        T0_guess = np.max(median_profile)       
        x0_guess = x_min + (x_max - x_min) / 2  
        R_guess = (x_max - x_min) / 2.5         
        Sigma_guess = 0.05                      
        p0 = [T0_guess, Sigma_guess, R_guess, x0_guess]
        
        ax.plot(x_pixels, median_profile, 'o', color='gray', alpha=0.6, markersize=4, label='Median Profile')
        
        try:
            bounds = ([0.1, 0.0001, 10, x_min], [1.5, 1.0, 100, x_max])
            popt, pcov = curve_fit(cylinder_transmittance, x_pixels, median_profile, p0=p0, bounds=bounds)
            T0_fit, Sigma_fit, R_fit, x0_fit = popt
            
            fit_curve = cylinder_transmittance(x_pixels, *popt)
            ax.plot(x_pixels, fit_curve, 'r-', linewidth=2.5, label='Cylinder Fit')
            
            mean_free_path = 1.0 / Sigma_fit
            physical_radius_cm = 0.5  
            pixel_size_cm = physical_radius_cm / R_fit
            mfp_cm = mean_free_path * pixel_size_cm
            sigma_cm = Sigma_fit / pixel_size_cm
            
            # --- UNCERTAINTY PROPAGATION FROM PCOV ---
            var_Sigma = pcov[1, 1]
            var_R = pcov[2, 2]
            cov_Sigma_R = pcov[1, 2]
            
            var_prod = (R_fit**2 * var_Sigma) + (Sigma_fit**2 * var_R) + (2 * R_fit * Sigma_fit * cov_Sigma_R)
            dsigma_cm = np.sqrt(max(0, var_prod)) / physical_radius_cm
            
            # Save structured record to list
            extracted_sigmas.append({
                "Sample_ID": name,
                "Exp_Mean": sigma_cm,
                "Exp_Std": dsigma_cm,
                "MFP_cm": mfp_cm,
                "Pixel_Size_cm": pixel_size_cm,
                "R_fit_px": R_fit
            })
            
            title = (f"{name}\n"
                     f"$\\Sigma$: {sigma_cm:.3f} $\\pm$ {dsigma_cm:.3f} cm⁻¹\n"
                     f"MFP: {mfp_cm:.2f} cm")
            ax.set_title(title)
            
        except RuntimeError:
            ax.set_title(f"{name}\nFIT FAILED")
            
        ax.set_xlabel("X / pixels")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.set_ylabel("Transmittance (-)")
            ax.legend()
            
    plt.tight_layout()
    plt.show()

    # ==============================================================================
    # 2b. EXTRACT AND FIT PROFILES (SET 2)
    # ==============================================================================
    cylinders_set2 = {
        "Nb": (460, 580),
        "Mo": (675, 820),
        "V": (870, 1000),
        "W": (1100, 1225),
        "Au": (1300, 1450)
    }
    
    fig2, axes2 = plt.subplots(1, len(cylinders_set2), figsize=(18, 4), sharey=True)
    print("Extracting profiles and fitting Set 2 cylinders...\n")
    
    for idx, (name, (x_min, x_max)) in enumerate(cylinders_set2.items()):
        ax = axes2[idx]
        
        if name == "Au":
            roi = transmittance_img2[2048-975:2048-905, x_min:x_max]
        else:
            roi = transmittance_img2[y_start:y_end, x_min:x_max]
            
        median_profile = np.median(roi, axis=0)
        x_pixels = np.arange(x_min, x_max)
        
        T0_guess = np.max(median_profile)
        x0_guess = x_min + (x_max - x_min) / 2  
        R_guess = (x_max - x_min) / 2.5         
        Sigma_guess = 0.05                      
        p0 = [T0_guess, Sigma_guess, R_guess, x0_guess]
        
        ax.plot(x_pixels, median_profile, 'o', color='gray', alpha=0.6, markersize=4, label='Median Profile')
        
        try:
            bounds = ([0.1, 0.0001, 10, x_min], [1.5, 1.0, 100, x_max])
            popt, pcov = curve_fit(cylinder_transmittance, x_pixels, median_profile, p0=p0, bounds=bounds)
            T0_fit, Sigma_fit, R_fit, x0_fit = popt
            
            fit_curve = cylinder_transmittance(x_pixels, *popt)
            ax.plot(x_pixels, fit_curve, 'c-', linewidth=2.5, label='Cylinder Fit')
            
            mean_free_path = 1.0 / Sigma_fit
            
            if name == "Y":
                physical_radius_cm = 0.4  
            else:
                physical_radius_cm = 0.5
                
            pixel_size_cm = physical_radius_cm / R_fit
            mfp_cm = mean_free_path * pixel_size_cm
            sigma_cm = Sigma_fit / pixel_size_cm
            
            # --- UNCERTAINTY PROPAGATION FROM PCOV ---
            var_Sigma = pcov[1, 1]
            var_R = pcov[2, 2]
            cov_Sigma_R = pcov[1, 2]
            
            var_prod = (R_fit**2 * var_Sigma) + (Sigma_fit**2 * var_R) + (2 * R_fit * Sigma_fit * cov_Sigma_R)
            dsigma_cm = np.sqrt(max(0, var_prod)) / physical_radius_cm
            
            # Save structured record to list
            extracted_sigmas.append({
                "Sample_ID": name,
                "Exp_Mean": sigma_cm,
                "Exp_Std": dsigma_cm,
                "MFP_cm": mfp_cm,
                "Pixel_Size_cm": pixel_size_cm,
                "R_fit_px": R_fit
            })
            
            title = (f"{name}\n"
                     f"$\\Sigma$: {sigma_cm:.3f} $\\pm$ {dsigma_cm:.3f} cm⁻¹")
            ax.set_title(title)
            
        except RuntimeError:
            ax.set_title(f"{name}\nFIT FAILED")
            
        ax.set_xlabel("X / pixels")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.set_ylabel("Transmittance (-)")
            ax.legend()
            
    plt.tight_layout()
    plt.show()

    # ==============================================================================
    # 3. GENERATE FINAL COMBINED BAR PLOT
    # ==============================================================================
    print("\nGenerating final cross-section bar plot for all cylinders...")
    
    # FIXED: Extract materials and sigmas from list of dicts
    plot_materials = [rec["Sample_ID"] for rec in extracted_sigmas]
    plot_sigmas = [rec["Exp_Mean"] for rec in extracted_sigmas]
    plot_errors = [rec["Exp_Std"] for rec in extracted_sigmas]

    plt.style.use('dark_background')
    fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
    fig_bar.patch.set_facecolor('black')
    ax_bar.set_facecolor('black')

    COLOR_CROCUS = '#00ffff'  # Pure Cyan
    x_pos = np.arange(len(plot_materials))
    bar_width = 0.5  

    # Plot single bars with error bars for CROCUS
    rects_bar = ax_bar.bar(
        x_pos, plot_sigmas, bar_width, 
        yerr=plot_errors, capsize=4, ecolor='white',
        label='CROCUS (Li6F:ZnS(Cu) 300um)', color=COLOR_CROCUS, edgecolor='white'
    )

    # Styling Axes
    ax_bar.set_ylabel('Macroscopic Cross Section $\Sigma$ ($cm^{-1}$)', color='#ffffff', fontsize=12)
    ax_bar.set_title('Calculated Macroscopic Cross Sections (CROCUS)', color='#ffffff', fontsize=14, pad=20)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(plot_materials, rotation=45, ha='right', color='#cccccc', fontsize=12, fontweight='bold')
    ax_bar.tick_params(axis='y', colors='#cccccc')

    # Grid and Spines
    ax_bar.grid(True, axis='y', linestyle='--', color='#444444', alpha=0.6, zorder=0)
    ax_bar.set_axisbelow(True)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_color('#ffffff')
    ax_bar.spines['bottom'].set_color('#ffffff')

    ax_bar.legend(frameon=False, labelcolor='white', fontsize=12)

    # Add numbers on top of bars
    for rect in rects_bar:
        height = rect.get_height()
        ax_bar.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 6),  # 6 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig("CROCUS_Cylinder_XS_BarPlot.png", transparent=True, dpi=350)
    plt.show()
    
    # ==============================================================================
    # 4. EXPORT MACROSCOPIC EXPERIMENTAL DATA TO CSV
    # ==============================================================================
    print("\n=======================================")
    print("EXPORTING MACROSCOPIC EXPERIMENTAL DATA")
    print("=======================================")

    df_exp = pd.DataFrame(extracted_sigmas)
    
    # Save to CSV for downstream C/E comparison scripts
    csv_filename = "macroscopic_exp_data.csv"
    df_exp.to_csv(csv_filename, index=False)
    
    print(f"Successfully exported {len(df_exp)} samples to '{csv_filename}'")
    print("\nExtracted Experimental Summary:")
    print(df_exp[["Sample_ID", "Exp_Mean", "Exp_Std", "MFP_cm"]].to_string(index=False))