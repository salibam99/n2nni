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

# Physical detector pixel size (cm/pixel)
PIXEL_SIZE_CM = 0.01059  

# Known physical radii dictionary (cm)
PHYSICAL_RADII = {
    "SS316L": 0.5,
    "Zn": 0.5,
    "Al": 0.5,
    "Ta": 0.5,
    "C": 0.5,
    "Nb": 0.5,
    "Mo": 0.5,
    "V": 0.4,   # Vanadium cylinder radius
    "W": 0.5,
    "Au": 0.5,
}

extracted_sigmas = []

class ExperimentCatalog:
    def __init__(self, master_directory):
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


def cylinder_transmittance_fixed_R(x_cm, T0, Sigma, x0_cm, R_phys):
    inside_sqrt = R_phys**2 - (x_cm - x0_cm)**2
    inside_sqrt = np.maximum(0.0, inside_sqrt)
    path_length = 2 * np.sqrt(inside_sqrt)
    return T0 * np.exp(-Sigma * path_length)


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
    return transmittance


if __name__ == "__main__":
    master_dir = r"C:\Users\miche\Desktop\PhD\Neutron_Imaging_project\2nd_CROCUS_Campaign\blackbody"
    catalog = ExperimentCatalog(master_dir)
    
    experiment_list = [
        ("SampleSet1_0cm", "OB", "DC2"),
        ("SampleSet2_0cm", "OB", "DC2")
    ]
    
    transmittance_results = {}
    ii = 0
    for sample, ob, dc in experiment_list:
        result = process_transmittance(catalog, sample, ob, dc, threshold=100, radius=3, show_plots=0)
        transmittance_results[sample] = result
        if ii == 0:
            transmittance_img = result
        else:
            transmittance_img2 = result
        ii += 1

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
    print("Extracting profiles and fitting Set 1 cylinders (Fixed Radius, Dual Mask)...\n")
    
    for idx, (name, (x_min, x_max)) in enumerate(cylinders_set1.items()):
        ax = axes1[idx]
        roi = transmittance_img[y_start:y_end, x_min:x_max]
        median_profile = np.median(roi, axis=0)
        x_pixels = np.arange(x_min, x_max)
        x_cm = x_pixels * PIXEL_SIZE_CM
        R_phys = PHYSICAL_RADII[name]
        
        ax.plot(x_pixels, median_profile, 'o', color='gray', alpha=0.6, markersize=4, label='Median Profile')
        
        try:
            # 1. Initial center guess
            x0_guess_cm = x_cm[np.argmin(median_profile)]
            
            # 2. Dual-Region Masking (Core + Outer Baseline, excluding blurred edges)
            dist_from_center = np.abs(x_cm - x0_guess_cm)
            fit_mask = (dist_from_center <= 0.80 * R_phys) | (dist_from_center >= 1.20 * R_phys)
            
            x_fit = x_cm[fit_mask]
            prof_fit = median_profile[fit_mask]
            
            # 3. Dynamic T0 from actual baseline points
            baseline_pts = median_profile[dist_from_center >= 1.20 * R_phys]
            T0_guess = float(np.median(baseline_pts)) if len(baseline_pts) > 0 else 1.0
            
            p0 = [T0_guess, 0.5, x0_guess_cm]
            bounds = ([T0_guess * 0.85, 0.0001, x_cm[0]], [T0_guess * 1.15, 10.0, x_cm[-1]])
            
            fit_func = lambda x, T0, Sigma, x0: cylinder_transmittance_fixed_R(x, T0, Sigma, x0, R_phys)
            popt, pcov = curve_fit(fit_func, x_fit, prof_fit, p0=p0, bounds=bounds)
            T0_fit, sigma_cm, x0_fit_cm = popt
            
            dsigma_cm = float(np.sqrt(pcov[1, 1]))
            mfp_cm = 1.0 / sigma_cm if sigma_cm > 0 else np.nan
            
            # Plot full theoretical curve
            fit_curve = cylinder_transmittance_fixed_R(x_cm, T0_fit, sigma_cm, x0_fit_cm, R_phys)
            ax.plot(x_pixels, fit_curve, 'r-', linewidth=2.5, label=f'Fixed R={R_phys}cm Fit')
            
            extracted_sigmas.append({
                "Sample_ID": name,
                "Exp_Mean": sigma_cm,
                "Exp_Std": dsigma_cm,
                "MFP_cm": mfp_cm,
                "Pixel_Size_cm": PIXEL_SIZE_CM,
                "R_fit_px": R_phys / PIXEL_SIZE_CM
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
    print("Extracting profiles and fitting Set 2 cylinders (Fixed Radius, Dual Mask)...\n")
    
    for idx, (name, (x_min, x_max)) in enumerate(cylinders_set2.items()):
        ax = axes2[idx]
        if name == "Au":
            roi = transmittance_img2[2048-975:2048-905, x_min:x_max]
        else:
            roi = transmittance_img2[y_start:y_end, x_min:x_max]
            
        median_profile = np.median(roi, axis=0)
        x_pixels = np.arange(x_min, x_max)
        x_cm = x_pixels * PIXEL_SIZE_CM
        R_phys = PHYSICAL_RADII[name]
        
        ax.plot(x_pixels, median_profile, 'o', color='gray', alpha=0.6, markersize=4, label='Median Profile')
        
        try:
            x0_guess_cm = x_cm[np.argmin(median_profile)]
            
            dist_from_center = np.abs(x_cm - x0_guess_cm)
            fit_mask = (dist_from_center <= 0.80 * R_phys) | (dist_from_center >= 1.20 * R_phys)
            
            x_fit = x_cm[fit_mask]
            prof_fit = median_profile[fit_mask]
            
            baseline_pts = median_profile[dist_from_center >= 1.20 * R_phys]
            T0_guess = float(np.median(baseline_pts)) if len(baseline_pts) > 0 else 1.0
            
            p0 = [T0_guess, 0.5, x0_guess_cm]
            bounds = ([T0_guess * 0.85, 0.0001, x_cm[0]], [T0_guess * 1.15, 10.0, x_cm[-1]])
            
            fit_func = lambda x, T0, Sigma, x0: cylinder_transmittance_fixed_R(x, T0, Sigma, x0, R_phys)
            popt, pcov = curve_fit(fit_func, x_fit, prof_fit, p0=p0, bounds=bounds)
            T0_fit, sigma_cm, x0_fit_cm = popt
            
            dsigma_cm = float(np.sqrt(pcov[1, 1]))
            mfp_cm = 1.0 / sigma_cm if sigma_cm > 0 else np.nan
            
            fit_curve = cylinder_transmittance_fixed_R(x_cm, T0_fit, sigma_cm, x0_fit_cm, R_phys)
            ax.plot(x_pixels, fit_curve, 'c-', linewidth=2.5, label=f'Fixed R={R_phys}cm Fit')
            
            extracted_sigmas.append({
                "Sample_ID": name,
                "Exp_Mean": sigma_cm,
                "Exp_Std": dsigma_cm,
                "MFP_cm": mfp_cm,
                "Pixel_Size_cm": PIXEL_SIZE_CM,
                "R_fit_px": R_phys / PIXEL_SIZE_CM
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
    # 4. EXPORT MACROSCOPIC EXPERIMENTAL DATA TO CSV
    # ==============================================================================
    df_exp = pd.DataFrame(extracted_sigmas)
    csv_filename = "macroscopic_exp_data2.csv"
    df_exp.to_csv(csv_filename, index=False)
    print(f"\nSuccessfully exported {len(df_exp)} samples to '{csv_filename}'")