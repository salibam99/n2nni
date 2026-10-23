# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:38:31 2026

@author: miche
"""

from pathlib import Path
import warnings
from astropy.io import fits
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit

# Global storage for extracted cross-section records
extracted_sigmas = []


class ExperimentCatalog:

  def __init__(self, master_directory):
    """Scans the master directory and indexes folder paths."""
    self.master_path = Path(master_directory)
    self.folders = {
        folder.name: folder
        for folder in self.master_path.iterdir()
        if folder.is_dir()
    }
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
        datt = (
            image_data[0, :, :] if image_data.ndim == 3 else image_data
        )  # Extract 2D slice if 3D
        list_img.append(datt)
    return np.array(list_img)

  def get_data(self, folder_name_string):
    """Loads all .fits files inside the specified catalog directory into a 3D array."""
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
    """Removes isolated outlier pixels (salt-and-pepper noise) using median neighborhood filtering."""
    box_size = (2 * radius) + 1
    median_img = median_filter(image, size=box_size)

    img_float = image.astype(np.float32)
    median_float = median_img.astype(np.float32)
    difference = np.abs(img_float - median_float)

    outlier_mask = difference > threshold
    cleaned_image = np.where(outlier_mask, median_img, image)
    return cleaned_image.astype(image.dtype)


# ==============================================================================
# IMAGE PLOTTING & ROI VISUALIZATION HELPERS
# ==============================================================================
def plot_image(
    image,
    title,
    vmin,
    vmax,
    cbar_label="Intensity / Counts",
    roi_box=None,
    roi_label="Background ROI",
):
  """Plots an image with an optional highlighted Region of Interest (ROI) bounding box."""
  fig, ax = plt.subplots(figsize=(7, 6))
  img_plot = ax.imshow(
      image, cmap="gray", origin="lower", vmin=vmin, vmax=vmax
  )
  cbar = plt.colorbar(img_plot, ax=ax, fraction=0.046, pad=0.04)
  cbar.set_label(cbar_label)

  if roi_box is not None:
    xmin, xmax, ymin, ymax = roi_box
    rect = patches.Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
        label=roi_label,
    )
    ax.add_patch(rect)
    ax.legend(loc="upper right")

  ax.set_title(title)
  ax.set_xlabel("X / pixels")
  ax.set_ylabel("Y / pixels")
  ax.set_xlim(350, 1700)
  ax.set_ylim(350, 1700)
  plt.tight_layout()
  plt.show()


def plot_image_save(
    image,
    title,
    vmin,
    vmax,
    cbar_label="Intensity / Counts",
    folder_name="name1",
    roi_box=None,
):
  """Saves a processed image with an optional ROI box to the processed_images folder."""
  fig, ax = plt.subplots(figsize=(7, 6))
  img_plot = ax.imshow(
      image, cmap="gray", origin="lower", vmin=vmin, vmax=vmax
  )
  cbar = plt.colorbar(img_plot, ax=ax, fraction=0.046, pad=0.04)
  cbar.set_label(cbar_label)

  if roi_box is not None:
    xmin, xmax, ymin, ymax = roi_box
    rect = patches.Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect)

  ax.set_title(title)
  ax.set_xlabel("X / pixels")
  ax.set_ylabel("Y / pixels")
  ax.set_xlim(350, 1700)
  ax.set_ylim(350, 1700)

  Path("processed_images").mkdir(parents=True, exist_ok=True)
  plt.savefig(f"processed_images/{folder_name}.png", dpi=350)
  plt.close()


def check_and_display_beam_roi(ob_img, s1_img, s2_img, roi_box):
  """Computes beam intensity in the selected ROI, calculates percentage differences

  relative to the Open Beam (OB), and generates a 3-panel visual display with
  the
  ROI rectangle drawn on all images.
  """
  xmin, xmax, ymin, ymax = roi_box
  width = xmax - xmin
  height = ymax - ymin

  roi_ob = ob_img[ymin:ymax, xmin:xmax]
  roi_s1 = s1_img[ymin:ymax, xmin:xmax]
  roi_s2 = s2_img[ymin:ymax, xmin:xmax]

  mean_ob = float(np.mean(roi_ob))
  std_ob = float(np.std(roi_ob))
  mean_s1 = float(np.mean(roi_s1))
  std_s1 = float(np.std(roi_s1))
  mean_s2 = float(np.mean(roi_s2))
  std_s2 = float(np.std(roi_s2))

  diff_s1_pct = ((mean_s1 - mean_ob) / mean_ob) * 100.0
  diff_s2_pct = ((mean_s2 - mean_ob) / mean_ob) * 100.0

  print("\n=======================================================")
  print("              BACKGROUND BEAM INTENSITY CHECK          ")
  print("=======================================================")
  print(f"ROI Coordinates: X[{xmin}:{xmax}], Y[{ymin}:{ymax}] (Area: {width*height} px)")
  print(f"Open Beam (OB) Background Mean:      {mean_ob:.2f} ± {std_ob:.2f} counts")
  print(
      f"SampleSet 1 Background Mean:         {mean_s1:.2f} ± {std_s1:.2f} counts "
      f"({abs(diff_s1_pct):.2f}% {'HIGHER' if diff_s1_pct >= 0 else 'LOWER'} than OB)"
  )
  print(
      f"SampleSet 2 Background Mean:         {mean_s2:.2f} ± {std_s2:.2f} counts "
      f"({abs(diff_s2_pct):.2f}% {'HIGHER' if diff_s2_pct >= 0 else 'LOWER'} than OB)"
  )
  print("=======================================================\n")

  # 3-Panel Side-by-Side Visualization
  fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=True, sharey=True)

  panels = [
      (
          axes[0],
          ob_img,
          "Open Beam (Reference)",
          f"Mean: {mean_ob:.1f} cts\nRef (0.00%)",
          "red",
      ),
      (
          axes[1],
          s1_img,
          "SampleSet 1",
          f"Mean: {mean_s1:.1f} cts\n{diff_s1_pct:+.2f}% vs OB",
          "lime" if abs(diff_s1_pct) <= 2.0 else "yellow",
      ),
      (
          axes[2],
          s2_img,
          "SampleSet 2",
          f"Mean: {mean_s2:.1f} cts\n{diff_s2_pct:+.2f}% vs OB",
          "lime" if abs(diff_s2_pct) <= 2.0 else "yellow",
      ),
  ]

  for ax, img, title_label, stat_text, box_color in panels:
    im = ax.imshow(img, cmap="inferno", origin="lower", vmin=10, vmax=300)
    rect = patches.Rectangle(
        (xmin, ymin),
        width,
        height,
        linewidth=2.5,
        edgecolor=box_color,
        facecolor="none",
        linestyle="--",
        label="Background ROI",
    )
    ax.add_patch(rect)

    ax.text(
        xmin + width / 2,
        ymax + 35,
        stat_text,
        color="white",
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="black",
            alpha=0.75,
            edgecolor=box_color,
        ),
    )

    ax.set_title(title_label, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("X / pixels")
    ax.set_xlim(350, 1700)
    ax.set_ylim(350, 1700)
    ax.legend(loc="upper right", framealpha=0.6)

  axes[0].set_ylabel("Y / pixels")
  fig.subplots_adjust(right=0.88)
  cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
  fig.colorbar(im, cax=cbar_ax, label="Dark-Corrected Intensity / Counts")

  plt.suptitle(
      "Beam Intensity & Stability Check Across Datasets",
      fontsize=14,
      fontweight="bold",
      y=0.98,
  )
  plt.show()

  return {
      "mean_ob": mean_ob,
      "mean_s1": mean_s1,
      "mean_s2": mean_s2,
      "diff_s1_pct": diff_s1_pct,
      "diff_s2_pct": diff_s2_pct,
  }


# ==============================================================================
# MAIN WORKFLOW SCRIPT
# ==============================================================================
if __name__ == "__main__":
  master_dir = (
      r"C:\Users\miche\Desktop\PhD\Neutron_Imaging_project\2nd_CROCUS_Campaign\blackbody"
  )
  catalog = ExperimentCatalog(master_dir)

  # 1. Load Dark Current (DC)
  dc_data = catalog.get_data("DC2")
  dc_median = np.median(dc_data, axis=0)

  # Preprocessing helper function
  def load_and_preprocess(folder_name, radius=3, threshold=100):
    raw_stack = catalog.get_data(folder_name)
    corrected_stack = raw_stack - dc_median
    median_frame = np.median(corrected_stack, axis=0)
    cleaned_frame = catalog.remove_outliers(
        median_frame, radius=radius, threshold=threshold
    )
    return cleaned_frame

  # 2. Preprocess Open Beam and Sample Stacks
  print("\n>>> Pre-processing Frames for Verification <<<")
  ob_clean = load_and_preprocess("OB", radius=3, threshold=100)
  sample1_clean = load_and_preprocess("SampleSet1_0cm", radius=3, threshold=100)
  sample2_clean = load_and_preprocess("SampleSet2_0cm", radius=3, threshold=100)

  # ==============================================================================
  # BACKGROUND ROI BEAM STABILITY INSPECTION
  # ==============================================================================
  roi_background = (500, 750, 500, 750)  # (xmin, xmax, ymin, ymax)

  roi_stats = check_and_display_beam_roi(
      ob_img=ob_clean,
      s1_img=sample1_clean,
      s2_img=sample2_clean,
      roi_box=roi_background,
  )

  # ==============================================================================
  # 3. CONDITIONAL SCALAR FLUX NORMALIZATION & TRANSMITTANCE
  # ==============================================================================
  DRIFT_THRESHOLD_PCT = 2.0  # Threshold in percentage

  print(">>> Evaluating Scalar Flux Normalization Factors <<<")

  # Evaluate SampleSet 1 drift
  if abs(roi_stats["diff_s1_pct"]) > DRIFT_THRESHOLD_PCT:
    scale_s1 = roi_stats["mean_ob"] / roi_stats["mean_s1"]
    status_s1 = f"Rescaled (Factor: {scale_s1:.4f})"
    print(
        f"  [RESCALE] SampleSet 1 drift ({roi_stats['diff_s1_pct']:+.2f}%) exceeds"
        f" ±{DRIFT_THRESHOLD_PCT}%. Applying scalar factor {scale_s1:.4f}"
    )
  else:
    scale_s1 = 1.0
    status_s1 = "Unscaled (Drift ≤ 2%)"
    print(
        f"  [STABLE]  SampleSet 1 drift ({roi_stats['diff_s1_pct']:+.2f}%) within"
        f" ±{DRIFT_THRESHOLD_PCT}%. No rescaling required."
    )

  # Evaluate SampleSet 2 drift
  if abs(roi_stats["diff_s2_pct"]) > DRIFT_THRESHOLD_PCT:
    scale_s2 = roi_stats["mean_ob"] / roi_stats["mean_s2"]
    status_s2 = f"Rescaled (Factor: {scale_s2:.4f})"
    print(
        f"  [RESCALE] SampleSet 2 drift ({roi_stats['diff_s2_pct']:+.2f}%) exceeds"
        f" ±{DRIFT_THRESHOLD_PCT}%. Applying scalar factor {scale_s2:.4f}"
    )
  else:
    scale_s2 = 1.0
    status_s2 = "Unscaled (Drift ≤ 2%)"
    print(
        f"  [STABLE]  SampleSet 2 drift ({roi_stats['diff_s2_pct']:+.2f}%) within"
        f" ±{DRIFT_THRESHOLD_PCT}%. No rescaling required."
    )

  # Calculate normalized transmittance fields (1e-6 added to prevent division by zero)
  transmittance_img = (sample1_clean * scale_s1) / (ob_clean + 1e-6)
  transmittance_img2 = (sample2_clean * scale_s2) / (ob_clean + 1e-6)

  # Display transmittance maps with ROI reference box overlaid
  plot_image(
      transmittance_img,
      title=f"Transmittance: SampleSet 1 [{status_s1}]",
      vmin=0.2,
      vmax=1.3,
      cbar_label="Transmittance (-)",
      roi_box=roi_background,
  )
  plot_image(
      transmittance_img2,
      title=f"Transmittance: SampleSet 2 [{status_s2}]",
      vmin=0.2,
      vmax=1.3,
      cbar_label="Transmittance (-)",
      roi_box=roi_background,
  )

  # ==============================================================================
  # 4. DEFINE CYLINDER TRANSMISSION MODEL
  # ==============================================================================
  def cylinder_transmittance(x, T0, Sigma, R, x0):
    """Theoretical transmittance through a solid cylinder."""
    inside_sqrt = R**2 - (x - x0) ** 2
    inside_sqrt = np.where(inside_sqrt > 0, inside_sqrt, 0)
    path_length = 2 * np.sqrt(inside_sqrt)
    return T0 * np.exp(-Sigma * path_length)

  # ==============================================================================
  # 5a. EXTRACT AND FIT PROFILES (SET 1)
  # ==============================================================================
  y_start, y_end = 1150, 1600
  cylinders_set1 = {
      "SS316L": (460, 600),
      "Zn": (650, 780),
      "Al": (870, 1090),
      "Ta": (1120, 1300),
      "C": (1360, 1500),
  }

  warnings.filterwarnings("ignore")
  fig1, axes1 = plt.subplots(
      1, len(cylinders_set1), figsize=(18, 4), sharey=True
  )
  print("\nExtracting profiles and fitting Set 1 cylinders...\n")

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

    ax.plot(
        x_pixels,
        median_profile,
        "o",
        color="gray",
        alpha=0.6,
        markersize=4,
        label="Median Profile",
    )

    try:
      bounds = ([0.1, 0.0001, 10, x_min], [1.5, 1.0, 100, x_max])
      popt, pcov = curve_fit(
          cylinder_transmittance, x_pixels, median_profile, p0=p0, bounds=bounds
      )
      T0_fit, Sigma_fit, R_fit, x0_fit = popt

      fit_curve = cylinder_transmittance(x_pixels, *popt)
      ax.plot(x_pixels, fit_curve, "r-", linewidth=2.5, label="Cylinder Fit")

      mean_free_path = 1.0 / Sigma_fit
      physical_radius_cm = 0.5
      pixel_size_cm = physical_radius_cm / R_fit
      mfp_cm = mean_free_path * pixel_size_cm
      sigma_cm = Sigma_fit / pixel_size_cm

      # Uncertainty propagation
      var_Sigma = pcov[1, 1]
      var_R = pcov[2, 2]
      cov_Sigma_R = pcov[1, 2]

      var_prod = (
          (R_fit**2 * var_Sigma)
          + (Sigma_fit**2 * var_R)
          + (2 * R_fit * Sigma_fit * cov_Sigma_R)
      )
      dsigma_cm = np.sqrt(max(0, var_prod)) / physical_radius_cm

      extracted_sigmas.append({
          "Sample_ID": name,
          "Exp_Mean": sigma_cm,
          "Exp_Std": dsigma_cm,
          "MFP_cm": mfp_cm,
          "Pixel_Size_cm": pixel_size_cm,
          "R_fit_px": R_fit,
      })

      title = (
          f"{name}\n"
          f"$\\Sigma$: {sigma_cm:.3f} $\\pm$ {dsigma_cm:.3f} cm⁻¹\n"
          f"MFP: {mfp_cm:.2f} cm"
      )
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
  # 5b. EXTRACT AND FIT PROFILES (SET 2)
  # ==============================================================================
  cylinders_set2 = {
      "Nb": (460, 580),
      "Mo": (675, 820),
      "V": (870, 1000),
      "W": (1100, 1225),
      "Au": (1300, 1450),
  }

  fig2, axes2 = plt.subplots(
      1, len(cylinders_set2), figsize=(18, 4), sharey=True
  )
  print("Extracting profiles and fitting Set 2 cylinders...\n")

  for idx, (name, (x_min, x_max)) in enumerate(cylinders_set2.items()):
    ax = axes2[idx]

    if name == "Au":
      roi = transmittance_img2[2048 - 975 : 2048 - 905, x_min:x_max]
    else:
      roi = transmittance_img2[y_start:y_end, x_min:x_max]

    median_profile = np.median(roi, axis=0)
    x_pixels = np.arange(x_min, x_max)

    T0_guess = np.max(median_profile)
    x0_guess = x_min + (x_max - x_min) / 2
    R_guess = (x_max - x_min) / 2.5
    Sigma_guess = 0.05
    p0 = [T0_guess, Sigma_guess, R_guess, x0_guess]

    ax.plot(
        x_pixels,
        median_profile,
        "o",
        color="gray",
        alpha=0.6,
        markersize=4,
        label="Median Profile",
    )

    try:
      bounds = ([0.1, 0.0001, 10, x_min], [1.5, 1.0, 100, x_max])
      popt, pcov = curve_fit(
          cylinder_transmittance, x_pixels, median_profile, p0=p0, bounds=bounds
      )
      T0_fit, Sigma_fit, R_fit, x0_fit = popt

      fit_curve = cylinder_transmittance(x_pixels, *popt)
      ax.plot(x_pixels, fit_curve, "c-", linewidth=2.5, label="Cylinder Fit")

      mean_free_path = 1.0 / Sigma_fit
      physical_radius_cm = 0.4 if name == "Y" else 0.5

      pixel_size_cm = physical_radius_cm / R_fit
      mfp_cm = mean_free_path * pixel_size_cm
      sigma_cm = Sigma_fit / pixel_size_cm

      # Uncertainty propagation
      var_Sigma = pcov[1, 1]
      var_R = pcov[2, 2]
      cov_Sigma_R = pcov[1, 2]

      var_prod = (
          (R_fit**2 * var_Sigma)
          + (Sigma_fit**2 * var_R)
          + (2 * R_fit * Sigma_fit * cov_Sigma_R)
      )
      dsigma_cm = np.sqrt(max(0, var_prod)) / physical_radius_cm

      extracted_sigmas.append({
          "Sample_ID": name,
          "Exp_Mean": sigma_cm,
          "Exp_Std": dsigma_cm,
          "MFP_cm": mfp_cm,
          "Pixel_Size_cm": pixel_size_cm,
          "R_fit_px": R_fit,
      })

      title = (
          f"{name}\n"
          f"$\\Sigma$: {sigma_cm:.3f} $\\pm$ {dsigma_cm:.3f} cm⁻¹\n"
          f"MFP: {mfp_cm:.2f} cm"
      )
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
  # 6. COMBINED MACROSCOPIC CROSS-SECTION BAR PLOT
  # ==============================================================================
  print("\nGenerating final cross-section bar plot for all cylinders...")

  plot_materials = [rec["Sample_ID"] for rec in extracted_sigmas]
  plot_sigmas = [rec["Exp_Mean"] for rec in extracted_sigmas]
  plot_errors = [rec["Exp_Std"] for rec in extracted_sigmas]

  plt.style.use("dark_background")
  fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
  fig_bar.patch.set_facecolor("black")
  ax_bar.set_facecolor("black")

  COLOR_CROCUS = "#00ffff"  # Pure Cyan
  x_pos = np.arange(len(plot_materials))
  bar_width = 0.5

  rects_bar = ax_bar.bar(
      x_pos,
      plot_sigmas,
      bar_width,
      yerr=plot_errors,
      capsize=4,
      ecolor="white",
      label="CROCUS (Li6F:ZnS(Cu) 300um)",
      color=COLOR_CROCUS,
      edgecolor="white",
  )

  ax_bar.set_ylabel(
      r"Macroscopic Cross Section $\Sigma$ ($cm^{-1}$)",
      color="#ffffff",
      fontsize=12,
  )
  ax_bar.set_title(
      "Calculated Macroscopic Cross Sections (CROCUS)",
      color="#ffffff",
      fontsize=14,
      pad=20,
  )
  ax_bar.set_xticks(x_pos)
  ax_bar.set_xticklabels(
      plot_materials,
      rotation=45,
      ha="right",
      color="#cccccc",
      fontsize=12,
      fontweight="bold",
  )
  ax_bar.tick_params(axis="y", colors="#cccccc")

  ax_bar.grid(
      True, axis="y", linestyle="--", color="#444444", alpha=0.6, zorder=0
  )
  ax_bar.set_axisbelow(True)
  ax_bar.spines["top"].set_visible(False)
  ax_bar.spines["right"].set_visible(False)
  ax_bar.spines["left"].set_color("#ffffff")
  ax_bar.spines["bottom"].set_color("#ffffff")

  ax_bar.legend(frameon=False, labelcolor="white", fontsize=12)

  for rect in rects_bar:
    height = rect.get_height()
    ax_bar.annotate(
        f"{height:.3f}",
        xy=(rect.get_x() + rect.get_width() / 2, height),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        color="white",
        fontsize=10,
        fontweight="bold",
    )

  plt.tight_layout()
  plt.savefig("CROCUS_Cylinder_XS_BarPlot.png", transparent=True, dpi=350)
  plt.show()

  # ==============================================================================
  # 7. EXPORT DATA TO CSV
  # ==============================================================================
  print("\n=======================================")
  print("EXPORTING MACROSCOPIC EXPERIMENTAL DATA")
  print("=======================================")

  df_exp = pd.DataFrame(extracted_sigmas)
  csv_filename = "macroscopic_exp_data.csv"
  df_exp.to_csv(csv_filename, index=False)

  print(f"Successfully exported {len(df_exp)} samples to '{csv_filename}'")
  print("\nExtracted Experimental Summary:")
  print(
      df_exp[["Sample_ID", "Exp_Mean", "Exp_Std", "MFP_cm"]].to_string(
          index=False
      )
  )