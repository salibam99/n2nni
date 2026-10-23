# -*- coding: utf-8 -*-
"""
Neutron imaging: flat-field correction + macroscopic cross-section extraction
from cylindrical samples (CROCUS 2nd campaign).

Fitting strategy
----------------
R is held FIXED at the metrology value. Sigma and R are almost perfectly
anti-correlated through T_min = T0*exp(-2*Sigma*R), so a free R absorbs the
edge blur (detector PSF + beam penumbra + small-angle scatter halo) by growing,
and Sigma shrinks to compensate. The free-R fit is still run as a *diagnostic*
only, to quantify that inflation.

The Sigma fit uses only the deepest part of the dip:
  - depth criterion : (T0 - T) >= CORE_DEPTH_FRAC * (T0 - T_min)
  - geometric caps  : |x - x0| <= CORE_MAX_FRAC * R  and  <= R - EDGE_EXCLUSION
A log-space linear fit of -ln(T/T0) against the chord length is run alongside as
a cross-check; its intercept flags scattered-background contamination.

Created on Wed May  6 16:38:31 2026
@author: miche
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit

# =============================================================================
# CONFIGURATION
# =============================================================================

MASTER_DIR = r"C:\Users\miche\Desktop\PhD\Neutron_Imaging_project\2nd_CROCUS_Campaign\blackbody"
CSV_FILENAME = "macroscopic_exp_data_0to1_trial.csv"

PIXEL_SIZE_CM = 0.01059          # detector pixel pitch (cm/pixel)
DETECTOR_SIZE_PX = 2048

# Transmittance display range: physically T lies in [0, 1]
T_VMIN, T_VMAX = 0.0, 1.0
CLIP_DISPLAY_TO_UNIT = True      # clip *for plotting only*, never for fitting

# Outlier (salt-and-pepper) removal
OUTLIER_RADIUS = 3
OUTLIER_THRESHOLD = 100

# --- Fit region ---------------------------------------------------------------
# Keep only points whose attenuation depth reaches this fraction of the maximum
# depth. For a weak absorber this is purely geometric (0.85 -> |x-x0| < 0.53 R);
# for a strong absorber the dip saturates, so the geometric caps below take over.
CORE_DEPTH_FRAC = 0.85
CORE_MAX_FRAC = 0.70             # hard cap on |x - x0| / R
EDGE_EXCLUSION_PX = 10          # extra guard band inside the geometric edge (px)
BASELINE_FRAC = 1.20             # |x - x0| >= 1.2 R defines the open-beam level
MIN_FIT_POINTS = 5

FIX_T0_TO_BASELINE = True        # True: T0 from the baseline median, not fitted
RUN_FREE_R_DIAGNOSTIC = True     # fit R as well, for comparison only

# Known physical radii (cm)
PHYSICAL_RADII = {
    "SS316L": 0.5,
    "Zn": 0.5,
    "Al": 0.5,
    "Ta": 0.5,
    "C": 0.5,
    "Nb": 0.5,
    "Mo": 0.5,
    "V": 0.4,    # Vanadium cylinder radius = 0.4 cm
    "W": 0.5,
    "Au": 0.5,
}

DEFAULT_Y_RANGE = (1150, 1600)

CYLINDER_SETS = [
    {
        "label": "Set 1",
        "sample": "SampleSet1_0cm",
        "fit_color": "r",
        "cylinders": {
            "SS316L": {"x": (460, 600)},
            "Zn":     {"x": (650, 780)},
            "Al":     {"x": (870, 1090)},
            "Ta":     {"x": (1120, 1300)},
            "C":      {"x": (1360, 1500)},
        },
    },
    {
        "label": "Set 2",
        "sample": "SampleSet2_0cm",
        "fit_color": "c",
        "cylinders": {
            "Nb": {"x": (460, 580)},
            "Mo": {"x": (675, 820)},
            "V":  {"x": (870, 1000)},
            "W":  {"x": (1100, 1225)},
            "Au": {"x": (1300, 1450),
                   "y": (DETECTOR_SIZE_PX - 975, DETECTOR_SIZE_PX - 905)},
        },
    },
]

EXPERIMENT_LIST = [
    ("SampleSet1_0cm", "OB", "DC2"),
    ("SampleSet2_0cm", "OB", "DC2"),
]


# =============================================================================
# DATA ACCESS
# =============================================================================

class ExperimentCatalog:
    """Indexes the sub-folders of a master directory and loads their FITS stacks."""

    def __init__(self, master_directory):
        self.master_path = Path(master_directory)
        if not self.master_path.is_dir():
            raise FileNotFoundError(f"Master directory not found: {self.master_path}")

        self.folders = {f.name: f for f in sorted(self.master_path.iterdir()) if f.is_dir()}
        self._cache = {}

        print(f"Catalog initialized! Found {len(self.folders)} folders.")
        print(f"Available folders: {list(self.folders.keys())}\n")

    @staticmethod
    def _load_to_tensor(file_paths):
        frames = []
        for path in file_paths:
            with fits.open(path) as hdul:
                data = hdul[0].data
                frames.append(data[0] if data.ndim == 3 else data)
        return np.asarray(frames, dtype=np.float32)

    def get_data(self, folder_name):
        if folder_name not in self.folders:
            raise ValueError(f"Folder '{folder_name}' not found in catalog!")
        if folder_name in self._cache:
            return self._cache[folder_name]

        files = sorted(self.folders[folder_name].rglob("*.fits"))
        if not files:
            print(f"Warning: no .fits files found in '{folder_name}'")
            return None

        print(f"--- Loading {len(files)} frames for '{folder_name}' ---")
        stack = self._load_to_tensor(files)
        self._cache[folder_name] = stack
        return stack

    def get_median_frame(self, folder_name):
        stack = self.get_data(folder_name)
        return None if stack is None else np.median(stack, axis=0)


def remove_outliers(image, radius=OUTLIER_RADIUS, threshold=OUTLIER_THRESHOLD):
    """Replace isolated outlier pixels by the local median."""
    median_img = median_filter(image, size=2 * radius + 1)
    outliers = np.abs(image - median_img) > threshold
    return np.where(outliers, median_img, image)


# =============================================================================
# PLOT HELPERS
# =============================================================================

def plot_image(image, title, vmin, vmax, cbar_label="Intensity / Counts",
               save_as=None, clip=False):
    if clip:
        image = np.clip(image, vmin, vmax)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(image, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(cbar_label)
    ax.set_xlabel("X / pixels")
    ax.set_ylabel("Y / pixels")
    ax.set_xlim(350, 1700)
    ax.set_ylim(350, 1700)
    ax.set_title(title)
    fig.tight_layout()

    if save_as is not None:
        out_dir = Path("processed_images")
        out_dir.mkdir(exist_ok=True)
        fig.savefig(out_dir / f"{save_as}.png", dpi=350)
        plt.close(fig)
    else:
        plt.show()


# =============================================================================
# PHYSICS
# =============================================================================

def chord_length(x_cm, x0_cm, R_phys):
    """Path length through a solid cylinder of radius R_phys (cm)."""
    return 2.0 * np.sqrt(np.maximum(0.0, R_phys ** 2 - (x_cm - x0_cm) ** 2))


def cylinder_transmittance(x_cm, T0, Sigma, x0_cm, R):
    return T0 * np.exp(-Sigma * chord_length(x_cm, x0_cm, R))


def process_transmittance(catalog, sample_folder, ob_folder, dc_folder, show_plots=0):
    """Dark-current subtraction + flat-field (open-beam) normalisation."""
    print(f"\n>>> Processing transmittance for: {sample_folder} <<<")

    dark = catalog.get_median_frame(dc_folder)
    sample = remove_outliers(catalog.get_median_frame(sample_folder) - dark)
    open_beam = remove_outliers(catalog.get_median_frame(ob_folder) - dark)

    if show_plots == 1:
        plot_image(sample, f"Sample: {sample_folder} (dark corrected)", vmin=10, vmax=300)
        plot_image(open_beam, f"Open beam: {ob_folder} (dark corrected)", vmin=10, vmax=300)

    with np.errstate(divide="ignore", invalid="ignore"):
        transmittance = np.where(open_beam > 0, sample / open_beam, np.nan)

    finite = np.isfinite(transmittance)
    print(f"    T statistics: median={np.median(transmittance[finite]):.3f}, "
          f"{np.count_nonzero(~finite)} invalid pixels")

    if show_plots in (1, 2):
        plot_image(transmittance, f"Transmittance: {sample_folder}",
                   vmin=T_VMIN, vmax=T_VMAX, cbar_label="Transmittance (-)",
                   save_as=sample_folder if show_plots == 2 else None,
                   clip=CLIP_DISPLAY_TO_UNIT)

    return transmittance


# =============================================================================
# PROFILE FITTING
# =============================================================================

def estimate_center_and_baseline(x_cm, profile, R_phys):
    """
    Robust starting estimates.

    x0 comes from the depth-weighted centroid of the dip (insensitive to the
    single noisy pixel that argmin would latch onto); T0 from the median of the
    points lying beyond BASELINE_FRAC * R.
    """
    valid = np.isfinite(profile)
    if valid.sum() < MIN_FIT_POINTS:
        raise RuntimeError("too few finite points in the profile")

    x_v, p_v = x_cm[valid], profile[valid]

    # First pass: crude center from the argmin, refined by a weighted centroid.
    x0 = x_v[np.argmin(p_v)]
    for _ in range(3):
        far = np.abs(x_v - x0) >= BASELINE_FRAC * R_phys
        T0 = float(np.median(p_v[far])) if far.sum() >= 3 else float(np.max(p_v))
        depth = np.clip(T0 - p_v, 0.0, None)
        heavy = depth >= 0.5 * depth.max()
        if not heavy.any() or depth[heavy].sum() <= 0:
            break
        x0 = float(np.average(x_v[heavy], weights=depth[heavy]))

    far = np.abs(x_v - x0) >= BASELINE_FRAC * R_phys
    T0 = float(np.median(p_v[far])) if far.sum() >= 3 else float(np.max(p_v))
    return x0, T0


def build_core_mask(x_cm, profile, x0_cm, T0, R_phys):
    """
    Select the deepest part of the dip, away from the blurred edge.

    Combines an intensity criterion (depth >= CORE_DEPTH_FRAC of the maximum
    depth) with two geometric caps, so that saturated dips from strong absorbers
    do not silently pull edge pixels into the fit.
    """
    valid = np.isfinite(profile)
    depth = np.where(valid, T0 - profile, np.nan)
    max_depth = np.nanmax(depth)
    if not np.isfinite(max_depth) or max_depth <= 0:
        raise RuntimeError("no measurable attenuation in this ROI")

    deep_enough = depth >= CORE_DEPTH_FRAC * max_depth

    dist = np.abs(x_cm - x0_cm)
    r_limit = min(CORE_MAX_FRAC * R_phys,
                  R_phys - EDGE_EXCLUSION_PX * PIXEL_SIZE_CM)
    inside = dist <= r_limit

    mask = valid & deep_enough & inside
    if mask.sum() < MIN_FIT_POINTS:      # fall back to the geometric core alone
        mask = valid & inside
    if mask.sum() < MIN_FIT_POINTS:
        raise RuntimeError(f"only {mask.sum()} points survive the core mask")
    return mask


def fit_sigma_fixed_R(x_cm, profile, mask, x0_guess, T0, R_phys):
    """Non-linear fit of Sigma (and x0) with T0 and R held fixed."""
    if FIX_T0_TO_BASELINE:
        def model(x, Sigma, x0):
            return cylinder_transmittance(x, T0, Sigma, x0, R_phys)

        p0 = [0.5, x0_guess]
        bounds = ([1e-4, x0_guess - 0.3 * R_phys], [50.0, x0_guess + 0.3 * R_phys])
        popt, pcov = curve_fit(model, x_cm[mask], profile[mask],
                               p0=p0, bounds=bounds, maxfev=20000)
        sigma, x0 = popt
        T0_out, dsigma = T0, float(np.sqrt(pcov[0, 0]))
    else:
        def model(x, T0_f, Sigma, x0):
            return cylinder_transmittance(x, T0_f, Sigma, x0, R_phys)

        p0 = [T0, 0.5, x0_guess]
        bounds = ([0.85 * T0, 1e-4, x0_guess - 0.3 * R_phys],
                  [1.15 * T0, 50.0, x0_guess + 0.3 * R_phys])
        popt, pcov = curve_fit(model, x_cm[mask], profile[mask],
                               p0=p0, bounds=bounds, maxfev=20000)
        T0_out, sigma, x0 = popt
        dsigma = float(np.sqrt(pcov[1, 1]))

    resid = profile[mask] - cylinder_transmittance(x_cm[mask], T0_out, sigma, x0, R_phys)
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return T0_out, sigma, x0, dsigma, rmse


def fit_sigma_linear(x_cm, profile, mask, x0_cm, T0, R_phys):
    """
    Beer-Lambert cross-check: -ln(T/T0) = Sigma * L + c.

    The intercept c is the useful diagnostic. c < 0 means the dip is shallower
    than a pure exponential, the signature of a scattered-neutron background
    filling in the sample; c ~ 0 means the core is clean.
    """
    L = chord_length(x_cm[mask], x0_cm, R_phys)
    T = profile[mask]
    ok = (T > 0) & (L > 0)
    if ok.sum() < MIN_FIT_POINTS:
        return np.nan, np.nan, np.nan

    y = -np.log(T[ok] / T0)
    coeffs, cov = np.polyfit(L[ok], y, 1, cov=True)
    sigma_lin, intercept = float(coeffs[0]), float(coeffs[1])
    dsigma_lin = float(np.sqrt(cov[0, 0]))
    return sigma_lin, dsigma_lin, intercept


def fit_free_radius(x_cm, profile, x0_guess, T0, R_phys):
    """
    Diagnostic only: let R float over the full profile.

    Returns (R_free, Sigma_free, corr(Sigma, R)). Expect R_free > R_phys and a
    correlation close to -1, which is exactly why R is fixed in the main fit.
    """
    valid = np.isfinite(profile)
    if valid.sum() < MIN_FIT_POINTS + 2:
        return np.nan, np.nan, np.nan

    def model(x, T0_f, Sigma, x0, R):
        return cylinder_transmittance(x, T0_f, Sigma, x0, R)

    p0 = [T0, 0.5, x0_guess, R_phys]
    bounds = ([0.85 * T0, 1e-4, x0_guess - 0.3 * R_phys, 0.4 * R_phys],
              [1.15 * T0, 50.0, x0_guess + 0.3 * R_phys, 2.5 * R_phys])
    try:
        popt, pcov = curve_fit(model, x_cm[valid], profile[valid],
                               p0=p0, bounds=bounds, maxfev=40000)
    except (RuntimeError, ValueError):
        return np.nan, np.nan, np.nan

    sigma_free, R_free = popt[1], popt[3]
    denom = np.sqrt(pcov[1, 1] * pcov[3, 3])
    corr = float(pcov[1, 3] / denom) if np.isfinite(denom) and denom > 0 else np.nan
    return float(R_free), float(sigma_free), corr


# =============================================================================
# PER-SET ANALYSIS
# =============================================================================

def analyse_cylinder_set(image, set_config):
    label = set_config["label"]
    cylinders = set_config["cylinders"]
    print(f"\nFitting {label} cylinders (fixed R, depth-selected core)...")

    records = []
    fig, axes = plt.subplots(1, len(cylinders), figsize=(18, 4.4), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (name, cfg) in zip(axes, cylinders.items()):
        x_min, x_max = cfg["x"]
        y_min, y_max = cfg.get("y", DEFAULT_Y_RANGE)
        R_phys = PHYSICAL_RADII[name]

        roi = image[y_min:y_max, x_min:x_max]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            profile = np.nanmedian(roi, axis=0)

        x_pixels = np.arange(x_min, x_max)
        x_cm = x_pixels * PIXEL_SIZE_CM

        ax.plot(x_pixels, profile, "o", color="0.6", alpha=0.6,
                markersize=4, label="Median profile")

        try:
            x0_guess, T0_base = estimate_center_and_baseline(x_cm, profile, R_phys)
            mask = build_core_mask(x_cm, profile, x0_guess, T0_base, R_phys)
            T0_fit, sigma, x0_fit, dsigma, rmse = fit_sigma_fixed_R(
                x_cm, profile, mask, x0_guess, T0_base, R_phys)
        except (RuntimeError, ValueError) as err:
            print(f"    {name}: FIT FAILED ({err})")
            ax.set_title(f"{name}\nFIT FAILED")
            ax.set_xlabel("X / pixels")
            ax.grid(True, alpha=0.3)
            continue

        sigma_lin, dsigma_lin, intercept = fit_sigma_linear(
            x_cm, profile, mask, x0_fit, T0_fit, R_phys)

        if RUN_FREE_R_DIAGNOSTIC:
            R_free, sigma_free, corr_sr = fit_free_radius(
                x_cm, profile, x0_guess, T0_base, R_phys)
        else:
            R_free = sigma_free = corr_sr = np.nan

        mfp = 1.0 / sigma if sigma > 0 else np.nan
        dmfp = dsigma / sigma ** 2 if sigma > 0 else np.nan

        # Highlight the points actually used
        ax.plot(x_pixels[mask], profile[mask], "o", color="k",
                markersize=4, label=f"Fitted core (n={mask.sum()})")
        ax.axhline(T0_fit, color="0.5", linestyle=":", linewidth=1)
        ax.plot(x_pixels, cylinder_transmittance(x_cm, T0_fit, sigma, x0_fit, R_phys),
                set_config["fit_color"], linewidth=2.5, label=f"Fixed R = {R_phys} cm")
        if np.isfinite(R_free):
            ax.plot(x_pixels,
                    cylinder_transmittance(x_cm, T0_fit, sigma_free, x0_fit, R_free),
                    color="orange", linestyle="--", linewidth=1.5,
                    label=f"Free R = {R_free:.3f} cm")

        records.append({
            "Sample_ID": name,
            "Exp_Mean": sigma,
            "Exp_Std": dsigma,
            "MFP_cm": mfp,
            "MFP_Std_cm": dmfp,
            "Sigma_linear": sigma_lin,
            "Sigma_linear_Std": dsigma_lin,
            "LogFit_Intercept": intercept,
            "T0_baseline": T0_fit,
            "Fit_RMSE": rmse,
            "N_fit_points": int(mask.sum()),
            "R_phys_cm": R_phys,
            "R_free_cm": R_free,
            "R_free_ratio": R_free / R_phys if np.isfinite(R_free) else np.nan,
            "Sigma_free_R": sigma_free,
            "Corr_Sigma_R": corr_sr,
            "Pixel_Size_cm": PIXEL_SIZE_CM,
            "R_fit_px": R_phys / PIXEL_SIZE_CM,
        })

        ax.set_title(f"{name}\n"
                     f"$\\Sigma$: {sigma:.3f} $\\pm$ {dsigma:.3f} cm$^{{-1}}$\n"
                     f"MFP: {mfp:.2f} cm")
        ax.set_xlabel("X / pixels")
        ax.set_ylim(T_VMIN, T_VMAX)
        ax.grid(True, alpha=0.3)

        print(f"    {name}: Sigma = {sigma:.4f} +/- {dsigma:.4f} cm^-1 "
              f"| linear {sigma_lin:.4f} (c = {intercept:+.4f}) "
              f"| free R = {R_free:.4f} cm ({R_free / R_phys:.2f}x, "
              f"corr = {corr_sr:+.2f}) | n = {mask.sum()}")

    axes[0].set_ylabel("Transmittance (-)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{label} - cylinder transmittance profiles")
    fig.tight_layout()
    plt.show()

    return records


# =============================================================================
# SUMMARY PLOT
# =============================================================================

def plot_cross_section_bars(records, filename="CROCUS_Cylinder_XS_BarPlot.png"):
    materials = [r["Sample_ID"] for r in records]
    sigmas = [r["Exp_Mean"] for r in records]
    errors = [r["Exp_Std"] for r in records]

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        x_pos = np.arange(len(materials))
        bars = ax.bar(x_pos, sigmas, 0.5, yerr=errors, capsize=4, ecolor="white",
                      color="#00ffff", edgecolor="white", zorder=3,
                      label="CROCUS (Li6F:ZnS(Cu) 300 um)")

        ax.set_ylabel("Macroscopic cross section $\\Sigma$ ($cm^{-1}$)",
                      color="white", fontsize=12)
        ax.set_title("Calculated macroscopic cross sections (CROCUS)",
                     color="white", fontsize=14, pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(materials, rotation=45, ha="right",
                           color="#cccccc", fontsize=12, fontweight="bold")
        ax.tick_params(axis="y", colors="#cccccc")
        ax.grid(True, axis="y", linestyle="--", color="#444444", alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("white")
        ax.legend(frameon=False, labelcolor="white", fontsize=12)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom",
                        color="white", fontsize=10, fontweight="bold")

        fig.tight_layout()
        fig.savefig(filename, transparent=True, dpi=350)
        plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    catalog = ExperimentCatalog(MASTER_DIR)

    transmittance_results = {}
    for sample, ob, dc in EXPERIMENT_LIST:
        print("\n" + "=" * 39)
        print(f"STARTING BATCH RUN FOR: {sample}")
        print("=" * 39)
        transmittance_results[sample] = process_transmittance(
            catalog, sample, ob, dc, show_plots=0)

    extracted_sigmas = []
    for set_config in CYLINDER_SETS:
        image = transmittance_results.get(set_config["sample"])
        if image is None:
            print(f"Skipping {set_config['label']}: no transmittance image available.")
            continue
        extracted_sigmas += analyse_cylinder_set(image, set_config)

    if not extracted_sigmas:
        print("\nNo successful fits - nothing to export.")
        return

    print("\nGenerating final cross-section bar plot for all cylinders...")
    plot_cross_section_bars(extracted_sigmas)

    print("\n" + "=" * 39)
    print("EXPORTING MACROSCOPIC EXPERIMENTAL DATA")
    print("=" * 39)

    df_exp = pd.DataFrame(extracted_sigmas)
    df_exp.to_csv(CSV_FILENAME, index=False)

    print(f"Successfully exported {len(df_exp)} samples to '{CSV_FILENAME}'")
    print("\nExtracted experimental summary:")
    print(df_exp[["Sample_ID", "Exp_Mean", "Exp_Std", "Sigma_linear",
                  "LogFit_Intercept", "R_free_ratio", "MFP_cm"]].to_string(index=False))


if __name__ == "__main__":
    main()