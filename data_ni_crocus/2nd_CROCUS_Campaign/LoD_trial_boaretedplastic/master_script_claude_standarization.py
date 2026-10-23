# -*- coding: utf-8 -*-
"""
Two-point L/D determination from a borated-plastic edge
========================================================
Two configurations of the same absorber:
  1. against the scintillator  (l1 small)  -> essentially the detector blur
  2. at the flight-tube distance (l2 large) -> geometry-dominated blur

Model (quadrature addition of blur contributions):
    B_tot(l)^2 = B_int^2 + (l * D/L)^2
Two measurements give a closed-form solution for both D/L and B_int:
    (D/L)^2 = (B2^2 - B1^2) / (l2^2 - l1^2)
    B_int^2 = B1^2 - l1^2 * (D/L)^2

BLUR METRIC
-----------
The edge is fitted with an error function (same as the Cd-edge analysis):
    ESF(x) = y0 + a * erf((x - x0)/t)
whose underlying Gaussian has sigma = t/sqrt(2). Several widths follow:
    FWHM    = 2.3548 * sigma
    W_10_90 = 2.5631 * sigma          (classic 10-90% edge width)
    W_full  = 4.0    * sigma          (approximate full 0-100% ramp)
BLUR_METRIC below selects which one is reported as B. This matters: the
geometric penumbra in B = l*D/L is conventionally the FULL transition
width, so using the 10-90 width underestimates B and therefore
OVERestimates L/D by ~1.56x. Both configurations must use the same metric.

Indexing: array[row, col]; with origin='lower' the imshow y-axis is rows
(1st index) and the x-axis is columns (2nd index).
"""
from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.special import erf
import warnings
import pandas as pd

# =====================================================================
# USER PARAMETERS
# =====================================================================
MASTER_DIR = (r"C:\Users\miche\Desktop\PhD\Neutron_Imaging_project"
              r"\2nd_CROCUS_Campaign\LoD_trial_boaretedplastic")

PIXEL_SIZE_MM = 0.1059      # <-- VERIFY THIS. It scales B1, B2 and hence L/D
                            # linearly. Calibrate against a known-length object.

L1_MM = 50.0                # absorber against the scintillator
L2_MM = 2800.0              # absorber at the flight-tube position

# HOW TO GET B_int
#   'direct'   : treat the near measurement AS the intrinsic blur (B_int = B1).
#                Correct when the absorber is essentially touching the screen,
#                and numerically far better conditioned than the subtraction.
#                D/L then comes from the far point alone.
#   'twopoint' : solve the 2x2 system for both B_int and D/L. Only use this
#                when l1 is genuinely large enough to matter; with l1 << l2 the
#                intercept B_int^2 = B1^2 - l1^2*(D/L)^2 is a difference of two
#                similar numbers and easily goes negative on small errors.
B_INT_MODE = "direct"

# Sampling geometry: a band of rows is averaged for each profile, and the
# band is stepped across several positions to get a spread on B.
SCINT_Y_CENTER, SCINT_X_START, SCINT_X_END = 1000, 900, 1500
FLIGHT_Y_CENTER, FLIGHT_X_START, FLIGHT_X_END = 1200, 700, 1700

LINE_WIDTH = 10             # rows averaged per profile
N_SAMPLE_LINES = 7          # profiles taken at different y, for statistics
LINE_SPACING = 40           # row spacing between sampling lines

# 'fwhm' | 'w1090' | 'full'  -- must be identical for both configurations
BLUR_METRIC = "full"

PLATEAU_FRAC = 0.10         # fraction of the window used to estimate plateaus
PLATEAU_TOL = 0.15          # warn if a plateau is not flat to within this


# =====================================================================
# DATA CATALOG
# =====================================================================
class ExperimentCatalog:
    def __init__(self, master_directory):
        self.master_path = Path(master_directory)
        self.folders = {f.name: f for f in self.master_path.iterdir() if f.is_dir()}
        print(f"Catalog initialized! Found {len(self.folders)} folders.")
        print(f"Available folders: {list(self.folders.keys())}\n")

    def _load_to_tensor(self, file_paths):
        if not file_paths:
            return None
        list_img = []
        for path in file_paths:
            with fits.open(path) as hdul:
                image_data = hdul[0].data
                datt = image_data[0, :, :] if image_data.ndim == 3 else image_data
                list_img.append(datt)
        return np.array(list_img)

    def get_data(self, folder_name_string):
        if folder_name_string not in self.folders:
            raise ValueError(f"Folder '{folder_name_string}' not found in catalog!")
        print(f"--- Loading Data for '{folder_name_string}' ---")
        all_files = list(self.folders[folder_name_string].rglob("*.fits"))
        if len(all_files) == 0:
            print(f"Warning: No .fits files found in {folder_name_string}")
            return None
        return self._load_to_tensor(all_files)

    def remove_outliers(self, image, radius, threshold):
        """Median-filter despeckle. NOTE: `threshold` is in the units of the
        image passed in -- a threshold of 70 does nothing to transmittance
        data whose range is ~0-1."""
        box_size = (2 * radius) + 1
        median_img = median_filter(image, size=box_size)
        difference = np.abs(image.astype(np.float32) - median_img.astype(np.float32))
        cleaned = np.where(difference > threshold, median_img, image)
        return cleaned.astype(image.dtype)


# =====================================================================
# EDGE FITTING
# =====================================================================
SIGMA_TO = {"fwhm": 2.35482, "w1090": 2.56310, "full": 4.0, "mtf10": 2*1.464}


def esf_erf(x, y0, a, x0, t):
    return y0 + a * erf((x - x0) / t)


def fit_edge(x_px, profile):
    """
    Fit ESF(x) = y0 + a*erf((x-x0)/t).
    Returns dict with x0, t, sigma, plateau levels and fit quality, or None.
    """
    prof = np.asarray(profile, dtype=float)
    y0_g = 0.5 * (prof.min() + prof.max())
    a_g = 0.5 * (prof.max() - prof.min())
    x0_g = x_px[np.argmax(np.abs(np.gradient(gaussian_filter1d(prof, 3))))]

    for a_try in (a_g, -a_g):          # either edge polarity
        try:
            popt, pcov = curve_fit(esf_erf, x_px, prof,
                                   p0=[y0_g, a_try, x0_g, 20.0], maxfev=10000)
            y0, a, x0, t = popt
            t = abs(t)
            if not (x_px[0] < x0 < x_px[-1]):
                continue
            if t > (x_px[-1] - x_px[0]) / 2:
                continue
            resid = prof - esf_erf(x_px, *popt)
            rms = float(np.sqrt(np.mean(resid**2)))
            return {"x0": x0, "t": t, "sigma": t / np.sqrt(2.0),
                    "low": y0 - abs(a), "high": y0 + abs(a),
                    "rms": rms, "popt": popt,
                    "t_err": float(np.sqrt(np.diag(pcov))[3])}
        except (RuntimeError, ValueError):
            continue
    return None


def blur_from_sigma(sigma_px, pixel_size_mm, metric=BLUR_METRIC):
    return sigma_px * SIGMA_TO[metric] * pixel_size_mm


def check_plateaus(profile, label):
    """
    Verify the window actually contains flat regions on BOTH sides.
    Without them the edge amplitude is undefined and the blur is biased.
    """
    prof = gaussian_filter1d(np.asarray(profile, float), 3)
    n = len(prof)
    k = max(10, int(n * PLATEAU_FRAC))
    left, right = prof[:k], prof[-k:]
    span = abs(left.mean() - right.mean())
    if span <= 0:
        print(f"  [{label}] WARNING: no contrast across the window.")
        return False
    left_slope = abs(left[-1] - left[0]) / span
    right_slope = abs(right[-1] - right[0]) / span
    ok = True
    if left_slope > PLATEAU_TOL:
        print(f"  [{label}] WARNING: left side is still sloping "
              f"({left_slope*100:.0f}% of contrast across the plateau window).\n"
              f"           Extend X_START further from the edge.")
        ok = False
    if right_slope > PLATEAU_TOL:
        print(f"  [{label}] WARNING: right side is still sloping "
              f"({right_slope*100:.0f}% of contrast across the plateau window).\n"
              f"           Extend X_END further from the edge.")
        ok = False
    return ok


# =====================================================================
# PROFILE EXTRACTION + PLOTTING
# =====================================================================
def measure_edge(image, title, y_center, x_start, x_end,
                 line_width=LINE_WIDTH, n_lines=N_SAMPLE_LINES,
                 spacing=LINE_SPACING, pixel_size_mm=PIXEL_SIZE_MM):
    """
    Extract several horizontal profiles at different y, fit each, and return
    (B_mm_mean, B_mm_std, diagnostics). Also produces the diagnostic figure.
    """
    x_px = np.arange(x_start, x_end)
    offsets = (np.arange(n_lines) - n_lines // 2) * spacing

    sigmas, fits_ok, profiles = [], [], []
    for off in offsets:
        yc = y_center + off
        r0, r1 = yc - line_width // 2, yc + line_width // 2
        prof = np.mean(image[r0:r1, x_start:x_end], axis=0)   # rows, cols
        profiles.append((yc, prof))
        res = fit_edge(x_px, prof)
        if res is not None:
            sigmas.append(res["sigma"])
            fits_ok.append((yc, res))

    if not sigmas:
        raise RuntimeError(f"{title}: no profile could be fitted.")

    sigmas = np.array(sigmas)
    B_vals = blur_from_sigma(sigmas, pixel_size_mm)
    B_mean, B_std = float(B_vals.mean()), float(B_vals.std())

    # ---- plateau sanity check on the central profile ----
    central = profiles[len(profiles) // 2][1]
    check_plateaus(central, title)

    # ---- figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.imshow(image, cmap="gray", origin="lower", vmin=0.0, vmax=1.2)
    for yc, _ in profiles:
        ax1.plot([x_start, x_end], [yc, yc], color="red", ls="--", lw=1, alpha=0.6)
    ax1.set_title(f"{title} - sampling lines")
    ax1.set_xlabel("X / pixels"); ax1.set_ylabel("Y / pixels")
    ax1.set_xlim(350, 1700); ax1.set_ylim(350, 1700)

    ax2.plot(x_px, central, color="0.6", lw=1, label="measured ESF (central line)")
    yc_c, res_c = fits_ok[len(fits_ok) // 2]
    ax2.plot(x_px, esf_erf(x_px, *res_c["popt"]), "r-", lw=2,
             label=f"erf fit: $\\sigma$ = {res_c['sigma']:.1f} px")
    # mark the reported blur width around the edge centre
    half = 0.5 * res_c["sigma"] * SIGMA_TO[BLUR_METRIC]
    ax2.axvspan(res_c["x0"] - half, res_c["x0"] + half, color="orange", alpha=0.2,
                label=f"B ({BLUR_METRIC}) = {2*half*pixel_size_mm:.2f} mm")
    ax2.axhline(res_c["low"], color="b", ls=":", lw=1, alpha=0.7)
    ax2.axhline(res_c["high"], color="b", ls=":", lw=1, alpha=0.7)
    ax2.set_title("Edge spread function and fit")
    ax2.set_xlabel("X position / pixels"); ax2.set_ylabel("Transmittance")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

    return B_mean, B_std, fits_ok


# =====================================================================
# TRANSMITTANCE
# =====================================================================
def process_transmittance(catalog, sample_folder, ob_folder, dc_folder,
                          threshold=100, radius=4, show_plots=False):
    print(f"\n>>> Processing Transmittance for: {sample_folder} <<<")
    dc = catalog.get_data(dc_folder)
    sample_data = catalog.get_data(sample_folder)
    sample_median = np.median(sample_data - dc, axis=0)
    sample_clean = catalog.remove_outliers(sample_median, radius=radius,
                                           threshold=threshold)

    ob_data = catalog.get_data(ob_folder)
    ob_median = np.median(ob_data - dc, axis=0)
    ob_clean = catalog.remove_outliers(ob_median, radius=radius,
                                       threshold=threshold)

    transmittance = sample_clean / (ob_clean + 1e-6)

    if show_plots:
        plt.figure(figsize=(7, 6))
        plt.imshow(transmittance, cmap="gray", origin="lower", vmin=0.0, vmax=1.2)
        plt.colorbar(label="Transmittance (-)")
        plt.title(f"Transmittance: {sample_folder}")
        plt.xlabel("X / pixels"); plt.ylabel("Y / pixels")
        plt.xlim(350, 1700); plt.ylim(350, 1700)
        plt.show()

    return transmittance


# =====================================================================
# TWO-POINT L/D
# =====================================================================
def ld_direct(l2, B2, B_int, dB2=0.0, dB_int=0.0):
    """
    Near-contact route. B_int is MEASURED (= B1), not inferred, so the far
    point alone gives the geometric term:
        B2^2 = B_int^2 + (l2 * D/L)^2
        =>  D/L = sqrt(B2^2 - B_int^2) / l2
    Well conditioned whenever B2 > B_int, which is the whole point of
    putting the second measurement far away.
    """
    geo_sq = B2**2 - B_int**2
    if geo_sq <= 0:
        print("ERROR: the far blur is not larger than the intrinsic blur.")
        return None, None
    geo = np.sqrt(geo_sq)          # purely geometric blur at l2
    R = geo / l2
    ld = 1.0 / R
    # propagate: d(geo)/dB2 = B2/geo, d(geo)/dB_int = -B_int/geo
    dgeo = np.hypot((B2 / geo) * dB2, (B_int / geo) * dB_int)
    dld = ld * dgeo / geo
    return ld, dld


def ld_two_point(l1, B1, l2, B2, dB1=0.0, dB2=0.0):
    """
    Solve B_tot^2 = B_int^2 + (l*D/L)^2 for two (l, B) pairs.
    Returns (L/D, B_int, dLD) with dLD propagated from dB1, dB2.
    """
    if l2 <= l1:
        raise ValueError("l2 must exceed l1.")
    r_sq = (B2**2 - B1**2) / (l2**2 - l1**2)
    if r_sq <= 0:
        print("ERROR: B2 <= B1 -- the far measurement is not blurrier than the "
              "near one. Check the metric, the distances, or the data.")
        return None, None, None
    R = np.sqrt(r_sq)
    ld = 1.0 / R
    b_int_sq = B1**2 - l1**2 * r_sq
    if b_int_sq < 0:
        print("WARNING: negative intrinsic blur squared; clipping to 0. "
              "The near measurement is smaller than the geometric term predicts.")
        b_int = 0.0
    else:
        b_int = np.sqrt(b_int_sq)

    # Error propagation: dR/dB1, dR/dB2
    denom = (l2**2 - l1**2)
    dR_dB1 = -B1 / (R * denom)
    dR_dB2 = B2 / (R * denom)
    dR = np.hypot(dR_dB1 * dB1, dR_dB2 * dB2)
    dld = ld * dR / R if R > 0 else 0.0
    return ld, b_int, dld


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    catalog = ExperimentCatalog(MASTER_DIR)

    experiment_list = [
        ("BoratedPlastic_scint", "OB_Li6_1", "DC"),      # near: detector blur
        ("BoratedPlastic_flightube", "OB_Li6_1", "DC"),  # far: geometry-dominated
    ]

    transmittance_results = {}
    for sample, ob, dc in experiment_list:
        print(f"\n=======================================")
        print(f"STARTING BATCH RUN FOR: {sample}")
        print(f"=======================================")
        transmittance_results[sample] = process_transmittance(
            catalog, sample, ob, dc, threshold=70, radius=4, show_plots=False)

    print("\n=======================================")
    print("EDGE MEASUREMENTS")
    print("=======================================")

    B1, dB1, fits1 = measure_edge(
        transmittance_results["BoratedPlastic_scint"],
        "Near: absorber on scintillator",
        SCINT_Y_CENTER, SCINT_X_START, SCINT_X_END)
    print(f"  B1 = {B1:.3f} +/- {dB1:.3f} mm   "
          f"({B1/PIXEL_SIZE_MM:.1f} px, metric = {BLUR_METRIC})")

    B2, dB2, fits2 = measure_edge(
        transmittance_results["BoratedPlastic_flightube"],
        "Far: absorber at flight tube",
        FLIGHT_Y_CENTER, FLIGHT_X_START, FLIGHT_X_END)
    print(f"  B2 = {B2:.3f} +/- {dB2:.3f} mm   "
          f"({B2/PIXEL_SIZE_MM:.1f} px, metric = {BLUR_METRIC})")

    # ---- L/D ----
    print("\n=======================================")
    print(f"L/D  (B_int mode = '{B_INT_MODE}')")
    print("=======================================")
    print(f"Point 1: l = {L1_MM:7.1f} mm,  B = {B1:.3f} +/- {dB1:.3f} mm")
    print(f"Point 2: l = {L2_MM:7.1f} mm,  B = {B2:.3f} +/- {dB2:.3f} mm")
    print("-" * 39)

    if B_INT_MODE == "direct":
        # B_int is measured directly at the near (contact) position.
        b_int = B1
        ld, dld = ld_direct(L2_MM, B2, b_int, dB2, dB1)
        if ld is not None:
            geo = np.sqrt(B2**2 - b_int**2)
            print(f"B_int (measured at l1)   : {b_int:.3f} mm "
                  f"({b_int/PIXEL_SIZE_MM:.1f} px)")
            print(f"Geometric blur at l2     : sqrt({B2:.3f}^2 - {b_int:.3f}^2)"
                  f" = {geo:.3f} mm")
            print(f"D/L                      : {geo:.3f}/{L2_MM:.0f}"
                  f" = {geo/L2_MM:.6f}")
            print(f"Blur metric              : {BLUR_METRIC}")
            print(f"Pixel size               : {PIXEL_SIZE_MM} mm")
            print(f"\n>>> L/D = {ld:.1f} +/- {dld:.1f} <<<")
    else:
        ld, b_int, dld = ld_two_point(L1_MM, B1, L2_MM, B2, dB1, dB2)
        if ld is not None:
            R = 1.0 / ld
            print(f"D/L                      : {R:.6f}")
            print(f"Geometric blur at l1     : {L1_MM*R:.3f} mm  "
                  f"(vs measured B1 = {B1:.3f} mm)")
            if L1_MM * R > B1:
                print("  ^ larger than B1 -> B_int^2 < 0. The near point is "
                      "sharper than\n    the fitted geometry allows. Either l1 "
                      "is smaller than assumed\n    (use B_INT_MODE='direct'), "
                      "or B2 is overestimated.")
            print(f"Intrinsic blur B_int     : {b_int:.3f} mm "
                  f"({b_int/PIXEL_SIZE_MM:.1f} px)")
            print(f"Blur metric              : {BLUR_METRIC}")
            print(f"Pixel size               : {PIXEL_SIZE_MM} mm")
            print(f"\n>>> L/D = {ld:.1f} +/- {dld:.1f} <<<")

    # ---- Metric sensitivity (recomputed consistently in the active mode) ----
    if ld is not None:
        print("\nSensitivity to the blur-width convention:")
        for m in ("fwhm", "w1090", "full","mtf10"):
            scale = SIGMA_TO[m] / SIGMA_TO[BLUR_METRIC]
            B1_m, B2_m = B1 * scale, B2 * scale
            if B_INT_MODE == "direct":
                ld_m, _ = ld_direct(L2_MM, B2_m, B1_m)
                b_m = B1_m
            else:
                ld_m, b_m, _ = ld_two_point(L1_MM, B1_m, L2_MM, B2_m)
            if ld_m:
                print(f"  {m:6s}: L/D = {ld_m:6.1f},  B_int = {b_m:.3f} mm")
        print("\nNOTE: L/D scales inversely with the width convention, so quote "
              "the metric alongside the number.")
        
    
    
        
        
        
        
        