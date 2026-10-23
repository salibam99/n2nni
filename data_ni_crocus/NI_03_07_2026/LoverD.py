# -*- coding: utf-8 -*-
"""
Cd-edge L/D analysis  --  sigma-based (primary) + MTF metrics (comparison)
==========================================================================
PRIMARY METRIC: the erf fit
    ESF(x) = y0 + a*erf((x - x0)/t)                     Eq. (18)
    the underlying LSF is Gaussian with  sigma = t / sqrt(2)
sigma is the complete description of the blur -- one parameter, no
frequency convention, no 10%-threshold sensitivity. Every width metric is
just a rescaling of it:
    FWHM     = 2.3548 * sigma
    W_10_90  = 2.5631 * sigma
    W_full   = 4.0000 * sigma        (approx. full transition width)
    u_10%MTF = 4.3958 * sigma        (= 2*pi*t/3.03485, angular convention)
Set BLUR_METRIC to choose which one is reported. 'sigma' reports sigma
itself. The results block prints L/D under ALL metrics so the convention
dependence is explicit rather than hidden.

Blur combination (Eq. 2, generalized exponent):
    u_tot^n = u_det^n + u_geo^n,   u_geo(z) = z * (D/L)
u_det is MEASURED near z = 0 (contact), not fitted.
Both n = 2 (common practice) and n_opt (fitted) are reported.

Method B (presampled MTF: smooth -> derivative -> baseline -> Hann -> FFT)
is retained purely as an independent cross-check of the absolute scale.

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

# =====================================================================
# USER PARAMETERS
# =====================================================================
X_SHEET_LEFT, X_SHEET_RIGHT = 500, 1090   # columns of the sheet ends
Z_LEFT_CM, Z_RIGHT_CM = 0.0, 8.0          # their distances from the scintillator

Y_START, Y_END = 750, 1100                # rows containing the edge
PIXEL_SIZE_MM = 0.1                       # 100 um pixels

EDGE_MARGIN_PX = 20      # step inside the sheet ends before fitting
COLUMN_BIN = 4           # columns averaged per ESF fit


Z_DET_MAX_CM = 0.5      # points with z <= this define u_det
Z_FAR_MIN_CM = 5.0       # far-distance linear fit region

# Which width to report and fit:
#   'sigma' | 'fwhm' | 'w1090' | 'full' | 'mtf10'
BLUR_METRIC = "mtf10"

LSF_SMOOTH_SIGMA = 1.5   # Gaussian pre-smoothing for Method B and LSF plots
LSF_HALF_WIN_PX = 25     # half-window for the centered LSF figure

L_OVER_D_EXPECTED = None   # nominal beamline value (None to skip)

# sigma -> width conversion factors
SIGMA_TO = {
    "sigma": 1.0,
    "fwhm":  2.35482,
    "w1090": 2.56310,
    "full":  4.0,
    "mtf10": 2.0 * np.pi * np.sqrt(2.0) / (2.0 * np.sqrt(np.log(10.0))),  # Line Pair Width= 2.928 sigma
}


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
        box_size = (2 * radius) + 1
        median_img = median_filter(image, size=box_size)
        difference = np.abs(image.astype(np.float32) - median_img.astype(np.float32))
        cleaned = np.where(difference > threshold, median_img, image)
        return cleaned.astype(image.dtype)


# =====================================================================
# PRIMARY: erf ESF fit -> sigma
# =====================================================================
def esf_erf(x, y0, a, x0, t):
    """Eq. (18)."""
    return y0 + a * erf((x - x0) / t)


def fit_esf(x_px, profile):
    """
    Fit Eq. (18). Returns (x0, t, sigma, sigma_err) or None.
    sigma = t/sqrt(2) is the Gaussian width of the underlying LSF.
    """
    prof = np.asarray(profile, dtype=float)
    y0_g = 0.5 * (prof.min() + prof.max())
    a_g = 0.5 * (prof.max() - prof.min())
    x0_g = x_px[np.argmax(np.abs(np.gradient(gaussian_filter1d(prof, 2))))]
    for a_try in (a_g, -a_g):          # handles either edge polarity
        try:
            popt, pcov = curve_fit(esf_erf, x_px, prof,
                                   p0=[y0_g, a_try, x0_g, 3.0], maxfev=8000)
            x0, t = popt[2], abs(popt[3])
            if x_px[0] < x0 < x_px[-1] and t < (x_px[-1] - x_px[0]) / 3:
                t_err = float(np.sqrt(np.diag(pcov))[3])
                return x0, t, t / np.sqrt(2.0), t_err / np.sqrt(2.0)
        except (RuntimeError, ValueError):
            continue
    return None


def width_from_sigma(sigma, metric=BLUR_METRIC):
    """Convert sigma to the selected width convention."""
    return sigma * SIGMA_TO[metric]


# =====================================================================
# CROSS-CHECK: presampled MTF
# =====================================================================
def presampled_mtf(profile, smooth_sigma=LSF_SMOOTH_SIGMA):
    prof = np.asarray(profile, dtype=float)
    smoothed = gaussian_filter1d(prof, sigma=smooth_sigma)
    lsf = np.abs(np.gradient(smoothed))

    n = len(lsf)
    tail = max(3, n // 10)
    baseline = 0.5 * (lsf[:tail].mean() + lsf[-tail:].mean())
    lsf = lsf - baseline
    lsf[lsf < 0] = 0.0
    if lsf.sum() <= 0:
        return None, None

    idx = np.arange(n)
    centroid = (idx * lsf).sum() / lsf.sum()
    lsf = np.roll(lsf, int(round(n / 2.0 - centroid))) * np.hanning(n)

    mtf = np.abs(np.fft.rfft(lsf))
    if mtf[0] <= 0:
        return None, None
    return np.fft.rfftfreq(n, d=1.0), mtf / mtf[0]


def sigma_from_presampled(profile, smooth_sigma=LSF_SMOOTH_SIGMA):
    """
    Return sigma (in px) implied by the presampled MTF's 10% crossing,
    so it is directly comparable with the erf sigma.
    For a Gaussian LSF:  MTF(f) = exp(-2*pi^2*sigma^2*f^2)
        MTF = 0.1  =>  sigma = sqrt(ln10 / 2) / (pi * f_0.1)
    """
    freqs, mtf = presampled_mtf(profile, smooth_sigma)
    if freqs is None:
        return None
    below = np.where(mtf < 0.1)[0]
    if len(below) == 0 or below[0] == 0:
        return None
    i = below[0]
    f1, f2, m1, m2 = freqs[i - 1], freqs[i], mtf[i - 1], mtf[i]
    f_01 = f1 + (0.1 - m1) * (f2 - f1) / (m2 - m1)
    if f_01 <= 0:
        return None
    sigma = np.sqrt(np.log(10.0) / 2.0) / (np.pi * f_01)
    # remove the pre-smoothing contribution (added in quadrature)
    sigma_sq = sigma**2 - smooth_sigma**2
    return np.sqrt(sigma_sq) if sigma_sq > 0 else None


# =====================================================================
# BLUR COMBINATION
# =====================================================================
def u_tot_model(z_cm, u_det, inv_LD, n):
    return (u_det**n + (inv_LD * z_cm)**n)**(1.0 / n)


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    master_dir = r"C:\Users\miche\Desktop\PhD\Neutron_Imaging_project\NI_03_07_2026"
    catalog = ExperimentCatalog(master_dir)

    sample_data = catalog.get_data("Cd_edge_dupplicated")
    dc = catalog.get_data("closed_shutter_DC_with_beam_dupplicated")
    sample_median = np.median(sample_data - dc, axis=0)

    ob = catalog.get_data("OB")
    ob_median = np.median(ob - dc, axis=0)

    transmittance = sample_median / (ob_median + 1e-6)
    transmittance = catalog.remove_outliers(transmittance, 1, threshold=1.3)

    PX_TO_UM = PIXEL_SIZE_MM * 1000.0

    # ---- Overview ----
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(transmittance, origin="lower", vmin=0.2, vmax=1.3)
    ax.axvline(X_SHEET_LEFT, color="r", ls="--", lw=1.2,
               label=f"z = {Z_LEFT_CM:g} cm (x={X_SHEET_LEFT})")
    ax.axvline(X_SHEET_RIGHT, color="orange", ls="--", lw=1.2,
               label=f"z = {Z_RIGHT_CM:g} cm (x={X_SHEET_RIGHT})")
    ax.axhline(Y_START, color="w", ls=":", lw=1)
    ax.axhline(Y_END, color="w", ls=":", lw=1)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Transmittance + z-calibration endpoints")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout(); plt.show()

    x_left, x_right = X_SHEET_LEFT, X_SHEET_RIGHT
    x_fit_start = x_left + EDGE_MARGIN_PX
    x_fit_end = x_right - EDGE_MARGIN_PX

    cropped_cd = transmittance[Y_START:Y_END, x_fit_start:x_fit_end]
    y_px = np.arange(Y_START, Y_END)
    n_rows, n_cols = cropped_cd.shape

    # ---- Per-bin erf fits ----
    x_cols, t_vals, sig_A, sig_A_err, sig_B, x0s, profiles = [], [], [], [], [], [], []
    print(f"\nFitting erf ESF (bins of {COLUMN_BIN} columns)...")
    for j in range(0, n_cols - COLUMN_BIN, COLUMN_BIN):
        profile = cropped_cd[:, j:j + COLUMN_BIN].mean(axis=1)
        res = fit_esf(y_px, profile)
        if res is None:
            continue
        x0, t, sigma, sigma_err = res
        x_cols.append(x_fit_start + j + COLUMN_BIN // 2)
        x0s.append(x0); t_vals.append(t)
        sig_A.append(sigma); sig_A_err.append(sigma_err)
        sB = sigma_from_presampled(profile)
        sig_B.append(sB if sB is not None else np.nan)
        profiles.append(profile)

    x_cols = np.array(x_cols); x0s = np.array(x0s); t_vals = np.array(t_vals)
    sig_A = np.array(sig_A); sig_A_err = np.array(sig_A_err); sig_B = np.array(sig_B)
    n_bins = n_cols // COLUMN_BIN
    print(f"erf fits successful        : {len(sig_A)} / {n_bins}")
    print(f"presampled MTF valid       : {np.isfinite(sig_B).sum()} / {n_bins}")

    z_cm = Z_LEFT_CM + (x_cols - x_left) / (x_right - x_left) * (Z_RIGHT_CM - Z_LEFT_CM)

    # ---- Figure 1: sigma vs z, the raw fitted quantity ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(z_cm, sig_A, yerr=sig_A_err, fmt="o", color="purple", ms=4,
                alpha=0.6, elinewidth=0.8, capsize=2,
                label="$\\sigma$ from erf fit")
    okB = np.isfinite(sig_B)
    if okB.any():
        ax.plot(z_cm[okB], sig_B[okB], "s", color="teal", ms=3, alpha=0.35,
                label="$\\sigma$ from presampled MTF (cross-check)")
    ax.set_xlabel("Distance from scintillator z / cm")
    ax.set_ylabel("Gaussian width $\\sigma$ / pixels")
    ax.set_title("Fitted edge width $\\sigma$ vs distance\n"
                 "(the raw quantity, before any width convention)")
    ax.grid(alpha=0.4); ax.legend()
    ax_um = ax.secondary_yaxis(
        "right", functions=(lambda px: px * PX_TO_UM, lambda um: um / PX_TO_UM))
    ax_um.set_ylabel("$\\sigma$ / $\\mu$m")
    plt.tight_layout(); plt.show()

    # ---- Centered LSF plot ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.cm.jet
    z_min, z_max = z_cm.min(), z_cm.max()
    for profile, x0, z in zip(profiles, x0s, z_cm):
        lsf = np.abs(np.gradient(gaussian_filter1d(profile, LSF_SMOOTH_SIGMA)))
        lsf /= lsf.max()
        yc = y_px - x0
        win = np.abs(yc) <= LSF_HALF_WIN_PX
        ax.plot(yc[win], lsf[win], color=cmap((z - z_min) / (z_max - z_min)),
                lw=1.0, alpha=0.30)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=z_min, vmax=z_max))
    sm.set_array([])
    fig.colorbar(sm, ax=ax).set_label("Distance from scintillator z / cm")
    ax.set_xlabel("Distance from edge center / pixels")
    ax.set_ylabel("LSF (peak-normalized)")
    ax.set_title("Line spread functions, centered on the fitted edge position")
    ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

    # ---- Work in the selected metric ----
    u_A = width_from_sigma(sig_A)
    u_det_metric = None

    det_mask = z_cm <= Z_DET_MAX_CM
    if np.count_nonzero(det_mask) < 3:
        det_mask = np.zeros_like(z_cm, dtype=bool)
        det_mask[np.argsort(z_cm)[:5]] = True
        print(f"\nNote: fewer than 3 points within {Z_DET_MAX_CM} cm; using the "
              f"5 nearest (z up to {z_cm[det_mask].max():.2f} cm) for u_det.")
    u_det_px = np.median(u_A[det_mask])
    u_det_err = np.std(u_A[det_mask]) / np.sqrt(np.count_nonzero(det_mask))
    sigma_det = np.median(sig_A[det_mask])

    # ---- Fits ----
    p2, c2 = curve_fit(lambda z, s: u_tot_model(z, u_det_px, s, 2.0),
                       z_cm, u_A, p0=[0.5], maxfev=5000)
    inv_LD_n2, inv_LD_n2_err = abs(p2[0]), np.sqrt(c2[0, 0])

    pn, cn = curve_fit(lambda z, s, n: u_tot_model(z, u_det_px, s, n),
                       z_cm, u_A, p0=[inv_LD_n2, 2.0],
                       bounds=([0, 0.5], [np.inf, 10.0]), maxfev=10000)
    inv_LD_opt, n_opt = abs(pn[0]), pn[1]
    perr_n = np.sqrt(np.diag(cn))

    far = z_cm >= Z_FAR_MIN_CM
    lin_slope = None
    if np.count_nonzero(far) > 5:
        lin_coef, lin_cov = np.polyfit(z_cm[far], u_A[far], 1, cov=True)
        lin_slope, lin_intercept = lin_coef
        lin_slope_err = np.sqrt(lin_cov[0, 0])

    def LD(slope_px_per_cm):
        return 1.0 / (slope_px_per_cm * PIXEL_SIZE_MM / 10.0)

    # ---- Figure 2: blur vs distance in the chosen metric ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(z_cm, u_A * PX_TO_UM, "o", color="purple", ms=4, alpha=0.5,
            label=f"$u_{{tot}}$ ({BLUR_METRIC} = {SIGMA_TO[BLUR_METRIC]:.3f}$\\sigma$)")

    zz = np.linspace(0, z_cm.max(), 200)
    ax.plot(zz, u_tot_model(zz, u_det_px, inv_LD_n2, 2.0) * PX_TO_UM, "k--", lw=2,
            label=f"$n$ = 2:  $L/D$ = {LD(inv_LD_n2):.0f}")
    ax.plot(zz, u_tot_model(zz, u_det_px, inv_LD_opt, n_opt) * PX_TO_UM, "r-", lw=2,
            label=f"$n_{{opt}}$ = {n_opt:.2f}:  $L/D$ = {LD(inv_LD_opt):.0f}")
    if lin_slope is not None:
        zz_far = np.linspace(Z_FAR_MIN_CM, z_cm.max(), 50)
        ax.plot(zz_far, (lin_slope * zz_far + lin_intercept) * PX_TO_UM, "b-",
                lw=1.5, alpha=0.8,
                label=f"linear (z $\\geq$ {Z_FAR_MIN_CM:g} cm):  $L/D$ = {LD(lin_slope):.0f}")
        ax.axvline(Z_FAR_MIN_CM, color="blue", ls=":", alpha=0.35)

    if L_OVER_D_EXPECTED:
        inv_LD_exp = 1.0 / (L_OVER_D_EXPECTED * PIXEL_SIZE_MM / 10.0)
        ax.plot(zz, u_tot_model(zz, u_det_px, inv_LD_exp, 2.0) * PX_TO_UM,
                color="green", ls="-.", lw=1.5, alpha=0.8,
                label=f"expected $L/D$ = {L_OVER_D_EXPECTED:.0f}")

    ax.axhline(u_det_px * PX_TO_UM, color="gray", lw=1, alpha=0.6)
    ax.annotate(f"detector blur = {u_det_px*PX_TO_UM:.0f} $\\mu$m",
                xy=(z_cm.max(), u_det_px * PX_TO_UM),
                xytext=(z_cm.max(), u_det_px * PX_TO_UM * 0.90),
                textcoords="data", fontsize=9, color="gray", ha="right")

    ax.set_xlabel("Distance from scintillator z / cm")
    ax.set_ylabel(f"Blur ({BLUR_METRIC}) / $\\mu$m")
    ax.set_title(f"Blur vs sample-detector distance  [metric: {BLUR_METRIC}]")
    ax.set_xlim(0, z_cm.max() * 1.02)
    top = np.nanpercentile(u_A * PX_TO_UM, 98) * 1.3
    if L_OVER_D_EXPECTED:
        top = max(top, u_tot_model(z_cm.max(), u_det_px, inv_LD_exp, 2.0) * PX_TO_UM * 1.1)
    ax.set_ylim(0, top)
    ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.4)
    ax_px = ax.secondary_yaxis(
        "right", functions=(lambda um: um / PX_TO_UM, lambda px: px * PX_TO_UM))
    ax_px.set_ylabel(f"Blur / pixels")
    plt.tight_layout(); plt.show()

    # =================================================================
    # RESULTS
    # =================================================================
    print("\n================= RESULTS =================")
    print(f"Reported metric            : {BLUR_METRIC} "
          f"({SIGMA_TO[BLUR_METRIC]:.4f} x sigma)")
    print(f"z-calibration              : x {x_left} -> {Z_LEFT_CM:g} cm, "
          f"x {x_right} -> {Z_RIGHT_CM:g} cm")
    print(f"\nRAW FITTED WIDTH (convention-free):")
    print(f"  sigma at contact         : {sigma_det:.3f} px "
          f"({sigma_det*PX_TO_UM:.0f} um)")
    print(f"  sigma at z = {z_cm.max():.1f} cm      : {sig_A[-1]:.3f} px "
          f"({sig_A[-1]*PX_TO_UM:.0f} um)")
    print(f"  sigma growth             : "
          f"{(sig_A[-1]/sigma_det - 1)*100:.0f} % across the range")
    print(f"\nIn the '{BLUR_METRIC}' metric:")
    print(f"  Detector blur u_det      : {u_det_px:.2f} +/- {u_det_err:.2f} px "
          f"({u_det_px*PX_TO_UM:.0f} um)")
    print(f"  n = 2   -> L/D           : {LD(inv_LD_n2):.0f} "
          f"+/- {LD(inv_LD_n2)*inv_LD_n2_err/inv_LD_n2:.0f}")
    print(f"  n_opt = {n_opt:.2f} -> L/D       : {LD(inv_LD_opt):.0f} "
          f"+/- {LD(inv_LD_opt)*perr_n[0]/inv_LD_opt:.0f}")
    if lin_slope is not None:
        print(f"  linear (far) -> L/D      : {LD(lin_slope):.0f} "
              f"+/- {LD(lin_slope)*lin_slope_err/lin_slope:.0f}")

    # ---- L/D under every metric ----
    print(f"\nL/D under each width convention (n = 2):")
    for m, fac in SIGMA_TO.items():
        u_m = sig_A * fac
        u_det_m = np.median(u_m[det_mask])
        try:
            pm, _ = curve_fit(lambda z, s: u_tot_model(z, u_det_m, s, 2.0),
                              z_cm, u_m, p0=[0.5], maxfev=5000)
            print(f"  {m:6s} ({fac:.3f} sigma): L/D = {LD(abs(pm[0])):6.0f},"
                  f"  u_det = {u_det_m*PX_TO_UM:5.0f} um")
        except RuntimeError:
            print(f"  {m:6s}: fit failed")
    print("\nNOTE: L/D scales inversely with the width convention. The RATIO of")
    print("      blur growth to blur floor is convention-independent -- that is")
    print("      the physically meaningful quantity.")

    if okB.any():
        ratio = np.nanmedian(sig_B[okB] / sig_A[okB])
        print(f"\nSigma ratio, presampled / erf (median): {ratio:.3f}")
        print("  ~1.0 confirms the erf fit captures the true edge width.")

    if L_OVER_D_EXPECTED:
        sigma_geo_expected = (z_cm.max() / L_OVER_D_EXPECTED * 10.0
                              / PIXEL_SIZE_MM / SIGMA_TO[BLUR_METRIC])
        sigma_geo_measured = np.sqrt(max(sig_A[-1]**2 - sigma_det**2, 0))
        print(f"\n--- Consistency with L/D = {L_OVER_D_EXPECTED:.0f} ---")
        print(f"  Expected sigma growth at z = {z_cm.max():.1f} cm: "
              f"{sigma_geo_expected:.2f} px")
        print(f"  Measured sigma growth               : "
              f"{sigma_geo_measured:.2f} px")
        print(f"  Ratio                               : "
              f"{sigma_geo_expected/max(sigma_geo_measured,1e-9):.1f}")
    print("===========================================")