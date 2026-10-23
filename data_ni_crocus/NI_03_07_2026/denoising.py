# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:42:51 2026

@author: miche
"""

"""
Standalone denoiser.  Run:   python denoise_pkl.py result.pkl

Loads the cached transmittance array, figures out its own settings,
denoises, plots, and writes denoised.pkl.  Nothing to tune on run 1.
"""

import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
import bm3d

# ---- the only knobs you should ever need -----------------------------
STRENGTH = 1.0        # >1 smoother, <1 more detail preserved
PROFILE = "np"        # "np" normal | "refilter" stronger | "lc" fast
T_RANGE = (0.0, 2.0)  # physically valid transmittance
# ----------------------------------------------------------------------
CROP = (slice(350, 1700), slice(350, 1700))     # cut the noisy outer band
UL_STEP = (slice(1240, 1460), slice(410, 640))  # upper-left step (nudge if off)
 


def load(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    arr = obj["result"] if isinstance(obj, dict) else obj
    arr = np.asarray(arr, dtype=np.float64)
    print(f"[load ] {path}  shape={arr.shape}  dtype={arr.dtype}")
    return arr


def clean(img):
    """Mask junk (I0~0 blowups, NaN) and crop to the valid bounding box."""
    m = np.isfinite(img) & (img > T_RANGE[0]) & (img < T_RANGE[1])
    bad = int((~m).sum())
    img = img.copy()
    img[~m] = np.median(img[m])
    rows, cols = np.where(m.any(1))[0], np.where(m.any(0))[0]
    roi = (slice(rows[0] + 10, rows[-1] - 10), slice(cols[0] + 10, cols[-1] - 10))
    print(f"[mask ] junk pixels {bad} ({100*bad/img.size:.2f} %), "
          f"valid range [{img[m].min():.3f}, {img[m].max():.3f}]")
    print(f"[crop ] rows {roi[0].start}:{roi[0].stop}  cols {roi[1].start}:{roi[1].stop}")
    return img, m, roi


def detrend(p):
    """Remove a fitted plane so gradients are not mistaken for noise."""
    p = np.asarray(p, float)
    yy, xx = np.mgrid[: p.shape[0], : p.shape[1]]
    A = np.column_stack([np.ones(p.size), xx.ravel(), yy.ravel()])
    c, *_ = np.linalg.lstsq(A, p.ravel(), rcond=None)
    return p - (A @ c).reshape(p.shape)


def find_flat_patch(img, size=128, stride=96):
    """
    Auto-pick a uniform region: lowest detrended variance, but taking the
    20th percentile rather than the minimum so we don't land on a
    degenerate/clipped area.
    """
    cands, stds = [], []
    for y in range(0, img.shape[0] - size, stride):
        for x in range(0, img.shape[1] - size, stride):
            sl = (slice(y, y + size), slice(x, x + size))
            s = float(np.std(detrend(img[sl])))
            if np.isfinite(s) and s > 0:
                cands.append(sl)
                stds.append(s)
    stds = np.array(stds)
    pick = cands[int(np.argsort(stds)[max(0, int(0.2 * len(stds)) - 1)])]
    print(f"[flat ] patch rows {pick[0].start}:{pick[0].stop} "
          f"cols {pick[1].start}:{pick[1].stop}  sigma={np.std(detrend(img[pick])):.5f}")
    return pick


def autocorr_lag1(img, patch):
    r = detrend(img[patch])
    F = np.fft.fft2(r)
    ac = np.real(np.fft.ifft2(np.abs(F) ** 2))
    ac /= ac[0, 0]
    return float(max(ac[0, 1], ac[1, 0]))


def noise_psd(img, patch, out_shape):
    """2-D noise PSD scaled to the bm3d convention: mean = N1*N2*sigma^2."""
    r = detrend(img[patch])
    m, n = r.shape
    var = float(r.var())
    w = np.outer(np.hanning(m), np.hanning(n))
    w /= np.sqrt(np.mean(w ** 2))
    P = np.abs(np.fft.fft2(r * w)) ** 2 / (m * n)
    P = np.fft.ifftshift(gaussian_filter(np.fft.fftshift(P), 2.0, mode="wrap"))
    N1, N2 = out_shape
    Pz = np.fft.ifftshift(zoom(np.fft.fftshift(P), (N1 / m, N2 / n), order=1))
    Pz = np.clip(Pz, Pz.max() * 1e-4, None)   # spectral nulls break the Wiener stage
    return Pz * (N1 * N2 * var) / Pz.mean()


def main(path="result.pkl"):
    raw = load(path)
    img, mask, roi = clean(raw)
    sub = np.ascontiguousarray(img[roi])

    flat = find_flat_patch(sub)
    sigma = float(np.std(detrend(sub[flat])))
    r1 = autocorr_lag1(sub, flat)

    if r1 > 0.2:
        print(f"[noise] lag-1 corr {r1:.2f} -> correlated, using 2-D PSD")
        arg = noise_psd(sub, flat, sub.shape) * STRENGTH ** 2
    else:
        print(f"[noise] lag-1 corr {r1:.2f} -> white, using scalar sigma")
        arg = sigma * STRENGTH

    print("[bm3d ] running ...")
    den = np.asarray(bm3d.bm3d(sub, arg, profile=PROFILE,
                               stage_arg=bm3d.BM3DStages.ALL_STAGES),
                     float).reshape(sub.shape)

    before = sigma
    after = float(np.std(detrend(den[flat])))
    print(f"[check] flat std {before:.5f} -> {after:.5f}   "
          f"(noise reduction {before/max(after,1e-12):.1f}x)")

    with open("denoised.pkl", "wb") as f:
        pickle.dump({"denoised": den, "original": sub, "roi": roi,
                     "sigma": sigma, "lag1": r1}, f)
    print("[save ] denoised.pkl")

    plot(sub, den, flat)
    return sub, den


def plot(sub, den, flat):
    vmin, vmax = np.percentile(sub, [1, 99])
    kw = dict(origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    resid = sub - den
    r = 3 * np.std(resid)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    for a, d, t in ((ax[0], sub, "Original"), (ax[1], den, "BM3D")):
        im = a.imshow(d, **kw); a.set_title(t); fig.colorbar(im, ax=a, fraction=0.046)
    im = ax[2].imshow(resid, origin="lower", cmap="coolwarm", vmin=-r, vmax=r)
    ax[2].set_title("Residual (no step edges = good)")
    fig.colorbar(im, ax=ax[2], fraction=0.046)

    row = sub.shape[0] // 2
    ax[3].plot(sub[row], lw=0.5, alpha=0.45, label="original")
    ax[3].plot(den[row], lw=1.5, label="BM3D")
    ax[3].set_title(f"Profile, row {row}"); ax[3].set_xlabel("x [px]")
    ax[3].set_ylabel("Transmittance"); ax[3].legend(); ax[3].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("denoise_check.png", dpi=110)
    print("[save ] denoise_check.png")
    plt.show()


def _to_sub(sl, roi, shape):
    """Original-image coords -> indices in the auto-cropped array."""
    out = []
    for s, r, n in zip(sl, roi, shape):
        a = max(0, s.start - r.start)
        b = min(n, s.stop - r.start)
        out.append(slice(a, b))
    return tuple(out)
 
 
def _single(data, title, fname, vmin, vmax, sl):
    """One image, one figure, original pixel coordinates on the axes."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
                   extent=(sl[1].start, sl[1].stop, sl[0].start, sl[0].stop))
    ax.set_title(title)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Transmittance")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    print(f"[save ] {fname}")
 
 
def final_figures():
    with open("denoised.pkl", "rb") as f:
        d = pickle.load(f)
    sub, den, roi = d["original"], d["denoised"], d["roi"]
 
    crop = _to_sub(CROP, roi, sub.shape)
    step = _to_sub(UL_STEP, roi, sub.shape)
 
    # --- full field, common grey scale so the two are comparable ---
    o_c, d_c = sub[crop], den[crop]
    vmin, vmax = np.percentile(o_c, [1, 99])
    _single(o_c, "Original (Transmittance)", "01_original.png", vmin, vmax, CROP)
    _single(d_c, "BM3D Denoised", "02_denoised.png", vmin, vmax, CROP)
 
    # --- upper-left step: rescale to the local range or the contrast vanishes ---
    o_z, d_z = sub[step], den[step]
    zmin, zmax = np.percentile(o_z, [1, 99])
    _single(o_z, "Original - upper-left step", "03_original_zoom.png",
            zmin, zmax, UL_STEP)
    _single(d_z, "BM3D - upper-left step", "04_denoised_zoom.png",
            zmin, zmax, UL_STEP)
 
    print(f"[step ] original  mean={o_z.mean():.4f}  std={o_z.std():.4f}")
    print(f"[step ] denoised  mean={d_z.mean():.4f}  std={d_z.std():.4f}")
    plt.show()
    
    
if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "result_pe_steps.pkl")
    final_figures()