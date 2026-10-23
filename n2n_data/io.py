"""Zero-copy access to raw FITS frames.

The camera writes uncompressed single-HDU FITS (int16 + BZERO=32768, a few
float32), so the pixel block can be memory-mapped directly and any crop read
without loading the whole 2048x2048 frame.
"""
from pathlib import Path

import numpy as np
from astropy.io import fits

BITPIX_DTYPE = {8: "u1", 16: ">i2", 32: ">i4", -32: ">f4", -64: ">f8"}


def frame_header(path):
    """Return the metadata needed to memory-map one frame later on."""
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        offset = hdul.fileinfo(0)["datLoc"]
    shape = tuple(hdr[f"NAXIS{i}"] for i in range(hdr["NAXIS"], 0, -1))
    shape = tuple(s for s in shape if s != 1)  # (1, H, W) -> (H, W)
    return {
        "offset": int(offset),
        "bitpix": int(hdr["BITPIX"]),
        "bzero": float(hdr.get("BZERO", 0.0)),
        "bscale": float(hdr.get("BSCALE", 1.0)),
        "shape": [int(s) for s in shape],
        "exposure": float(hdr.get("EXPOSURE", np.nan)),
        "date": str(hdr.get("DATE", "")),
    }


def read_frame(path, meta, region=None):
    """Read a frame (or a `region` = (y0, y1, x0, x1)) as float32 ADU."""
    raw = np.memmap(Path(path), dtype=BITPIX_DTYPE[meta["bitpix"]], mode="r",
                    offset=meta["offset"], shape=tuple(meta["shape"]))
    if region is not None:
        y0, y1, x0, x1 = region
        raw = raw[y0:y1, x0:x1]
    out = raw.astype(np.float32)
    if meta["bscale"] != 1.0:
        out *= meta["bscale"]
    if meta["bzero"]:
        out += meta["bzero"]
    return out
