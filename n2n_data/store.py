"""Read access to the catalogued raw frames (see build.py for the manifest).

Frames are memory-mapped straight from the original FITS files, so nothing is
copied and a FrameStore is cheap to pickle into DataLoader workers.
"""
import json
from pathlib import Path

import numpy as np

from .io import read_frame


class FrameStore:
    def __init__(self, manifest_path):
        manifest = json.loads(Path(manifest_path).read_text())
        self.root = Path(manifest["root"])
        self.frames = manifest["frames"]
        self.stacks = manifest["stacks"]
        self.shape = tuple(manifest["frame_shape"])

    def select(self, split=None, roles=None, min_frames=1):
        """Stack ids filtered by split name(s), role(s) and minimum frame count."""
        def match(value, want):
            return want is None or value == want or (not isinstance(want, str) and value in want)
        return [sid for sid, s in self.stacks.items()
                if match(s["split"], split) and match(s["role"], roles)
                and len(s["frames"]) >= min_frames]

    def read(self, f, region=None):
        """Global frame index `f` -> float32 raw ADU; `region` = (y0, y1, x0, x1)."""
        meta = self.frames[f]
        return read_frame(self.root / meta["path"], meta, region)

    def combine(self, stack_id, frame_ids, region=None, reduce="sum"):
        """Sum (or mean) of the stack's frames at positions `frame_ids`, raw ADU."""
        stack = self.stacks[stack_id]
        acc = None
        for i in frame_ids:
            img = self.read(stack["frames"][i], region)
            acc = img if acc is None else acc + img
        return acc / len(frame_ids) if reduce == "mean" else acc
