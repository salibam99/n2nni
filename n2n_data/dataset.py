"""Noise2Noise pairs drawn on the fly from the raw frames.

Each sample: pick a stack, draw 2k distinct frames from it at random, and return
    input  = sum of the first k frames  (raw ADU, no DC subtraction, no filtering)
    target = sum of the other k frames
cropped to the same random patch. Input and target never share a raw frame;
across samples frames are reused in new random combinations.

    from n2n_data import N2NPairs
    train = N2NPairs("catalog/manifest.json", split="train", k=1, patch=256)
    val   = N2NPairs("catalog/manifest.json", split="val",   k=1, patch=256)
    loader = DataLoader(train, batch_size=16, num_workers=8, shuffle=True)
    for inp, tgt in loader: ...          # (B, 1, patch, patch) float32

Val samples are fully deterministic (seeded per index), so the val set is the
same every epoch and every run. Stacks with fewer than 2k frames are skipped.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from .store import FrameStore


class N2NPairs(Dataset):
    def __init__(self, manifest, split="train", k=1, patch=256, roles=("object", "ob"),
                 samples_per_epoch=None, augment=None, reduce="sum", seed=0,
                 return_meta=False):
        self.store = manifest if isinstance(manifest, FrameStore) else FrameStore(manifest)
        self.k, self.patch, self.reduce = k, patch, reduce
        self.stack_ids = sorted(self.store.select(split, roles, min_frames=2 * k))
        if not self.stack_ids:
            raise ValueError(f"no {split} stacks with >= {2 * k} frames for roles {roles}")
        n = np.array([len(self.store.stacks[s]["frames"]) for s in self.stack_ids], float)
        self.weights = n / n.sum()  # every raw frame equally likely to be drawn
        self.samples_per_epoch = samples_per_epoch or (2000 if split == "train" else 200)
        self.augment = (split == "train") if augment is None else augment
        self.deterministic = split != "train"
        self.seed = seed
        self.return_meta = return_meta

    def __len__(self):
        return self.samples_per_epoch

    def _rng(self, idx):
        if self.deterministic:
            return np.random.default_rng([self.seed, idx])
        # differs per worker, per epoch and per index (DataLoader reseeds torch)
        return np.random.default_rng([torch.initial_seed() % 2**63, idx])

    def __getitem__(self, idx):
        rng = self._rng(idx)
        sid = self.stack_ids[rng.choice(len(self.stack_ids), p=self.weights)]
        n = len(self.store.stacks[sid]["frames"])
        perm = rng.permutation(n)
        a, b = sorted(perm[:self.k]), sorted(perm[self.k:2 * self.k])

        h, w = self.store.shape
        if self.patch is None:
            region = (0, h, 0, w)
        else:
            y, x = rng.integers(0, h - self.patch + 1), rng.integers(0, w - self.patch + 1)
            region = (y, y + self.patch, x, x + self.patch)
        inp = self.store.combine(sid, a, region, self.reduce)
        tgt = self.store.combine(sid, b, region, self.reduce)

        if self.augment:
            rot, flip = rng.integers(4), rng.integers(2)
            inp, tgt = (np.rot90(t, rot) for t in (inp, tgt))
            if flip:
                inp, tgt = inp[:, ::-1], tgt[:, ::-1]
        inp = torch.from_numpy(np.ascontiguousarray(inp))[None]
        tgt = torch.from_numpy(np.ascontiguousarray(tgt))[None]
        if not self.return_meta:
            return inp, tgt
        return inp, tgt, {"stack": sid, "input_frames": a, "target_frames": b,
                          "region": list(region)}
