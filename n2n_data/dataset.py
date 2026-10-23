"""Noise2Noise pairs drawn on the fly from the raw frames.

Each sample: pick a stack of n frames, draw k in [1, n // 2], draw 2k distinct
frames at random, and return
    input  = sum of k frames            (raw ADU, no DC subtraction, no filtering)
    target = sum of k *other* frames
cropped to the same random patch. Input and target never share a raw frame;
across samples frames are reused in new random combinations. Pass an int `k`
to use one fixed k instead (stacks with fewer than 2k frames are then skipped).

    from n2n_data import N2NPairs
    train = N2NPairs("catalog/manifest.json", split="train", patch=256)
    val   = N2NPairs("catalog/manifest.json", split="val",   patch=256)
    loader = DataLoader(train, batch_size=16, num_workers=8, shuffle=True)
    for inp, tgt in loader: ...          # (B, 1, patch, patch) float32

Val samples are fully deterministic (seeded per index), so the val set is the
same every epoch and every run.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from .store import FrameStore


class N2NPairs(Dataset):
    def __init__(self, manifest, split="train", k="auto", patch=256, roles=("object", "ob"),
                 samples_per_epoch=None, augment=None, reduce="sum", seed=0,
                 return_meta=False):
        self.store = manifest if isinstance(manifest, FrameStore) else FrameStore(manifest)
        self.k, self.patch, self.reduce = k, patch, reduce
        min_frames = 2 if k == "auto" else 2 * k
        self.stack_ids = sorted(self.store.select(split, roles, min_frames))
        if not self.stack_ids:
            raise ValueError(f"no {split} stacks with >= {min_frames} frames for roles {roles}")
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
        k = int(rng.integers(1, n // 2 + 1)) if self.k == "auto" else self.k
        perm = rng.permutation(n)
        a, b = sorted(perm[:k]), sorted(perm[k:2 * k])

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
        return inp, tgt, {"stack": sid, "k": k, "input_frames": a, "target_frames": b,
                          "region": list(region)}
