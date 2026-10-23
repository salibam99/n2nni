"""Scan the raw data tree and write a manifest (JSON catalog) of acquisition stacks.

    python -m n2n_data.build [--config data_config.yaml]

What it does, without touching or copying any image:
  1. reads every FITS header and hashes every file (to find byte-identical copies),
  2. groups frames into stacks = same folder + same file-name stem + same exposure,
  3. removes duplicate frames: each unique frame is kept in exactly one stack
     (larger stacks first, folders matching `prefer_last` lose ties),
  4. tags each stack as object / ob / dc from its name,
  5. links every non-DC stack to the closest-in-time DC (and objects to an OB)
     for flat-field normalization later,
  6. assigns train/val by whole stack; existing assignments are kept on rebuild
     so the val set stays constant.
"""
import argparse
import hashlib
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from .io import frame_header

FRAME_SUFFIX = re.compile(r"(?:[_\s-]*\d{4})+(?:\s*-\s*Copy)?$", re.I)


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def stem_of(path):
    return FRAME_SUFFIX.sub("", path.stem).rstrip(" _-") or path.stem


def guess_role(text):
    text = text.lower()
    if re.search(r"(^|[^a-z])(dc|dark)", text):
        return "dc"
    if re.search(r"(^|[^a-z])ob([^a-z]|$)", text) or "open_beam" in text:
        return "ob"
    return "object"


def timestamp(date):
    try:
        return datetime.fromisoformat(date).timestamp()
    except ValueError:
        return float("nan")


def scan(root, exclude):
    paths = sorted(p for p in root.rglob("*.fits")
                   if not any(x in str(p.relative_to(root)) for x in exclude))

    def info(p):
        return {"path": str(p.relative_to(root)), "md5": md5(p), **frame_header(p)}

    with ThreadPoolExecutor(16) as pool:
        return list(pool.map(info, paths))


def group(frames, cfg, dropped):
    """Group frames into stacks and drop duplicate frames."""
    groups = defaultdict(list)
    for i, fr in enumerate(frames):
        p = Path(fr["path"])
        groups[(str(p.parent), stem_of(p), fr["exposure"])].append(i)

    exposures = defaultdict(set)
    for folder, stem, exp in groups:
        exposures[(folder, stem)].add(exp)

    candidates = []
    for (folder, stem, exp), idx in groups.items():
        sid = f"{folder}/{stem}"
        if len(exposures[(folder, stem)]) > 1:
            sid += f"@{exp:g}s"
        idx.sort(key=lambda i: (frames[i]["date"], frames[i]["path"]))
        n_unique = len({frames[i]["md5"] for i in idx})
        penalty = any(p in sid for p in cfg["prefer_last"])
        candidates.append((-n_unique, penalty, sid, folder, stem, exp, idx))

    seen, stacks = {}, {}
    for _, _, sid, folder, stem, exp, idx in sorted(candidates):
        keep = []
        for i in idx:
            h = frames[i]["md5"]
            if h in seen:
                dropped.append({"path": frames[i]["path"], "reason": f"duplicate of {seen[h]}"})
            else:
                seen[h] = frames[i]["path"]
                keep.append(i)
        if len(keep) < cfg["min_frames"]:
            for i in keep:
                dropped.append({"path": frames[i]["path"],
                                "reason": f"stack has {len(keep)} unique frame(s) < min_frames"})
            continue
        session = folder.split("/")[0]
        role = guess_role(f"{Path(folder).name}/{stem}")
        for pattern, forced in cfg["role_overrides"].items():
            if pattern in sid:
                role = forced
        stacks[sid] = {
            "session": session, "role": role, "exposure": exp,
            "time": float(np.nanmedian([timestamp(frames[i]["date"]) for i in keep])),
            "frames": keep,
        }
    return stacks


def nearest(stacks, sid, role, exclude=()):
    """Closest-in-time stack of `role`, preferring the same exposure."""
    me = stacks[sid]
    pool = [c for c, s in stacks.items() if s["role"] == role and c != sid]
    clean = [c for c in pool if not any(x in c.lower() for x in exclude)]
    pool = clean or pool
    if not pool:
        return None, None
    best = min(pool, key=lambda c: (stacks[c]["exposure"] != me["exposure"],
                                    abs(stacks[c]["time"] - me["time"])))
    return best, round(abs(stacks[best]["time"] - me["time"]) / 3600, 2)


def assign_splits(stacks, cfg, previous):
    """Fixed val set: keep earlier assignments, then fill up to val_fraction of frames."""
    trainable = [s for s, v in stacks.items() if v["role"] in ("object", "ob")]
    for sid, st in stacks.items():
        st["split"] = previous.get(sid, "val" if sid in cfg["val_stacks"] else "train")
    if any(sid in previous for sid in trainable):
        new = [s for s in trainable if s not in previous and s not in cfg["val_stacks"]]
    else:
        new = [s for s in trainable if s not in cfg["val_stacks"]]
    total = sum(len(stacks[s]["frames"]) for s in trainable)
    val = sum(len(stacks[s]["frames"]) for s in trainable if stacks[s]["split"] == "val")
    rng = np.random.default_rng(cfg["seed"])
    target = cfg["val_fraction"] * total
    for sid in rng.permutation(sorted(new)):
        n = len(stacks[sid]["frames"])
        if val + n <= target:  # never overshoot the requested fraction
            stacks[sid]["split"] = "val"
            val += n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="data_config.yaml")
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    root = (cfg_path.parent / cfg["root"]).resolve()
    out = (cfg_path.parent / cfg["manifest"]).resolve()

    previous = {}
    if out.exists():
        previous = {sid: s["split"] for sid, s in json.loads(out.read_text())["stacks"].items()}

    print(f"scanning {root} ...")
    frames = scan(root, cfg["exclude"])
    dropped = []
    stacks = group(frames, cfg, dropped)

    for sid, st in stacks.items():
        st["dc"], st["dc_dt_hours"] = nearest(stacks, sid, "dc", cfg["dc_exclude"])
        st["ob"], st["ob_dt_hours"] = (nearest(stacks, sid, "ob") if st["role"] == "object"
                                       else (None, None))
    for sid, link in cfg["links"].items():
        for key, target in link.items():
            if target not in stacks:
                raise SystemExit(f"links: {sid} -> {key}={target} is not a known stack")
            stacks[sid][key] = target
            stacks[sid][f"{key}_dt_hours"] = round(
                abs(stacks[target]["time"] - stacks[sid]["time"]) / 3600, 2)
    assign_splits(stacks, cfg, previous)

    shapes = {tuple(f["shape"]) for f in frames}
    if len(shapes) != 1:
        raise SystemExit(f"mixed frame shapes: {shapes}")
    # re-index so the manifest only lists frames that are used
    used = sorted(i for st in stacks.values() for i in st["frames"])
    remap = {old: new for new, old in enumerate(used)}
    for st in stacks.values():
        st["frames"] = [remap[i] for i in st["frames"]]

    manifest = {
        "root": str(root),
        "frame_shape": list(shapes.pop()),
        "frames": [frames[i] for i in used],
        "stacks": dict(sorted(stacks.items())),
        "dropped": dropped,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1))

    print(f"\n{'split':5s} {'role':6s} {'n':>3s} {'exp':>5s}  stack  [dc | ob]")
    for sid, st in manifest["stacks"].items():
        print(f"{st['split']:5s} {st['role']:6s} {len(st['frames']):3d} {st['exposure']:5g}  {sid}"
              f"  [{st['dc']} {st['dc_dt_hours']}h | {st['ob']} {st['ob_dt_hours']}h]")
    for split in ("train", "val"):
        sel = [s for s in manifest["stacks"].values() if s["split"] == split and s["role"] != "dc"]
        print(f"{split}: {len(sel)} stacks, {sum(len(s['frames']) for s in sel)} frames (object+ob)")
    print(f"{len(used)} unique frames kept, {len(dropped)} files dropped -> {out}")


if __name__ == "__main__":
    main()
