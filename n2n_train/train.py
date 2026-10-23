"""Train a 2D U-Net with Noise2Noise pairs.

    python -m n2n_train.train --loss l1 --name unet_l1
    python -m n2n_train.train --loss l2 --name unet_l2

Scaling: each input patch is divided by its own mean; the target is divided by
the *same* number (computed from the input only), and predictions are multiplied
back at inference, so outputs are in raw ADU.

Outputs go to runs/<name>/: config.json, log.csv (train loss + every registered
loss on val, per epoch), last.pt and best.pt (lowest val value of the training loss).
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from n2n_data import FrameStore, N2NPairs
from .losses import LOSSES
from .unet import UNet


def input_scale(x):
    """Per-sample mean of the input, shape (B, 1, 1, 1)."""
    return x.mean(dim=(1, 2, 3), keepdim=True)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loss", required=True, choices=sorted(LOSSES))
    ap.add_argument("--name", help="run name (default: unet_<loss>)")
    ap.add_argument("--manifest", default="catalog/manifest.json")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--k", default="auto", help="'auto' (1..n/2 per stack) or a fixed int")
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--samples", type=int, default=4000, help="training pairs per epoch")
    ap.add_argument("--val-samples", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=32, help="U-Net channels at the top level")
    ap.add_argument("--depth", type=int, default=4, help="U-Net down/up levels")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast on GPU")
    args = ap.parse_args()
    args.k = args.k if args.k == "auto" else int(args.k)
    args.name = args.name or f"unet_{args.loss}"
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Path(args.out) / args.name
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(vars(args), indent=1))

    store = FrameStore(args.manifest)
    train = N2NPairs(store, "train", k=args.k, patch=args.patch, samples_per_epoch=args.samples)
    val = N2NPairs(store, "val", k=args.k, patch=args.patch, samples_per_epoch=args.val_samples)
    loader = DataLoader(train, batch_size=args.batch, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val, batch_size=args.batch, num_workers=args.workers)

    model = UNet(base=args.base, depth=args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = LOSSES[args.loss]
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda")
    print(f"{run}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params on {device}, "
          f"{len(train.stack_ids)} train / {len(val.stack_ids)} val stacks")

    fields = ["epoch", "lr", "train_loss"] + [f"val_{name}" for name in LOSSES] + ["seconds"]
    log_path = run / "log.csv"
    with open(log_path, "w", newline="") as fh:
        csv.writer(fh).writerow(fields)
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for inp, tgt in loader:
            inp, tgt = inp.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
            s = input_scale(inp)
            with torch.autocast(**amp):
                pred = model(inp / s)
            loss = loss_fn(pred.float(), tgt / s)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(inp)
        sched.step()

        model.eval()
        val_sums = dict.fromkeys(LOSSES, 0.0)
        with torch.no_grad():
            for inp, tgt in val_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                s = input_scale(inp)
                with torch.autocast(**amp):
                    pred = model(inp / s)
                for name, fn in LOSSES.items():
                    val_sums[name] += fn(pred.float(), tgt / s).item() * len(inp)
        val_losses = {name: v / len(val) for name, v in val_sums.items()}

        row = [epoch, sched.get_last_lr()[0], total / len(train)] + \
              [val_losses[name] for name in LOSSES] + [round(time.time() - t0, 1)]
        with open(log_path, "a", newline="") as fh:
            csv.writer(fh).writerow(row)
        ckpt = {"model": model.state_dict(), "config": vars(args), "epoch": epoch}
        torch.save(ckpt, run / "last.pt")
        if val_losses[args.loss] < best:
            best = val_losses[args.loss]
            torch.save(ckpt, run / "best.pt")
        print(f"epoch {epoch:3d}  train {args.loss} {total / len(train):.5f}  " +
              "  ".join(f"val {n} {v:.5f}" for n, v in val_losses.items()) +
              f"  ({row[-1]}s)")


if __name__ == "__main__":
    main()
