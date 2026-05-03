"""
probe_epochs.py  --  Track how CLS representations evolve across training epochs.

For each epoch checkpoint produced by vit_train.py --save-every N, this script:
  1. Collects CLS activations on the steer_probe split.
  2. Fits a Ridge linear probe at every layer for each action axis -> R2 (L, 3).
  3. Computes the contrastive steering direction per axis per layer and measures
     its cosine similarity to the final-epoch direction (representation stability).

Outputs saved to --out-dir:
  probe_r2_heatmap.png      -- (layer x epoch) R2 heatmap, one panel per axis
  probe_r2_vs_epoch.png     -- best-layer R2 vs epoch, all axes on one plot
  direction_stability.png   -- cosine sim to final-epoch direction (layer x epoch)

Usage:
    py steering/probe_epochs.py --mode continuous --size tiny
    py steering/probe_epochs.py --mode continuous --size tiny --ckpt-dir checkpoints
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "models"))
sys.path.insert(0, _HERE)

from vit_policy import ViTContinuous, IMAGENET_MEAN, IMAGENET_STD
from vit_train import LekiwiDataset
from steer import collect_activations, compute_directions, score_layers, AXIS_NAMES


def find_epoch_checkpoints(ckpt_dir: str, mode: str, size: str) -> list[tuple[int, str]]:
    pattern = os.path.join(ckpt_dir, f"vit_{mode}_{size}_epoch*.pth")
    results = []
    for p in glob.glob(pattern):
        m = re.search(r"_epoch(\d+)\.pth$", p)
        if m:
            results.append((int(m.group(1)), p))
    return sorted(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track representation evolution across training epochs"
    )
    parser.add_argument("--ckpt-dir",    default="./checkpoints")
    parser.add_argument("--mode",        default="continuous",
                        choices=["continuous", "categorical"])
    parser.add_argument("--size",        default="tiny")
    parser.add_argument("--data-path",   default="./data/dataset_full.npz")
    parser.add_argument("--splits-dir",  default="./data/splits")
    parser.add_argument("--out-dir",     default="./steering/results")
    parser.add_argument("--probe-layer", type=int, default=6,
                        help="Fixed layer for the layer-specific R2 vs epoch plot")
    parser.add_argument("--batch-size",  type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    epoch_ckpts = find_epoch_checkpoints(args.ckpt_dir, args.mode, args.size)
    if not epoch_ckpts:
        print(f"No epoch checkpoints found matching "
              f"vit_{args.mode}_{args.size}_epoch*.pth in {args.ckpt_dir}")
        print("Re-run training with --save-every N to generate them.")
        return
    print(f"Found {len(epoch_ckpts)} epoch checkpoints: "
          f"epochs {[e for e, _ in epoch_ckpts]}")

    # Build probe loader once -- reused for every checkpoint
    tf = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    idx = np.load(os.path.join(args.splits_dir, "steer_probe_idx.npy"))
    probe_ds = LekiwiDataset(args.data_path, idx, mode="continuous", transform=tf)
    probe_loader = DataLoader(
        probe_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    epochs      = []
    all_r2      = []   # list of (L, 3)
    all_dirs    = []   # list of dict[axis -> (L, D)]

    for epoch, ckpt_path in epoch_ckpts:
        print(f"\nEpoch {epoch:3d}  ({ckpt_path})")
        ckpt     = torch.load(ckpt_path, map_location=device, weights_only=False)
        backbone = ckpt.get("backbone", f"vit_{args.size}_patch16_224")
        model    = ViTContinuous(pretrained=False, backbone=backbone)
        model.load_state_dict(ckpt["state_dict"])
        model.eval().to(device)

        acts, labels = collect_activations(model, probe_loader, device)
        r2           = score_layers(acts, labels)           # (L, 3)
        dirs         = compute_directions(acts, labels)     # dict axis -> (L, D)

        epochs.append(epoch)
        all_r2.append(r2)
        all_dirs.append(dirs)
        print(f"  best R2  vx={r2[:, 0].max():.3f}  "
              f"vy={r2[:, 1].max():.3f}  vtheta={r2[:, 2].max():.3f}")

    epochs  = np.array(epochs)
    r2_arr  = np.stack(all_r2, axis=0)   # (T, L, 3)
    n_layers = r2_arr.shape[1]

    # -- Plot 1: R2 heatmap (layer x epoch) per axis --------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ai, aname in enumerate(AXIS_NAMES):
        ax   = axes[ai]
        data = r2_arr[:, :, ai].T        # (L, T)
        im   = ax.imshow(
            data, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="viridis",
            extent=[epochs[0] - 0.5, epochs[-1] + 0.5, -0.5, n_layers - 0.5],
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Layer")
        ax.set_title(f"Linear probe R2: {aname}")
        ax.set_yticks(range(n_layers))
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path = os.path.join(args.out_dir, "probe_r2_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {path}")

    # -- Plot 2: Best-layer R2 vs epoch, all axes -----------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["steelblue", "tomato", "darkorange"]
    for ai, aname in enumerate(AXIS_NAMES):
        ax.plot(epochs, r2_arr[:, :, ai].max(axis=1),
                marker="o", label=aname, color=colors[ai])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Best-layer R2")
    ax.set_title("Peak linear probe R2 vs training epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(args.out_dir, "probe_r2_vs_epoch.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # -- Plot 3: Direction stability (cosine sim to final-epoch direction) -----
    final_dirs = all_dirs[-1]
    fig, axes  = plt.subplots(1, 3, figsize=(18, 5))
    for ai, aname in enumerate(AXIS_NAMES):
        ax        = axes[ai]
        final_vec = final_dirs[aname]   # (L, D)
        sims = []
        for t_dirs in all_dirs:
            vec = t_dirs[aname]         # (L, D)
            cos = np.sum(vec * final_vec, axis=-1) / (
                np.linalg.norm(vec, axis=-1).clip(min=1e-8) *
                np.linalg.norm(final_vec, axis=-1).clip(min=1e-8)
            )
            sims.append(cos)
        sims = np.stack(sims, axis=0).T   # (L, T)
        im = ax.imshow(
            sims, aspect="auto", origin="lower", vmin=-1, vmax=1, cmap="RdBu_r",
            extent=[epochs[0] - 0.5, epochs[-1] + 0.5, -0.5, n_layers - 0.5],
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Layer")
        ax.set_title(f"Direction cosine sim to final epoch: {aname}")
        ax.set_yticks(range(n_layers))
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path = os.path.join(args.out_dir, "direction_stability.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # -- Plot 4: Fixed-layer R2 vs epoch (one line per axis) --------------------
    layer = min(args.probe_layer, n_layers - 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    for ai, aname in enumerate(AXIS_NAMES):
        ax.plot(epochs, r2_arr[:, layer, ai],
                marker="o", label=aname, color=colors[ai])
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"R2 at layer {layer}")
    ax.set_title(f"Linear probe R2 at layer {layer} vs training epoch")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(args.out_dir, f"probe_r2_layer{layer}_vs_epoch.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    print(f"\nDone. All plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
