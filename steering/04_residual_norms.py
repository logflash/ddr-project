"""
04_residual_norms.py  --  Residual contribution norm per transformer block.

For each block, computes mean ||CLS_output - CLS_input|| on the steer_probe set.
Near-zero norm would indicate an identity-like transformation; consistently
non-zero norms at layers 9-11 refute the identity-transformation objection.

Question: Are the causally decisive layers (9-11) doing non-trivial work,
or are they near-identity transformations?

Usage:
    python steering/04_residual_norms.py
"""

from __future__ import annotations

import os
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

from vit_policy import ViTContinuous, IMAGENET_MEAN, IMAGENET_STD
from vit_train import LekiwiDataset

CKPT       = "./checkpoints/vit_continuous_tiny_best.pth"
DATA_PATH  = "./data/dataset_full.npz"
SPLITS_DIR = "./data/splits"
OUT_DIR    = "./steering/results"
BATCH_SIZE = 64

HIGHLIGHT = (0.95, 0.88, 0.55)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt     = torch.load(CKPT, map_location=device, weights_only=False)
    fname    = os.path.basename(CKPT)
    fallback = "vit_tiny_patch16_224" if "tiny" in fname else "vit_small_patch16_224"
    backbone = ckpt.get("backbone", fallback)
    model    = ViTContinuous(pretrained=False, backbone=backbone)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    print(f"Loaded {backbone}  (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

    # ── Probe loader ──────────────────────────────────────────────────────────
    tf  = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    idx = np.load(os.path.join(SPLITS_DIR, "steer_eval_idx.npy"))
    ds  = LekiwiDataset(DATA_PATH, idx, mode="continuous", transform=tf)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Hook: capture (input, output) per block ───────────────────────────────
    n_layers  = len(model.backbone.blocks)
    norm_sums = [0.0] * n_layers
    counts    = [0]   * n_layers

    hooks = []
    for layer_idx, block in enumerate(model.backbone.blocks):
        def _hook(module, inp, out, li=layer_idx):
            cls_in  = inp[0][:, 0, :]
            cls_out = out[:, 0, :]
            norms   = (cls_out - cls_in).norm(dim=-1)
            norm_sums[li] += norms.sum().item()
            counts[li]    += norms.shape[0]
        hooks.append(block.register_forward_hook(_hook))

    print(f"\nCollecting residual norms ({len(idx)} samples)...")
    with torch.no_grad():
        for imgs, _ in loader:
            model(imgs.to(device))

    for h in hooks:
        h.remove()

    mean_norms = [norm_sums[l] / counts[l] for l in range(n_layers)]
    layers     = np.arange(n_layers)

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\nCLS residual contribution norm per layer  (N={counts[0]} samples):")
    print(f"  {'layer':>5}  {'mean ||out−in||':>16}")
    for l, n in enumerate(mean_norms):
        print(f"  {l:>5}  {n:>16.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [HIGHLIGHT if l >= 9 else "steelblue" for l in layers]
    ax.bar(layers, mean_norms, color=colors, alpha=0.9, edgecolor="white", linewidth=0.5)
    ax.axvspan(8.5, 11.5, color=HIGHLIGHT, alpha=0.25, zorder=0)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Mean ‖CLS output − CLS input‖", fontsize=11)
    ax.set_title("Residual contribution norm per block", fontsize=12)
    ax.set_xticks(layers)
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(0, max(mean_norms) * 1.22)
    ax.grid(True, alpha=0.3, axis="y")
    ax.text(10, max(mean_norms) * 1.04, "layers 9–11\nnot identity",
            ha="center", va="bottom", fontsize=8.5, color="#7a5a00",
            bbox=dict(boxstyle="round,pad=0.3", fc=HIGHLIGHT, ec="#c8a000", alpha=0.9))
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "residual_norms.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
