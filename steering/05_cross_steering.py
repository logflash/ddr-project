"""
05_cross_steering.py  --  Cross-dimensional steering effect matrix.

At layer 9 (the causal locus), applies each axis's contrastive steering
direction and measures the mean output shift for all three axes.

Rows = steered axis, Columns = measured axis.
Values are normalized by the diagonal (self-effect), so diagonal = 1.0
and off-diagonal shows spillover as a fraction of the intended effect.

Question: Is layer-9 steering specific to the intended action dimension,
or does steering one axis entangle others?

Usage:
    python steering/05_cross_steering.py
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
sys.path.insert(0, _HERE)

from vit_policy import (ViTContinuous, collect_cls_streams,
                        make_steer_hook, IMAGENET_MEAN, IMAGENET_STD)
from vit_train import LekiwiDataset
from steer import AXIS_IDX, AXIS_NAMES, CONTRAST_THRESH

CKPT       = "./checkpoints/vit_continuous_tiny_best.pth"
DATA_PATH  = "./data/dataset_full.npz"
SPLITS_DIR = "./data/splits"
OUT_DIR    = "./steering/results"
BATCH_SIZE = 64
LAYER      = 9
ALPHA      = 8.0


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

    tf = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    # ── Compute contrastive directions on steer_probe ─────────────────────────
    probe_idx    = np.load(os.path.join(SPLITS_DIR, "steer_probe_idx.npy"))
    probe_ds     = LekiwiDataset(DATA_PATH, probe_idx, mode="continuous", transform=tf)
    probe_loader = DataLoader(probe_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\nCollecting probe activations ({len(probe_idx)} samples)...")
    all_acts, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in probe_loader:
            all_acts.append(collect_cls_streams(model, imgs.to(device)).numpy())
            all_labels.append(labels.numpy())
    probe_acts   = np.concatenate(all_acts,   axis=0)   # (N, L, D)
    probe_labels = np.concatenate(all_labels, axis=0)   # (N, 3)

    directions = {}
    for axis_name, axis_i in AXIS_IDX.items():
        col      = probe_labels[:, axis_i]
        pos_mask = col > 0.5      if axis_name == "vx" else col >  CONTRAST_THRESH
        neg_mask = col < 0.5      if axis_name == "vx" else col < -CONTRAST_THRESH
        mean_pos = probe_acts[pos_mask].mean(axis=0)   # (L, D)
        mean_neg = probe_acts[neg_mask].mean(axis=0)
        raw      = mean_pos - mean_neg
        norms    = np.linalg.norm(raw, axis=-1, keepdims=True).clip(min=1e-8)
        directions[axis_name] = raw / norms
        print(f"  {axis_name}: {pos_mask.sum()} pos  {neg_mask.sum()} neg")

    # ── Collect baseline outputs on steer_eval ────────────────────────────────
    eval_idx    = np.load(os.path.join(SPLITS_DIR, "steer_eval_idx.npy"))
    eval_ds     = LekiwiDataset(DATA_PATH, eval_idx, mode="continuous", transform=tf)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\nCollecting eval baselines ({len(eval_idx)} samples)...")
    imgs_batches, baseline_all = [], []
    with torch.no_grad():
        for imgs, _ in eval_loader:
            imgs = imgs.to(device)
            imgs_batches.append(imgs)
            baseline_all.append(model(imgs).cpu().numpy())
    baseline_all = np.concatenate(baseline_all, axis=0)   # (N, 3)
    baseline_mean = baseline_all.mean(axis=0)              # (3,)

    # ── Build 3×3 cross-steering matrix ──────────────────────────────────────
    print(f"\nSweeping cross-steering at layer {LAYER}, alpha=±{ALPHA}...")
    raw_matrix = np.zeros((3, 3))   # raw_matrix[i, j] = mean delta in axis j when steering axis i

    for i, steer_axis in enumerate(AXIS_NAMES):
        vec = torch.tensor(directions[steer_axis][LAYER], dtype=torch.float32).to(device)

        # forward: +alpha
        hook = model.backbone.blocks[LAYER].register_forward_hook(
            make_steer_hook(vec, alpha=ALPHA)
        )
        with torch.no_grad():
            fwd = np.concatenate([model(imgs).cpu().numpy() for imgs in imgs_batches])
        hook.remove()

        # backward: -alpha
        hook = model.backbone.blocks[LAYER].register_forward_hook(
            make_steer_hook(vec, alpha=-ALPHA)
        )
        with torch.no_grad():
            bwd = np.concatenate([model(imgs).cpu().numpy() for imgs in imgs_batches])
        hook.remove()

        # signed mean delta: average of fwd shift and negated bwd shift
        fwd_delta = fwd.mean(axis=0) - baseline_mean   # (3,)
        bwd_delta = baseline_mean - bwd.mean(axis=0)   # (3,) — negated so positive = intended

        raw_matrix[i] = (fwd_delta + bwd_delta) / 2

        print(f"  steer={steer_axis}  "
              + "  ".join(f"Δ{n}={raw_matrix[i, j]:+.4f}" for j, n in enumerate(AXIS_NAMES)))

    # Normalize each row by its diagonal (self-effect)
    norm_matrix = raw_matrix.copy()
    for i in range(3):
        self_effect = raw_matrix[i, i]
        if abs(self_effect) > 1e-8:
            norm_matrix[i] /= self_effect

    print("\nNormalized cross-steering matrix (diagonal = 1.0):")
    header = f"  {'steer \\ measure':>15}  " + "  ".join(f"{n:>8}" for n in AXIS_NAMES)
    print(header)
    for i, steer_axis in enumerate(AXIS_NAMES):
        row = f"  {steer_axis:>15}  " + "  ".join(f"{norm_matrix[i, j]:>8.4f}" for j in range(3))
        print(row)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))

    vmax = max(abs(norm_matrix).max(), 1.0)
    im = ax.imshow(norm_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(AXIS_NAMES, fontsize=11)
    ax.set_yticklabels(AXIS_NAMES, fontsize=11)
    ax.set_xlabel("Measured axis", fontsize=11)
    ax.set_ylabel("Steered axis", fontsize=11)
    ax.set_title(f"Cross-steering effect at layer {LAYER}  (|α| = {ALPHA:.1f})", fontsize=12)

    # annotate cells
    for i in range(3):
        for j in range(3):
            val = norm_matrix[i, j]
            color = "white" if abs(val) > 0.6 * vmax else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=12, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Normalized effect (self = 1.0)")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "cross_steering.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
