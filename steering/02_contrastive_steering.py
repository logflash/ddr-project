"""
02_contrastive_steering.py  --  Per-layer contrastive steering success.

For each layer, adds a learned contrastive direction (scaled by ALPHA) to the
[CLS] token and measures the conditional success rate on steer_eval.

Question: How well can we directionally manipulate each action dim by adding
learned vectors to layer-specific [CLS] tokens?

Usage:
    python steering/02_contrastive_steering.py
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
from steer import AXIS_IDX, AXIS_NAMES, CONTRAST_THRESH, BOUNDARY

CKPT        = "./checkpoints/vit_continuous_tiny_best.pth"
DATA_PATH   = "./data/dataset_full.npz"
SPLITS_DIR  = "./data/splits"
OUT_DIR     = "./steering/results"
BATCH_SIZE  = 64
ALPHA       = 8.0

AXIS_COLORS = {"vx": "steelblue", "vy": "tomato", "vtheta": "darkorange"}


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
    n_layers = probe_acts.shape[1]

    directions = {}
    for axis_name, axis_i in AXIS_IDX.items():
        col      = probe_labels[:, axis_i]
        pos_mask = col > 0.5       if axis_name == "vx" else col >  CONTRAST_THRESH
        neg_mask = col < 0.5       if axis_name == "vx" else col < -CONTRAST_THRESH
        mean_pos = probe_acts[pos_mask].mean(axis=0)   # (L, D)
        mean_neg = probe_acts[neg_mask].mean(axis=0)
        raw      = mean_pos - mean_neg                 # (L, D)
        norms    = np.linalg.norm(raw, axis=-1, keepdims=True).clip(min=1e-8)
        directions[axis_name] = raw / norms            # unit vectors per layer
        print(f"  {axis_name}: {pos_mask.sum()} pos  {neg_mask.sum()} neg  probe samples")

    # ── Evaluate per-layer success on steer_eval ──────────────────────────────
    eval_idx    = np.load(os.path.join(SPLITS_DIR, "steer_eval_idx.npy"))
    eval_ds     = LekiwiDataset(DATA_PATH, eval_idx, mode="continuous", transform=tf)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\nCollecting eval baselines ({len(eval_idx)} samples)...")
    base_all, imgs_batches = [], []
    with torch.no_grad():
        for imgs, _ in eval_loader:
            imgs = imgs.to(device)
            imgs_batches.append(imgs)
            base_all.append(model(imgs).cpu().numpy())
    base_all = np.concatenate(base_all, axis=0)   # (N, 3)

    print(f"\nSweeping layers at alpha=±{ALPHA}...")
    success_per_layer = {}
    for axis_name, axis_i in AXIS_IDX.items():
        boundary = BOUNDARY[axis_name]
        neg_elig = base_all[:, axis_i] <= boundary
        pos_elig = base_all[:, axis_i] >  boundary
        rates = []
        for layer_idx in range(n_layers):
            vec = torch.tensor(directions[axis_name][layer_idx],
                               dtype=torch.float32).to(device)

            # forward: +alpha, neg→pos
            hook = model.backbone.blocks[layer_idx].register_forward_hook(
                make_steer_hook(vec, alpha=ALPHA)
            )
            with torch.no_grad():
                fwd = np.concatenate([
                    model(imgs).cpu().numpy()[:, axis_i] for imgs in imgs_batches
                ])
            hook.remove()
            fwd_rate = float((fwd[neg_elig] > boundary).mean()) \
                       if neg_elig.sum() > 0 else 0.0

            # backward: -alpha, pos→neg
            hook = model.backbone.blocks[layer_idx].register_forward_hook(
                make_steer_hook(vec, alpha=-ALPHA)
            )
            with torch.no_grad():
                bwd = np.concatenate([
                    model(imgs).cpu().numpy()[:, axis_i] for imgs in imgs_batches
                ])
            hook.remove()
            bwd_rate = float((bwd[pos_elig] <= boundary).mean()) \
                       if pos_elig.sum() > 0 else 0.0

            rate = (fwd_rate + bwd_rate) / 2
            rates.append(rate)
            print(f"  {axis_name}  layer {layer_idx:2d}  fwd={fwd_rate*100:.1f}%  bwd={bwd_rate*100:.1f}%  avg={rate*100:.1f}%")
        success_per_layer[axis_name] = np.array(rates)

    # ── Plot ──────────────────────────────────────────────────────────────────
    HIGHLIGHT = (0.95, 0.88, 0.55)
    layers = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for axis_name in AXIS_NAMES:
        ax.plot(layers, success_per_layer[axis_name], marker="o", label=axis_name,
                color=AXIS_COLORS[axis_name], linewidth=1.8)
    ax.axvspan(8.5, 11.5, color=HIGHLIGHT, alpha=0.25, zorder=0)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Reversal success rate", fontsize=11)
    ax.set_title(f"Contrastive steering reversal per layer  (|α| = {ALPHA:.1f})", fontsize=12)
    ax.set_xticks(layers)
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "contrastive_steering.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
