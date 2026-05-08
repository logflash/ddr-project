"""
activation_patch.py  --  Bidirectional activation patching for causal analysis.

For each layer, patches the mean [CLS] token between conditions and measures
how much the output changes:
  neg→pos: patch neg eval frames with mean pos probe [CLS]
  pos→neg: patch pos eval frames with mean neg probe [CLS]
Both effects are normalized to [0, 1] and averaged.

Normalized patching effect:
    0  = patching had no effect
    1  = output fully shifted to target condition

Question: Can we directionally manipulate each action dim by copying
layer-specific [CLS] tokens between images?

Usage:
    python steering/activation_patch.py
"""

from __future__ import annotations

import os
import sys
from typing import Callable

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "models"))
sys.path.insert(0, _HERE)

from torchvision import transforms
from vit_policy import (ViTContinuous, collect_cls_streams,
                        IMAGENET_MEAN, IMAGENET_STD)
from vit_train import LekiwiDataset
from steer import AXIS_IDX, AXIS_NAMES, CONTRAST_THRESH, BOUNDARY

CKPT        = "./checkpoints/vit_continuous_tiny_best.pth"
DATA_PATH   = "./data/dataset_full.npz"
SPLITS_DIR  = "./data/splits"
OUT_DIR     = "./steering/results"
BATCH_SIZE  = 64


# ── Hooks ─────────────────────────────────────────────────────────────────────

def make_patch_hook(cls_vec: torch.Tensor) -> Callable:
    """Replace the CLS token (position 0) with cls_vec for every sample in batch."""
    def _hook(module, inp, out):
        out = out.clone()
        out[:, 0, :] = cls_vec.to(out.device).unsqueeze(0).expand(out.shape[0], -1)
        return out
    return _hook


# ── Collection helpers ────────────────────────────────────────────────────────

@torch.no_grad()
def collect_all(
    model: ViTContinuous,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns streams (N, L, D), outputs (N, 3), labels (N, 3)."""
    all_streams, all_outputs, all_labels = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        streams = collect_cls_streams(model, imgs)
        outputs = model(imgs).cpu().numpy()
        all_streams.append(streams.numpy())
        all_outputs.append(outputs)
        all_labels.append(labels.numpy())
    return (
        np.concatenate(all_streams, axis=0),
        np.concatenate(all_outputs, axis=0),
        np.concatenate(all_labels,  axis=0),
    )


@torch.no_grad()
def run_patched_subset(
    model:     ViTContinuous,
    dataset:   LekiwiDataset,
    indices:   np.ndarray,
    patch_vec: torch.Tensor,
    layer_idx: int,
    device:    torch.device,
) -> np.ndarray:
    """Run a subset with CLS patched at layer_idx. Returns (N, 3)."""
    loader = DataLoader(Subset(dataset, indices.tolist()),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_preds = []
    for imgs, _ in loader:
        imgs = imgs.to(device)
        hook = model.backbone.blocks[layer_idx].register_forward_hook(
            make_patch_hook(patch_vec)
        )
        preds = model(imgs).cpu().numpy()
        hook.remove()
        all_preds.append(preds)
    return np.concatenate(all_preds, axis=0)



# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    # ── Load model ───────────────────────────────────────────────────────────
    ckpt     = torch.load(CKPT, map_location=device, weights_only=False)
    assert ckpt["mode"] == "continuous"
    fname    = os.path.basename(CKPT)
    fallback = "vit_tiny_patch16_224" if "tiny" in fname else "vit_small_patch16_224"
    backbone = ckpt.get("backbone", fallback)
    model    = ViTContinuous(pretrained=False, backbone=backbone)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    print(f"Loaded {backbone}  (epoch {ckpt['epoch']})")

    tf = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    # ── Probe set ────────────────────────────────────────────────────────────
    probe_idx = np.load(os.path.join(SPLITS_DIR, "steer_probe_idx.npy"))
    probe_ds  = LekiwiDataset(DATA_PATH, probe_idx, mode="continuous", transform=tf)
    probe_loader = DataLoader(probe_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    print("\nCollecting probe activations...")
    probe_streams, _, probe_labels = collect_all(model, probe_loader, device)
    n_layers = probe_streams.shape[1]

    mean_pos_cls, mean_neg_cls = {}, {}
    for axis_name, axis_i in AXIS_IDX.items():
        col = probe_labels[:, axis_i]
        pos_mask = col > 0.5       if axis_name == "vx" else col >  CONTRAST_THRESH
        neg_mask = col < 0.5       if axis_name == "vx" else col < -CONTRAST_THRESH
        mean_pos_cls[axis_name] = probe_streams[pos_mask].mean(axis=0)
        mean_neg_cls[axis_name] = probe_streams[neg_mask].mean(axis=0)
        print(f"  {axis_name}: {pos_mask.sum()} pos  {neg_mask.sum()} neg  probe samples")

    # ── Eval set ─────────────────────────────────────────────────────────────
    eval_idx = np.load(os.path.join(SPLITS_DIR, "steer_eval_idx.npy"))
    eval_ds  = LekiwiDataset(DATA_PATH, eval_idx, mode="continuous", transform=tf)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    print("\nCollecting eval baselines...")
    _, eval_outputs, eval_labels = collect_all(model, eval_loader, device)

    # ── Bidirectional patching sweep ─────────────────────────────────────────
    results = {}
    for axis_name, axis_i in AXIS_IDX.items():
        col      = eval_labels[:, axis_i]
        boundary = BOUNDARY[axis_name]
        neg_mask = col < 0.5       if axis_name == "vx" else col < -CONTRAST_THRESH
        pos_mask = col > 0.5       if axis_name == "vx" else col >  CONTRAST_THRESH

        neg_indices = np.where(neg_mask)[0]
        pos_indices = np.where(pos_mask)[0]
        if len(neg_indices) == 0:
            print(f"  [warn] {axis_name}: no neg eval samples, skipping")
            continue

        neg_baseline = eval_outputs[neg_mask, axis_i].mean()
        pos_baseline = eval_outputs[pos_mask, axis_i].mean() if pos_mask.any() else 1.0
        denom = pos_baseline - neg_baseline
        print(f"\n  axis={axis_name}  neg_baseline={neg_baseline:.3f}  "
              f"pos_baseline={pos_baseline:.3f}  n_neg={len(neg_indices)}")

        effects = []
        for layer in range(n_layers):
            # neg→pos
            fwd_vec     = torch.tensor(mean_pos_cls[axis_name][layer], dtype=torch.float32)
            fwd_patched = run_patched_subset(model, eval_ds, neg_indices, fwd_vec, layer, device)
            fwd_effect  = (fwd_patched[:, axis_i].mean() - neg_baseline) / denom \
                          if abs(denom) > 1e-6 else 0.0

            # pos→neg
            bwd_effect = 0.0
            if len(pos_indices) > 0:
                bwd_vec     = torch.tensor(mean_neg_cls[axis_name][layer], dtype=torch.float32)
                bwd_patched = run_patched_subset(model, eval_ds, pos_indices, bwd_vec, layer, device)
                bwd_effect  = (pos_baseline - bwd_patched[:, axis_i].mean()) / denom \
                              if abs(denom) > 1e-6 else 0.0

            effect = (fwd_effect + bwd_effect) / 2
            effects.append(float(effect))
            print(f"    layer {layer:2d}  effect={effect:+.3f}")

        results[axis_name] = {
            "effects":      np.array(effects),
            "neg_baseline": neg_baseline,
            "pos_baseline": pos_baseline,
        }

    # ── Plot ─────────────────────────────────────────────────────────────────
    HIGHLIGHT   = (0.95, 0.88, 0.55)
    axis_colors = {"vx": "steelblue", "vy": "tomato", "vtheta": "darkorange"}
    layers = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for axis_name in AXIS_NAMES:
        if axis_name not in results:
            continue
        ax.plot(layers, results[axis_name]["effects"], marker="o", label=axis_name,
                color=axis_colors[axis_name], linewidth=1.8)
    ax.axvspan(8.5, 11.5, color=HIGHLIGHT, alpha=0.25, zorder=0)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Normalized causal effect", fontsize=11)
    ax.set_title("Activation patching effect per layer", fontsize=12)
    ax.set_xticks(layers)
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "activation_patch.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {path}")

    print(f"\nDone. Results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
