"""
residual_norms.py  --  Measure residual contribution norm at each transformer block.

For each block, computes mean ||output - input|| on the CLS token over the
steer_probe set. Near-zero norm would indicate an identity-like transformation;
consistently non-zero norms confirm the blocks are doing non-trivial work.

The combined figure overlays the residual norms with the activation patching
causal effects (parsed from steering/results/activation_patch.txt) to show that
the blocks with the largest residual contributions are also the causally decisive
ones — directly refuting the identity-transformation objection.

Outputs saved to --out-dir:
    residual_norms.png  --  combined figure: residual norms + patching effects
    residual_norms.txt  --  numeric table

Usage:
    python steering/residual_norms.py
    python steering/residual_norms.py --ckpt checkpoints/vit_continuous_tiny_best.pth
"""

from __future__ import annotations

import argparse
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

from vit_policy import ViTContinuous, IMAGENET_MEAN, IMAGENET_STD
from vit_train import LekiwiDataset


# ── Patching results parser ────────────────────────────────────────────────────

def parse_patch_effects(txt_path: str) -> dict[str, list[float]] | None:
    """Parse per-layer patching effects from activation_patch.txt. Returns None if missing."""
    if not os.path.exists(txt_path):
        return None
    effects: dict[str, list[float]] = {}
    current_axis = None
    with open(txt_path) as f:
        for line in f:
            m = re.search(r"axis=(\w+)\s+neg_baseline", line)
            if m:
                current_axis = m.group(1)
                effects[current_axis] = []
                continue
            m = re.search(r"layer\s+\d+\s+effect=([+-]?\d+\.\d+)", line)
            if m and current_axis:
                effects[current_axis].append(float(m.group(1)))
    return effects if effects else None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Residual contribution norm per transformer block"
    )
    parser.add_argument("--ckpt",         default="./checkpoints/vit_continuous_tiny_best.pth")
    parser.add_argument("--data-path",    default="./data/dataset_full.npz")
    parser.add_argument("--splits-dir",   default="./data/splits")
    parser.add_argument("--out-dir",      default="./steering/results")
    parser.add_argument("--patch-txt",    default="./steering/results/activation_patch.txt",
                        help="Path to activation_patch.txt for the combined figure")
    parser.add_argument("--batch-size",   type=int, default=64)
    parser.add_argument("--num-workers",  type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    # ── Load model ───────────────────────────────────────────────────────────
    ckpt     = torch.load(args.ckpt, map_location=device, weights_only=False)
    fname    = os.path.basename(args.ckpt)
    fallback = "vit_tiny_patch16_224" if "tiny" in fname else "vit_small_patch16_224"
    backbone = ckpt.get("backbone", fallback)
    model    = ViTContinuous(pretrained=False, backbone=backbone)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    print(f"Loaded {backbone} from {args.ckpt}  (epoch {ckpt['epoch']})")

    # ── Probe loader ─────────────────────────────────────────────────────────
    tf  = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    idx = np.load(os.path.join(args.splits_dir, "steer_probe_idx.npy"))
    ds  = LekiwiDataset(args.data_path, idx, mode="continuous", transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

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

    # ── Save text ─────────────────────────────────────────────────────────────
    txt_path = os.path.join(args.out_dir, "residual_norms.txt")
    with open(txt_path, "w") as f:
        f.write(f"CLS residual contribution norm — {args.ckpt}\n")
        f.write("layer  mean_norm\n")
        for l, n in enumerate(mean_norms):
            f.write(f"{l:>5}  {n:.6f}\n")
    print(f"\nSaved {txt_path}")

    # ── Parse activation patching effects ────────────────────────────────────
    patch_effects = parse_patch_effects(args.patch_txt)
    if patch_effects:
        print(f"Loaded patching effects from {args.patch_txt}")
    else:
        print(f"[warn] {args.patch_txt} not found — combined figure will show norms only")

    # ── Plot ──────────────────────────────────────────────────────────────────
    has_patch = patch_effects is not None
    HIGHLIGHT = (0.95, 0.88, 0.55)   # warm yellow for layers 9-11

    # Figure 1: residual norms
    fig1, ax = plt.subplots(figsize=(7, 4.5))
    colors = [HIGHLIGHT if l >= 9 else "steelblue" for l in layers]
    ax.bar(layers, mean_norms, color=colors, alpha=0.9, edgecolor="white", linewidth=0.5)
    ax.axvspan(8.5, 11.5, color=HIGHLIGHT, alpha=0.25, zorder=0)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Mean ‖CLS output − CLS input‖", fontsize=11)
    ax.set_title("Residual contribution norm per block", fontsize=12)
    ax.set_xticks(layers)
    ax.set_xticklabels([str(l) for l in layers])
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(0, max(mean_norms) * 1.22)
    ax.text(10, max(mean_norms) * 1.04, "layers 9–11\nnot identity",
            ha="center", va="bottom", fontsize=8.5, color="#7a5a00",
            bbox=dict(boxstyle="round,pad=0.3", fc=HIGHLIGHT, ec="#c8a000", alpha=0.9))
    fig1.tight_layout()
    png_path = os.path.join(args.out_dir, "residual_norms.png")
    fig1.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved {png_path}")

    # Figure 2: activation patching effects
    if has_patch:
        fig2, ax2 = plt.subplots(figsize=(7, 4.5))
        axis_colors = {"vx": "steelblue", "vy": "tomato", "vtheta": "darkorange"}
        for axis_name, effs in patch_effects.items():
            ls = np.arange(len(effs))
            ax2.plot(ls, effs, marker="o", label=axis_name,
                     color=axis_colors.get(axis_name, "gray"), linewidth=1.8)
        ax2.axvspan(8.5, 11.5, color=HIGHLIGHT, alpha=0.25, zorder=0)
        ax2.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax2.set_xlabel("Layer patched", fontsize=11)
        ax2.set_ylabel("Normalized causal effect", fontsize=11)
        ax2.set_title("Activation patching effect per layer", fontsize=12)
        ax2.set_xticks(layers)
        ax2.set_xticklabels([str(l) for l in layers])
        ax2.set_xlim(-0.5, 11.5)
        ax2.set_ylim(0, 1.15)
        ax2.legend(fontsize=9, loc="lower right")
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        patch_png_path = os.path.join(args.out_dir, "activation_patch_plot.png")
        fig2.savefig(patch_png_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved {patch_png_path}")


if __name__ == "__main__":
    main()
