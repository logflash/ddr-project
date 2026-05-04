"""
activation_patch.py  --  Activation patching for causal analysis.

For each axis and each layer L, patches the mean CLS token from "positive"
probe examples into "negative" eval examples and measures how much of the
positive-class output is recovered.

Normalized patching effect:
    (patched_mean - neg_baseline) / (pos_baseline - neg_baseline)
    0  = patching had no effect
    1  = patching fully recovered the positive output

The layer where this first approaches 1 is the causal locus -- the earliest
point in the network where the action-relevant representation lives.

Also reports per-layer patching success rate: fraction of neg eval frames
where patching causes the output to cross the decision boundary.

Outputs saved to --out-dir:
    activation_patch.png   -- normalized effect + success rate per layer per axis

Usage:
    python steering/activation_patch.py --ckpt checkpoints/vit_continuous_tiny_best.pth
"""

from __future__ import annotations

import argparse
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
                        make_steer_hook, IMAGENET_MEAN, IMAGENET_STD)
from vit_train import LekiwiDataset
from steer import AXIS_IDX, AXIS_NAMES, CONTRAST_THRESH, BOUNDARY


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
    """
    Returns:
        streams  (N, L, D)  CLS at every layer
        outputs  (N, 3)     model predictions
        labels   (N, 3)     ground-truth normalised labels
    """
    all_streams, all_outputs, all_labels = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        streams = collect_cls_streams(model, imgs)       # (B, L, D) on CPU
        outputs = model(imgs).cpu().numpy()              # (B, 3)
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
    patch_vec: torch.Tensor,   # (D,)
    layer_idx: int,
    batch_size: int,
    device:    torch.device,
) -> np.ndarray:
    """Run a subset of dataset with CLS patched at layer_idx. Returns (N, 3)."""
    loader = DataLoader(Subset(dataset, indices.tolist()),
                        batch_size=batch_size, shuffle=False, num_workers=0)
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


@torch.no_grad()
def run_steered_subset(
    model:      ViTContinuous,
    dataset:    LekiwiDataset,
    indices:    np.ndarray,
    steer_vec:  torch.Tensor,   # (D,) unit vector
    alpha:      float,
    layer_idx:  int,
    batch_size: int,
    device:     torch.device,
) -> np.ndarray:
    """Run a subset with alpha * steer_vec ADDED to CLS at layer_idx. Returns (N, 3)."""
    loader = DataLoader(Subset(dataset, indices.tolist()),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    all_preds = []
    for imgs, _ in loader:
        imgs = imgs.to(device)
        hook = model.backbone.blocks[layer_idx].register_forward_hook(
            make_steer_hook(steer_vec, alpha=alpha)
        )
        preds = model(imgs).cpu().numpy()
        hook.remove()
        all_preds.append(preds)
    return np.concatenate(all_preds, axis=0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Activation patching causal analysis"
    )
    parser.add_argument("--ckpt",        default="./checkpoints/vit_continuous_tiny_best.pth")
    parser.add_argument("--data-path",   default="./data/dataset_full.npz")
    parser.add_argument("--splits-dir",  default="./data/splits")
    parser.add_argument("--out-dir",     default="./steering/results")
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

    # ── Load model ───────────────────────────────────────────────────────────
    ckpt     = torch.load(args.ckpt, map_location=device, weights_only=False)
    assert ckpt["mode"] == "continuous"
    fname    = os.path.basename(args.ckpt)
    fallback = "vit_tiny_patch16_224" if "tiny" in fname else "vit_small_patch16_224"
    backbone = ckpt.get("backbone", fallback)
    model    = ViTContinuous(pretrained=False, backbone=backbone)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    print(f"Loaded {backbone} from {args.ckpt}  (epoch {ckpt['epoch']})")

    tf = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    # ── Probe set: collect streams and mean pos CLS per axis ─────────────────
    probe_idx = np.load(os.path.join(args.splits_dir, "steer_probe_idx.npy"))
    probe_ds  = LekiwiDataset(args.data_path, probe_idx, mode="continuous", transform=tf)
    probe_loader = DataLoader(probe_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    print("\nCollecting probe activations...")
    probe_streams, probe_outputs, probe_labels = collect_all(model, probe_loader, device)
    # probe_streams: (N_probe, L, D)

    n_layers = probe_streams.shape[1]

    # mean positive CLS per axis per layer: dict axis -> (L, D)
    mean_pos_cls = {}
    for axis_name, axis_i in AXIS_IDX.items():
        col = probe_labels[:, axis_i]
        if axis_name == "vx":
            pos_mask = col > 0.5
        else:
            pos_mask = col > CONTRAST_THRESH
        mean_pos_cls[axis_name] = probe_streams[pos_mask].mean(axis=0)  # (L, D)
        print(f"  {axis_name}: {pos_mask.sum()} pos probe samples")

    # ── Eval set: collect baseline outputs and labels ────────────────────────
    eval_idx = np.load(os.path.join(args.splits_dir, "steer_eval_idx.npy"))
    eval_ds  = LekiwiDataset(args.data_path, eval_idx, mode="continuous", transform=tf)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    print("\nCollecting eval baselines...")
    _, eval_outputs, eval_labels = collect_all(model, eval_loader, device)
    # eval_outputs: (N_eval, 3)

    # ── Patching sweep per axis ──────────────────────────────────────────────
    results = {}

    for axis_name, axis_i in AXIS_IDX.items():
        col      = eval_labels[:, axis_i]
        boundary = BOUNDARY[axis_name]
        if axis_name == "vx":
            neg_mask = col < 0.5
            pos_mask = col > 0.5
        else:
            neg_mask = col < -CONTRAST_THRESH
            pos_mask = col >  CONTRAST_THRESH

        neg_indices = np.where(neg_mask)[0]
        n_neg = len(neg_indices)
        if n_neg == 0:
            print(f"  [warn] {axis_name}: no neg eval samples, skipping")
            continue

        neg_baseline = eval_outputs[neg_mask, axis_i].mean()
        pos_baseline = eval_outputs[pos_mask, axis_i].mean() if pos_mask.any() else 1.0
        denom = pos_baseline - neg_baseline
        print(f"\n  axis={axis_name}  neg_baseline={neg_baseline:.3f}  "
              f"pos_baseline={pos_baseline:.3f}  n_neg={n_neg}")

        effects      = []
        success_rates = []

        for layer in range(n_layers):
            patch_vec = torch.tensor(
                mean_pos_cls[axis_name][layer], dtype=torch.float32
            )
            patched = run_patched_subset(
                model, eval_ds, neg_indices, patch_vec, layer,
                args.batch_size, device,
            )                                         # (n_neg, 3)
            patched_mean = patched[:, axis_i].mean()
            effect = (patched_mean - neg_baseline) / denom if abs(denom) > 1e-6 else 0.0
            success = ((patched[:, axis_i] > boundary)).mean()

            effects.append(float(effect))
            success_rates.append(float(success))
            print(f"    layer {layer:2d}  effect={effect:+.3f}  "
                  f"success={success*100:.1f}%")

        results[axis_name] = {
            "effects":       np.array(effects),
            "success_rates": np.array(success_rates),
            "neg_baseline":  neg_baseline,
            "pos_baseline":  pos_baseline,
        }

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    colors = ["steelblue", "tomato", "darkorange"]
    layers = np.arange(n_layers)

    for ai, axis_name in enumerate(AXIS_NAMES):
        if axis_name not in results:
            continue
        r = results[axis_name]

        # Top row: normalized patching effect
        ax = axes[0, ai]
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", label="full recovery")
        ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
        ax.plot(layers, r["effects"], marker="o", color=colors[ai])
        ax.set_xlabel("Layer patched")
        ax.set_ylabel("Normalized patching effect")
        ax.set_title(f"Causal effect: {axis_name}\n"
                     f"neg={r['neg_baseline']:.3f}  pos={r['pos_baseline']:.3f}")
        ax.set_ylim(-0.1, 1.2)
        ax.set_xticks(layers)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Bottom row: patching success rate
        ax = axes[1, ai]
        ax.plot(layers, r["success_rates"] * 100, marker="s", color=colors[ai])
        ax.set_xlabel("Layer patched")
        ax.set_ylabel("Success rate (%)")
        ax.set_title(f"Boundary-crossing rate after patching: {axis_name}")
        ax.set_ylim(0, 100)
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(args.out_dir, "activation_patch.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {path}")

    # ── Causal sufficiency test ───────────────────────────────────────────────
    # At the causal layer (argmax of full-patching effect), sweep alpha while
    # ADDING only the contrastive direction to the neg CLS (not replacing it).
    # If recovery approaches the full-patch baseline, the direction is causally sufficient.
    print("\n-- Causal sufficiency test --")
    ALPHA_STEPS = 15
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

    for ai, axis_name in enumerate(AXIS_NAMES):
        if axis_name not in results:
            continue
        r         = results[axis_name]
        axis_i    = AXIS_IDX[axis_name]
        boundary  = BOUNDARY[axis_name]
        causal_layer  = int(np.argmax(r["effects"]))
        full_recovery = float(r["effects"][causal_layer])

        # Contrastive direction at causal layer from probe set
        col_probe = probe_labels[:, axis_i]
        if axis_name == "vx":
            pos_mask_p = col_probe > 0.5
            neg_mask_p = col_probe < 0.5
        else:
            pos_mask_p = col_probe >  CONTRAST_THRESH
            neg_mask_p = col_probe < -CONTRAST_THRESH

        mean_pos_l = probe_streams[pos_mask_p, causal_layer, :].mean(0)  # (D,)
        mean_neg_l = probe_streams[neg_mask_p, causal_layer, :].mean(0)
        raw_diff   = mean_pos_l - mean_neg_l
        natural_alpha = float(np.linalg.norm(raw_diff))
        steer_vec = torch.tensor(
            raw_diff / (natural_alpha + 1e-8), dtype=torch.float32
        )

        # Neg eval indices
        col_eval = eval_labels[:, axis_i]
        neg_mask_e = col_eval < (0.5 if axis_name == "vx" else -CONTRAST_THRESH)
        neg_indices = np.where(neg_mask_e)[0]

        neg_baseline = r["neg_baseline"]
        denom        = r["pos_baseline"] - neg_baseline
        alphas       = np.linspace(0, 2.0 * natural_alpha, ALPHA_STEPS + 1)

        recoveries, suc_rates = [], []
        print(f"\n  axis={axis_name}  causal_layer={causal_layer}  "
              f"natural_alpha={natural_alpha:.2f}  full_recovery={full_recovery:.3f}")

        for alpha in alphas:
            patched      = run_steered_subset(
                model, eval_ds, neg_indices, steer_vec, float(alpha),
                causal_layer, args.batch_size, device,
            )
            patched_mean = patched[:, axis_i].mean()
            recovery     = (patched_mean - neg_baseline) / denom if abs(denom) > 1e-6 else 0.0
            success      = float((patched[:, axis_i] > boundary).mean())
            recoveries.append(float(recovery))
            suc_rates.append(success)
            print(f"    alpha={alpha:6.2f}  recovery={recovery:+.3f}  "
                  f"success={success*100:.1f}%")

        ax = axes2[ai]
        ax.axhline(full_recovery, color="gray", linestyle="--", linewidth=1.5,
                   label=f"full patch ({full_recovery:.2f})")
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8,
                   label="perfect (1.0)")
        ax.plot(alphas, recoveries, marker="o", color=colors[ai],
                label="steering dir only")
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Normalized recovery")
        ax.set_title(f"Causal sufficiency: {axis_name}  (layer {causal_layer})\n"
                     f"natural alpha = {natural_alpha:.1f}")
        ax.set_ylim(-0.1, 1.3)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig2.tight_layout()
    path2 = os.path.join(args.out_dir, "activation_patch_sufficiency.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"\nSaved {path2}")
    print(f"Done. Results saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
