"""
steer.py  –  Contrastive steering for the continuous ViT policy.

Pipeline
--------
1. Load checkpoint (read-only) and reconstruct model.
2. Collect CLS residual stream at every layer on the steer_probe set.
3. For each axis (vx, vy, vtheta) and each layer, compute a contrastive
   steering direction:  mean_act[axis > +thresh] - mean_act[axis < -thresh]
4. Score every layer via linear-probe R² to find the most informative layer
   for each axis.
5. On steer_eval, sweep alpha (steering magnitude) and measure the mean shift
   in the target output axis (dose-response).
6. Print a summary table and save plots to --out-dir.

Usage
-----
    python steering/steer.py --ckpt checkpoints/vit_continuous_tiny_best.pth
    python steering/steer.py --ckpt checkpoints/vit_continuous_tiny_best.pth \\
                             --axis vy --layer 8 --alpha-max 20
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "models"))

from torchvision import transforms
from vit_policy import (ViTContinuous, collect_cls_streams, make_steer_hook,
                        IMAGENET_MEAN, IMAGENET_STD)
from vit_train import LekiwiDataset

# ── Constants ─────────────────────────────────────────────────────────────────
AXIS_IDX   = {"vx": 0, "vy": 1, "vtheta": 2}
AXIS_NAMES = ["vx", "vy", "vtheta"]
CONTRAST_THRESH = 0.4   # normalised units; labels are in [-1, 1]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device) -> tuple[ViTContinuous, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    assert ckpt["mode"] == "continuous", \
        f"Checkpoint mode is '{ckpt['mode']}', expected 'continuous'."
    fname = os.path.basename(ckpt_path)
    _fallback = "vit_tiny_patch16_224" if "tiny" in fname else "vit_small_patch16_224"
    backbone = ckpt.get("backbone", _fallback)
    model = ViTContinuous(pretrained=False, backbone=backbone)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    print(f"Loaded {backbone} from {ckpt_path}  (epoch {ckpt['epoch']}, "
          f"val_loss={ckpt['val_loss']:.4f})")
    return model, ckpt


# ── Activation collection ─────────────────────────────────────────────────────

@torch.no_grad()
def collect_activations(
    model: ViTContinuous,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the full probe/eval set through the model and collect:
      acts   (N, L, D)  –  CLS token at every layer output
      labels (N, 3)     –  normalised [vx_n, vy_n, vtheta_n]
    """
    all_acts, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        streams = collect_cls_streams(model, imgs)   # (B, L, D) on CPU
        all_acts.append(streams.numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_acts, axis=0), np.concatenate(all_labels, axis=0)


# ── Steering directions ───────────────────────────────────────────────────────

def compute_directions(
    acts: np.ndarray,    # (N, L, D)
    labels: np.ndarray,  # (N, 3)
) -> dict[str, np.ndarray]:
    """
    For each axis, compute a unit-normalised contrastive direction at every layer.

    Returns  dirs[axis]  shaped (L, D).
    """
    dirs = {}
    for axis, idx in AXIS_IDX.items():
        col = labels[:, idx]
        # vx is in [0, 1] (never negative), so use a centred split
        if axis == "vx":
            pos_mask = col >  0.5
            neg_mask = col <  0.5
        else:
            pos_mask = col >  CONTRAST_THRESH
            neg_mask = col < -CONTRAST_THRESH
        n_pos, n_neg = pos_mask.sum(), neg_mask.sum()
        if n_pos == 0 or n_neg == 0:
            print(f"  [warn] axis={axis}: no contrast examples "
                  f"(pos={n_pos}, neg={n_neg}) — skipping")
            dirs[axis] = np.zeros((acts.shape[1], acts.shape[2]))
            continue
        mean_pos = acts[pos_mask].mean(axis=0)   # (L, D)
        mean_neg = acts[neg_mask].mean(axis=0)
        raw = mean_pos - mean_neg                 # (L, D)
        norms = np.linalg.norm(raw, axis=-1, keepdims=True).clip(min=1e-8)
        dirs[axis] = raw / norms                  # unit vectors per layer
        print(f"  axis={axis}  pos={n_pos}  neg={n_neg}  "
              f"mean_dir_norm={np.linalg.norm(raw, axis=-1).mean():.3f}")
    return dirs


# ── Layer scoring via linear probe ────────────────────────────────────────────

def score_layers(
    acts: np.ndarray,    # (N, L, D)
    labels: np.ndarray,  # (N, 3)
) -> np.ndarray:
    """
    Fit a Ridge regression at every layer to predict each action axis.
    Returns r2 shaped (L, 3).
    """
    n_layers = acts.shape[1]
    r2 = np.zeros((n_layers, 3))
    for layer in range(n_layers):
        X = acts[:, layer, :]   # (N, D)
        for axis_i in range(3):
            y = labels[:, axis_i]
            reg = Ridge(alpha=1.0).fit(X, y)
            r2[layer, axis_i] = r2_score(y, reg.predict(X))
    return r2


# ── Steering evaluation ────────────────────────────────────────────────────────

# Decision boundaries for success-rate metric (normalised units)
BOUNDARY = {"vx": 0.5, "vy": 0.0, "vtheta": 0.0}


@torch.no_grad()
def evaluate_steering(
    model:       ViTContinuous,
    loader:      DataLoader,
    steer_vec:   torch.Tensor,   # (D,)
    layer_idx:   int,
    alphas:      list[float],
    target_axis: int,
    axis_name:   str,
    device:      torch.device,
) -> dict[str, np.ndarray]:
    """
    For each alpha, apply alpha * steer_vec to the CLS token at layer_idx.

    Returns dict with keys:
      alphas          - sweep values
      mean_delta      - mean shift in target axis (N_alpha,)
      mean_baseline   - scalar mean baseline for target axis
      success_rate    - (N_alpha,) fraction of frames where output crosses
                        the decision boundary in the expected direction
                        (baseline <= boundary AND steered > boundary)
      mean_delta_all  - (N_alpha, 3) mean delta for all three axes
                        (axis specificity metric)
    """
    steer_vec = steer_vec.to(device)
    boundary  = BOUNDARY[axis_name]

    mean_deltas      = []
    mean_delta_all   = []
    success_rates    = []
    mean_baseline    = None

    for alpha in alphas:
        deltas_all, successes, baselines = [], [], []
        for imgs, _ in loader:
            imgs = imgs.to(device)

            # baseline: all 3 axes
            base_all = model(imgs).cpu().numpy()         # (B, 3)
            base_tgt = base_all[:, target_axis]

            # steered: all 3 axes
            hook = model.backbone.blocks[layer_idx].register_forward_hook(
                make_steer_hook(steer_vec, alpha=alpha)
            )
            steer_all = model(imgs).cpu().numpy()        # (B, 3)
            hook.remove()

            deltas_all.append(steer_all - base_all)      # (B, 3)

            steer_tgt = steer_all[:, target_axis]
            # bidirectional conditional success rate:
            #   alpha >= 0: of frames starting below boundary, fraction that cross up
            #   alpha <  0: of frames starting above boundary, fraction that cross down
            if alpha >= 0:
                eligible = base_tgt <= boundary
                crossed  = steer_tgt > boundary
            else:
                eligible = base_tgt > boundary
                crossed  = steer_tgt <= boundary
            n_eligible = eligible.sum()
            rate = float(crossed[eligible].mean()) if n_eligible > 0 else 0.0
            successes.append(rate)
            baselines.append(base_tgt)

        all_deltas = np.concatenate(deltas_all, axis=0)  # (N, 3)
        mean_deltas.append(float(all_deltas[:, target_axis].mean()))
        mean_delta_all.append(all_deltas.mean(axis=0))   # (3,)
        success_rates.append(float(np.mean(successes)))
        if mean_baseline is None:
            mean_baseline = float(np.concatenate(baselines).mean())

    return {
        "alphas":         np.array(alphas),
        "mean_delta":     np.array(mean_deltas),
        "mean_baseline":  mean_baseline,
        "success_rate":   np.array(success_rates),
        "mean_delta_all": np.array(mean_delta_all),   # (N_alpha, 3)
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_layer_scores(r2: np.ndarray, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, name in enumerate(AXIS_NAMES):
        ax.plot(r2[:, i], marker="o", label=name)
    ax.set_xlabel("Layer")
    ax.set_ylabel("R² (linear probe)")
    ax.set_title("Linear probe R² per layer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "layer_probe_r2.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_dose_response(results: dict, axis_name: str, layer: int, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # Left: mean delta (target axis)
    ax = axes[0]
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.plot(results["alphas"], results["mean_delta"], marker="o", color="steelblue")
    ax.set_xlabel("Steering magnitude alpha")
    ax.set_ylabel(f"Mean delta {axis_name} (normalised)")
    ax.set_title(f"Dose-response: {axis_name} at layer {layer}\n"
                 f"baseline mean = {results['mean_baseline']:.3f}")
    ax.grid(True, alpha=0.3)

    # Middle: bidirectional conditional success rate
    ax = axes[1]
    ax.axhline(100, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0,   color="gray", linewidth=0.8, linestyle=":")
    ax.plot(results["alphas"], results["success_rate"] * 100, marker="s", color="seagreen")
    ax.set_xlabel("Steering magnitude alpha")
    ax.set_ylabel("Conditional success rate (%)")
    ax.set_title(f"Bidirectional conditional success rate: {axis_name}\n"
                 f"boundary = {BOUNDARY[axis_name]:.1f}  "
                 f"(eligible frames only, direction follows sign of alpha)")
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)

    # Right: axis specificity (mean delta for all 3 axes)
    ax = axes[2]
    colors = ["steelblue", "tomato", "darkorange"]
    for i, name in enumerate(AXIS_NAMES):
        ls = "-" if name == axis_name else "--"
        lw = 2.0 if name == axis_name else 1.2
        ax.plot(results["alphas"], results["mean_delta_all"][:, i],
                marker="o", label=name, color=colors[i], linestyle=ls, linewidth=lw)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Steering magnitude alpha")
    ax.set_ylabel("Mean delta (normalised)")
    ax.set_title(f"Axis specificity when steering {axis_name}\n(dashed = off-target axes)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, f"dose_response_{axis_name}_layer{layer}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Contrastive steering for ViTContinuous")
    parser.add_argument("--ckpt",        default="./checkpoints/vit_continuous_tiny_best.pth")
    parser.add_argument("--data-path",   default="./data/dataset_full.npz")
    parser.add_argument("--splits-dir",  default="./data/splits")
    parser.add_argument("--out-dir",     default="./steering/results")
    parser.add_argument("--axis",        default=None, choices=["vx", "vy", "vtheta"],
                        help="Axis to steer (default: all three, one plot each)")
    parser.add_argument("--layer",       type=int, default=None,
                        help="Layer index to steer (default: auto = best probe layer)")
    parser.add_argument("--alpha-max",   type=float, default=15.0)
    parser.add_argument("--alpha-min",   type=float, default=None,
                        help="Minimum alpha (default: -alpha-max)")
    parser.add_argument("--alpha-steps", type=int,   default=20)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--num-workers", type=int,   default=0)
    parser.add_argument("--save-vecs",   default="",
                        help="Save steering vectors to this path "
                             "(e.g. ./checkpoints/steering_vecs_layer9.pth). "
                             "Empty = don't save.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    # ── 1. Load model ────────────────────────────────────────────────────────
    model, _ = load_model(args.ckpt, device)

    tf = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def make_loader(split_file: str) -> DataLoader:
        idx = np.load(os.path.join(args.splits_dir, split_file))
        ds  = LekiwiDataset(args.data_path, idx, mode="continuous", transform=tf)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers)

    # ── 2. Collect activations on steer_probe ────────────────────────────────
    print("\n-- Collecting probe activations --")
    probe_loader = make_loader("steer_probe_idx.npy")
    probe_acts, probe_labels = collect_activations(model, probe_loader, device)
    print(f"  probe_acts: {probe_acts.shape}   probe_labels: {probe_labels.shape}")

    # ── 3. Compute contrastive directions ────────────────────────────────────
    print("\n-- Computing steering directions --")
    directions = compute_directions(probe_acts, probe_labels)

    # ── 4. Score layers ──────────────────────────────────────────────────────
    print("\n-- Scoring layers via linear probe --")
    r2 = score_layers(probe_acts, probe_labels)
    plot_layer_scores(r2, args.out_dir)

    print("\nR² per layer:")
    header = f"  {'layer':>5}  " + "  ".join(f"{n:>8}" for n in AXIS_NAMES)
    print(header)
    for l in range(r2.shape[0]):
        row = f"  {l:>5}  " + "  ".join(f"{r2[l, i]:>8.4f}" for i in range(3))
        print(row)

    best_layers = {name: int(r2[:, i].argmax()) for i, name in enumerate(AXIS_NAMES)}
    print(f"\nBest layers: { {k: v for k, v in best_layers.items()} }")

    # ── 5. Evaluate steering on steer_eval ───────────────────────────────────
    print("\n-- Evaluating steering on steer_eval --")
    eval_loader = make_loader("steer_eval_idx.npy")
    alpha_min = args.alpha_min if args.alpha_min is not None else -args.alpha_max
    alphas = list(np.linspace(alpha_min, args.alpha_max, args.alpha_steps + 1))

    axes_to_steer = [args.axis] if args.axis else AXIS_NAMES
    used_layers: dict[str, int] = {}
    for axis_name in axes_to_steer:
        axis_i = AXIS_IDX[axis_name]
        layer  = args.layer if args.layer is not None else best_layers[axis_name]
        used_layers[axis_name] = layer
        vec    = torch.tensor(directions[axis_name][layer], dtype=torch.float32)
        print(f"\n  axis={axis_name}  layer={layer}  |vec|={vec.norm():.4f}")

        results = evaluate_steering(
            model, eval_loader, vec, layer, alphas, axis_i, axis_name, device
        )
        plot_dose_response(results, axis_name, layer, args.out_dir)

        print(f"  alpha -> mean_delta_{axis_name}  cond_success  [delta_vx  delta_vy  delta_vtheta]:")
        for a, d, s, dall in zip(results["alphas"], results["mean_delta"],
                                  results["success_rate"], results["mean_delta_all"]):
            off = "  ".join(f"{dall[i]:+.4f}" for i in range(3))
            print(f"    alpha={a:6.2f}  delta={d:+.4f}  cond_success={s*100:5.1f}%  [{off}]")

    # ── Save steering vectors ────────────────────────────────────────────────
    if args.save_vecs:
        vecs_ckpt = {
            "ckpt":   args.ckpt,
            "layers": used_layers,
        }
        for axis_name in axes_to_steer:
            layer = used_layers[axis_name]
            vecs_ckpt[axis_name] = directions[axis_name][layer].astype(np.float32)
        os.makedirs(os.path.dirname(os.path.abspath(args.save_vecs)), exist_ok=True)
        torch.save(vecs_ckpt, args.save_vecs)
        print(f"\nSaved steering vectors to {args.save_vecs}")
        for axis_name in axes_to_steer:
            layer = used_layers[axis_name]
            v = vecs_ckpt[axis_name]
            print(f"  {axis_name}  layer={layer}  shape={v.shape}  norm={np.linalg.norm(v):.4f}")

    print(f"\nDone. Results saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
