#!/usr/bin/env python
"""
debug_policy.py

End-to-end sanity check for the ViTContinuous deployment pipeline.
Tests whether run_vit_policy.py's preprocessing matches training,
and whether the model actually predicts meaningful actions on real data.

Usage:
    python debug_policy.py
    python debug_policy.py --model checkpoints/vit_continuous_tiny_best.pth
    python debug_policy.py --n 500
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, ".")
from models.vit_policy import (
    IMAGENET_MEAN, IMAGENET_STD,
    VX_MAX, VY_MAX, VTHETA_MAX,
    ViTContinuous,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def stat(name: str, t: torch.Tensor, scale: float = 1.0, unit: str = "") -> None:
    print(
        f"  {name:18s}: mean={t.mean()*scale:+8.4f}  std={t.std()*scale:7.4f}"
        f"  min={t.min()*scale:+8.4f}  max={t.max()*scale:+8.4f}{unit}"
    )


# ── Preprocessing paths ───────────────────────────────────────────────────────

def preprocess_training(img_hwc_rgb_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    """Exact replication of vit_train.py LekiwiDataset.__getitem__ + val transform."""
    t = torch.from_numpy(img_hwc_rgb_uint8).float().div(255.0).permute(2, 0, 1)
    t = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(t)
    return t.unsqueeze(0).to(device)


def preprocess_deployment(img_hwc_bgr_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    """Exact copy of preprocess() in run_vit_policy.py."""
    rgb = cv2.cvtColor(img_hwc_bgr_uint8, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    t   = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
    _MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    _STD  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    t = (t - _MEAN) / _STD
    return t.unsqueeze(0).to(device)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/vit_continuous_tiny_best.pth")
    parser.add_argument("--data",  default="data/dataset_full.npz")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--n",     type=int, default=300, help="Val images to test")
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    section("Device")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("  CPU (no CUDA detected)")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    section("Checkpoint")
    ckpt = torch.load(args.model, map_location=device)
    backbone = ckpt.get("backbone", "vit_tiny_patch16_224")
    print(f"  path    : {args.model}")
    print(f"  backbone: {backbone}")
    print(f"  mode    : {ckpt.get('mode')}")
    print(f"  epoch   : {ckpt.get('epoch')}")
    print(f"  val_loss: {ckpt.get('val_loss', float('nan')):.6f}")
    m = ckpt.get("val_metrics", {})
    print(f"  acc_vx={m.get('acc_vx',0):.3f}  acc_vy={m.get('acc_vy',0):.3f}  acc_vtheta={m.get('acc_vtheta',0):.3f}")
    print(f"  mse_vx={m.get('mse_vx',0):.4f}  mse_vy={m.get('mse_vy',0):.4f}  mse_vtheta={m.get('mse_vtheta',0):.4f}")

    model = ViTContinuous(pretrained=False, backbone=backbone)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params  : {n_params/1e6:.1f}M")

    head_bias = model.head.bias.detach().cpu()
    print(f"  head bias (normalised): vx={head_bias[0]:+.4f}  vy={head_bias[1]:+.4f}  vtheta={head_bias[2]:+.4f}")
    print(f"  head bias (physical):   vx={head_bias[0]*VX_MAX:+.4f} m/s  "
          f"vy={head_bias[1]*VY_MAX:+.4f} m/s  vtheta={head_bias[2]*VTHETA_MAX:+.1f} deg/s")

    # ── Data ──────────────────────────────────────────────────────────────────
    section("Dataset")
    data    = np.load(args.data)
    val_idx = np.load(f"{args.splits_dir}/val_idx.npy")[:args.n]
    print(f"  full dataset: {data['vx'].shape[0]} samples")
    print(f"  testing on : {len(val_idx)} val samples")

    imgs    = data["img"][val_idx]      # (N, 224, 224, 3) uint8 RGB
    true_vx = data["vx"][val_idx]       # (N,) float32  [m/s]
    true_vy = data["vy"][val_idx]
    true_vt = data["vtheta"][val_idx]

    vx_uniq = sorted(set(np.round(true_vx, 3).tolist()))
    vy_uniq = sorted(set(np.round(true_vy, 3).tolist()))
    vt_uniq = sorted(set(np.round(true_vt, 3).tolist()))
    print(f"  true vx unique values: {vx_uniq}")
    print(f"  true vy unique values: {vy_uniq}")
    print(f"  true vt unique values: {vt_uniq}")
    print(f"  vx>0.01: {(true_vx > 0.01).mean()*100:.1f}%  "
          f"vy!=0: {(np.abs(true_vy) > 0.01).mean()*100:.1f}%  "
          f"vt!=0: {(np.abs(true_vt) > 0.01).mean()*100:.1f}%")

    # ── Preprocessing comparison ───────────────────────────────────────────────
    section("Preprocessing: training path vs deployment path")
    # Take one sample, simulate the two paths
    sample_rgb = imgs[0]   # (224, 224, 3) uint8 RGB  (already 224 from npz)
    sample_bgr = cv2.cvtColor(sample_rgb, cv2.COLOR_RGB2BGR)  # simulate robot camera

    t_train = preprocess_training(sample_rgb, device)    # npz → train path
    t_deploy = preprocess_deployment(sample_bgr, device)  # bgr → deploy path

    diff = (t_train - t_deploy).abs()
    print(f"  Pixel-wise absolute difference (train_path vs deploy_path):")
    print(f"    mean={diff.mean():.5f}  max={diff.max():.5f}")
    if diff.max() < 0.01:
        print("    ✓ Preprocessing is effectively identical")
    else:
        print("    ✗ Significant mismatch — possible preprocessing bug!")
        print(f"    train  tensor sample [0,0,:3]: {t_train[0,:,0,0].cpu().tolist()}")
        print(f"    deploy tensor sample [0,0,:3]: {t_deploy[0,:,0,0].cpu().tolist()}")

    # ── Inference: training path ──────────────────────────────────────────────
    section(f"Inference on {len(val_idx)} val images  [training preprocessing]")
    t0 = time.perf_counter()
    p_vx, p_vy, p_vt = [], [], []
    with torch.no_grad():
        for img_rgb in tqdm(imgs, desc="train path", unit="frame"):
            t   = preprocess_training(img_rgb, device)
            out = model.predict(t)
            p_vx.append(out["vx"].item())
            p_vy.append(out["vy"].item())
            p_vt.append(out["vtheta"].item())
    elapsed = time.perf_counter() - t0

    p_vx = torch.tensor(p_vx)
    p_vy = torch.tensor(p_vy)
    p_vt = torch.tensor(p_vt)
    tv   = torch.tensor(true_vx)
    tvy  = torch.tensor(true_vy)
    tvt  = torch.tensor(true_vt)

    print(f"  Inference time: {elapsed:.2f}s total  ({elapsed/len(val_idx)*1000:.1f} ms/frame)")
    print()
    print("  Predictions (physical units):")
    stat("vx pred",     p_vx, unit=" m/s")
    stat("vy pred",     p_vy, unit=" m/s")
    stat("vtheta pred", p_vt, unit=" deg/s")
    print()
    print("  Ground truth:")
    stat("vx true",     tv,  unit=" m/s")
    stat("vy true",     tvy, unit=" m/s")
    stat("vtheta true", tvt, unit=" deg/s")

    # Per-axis accuracy (same threshold as training metrics)
    vx_acc  = ((p_vx  > 0.15) == (tv  > 0.15)).float().mean()
    vy_acc  = ((p_vy  > -0.15) & (p_vy < 0.15) ) == ((tvy > -0.15) & (tvy < 0.15))
    vt_acc  = ((p_vt  > -22.5) & (p_vt < 22.5) ) == ((tvt > -22.5) & (tvt < 22.5))
    print()
    print(f"  Direction accuracy  vx: {vx_acc*100:.1f}%"
          f"   vy: {vy_acc.float().mean()*100:.1f}%"
          f"   vtheta: {vt_acc.float().mean()*100:.1f}%")

    # Does the model actually vary its output?
    print()
    print("  Output variance check (model must respond to different inputs):")
    print(f"    vx std={p_vx.std():.4f}  vy std={p_vy.std():.4f}  vtheta std={p_vt.std():.4f}")
    if p_vx.std() < 0.005:
        print("    ✗ vx is nearly constant — model ignores image content for vx!")
    else:
        print("    ✓ vx varies across images")

    # ── Inference: deployment path ────────────────────────────────────────────
    section(f"Inference on {len(val_idx)} val images  [deployment preprocessing]")
    p_vx_d, p_vy_d, p_vt_d = [], [], []
    with torch.no_grad():
        for img_rgb in tqdm(imgs, desc="deploy path", unit="frame"):
            bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)   # simulate robot camera
            t   = preprocess_deployment(bgr, device)
            out = model.predict(t)
            p_vx_d.append(out["vx"].item())
            p_vy_d.append(out["vy"].item())
            p_vt_d.append(out["vtheta"].item())

    p_vx_d = torch.tensor(p_vx_d)
    p_vy_d = torch.tensor(p_vy_d)
    p_vt_d = torch.tensor(p_vt_d)

    print("  Predictions (physical units):")
    stat("vx pred",     p_vx_d, unit=" m/s")
    stat("vy pred",     p_vy_d, unit=" m/s")
    stat("vtheta pred", p_vt_d, unit=" deg/s")

    vx_acc_d = ((p_vx_d > 0.15) == (tv > 0.15)).float().mean()
    print(f"\n  Direction accuracy  vx: {vx_acc_d*100:.1f}%")

    deploy_vs_train = (p_vx_d - p_vx).abs()
    print(f"\n  Prediction shift (deploy vs train path):  "
          f"mean={deploy_vs_train.mean():.4f}  max={deploy_vs_train.max():.4f} m/s")
    if deploy_vs_train.mean() > 0.02:
        print("  ✗ Deployment preprocessing produces noticeably different predictions!")
    else:
        print("  ✓ Both preprocessing paths give consistent predictions")

    # ── Conditional breakdown ─────────────────────────────────────────────────
    section("Prediction breakdown by ground-truth motion type")
    fwd_mask  = tv  > 0.01
    stop_mask = tv <= 0.01

    if fwd_mask.sum() > 0:
        print(f"  When robot IS moving forward ({fwd_mask.sum()} samples):")
        stat("  vx pred",     p_vx[fwd_mask],  unit=" m/s")
        stat("  vtheta pred", p_vt[fwd_mask],  unit=" deg/s")
    if stop_mask.sum() > 0:
        print(f"  When robot is STOPPED vx ({stop_mask.sum()} samples):")
        stat("  vx pred",     p_vx[stop_mask], unit=" m/s")

    # ── Summary ───────────────────────────────────────────────────────────────
    section("Summary")
    issues = []
    if diff.max() > 0.01:
        issues.append("preprocessing mismatch between training and deployment paths")
    if p_vx.std() < 0.005:
        issues.append("model predicts nearly constant vx (ignoring image content)")
    if p_vx[fwd_mask].mean() < 0.05 and fwd_mask.sum() > 0:
        issues.append(f"model underpredicts vx on forward frames (mean={p_vx[fwd_mask].mean():.3f} m/s, expected ~0.3)")
    if deploy_vs_train.mean() > 0.02:
        issues.append(f"deployment preprocessing shifts predictions by {deploy_vs_train.mean():.3f} m/s mean")

    if issues:
        print("  Issues found:")
        for iss in issues:
            print(f"    ✗ {iss}")
    else:
        print("  ✓ No preprocessing or decoding bugs detected.")
        print("  If the robot isn't moving forward, the cause is likely domain shift")
        print("  (deployment environment looks different from training data).")


if __name__ == "__main__":
    main()
