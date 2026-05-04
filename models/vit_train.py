"""
vit_train.py

Trains either ViTCategorical or ViTContinuous on dataset_full.npz using the
pre-computed index splits from data/splits/.

Usage:
    python models/vit_train.py --mode categorical
    python models/vit_train.py --mode continuous
    python models/vit_train.py --mode categorical --epochs 50 --batch-size 64
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from vit_policy import (
    VX_MAX, VY_MAX, VTHETA_MAX,
    IMAGENET_MEAN, IMAGENET_STD,
    DEFAULT_BACKBONE,
    ViTCategorical, ViTContinuous,
)

EPS = 0.01  # threshold for "near zero" when computing categorical labels


# ── Dataset ───────────────────────────────────────────────────────────────────

class LekiwiDataset(Dataset):
    """
    Loads images and actions from dataset_full.npz, optionally subsetted
    by a pre-computed index array.

    mode='categorical': labels are (vx_class, vy_class, vtheta_class) int64
    mode='continuous':  labels are (vx_n, vy_n, vtheta_n) float32 in [-1,1]
    """

    def __init__(
        self,
        npz_path: str,
        indices: np.ndarray,
        mode: str = "categorical",
        transform=None,
    ):
        data = np.load(npz_path)
        self.img    = data["img"][indices]     # (N, H, W, 3) uint8
        self.vx     = data["vx"][indices]      # (N,) float32
        self.vy     = data["vy"][indices]
        self.vtheta = data["vtheta"][indices]
        self.mode      = mode
        self.transform = transform

    def __len__(self) -> int:
        return len(self.vx)

    def __getitem__(self, idx: int):
        # uint8 HWC → float32 CHW in [0, 1]
        img = torch.from_numpy(self.img[idx]).float().div(255.0).permute(2, 0, 1)
        if self.transform is not None:
            img = self.transform(img)

        vx, vy, vth = float(self.vx[idx]), float(self.vy[idx]), float(self.vtheta[idx])

        if self.mode == "categorical":
            vx_c  = 0 if vx  < EPS  else 1
            vy_c  = 0 if vy  < -EPS else (1 if vy  < EPS  else 2)
            vth_c = 0 if vth < -EPS else (1 if vth < EPS  else 2)
            label = torch.tensor([vx_c, vy_c, vth_c], dtype=torch.long)
        else:
            label = torch.tensor(
                [vx / VX_MAX, vy / VY_MAX, vth / VTHETA_MAX],
                dtype=torch.float32,
            )

        return img, label


def make_loaders(
    npz_path: str,
    splits_dir: str,
    mode: str,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    # Training augmentation: color jitter only (no flips/rotation — they would
    # change the spatial meaning of vy/vtheta labels)
    train_tf = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        normalize,
    ])
    val_tf = normalize

    train_idx = np.load(os.path.join(splits_dir, "train_idx.npy"))
    val_idx   = np.load(os.path.join(splits_dir, "val_idx.npy"))

    train_ds = LekiwiDataset(npz_path, train_idx, mode, train_tf)
    val_ds   = LekiwiDataset(npz_path, val_idx,   mode, val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ── Loss and metrics ──────────────────────────────────────────────────────────

def categorical_loss(
    logits: dict[str, torch.Tensor],
    labels: torch.Tensor,          # (B, 3) int64: [vx_c, vy_c, vth_c]
) -> torch.Tensor:
    ce = nn.CrossEntropyLoss()
    return (
        ce(logits["vx"],     labels[:, 0]) +
        ce(logits["vy"],     labels[:, 1]) +
        ce(logits["vtheta"], labels[:, 2])
    )


def categorical_accuracy(
    logits: dict[str, torch.Tensor],
    labels: torch.Tensor,
) -> dict[str, float]:
    vx_acc  = (logits["vx"]    .argmax(1) == labels[:, 0]).float().mean().item()
    vy_acc  = (logits["vy"]    .argmax(1) == labels[:, 1]).float().mean().item()
    vth_acc = (logits["vtheta"].argmax(1) == labels[:, 2]).float().mean().item()
    all_acc = (
        (logits["vx"]    .argmax(1) == labels[:, 0]) &
        (logits["vy"]    .argmax(1) == labels[:, 1]) &
        (logits["vtheta"].argmax(1) == labels[:, 2])
    ).float().mean().item()
    return {"vx": vx_acc, "vy": vy_acc, "vtheta": vth_acc, "all": all_acc}


def continuous_loss(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(preds, labels)


def continuous_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        mse = (preds - labels).pow(2).mean(0)  # (3,)
        # "accuracy" = fraction where argmax of quantised prediction matches label
        # quantise continuous pred to nearest class for quick comparison
        def quant_acc(pred_col: torch.Tensor, label_col: torch.Tensor, n_classes: int) -> float:
            if n_classes == 2:
                pred_cls = (pred_col > 0.5).long()
                true_cls = (label_col > 0.5).long()
            else:
                boundaries = torch.tensor([-0.5, 0.5], device=pred_col.device)
                pred_cls = torch.bucketize(pred_col, boundaries)
                true_cls = torch.bucketize(label_col, boundaries)
            return (pred_cls == true_cls).float().mean().item()

        return {
            "mse_vx":     mse[0].item(),
            "mse_vy":     mse[1].item(),
            "mse_vtheta": mse[2].item(),
            "acc_vx":     quant_acc(preds[:, 0], labels[:, 0], 2),
            "acc_vy":     quant_acc(preds[:, 1], labels[:, 1], 3),
            "acc_vtheta": quant_acc(preds[:, 2], labels[:, 2], 3),
        }


# ── Training loop ─────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    mode: str,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
    desc: str = "",
) -> tuple[float, dict]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_metrics: dict[str, list] = {}
    n_batches = 0

    bar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    with torch.set_grad_enabled(training):
        for imgs, labels in bar:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                if mode == "categorical":
                    logits = model(imgs)
                    loss   = categorical_loss(logits, labels)
                    m      = categorical_accuracy(logits, labels)
                else:
                    preds = model(imgs)
                    loss  = continuous_loss(preds, labels)
                    m     = continuous_metrics(preds, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            total_loss += loss.item()
            n_batches  += 1
            for k, v in m.items():
                all_metrics.setdefault(k, []).append(v)

            bar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    avg_loss    = total_loss / max(n_batches, 1)
    avg_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    return avg_loss, avg_metrics


def train(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.mode == "categorical":
        model = ViTCategorical(pretrained=not args.no_pretrain, backbone=args.backbone)
    else:
        model = ViTContinuous(pretrained=not args.no_pretrain, backbone=args.backbone)
    model = model.to(device)
    print(f"Backbone: {args.backbone}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = make_loaders(
        npz_path    = args.data_path,
        splits_dir  = args.splits_dir,
        mode        = args.mode,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    # ── Optimiser: lower LR for backbone, higher for head ────────────────────
    backbone_params = list(model.backbone.parameters())
    if args.mode == "categorical":
        head_params = (list(model.head_vx.parameters()) +
                       list(model.head_vy.parameters()) +
                       list(model.head_vtheta.parameters()))
    else:
        head_params = list(model.head.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params,     "lr": args.head_lr},
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    # ── Checkpoint dir ────────────────────────────────────────────────────────
    os.makedirs(args.ckpt_dir, exist_ok=True)
    # e.g. vit_categorical_small_best.pth or vit_continuous_tiny_best.pth
    size_tag = "tiny" if "tiny" in args.backbone else "small" if "small" in args.backbone else "base"
    if args.no_pretrain:
        size_tag = f"{size_tag}_npt"
    ckpt_path = os.path.join(args.ckpt_dir, f"vit_{args.mode}_{size_tag}_best.pth")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_m = run_epoch(model, train_loader, optimizer, args.mode, device, scaler, desc=f"Ep {epoch:3d} train")
        val_loss,   val_m   = run_epoch(model, val_loader,   None,      args.mode, device, desc=f"Ep {epoch:3d} val  ")
        scheduler.step()

        elapsed = time.time() - t0

        if args.mode == "categorical":
            print(
                f"Epoch {epoch:3d}/{args.epochs}  "
                f"loss {train_loss:.4f}/{val_loss:.4f}  "
                f"acc(all) {train_m['all']:.3f}/{val_m['all']:.3f}  "
                f"vx {val_m['vx']:.3f}  vy {val_m['vy']:.3f}  vth {val_m['vtheta']:.3f}  "
                f"[{elapsed:.0f}s]"
            )
        else:
            print(
                f"Epoch {epoch:3d}/{args.epochs}  "
                f"loss {train_loss:.4f}/{val_loss:.4f}  "
                f"acc vx {val_m['acc_vx']:.3f}  vy {val_m['acc_vy']:.3f}  "
                f"vth {val_m['acc_vtheta']:.3f}  "
                f"[{elapsed:.0f}s]"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch":      epoch,
                "mode":       args.mode,
                "backbone":   args.backbone,
                "state_dict": model.state_dict(),
                "val_loss":   val_loss,
                "val_metrics": val_m,
            }, ckpt_path)
            print(f"  ✓ saved best checkpoint (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping after {epoch} epochs (no improvement for {args.patience}).")
                break

        if args.save_every > 0 and epoch % args.save_every == 0:
            ep_path = os.path.join(
                args.ckpt_dir, f"vit_{args.mode}_{size_tag}_epoch{epoch:03d}.pth"
            )
            torch.save({
                "epoch":       epoch,
                "mode":        args.mode,
                "backbone":    args.backbone,
                "state_dict":  model.state_dict(),
                "val_loss":    val_loss,
                "val_metrics": val_m,
            }, ep_path)
            print(f"  [epoch ckpt] {ep_path}")

    print(f"\nBest val loss: {best_val_loss:.4f}  →  {ckpt_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train ViT behavior-cloning policy")
    parser.add_argument("--mode",         choices=["categorical", "continuous"],
                        required=True)
    parser.add_argument("--data-path",    default="./data/dataset_full.npz")
    parser.add_argument("--splits-dir",   default="./data/splits")
    parser.add_argument("--ckpt-dir",     default="./checkpoints")
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--backbone-lr",  type=float, default=1e-4)
    parser.add_argument("--head-lr",      type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience",     type=int,   default=7)
    parser.add_argument("--num-workers",  type=int,   default=4)
    parser.add_argument("--backbone",     default=DEFAULT_BACKBONE,
                        help="timm backbone name (default: vit_small_patch16_224)")
    parser.add_argument("--save-every",   type=int,   default=0,
                        help="Save an epoch checkpoint every N epochs (0=disabled). "
                             "Files: vit_{mode}_{size}_epoch{N:03d}.pth")
    parser.add_argument("--no-pretrain",  action="store_true",
                        help="Train backbone from scratch instead of ImageNet weights")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
