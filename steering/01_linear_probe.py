"""
linear_probe.py  --  Linear decodability of action dimensions per ViT layer.

For each layer, fits a Ridge regression (192 → 1) on the CLS token from the
steer_probe set and reports R² for vx, vy, and vtheta.

Question: Can a linear model predict each action dimension from the [CLS]
token at a given layer?

Usage:
    python steering/linear_probe.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "models"))

from vit_policy import ViTContinuous, collect_cls_streams, IMAGENET_MEAN, IMAGENET_STD
from vit_train import LekiwiDataset

CKPT       = "./checkpoints/vit_continuous_tiny_best.pth"
DATA_PATH  = "./data/dataset_full.npz"
SPLITS_DIR = "./data/splits"
OUT_DIR    = "./steering/results"
BATCH_SIZE = 64

AXIS_NAMES  = ["vx", "vy", "vtheta"]
AXIS_COLORS = {"vx": "steelblue", "vy": "tomato", "vtheta": "darkorange"}
BOUNDARY    = {"vx": 0.5, "vy": 0.0, "vtheta": 0.0}


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

    # ── Collect CLS streams on steer_probe ───────────────────────────────────
    tf  = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    idx = np.load(os.path.join(SPLITS_DIR, "steer_probe_idx.npy"))
    ds  = LekiwiDataset(DATA_PATH, idx, mode="continuous", transform=tf)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    def collect_streams(loader):
        all_acts, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                streams = collect_cls_streams(model, imgs.to(device))
                all_acts.append(streams.numpy())
                all_labels.append(labels.numpy())
        return (np.concatenate(all_acts, axis=0),
                np.concatenate(all_labels, axis=0))

    print(f"\nCollecting CLS streams on steer_probe ({len(idx)} samples)...")
    acts_train, labels_train = collect_streams(loader)
    n_layers = acts_train.shape[1]
    print(f"  train: {acts_train.shape}")

    eval_idx    = np.load(os.path.join(SPLITS_DIR, "steer_eval_idx.npy"))
    eval_ds     = LekiwiDataset(DATA_PATH, eval_idx, mode="continuous", transform=tf)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Collecting CLS streams on steer_eval ({len(eval_idx)} samples)...")
    acts_eval, labels_eval = collect_streams(eval_loader)
    print(f"  eval:  {acts_eval.shape}")

    # ── Masks: only non-neutral samples for vy/vtheta ────────────────────────
    # For vy and vtheta, samples near zero dominate and inflate R².
    # CONTRAST_THRESH filters to samples that are clearly positive or negative.
    # vx spans [0,1] with no neutral zone, so all samples are kept.
    CONTRAST_THRESH = 0.4
    NONZERO_MASK = {
        "vx":     np.ones(len(labels_train),  dtype=bool),   # keep all
        "vy":     np.abs(labels_train[:, 1]) > CONTRAST_THRESH,
        "vtheta": np.abs(labels_train[:, 2]) > CONTRAST_THRESH,
    }
    NONZERO_MASK_EVAL = {
        "vx":     np.ones(len(labels_eval),   dtype=bool),
        "vy":     np.abs(labels_eval[:, 1])  > CONTRAST_THRESH,
        "vtheta": np.abs(labels_eval[:, 2])  > CONTRAST_THRESH,
    }
    for name in ["vy", "vtheta"]:
        print(f"  {name}: {NONZERO_MASK[name].sum()} probe / "
              f"{NONZERO_MASK_EVAL[name].sum()} eval non-neutral samples")

    def class_weights_2(labels_col, boundary):
        """Binary balance: above/below boundary."""
        pos = labels_col > boundary
        n_pos, n_neg = pos.sum(), (~pos).sum()
        w = np.where(pos, 1.0 / n_pos, 1.0 / n_neg)
        return w / w.sum() * len(w)

    def class_weights_3(labels_col, thresh):
        """3-class balance: negative / neutral / positive, each weighted 1/3."""
        pos  = labels_col >  thresh
        neg  = labels_col < -thresh
        zero = ~pos & ~neg
        w = np.ones(len(labels_col))
        for mask in (pos, neg, zero):
            n = mask.sum()
            if n > 0:
                w[mask] = 1.0 / n
        return w / w.sum() * len(w)

    def get_weights(labels_col, name, mask=None):
        col = labels_col if mask is None else labels_col[mask]
        if name == "vx":
            return class_weights_2(col, BOUNDARY[name])
        else:
            return class_weights_3(col, CONTRAST_THRESH)

    # ── Fit on probe, evaluate on eval ───────────────────────────────────────
    r2        = np.zeros((n_layers, 3))
    r2_bal    = np.zeros((n_layers, 3))
    r2_nz     = np.zeros((n_layers, 3))
    r2_nz_bal = np.zeros((n_layers, 3))
    for layer in range(n_layers):
        X_train_full, X_eval_full = acts_train[:, layer, :], acts_eval[:, layer, :]
        for ai, name in enumerate(AXIS_NAMES):
            m_tr = NONZERO_MASK[name]
            m_ev = NONZERO_MASK_EVAL[name]
            w_tr_full = get_weights(labels_train[:, ai],    name)
            w_ev_full = get_weights(labels_eval[:, ai],     name)
            w_tr_nz   = get_weights(labels_train[m_tr, ai], name)
            w_ev_nz   = get_weights(labels_eval[m_ev, ai],  name)
            # standard R²
            reg = Ridge(alpha=1.0).fit(X_train_full, labels_train[:, ai])
            r2[layer, ai] = r2_score(labels_eval[:, ai], reg.predict(X_eval_full))
            # balanced R² (all samples, inverse class-frequency weights)
            reg_bal = Ridge(alpha=1.0).fit(X_train_full, labels_train[:, ai],
                                           sample_weight=w_tr_full)
            r2_bal[layer, ai] = r2_score(labels_eval[:, ai], reg_bal.predict(X_eval_full),
                                         sample_weight=w_ev_full)
            # non-neutral R²
            reg_nz = Ridge(alpha=1.0).fit(X_train_full[m_tr], labels_train[m_tr, ai])
            yhat_nz = reg_nz.predict(X_eval_full[m_ev])
            r2_nz[layer, ai] = r2_score(labels_eval[m_ev, ai], yhat_nz)
            # non-neutral + balanced
            reg_nz_bal = Ridge(alpha=1.0).fit(X_train_full[m_tr], labels_train[m_tr, ai],
                                              sample_weight=w_tr_nz)
            r2_nz_bal[layer, ai] = r2_score(labels_eval[m_ev, ai],
                                            reg_nz_bal.predict(X_eval_full[m_ev]),
                                            sample_weight=w_ev_nz)


    layers = np.arange(n_layers)

    for label, arr in [("R²", r2), ("Balanced R²", r2_bal), ("Non-neutral R²", r2_nz), ("Non-neutral balanced R²", r2_nz_bal)]:
        print(f"\n{label} per layer (out-of-sample):")
        header = f"  {'layer':>5}  " + "  ".join(f"{n:>8}" for n in AXIS_NAMES)
        print(header)
        for l in range(n_layers):
            row = f"  {l:>5}  " + "  ".join(f"{arr[l, i]:>8.4f}" for i in range(3))
            print(row)

    # ── R² plots ──────────────────────────────────────────────────────────────
    for fname, arr, title, ylabel in [
        ("linear_probe_r2.png",        r2,        "Linear probe R² per layer",                         "R²"),
        ("linear_probe_r2_bal.png",    r2_bal,    "Linear probe R² per layer (balanced)",               "R²"),
        ("linear_probe_r2_nz.png",     r2_nz,     "Linear probe R² per layer (non-neutral)",            "R²"),
        ("linear_probe_r2_nz_bal.png", r2_nz_bal, "Linear probe R² per layer (non-neutral, balanced)",  "R²"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, name in enumerate(AXIS_NAMES):
            ax.plot(layers, arr[:, i], marker="o", label=name,
                    color=AXIS_COLORS[name], linewidth=1.8)
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(layers)
        ax.set_xticklabels([str(l) for l in layers])
        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(OUT_DIR, fname)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")

    # ── Linear classification probe ───────────────────────────────────────────
    # Binarize labels using per-axis decision boundaries
    bin_train = np.stack([labels_train[:, i] > BOUNDARY[n]
                          for i, n in enumerate(AXIS_NAMES)], axis=1).astype(int)
    bin_eval  = np.stack([labels_eval[:, i]  > BOUNDARY[n]
                          for i, n in enumerate(AXIS_NAMES)], axis=1).astype(int)

    f1 = np.zeros((n_layers, 3))
    for layer in range(n_layers):
        X_train, X_eval = acts_train[:, layer, :], acts_eval[:, layer, :]
        for ai in range(3):
            clf  = LogisticRegression(max_iter=1000, C=1.0).fit(X_train, bin_train[:, ai])
            yhat = clf.predict(X_eval)
            f1[layer, ai] = f1_score(bin_eval[:, ai], yhat)

    print(f"\nClassifier F1 per layer (out-of-sample):")
    header = f"  {'layer':>5}  " + "  ".join(f"{n:>8}" for n in AXIS_NAMES)
    print(header)
    for l in range(n_layers):
        row = f"  {l:>5}  " + "  ".join(f"{f1[l, i]:>8.4f}" for i in range(3))
        print(row)

    # ── F1 plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, name in enumerate(AXIS_NAMES):
        ax.plot(layers, f1[:, i], marker="o", label=name,
                color=AXIS_COLORS[name], linewidth=1.8)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("F1 score", fontsize=11)
    ax.set_title("Linear classification probe F1 per layer", fontsize=12)
    ax.set_xticks(layers)
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "linear_probe_f1.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
