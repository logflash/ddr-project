"""
train_sae.py  --  Sparse Autoencoder on layer-9 CLS activations.

Trains an SAE with expansion * d_model features (default 8*192=1536) on CLS
activations from the train split at a given layer. Then identifies features
correlated with vx/vy/vtheta on the held-out steer_probe set, and compares
each top feature's decoder direction to the contrastive steering vector.

Outputs:
    checkpoints/sae_layer9.pth              -- SAE weights
    sae/results/feature_scatter.png         -- top feature activations vs labels
    sae/results/feature_decoder_cosine.png  -- decoder direction vs steering vec

Usage:
    py sae/train_sae.py
    py sae/train_sae.py --l1 5e-4 --epochs 100
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "models"))

from torchvision import transforms
from vit_policy import ViTContinuous, collect_cls_streams, IMAGENET_MEAN, IMAGENET_STD
from vit_train import LekiwiDataset

AXIS_NAMES     = ["vx", "vy", "vtheta"]
CONTRAST_THRESH = 0.4


# -- SAE model ----------------------------------------------------------------

class SparseAutoencoder(nn.Module):
    """
    TopK SAE: h = topk(ReLU((x - b_pre) @ W_enc + b_enc), k)
              x_hat = h @ W_dec + b_dec

    Exactly `k` features are active per sample. No L1 needed.
    Decoder rows are kept unit-norm after each gradient step.
    Falls back to plain ReLU when k <= 0.
    """
    def __init__(self, d_model: int, n_features: int, k: int = 32):
        super().__init__()
        self.d_model    = d_model
        self.n_features = n_features
        self.k          = k

        self.b_pre = nn.Parameter(torch.zeros(d_model))
        self.W_enc = nn.Parameter(torch.empty(d_model, n_features))
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.W_dec = nn.Parameter(torch.empty(n_features, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = F.relu((x - self.b_pre) @ self.W_enc + self.b_enc)
        if self.k <= 0:
            return pre
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        h = torch.zeros_like(pre)
        h.scatter_(-1, topk_idx, topk_vals)
        return h

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        return h, self.decode(h)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)


# -- activation collection ----------------------------------------------------

@torch.no_grad()
def collect_layer_acts(
    model:  ViTContinuous,
    loader: DataLoader,
    layer:  int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (N, D) CLS activations at `layer` and (N, 3) labels."""
    all_acts, all_labels = [], []
    for imgs, labels in loader:
        imgs    = imgs.to(device)
        streams = collect_cls_streams(model, imgs)          # (B, L, D) on CPU
        all_acts.append(streams[:, layer, :].numpy())
        all_labels.append(labels.numpy())
    return (
        np.concatenate(all_acts,   axis=0),
        np.concatenate(all_labels, axis=0),
    )


# -- SAE training -------------------------------------------------------------

def train_sae(
    acts:       torch.Tensor,
    sae:        SparseAutoencoder,
    device:     torch.device,
    epochs:     int   = 50,
    batch_size: int   = 512,
    lr:         float = 1e-3,
    l1_coeff:   float = 1e-3,
) -> SparseAutoencoder:
    sae    = sae.to(device)
    opt    = torch.optim.Adam(sae.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(acts), batch_size=batch_size, shuffle=True)
    use_l1 = sae.k <= 0   # TopK needs no L1

    for epoch in range(1, epochs + 1):
        total_loss = total_mse = total_l1 = 0.0
        n_batches  = 0

        for (batch,) in loader:
            batch    = batch.to(device)
            h, x_hat = sae(batch)
            mse      = (batch - x_hat).pow(2).mean()
            l1       = h.abs().mean()
            loss     = mse + (l1_coeff * l1 if use_l1 else 0.0)

            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.normalize_decoder()

            total_loss += loss.item()
            total_mse  += mse.item()
            total_l1   += l1.item()
            n_batches  += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{epochs}  "
                  f"mse={total_mse/n_batches:.5f}  "
                  f"l1={total_l1/n_batches:.5f}")
    return sae


# -- feature analysis ---------------------------------------------------------

def compute_correlations(
    h_probe: np.ndarray,   # (N, F)
    labels:  np.ndarray,   # (N, 3)
) -> np.ndarray:
    """Return Pearson r matrix shaped (F, 3). Dead features get r=0."""
    N         = h_probe.shape[0]
    h_c       = h_probe - h_probe.mean(axis=0, keepdims=True)
    y_c       = labels  - labels .mean(axis=0, keepdims=True)
    h_std     = h_probe.std(axis=0)                             # (F,)
    y_std     = labels .std(axis=0)                             # (3,)
    cov       = (h_c.T @ y_c) / N                              # (F, 3)
    denom     = np.outer(h_std, y_std)                          # (F, 3)
    return np.where(denom > 1e-6, cov / denom, 0.0)            # (F, 3)


# -- main ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SAE on layer CLS activations and find action features"
    )
    parser.add_argument("--ckpt",        default="./checkpoints/vit_continuous_tiny_best.pth")
    parser.add_argument("--data-path",   default="./data/dataset_full.npz")
    parser.add_argument("--splits-dir",  default="./data/splits")
    parser.add_argument("--sae-ckpt",    default="./checkpoints/sae_layer9.pth")
    parser.add_argument("--out-dir",     default="./sae/results")
    parser.add_argument("--layer",       type=int,   default=9)
    parser.add_argument("--expansion",   type=int,   default=8,
                        help="n_features = expansion * d_model  (default: 8)")
    parser.add_argument("--topk",        type=int,   default=32,
                        help="TopK active features per sample (0 = use ReLU+L1)")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch-size",  type=int,   default=512)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--l1",          type=float, default=1e-3)
    parser.add_argument("--top-k",       type=int,   default=5)
    parser.add_argument("--num-workers", type=int,   default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    # -- Load ViT model -------------------------------------------------------
    ckpt     = torch.load(args.ckpt, map_location=device, weights_only=False)
    fname    = os.path.basename(args.ckpt)
    fallback = "vit_tiny_patch16_224" if "tiny" in fname else "vit_small_patch16_224"
    backbone = ckpt.get("backbone", fallback)
    model    = ViTContinuous(pretrained=False, backbone=backbone)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    print(f"Loaded {backbone} from {args.ckpt}  (epoch {ckpt['epoch']})")

    d_model    = model.embed_dim
    n_features = args.expansion * d_model
    print(f"SAE: d_model={d_model}  n_features={n_features}  layer={args.layer}  "
          f"topk={args.topk if args.topk > 0 else 'ReLU+L1'}")

    tf = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def make_loader(split_file: str) -> DataLoader:
        idx = np.load(os.path.join(args.splits_dir, split_file))
        ds  = LekiwiDataset(args.data_path, idx, mode="continuous", transform=tf)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers)

    # -- Collect activations --------------------------------------------------
    print(f"\nCollecting train activations at layer {args.layer}...")
    train_acts, _ = collect_layer_acts(
        model, make_loader("train_idx.npy"), args.layer, device
    )
    print(f"  train_acts: {train_acts.shape}")

    print(f"Collecting probe activations at layer {args.layer}...")
    probe_acts, probe_labels = collect_layer_acts(
        model, make_loader("steer_probe_idx.npy"), args.layer, device
    )
    print(f"  probe_acts: {probe_acts.shape}")

    # -- Train SAE ------------------------------------------------------------
    mode_str = f"topk={args.topk}" if args.topk > 0 else f"l1={args.l1}"
    print(f"\nTraining SAE  ({mode_str}  epochs={args.epochs})...")
    sae          = SparseAutoencoder(d_model, n_features, k=args.topk)
    train_tensor = torch.tensor(train_acts, dtype=torch.float32)
    sae = train_sae(train_tensor, sae, device,
                    epochs=args.epochs, batch_size=args.batch_size,
                    lr=args.lr, l1_coeff=args.l1)
    sae.eval()

    torch.save({
        "d_model":    d_model,
        "n_features": n_features,
        "layer":      args.layer,
        "k":          args.topk,
        "l1_coeff":   args.l1,
        "state_dict": sae.state_dict(),
    }, args.sae_ckpt)
    print(f"Saved SAE to {args.sae_ckpt}")

    # -- SAE statistics on probe set ------------------------------------------
    probe_t = torch.tensor(probe_acts, dtype=torch.float32).to(device)
    with torch.no_grad():
        h_probe_t, x_hat_t = sae(probe_t)
    h_probe  = h_probe_t.cpu().numpy()       # (N, F)
    x_hat    = x_hat_t.cpu().numpy()

    recon_mse = float(((probe_acts - x_hat) ** 2).mean())
    l0        = float((h_probe > 0).sum(axis=1).mean())
    n_dead    = int((h_probe.max(axis=0) == 0).sum())
    print(f"\nSAE probe stats:")
    print(f"  Reconstruction MSE    : {recon_mse:.5f}")
    print(f"  Mean L0 (active/sample): {l0:.1f} / {n_features}")
    print(f"  Dead features         : {n_dead} / {n_features}")

    # -- Feature correlations with action axes --------------------------------
    corr = compute_correlations(h_probe, probe_labels)   # (F, 3)

    print(f"\nTop {args.top_k} features per axis (by |Pearson r|):")
    top_features: dict[str, list[tuple[int, float]]] = {}
    for ai, aname in enumerate(AXIS_NAMES):
        top_idx = np.argsort(np.abs(corr[:, ai]))[::-1][:args.top_k]
        top_features[aname] = [(int(i), float(corr[i, ai])) for i in top_idx]
        print(f"\n  {aname}:")
        for rank, (feat_i, r) in enumerate(top_features[aname]):
            act_rate = float((h_probe[:, feat_i] > 0).mean())
            mean_act = float(h_probe[:, feat_i][h_probe[:, feat_i] > 0].mean())  \
                if act_rate > 0 else 0.0
            print(f"    #{rank+1}  feature {feat_i:4d}  r={r:+.3f}  "
                  f"active={act_rate*100:.1f}%  mean_activation={mean_act:.3f}")

    # -- Scatter plots: top features vs action labels -------------------------
    TOP_PLOT = min(3, args.top_k)
    colors   = ["steelblue", "tomato", "darkorange"]
    fig, axes = plt.subplots(3, TOP_PLOT, figsize=(5 * TOP_PLOT, 12))
    if TOP_PLOT == 1:
        axes = axes[:, np.newaxis]

    for ai, aname in enumerate(AXIS_NAMES):
        for rank in range(TOP_PLOT):
            feat_i, r = top_features[aname][rank]
            ax = axes[ai, rank]
            ax.scatter(probe_labels[:, ai], h_probe[:, feat_i],
                       alpha=0.3, s=8, color=colors[ai])
            ax.set_xlabel(f"{aname} (normalised)")
            ax.set_ylabel(f"Feature {feat_i} activation")
            ax.set_title(f"{aname}  rank {rank+1}: feature {feat_i}\nr={r:+.3f}")
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(args.out_dir, "feature_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {path}")

    # -- Decoder direction vs contrastive steering direction ------------------
    steer_dirs: dict[str, np.ndarray] = {}
    for ai, aname in enumerate(AXIS_NAMES):
        col      = probe_labels[:, ai]
        pos_mask = col > 0.5 if aname == "vx" else col >  CONTRAST_THRESH
        neg_mask = col < 0.5 if aname == "vx" else col < -CONTRAST_THRESH
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue
        raw = probe_acts[pos_mask].mean(0) - probe_acts[neg_mask].mean(0)
        steer_dirs[aname] = raw / (np.linalg.norm(raw) + 1e-8)

    W_dec = sae.W_dec.detach().cpu().numpy()   # (F, D) unit-norm rows

    print(f"\nDecoder cosine sim to contrastive steering direction:")
    for aname in AXIS_NAMES:
        if aname not in steer_dirs:
            continue
        sv = steer_dirs[aname]
        print(f"\n  {aname}:")
        for rank, (feat_i, r) in enumerate(top_features[aname]):
            cos = float(np.dot(W_dec[feat_i], sv))
            print(f"    #{rank+1}  feature {feat_i:4d}  pearson_r={r:+.3f}  "
                  f"decoder_cos={cos:+.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ai, aname in enumerate(AXIS_NAMES):
        if aname not in steer_dirs:
            continue
        ax      = axes[ai]
        sv      = steer_dirs[aname]
        ranked  = top_features[aname]
        cosines = [float(np.dot(W_dec[f], sv)) for f, _ in ranked]
        xlabels = [f"f{f}\nr={r:+.2f}" for f, r in ranked]
        ax.bar(range(len(ranked)), cosines, color=colors[ai], alpha=0.8)
        ax.set_xticks(range(len(ranked)))
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_ylabel("Decoder cosine sim to steering vec")
        ax.set_title(f"Top {args.top_k} {aname} features\nvs contrastive steering direction")
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(args.out_dir, "feature_decoder_cosine.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {path}")

    print(f"\nDone. Results in {args.out_dir}/")


if __name__ == "__main__":
    main()
