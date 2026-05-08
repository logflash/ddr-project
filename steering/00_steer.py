"""
00_steer.py  --  Compute and save contrastive steering vectors.

Computes unit-norm contrastive directions (mean_pos - mean_neg) at each layer
on the steer_probe set, then saves the vectors for the chosen layer.

Usage:
    python steering/00_steer.py --layer 9
    python steering/00_steer.py --layer 6
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "models"))
sys.path.insert(0, _HERE)

from vit_policy import IMAGENET_MEAN, IMAGENET_STD
from vit_train import LekiwiDataset
from steer import AXIS_NAMES, load_model, collect_activations, compute_directions

DATA_PATH  = "./data/dataset_full.npz"
SPLITS_DIR = "./data/splits"
BATCH_SIZE = 64


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",  default="./checkpoints/vit_continuous_tiny_best.pth")
    parser.add_argument("--layer", type=int, required=True,
                        help="Layer index to save steering vectors for")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    model, _ = load_model(args.ckpt, device)

    tf  = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    idx = np.load(os.path.join(SPLITS_DIR, "steer_probe_idx.npy"))
    ds  = LekiwiDataset(DATA_PATH, idx, mode="continuous", transform=tf)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\nCollecting probe activations ({len(idx)} samples)...")
    probe_acts, probe_labels = collect_activations(model, loader, device)

    print("\nComputing steering directions...")
    directions = compute_directions(probe_acts, probe_labels)

    out_path = args.ckpt.replace(".pth", f"_steer_vecs_layer{args.layer}.pth")
    vecs_ckpt = {"ckpt": args.ckpt, "layers": {name: args.layer for name in AXIS_NAMES}}
    for name in AXIS_NAMES:
        v = directions[name][args.layer].astype(np.float32)
        vecs_ckpt[name] = v
        print(f"  {name}  layer={args.layer}  norm={np.linalg.norm(v):.4f}")

    torch.save(vecs_ckpt, out_path)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
