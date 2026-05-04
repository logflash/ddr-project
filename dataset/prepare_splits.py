"""
prepare_splits.py

Creates stratified train / val / steer_probe / steer_eval index splits from
dataset_full.npz, then saves index arrays to data/splits/.

Split fractions (of total N):
  train       70 %   ViT training + SAE activation collection
  val         15 %   ViT early stopping / model selection
  steer_probe  7.5%  find contrastive steering directions & train linear probes
  steer_eval   7.5%  blind evaluation of whether steering generalises

Stratification key: (source_id, vx_class, vy_class, vtheta_class)
  Ensures every split has proportional representation of each data source
  AND each motion type within each source.

Rare strata (< MIN_STRATUM_SIZE samples) are merged into a single "other"
bucket before splitting to avoid sklearn errors on tiny groups.

Usage:
    python dataset/prepare_splits.py
    python dataset/prepare_splits.py --data-path ./data/dataset_full.npz
"""

import argparse
import os

import numpy as np
from sklearn.model_selection import train_test_split

# Source index -> name mapping (must match DATASETS order in construct_dataset.py)
SOURCE_NAMES = ["strafe", "rotation", "strafe_rotation", "straight", "CW"]

EPS = 0.01             # threshold for "near zero" actions
MIN_STRATUM_SIZE = 10  # strata smaller than this are merged into bucket -1


def action_stratum(vx: float, vy: float, vtheta: float) -> int:
    vx_c  = 0 if vx < EPS else 1
    vy_c  = 0 if vy < -EPS else (1 if vy < EPS else 2)
    vth_c = 0 if vtheta < -EPS else (1 if vtheta < EPS else 2)
    return vx_c * 9 + vy_c * 3 + vth_c  # unique int in [0, 17]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path",  default="./data/dataset_full.npz")
    parser.add_argument("--output-dir", default="./data/splits")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac",   type=float, default=0.15)
    parser.add_argument("--probe-frac", type=float, default=0.075)
    # steer_eval gets 1 - train - val - probe
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    steer_frac = 1.0 - args.train_frac - args.val_frac - args.probe_frac
    assert steer_frac > 0, "fractions must sum to < 1"

    data = np.load(args.data_path)
    vx, vy, vtheta = data["vx"], data["vy"], data["vtheta"]
    source = data["source"]
    N = int(vx.shape[0])
    print(f"Loaded {N} samples from {args.data_path}")

    # ── Build combined (source, action) strata ────────────────────────────────
    n_action_strata = 18  # 2 * 3 * 3
    raw_strata = np.array([
        int(source[i]) * n_action_strata + action_stratum(vx[i], vy[i], vtheta[i])
        for i in range(N)
    ])

    unique, counts = np.unique(raw_strata, return_counts=True)
    print("\nStrata distribution (source × action):")
    vx_names  = ["vx=0  ", "vx=fwd"]
    vy_names  = ["vy=R  ", "vy=0  ", "vy=L  "]  # vy<0 = right, vy>0 = left (LeKiwi convention)
    vth_names = ["vt=CW ", "vt=0  ", "vt=CCW"]
    for s, c in zip(unique, counts):
        src_id  = s // n_action_strata
        act_id  = s % n_action_strata
        vx_c, vy_c, vth_c = act_id // 9, (act_id % 9) // 3, act_id % 3
        src_name = SOURCE_NAMES[src_id] if src_id < len(SOURCE_NAMES) else str(src_id)
        tag = f"{src_name:16s} {vx_names[vx_c]} {vy_names[vy_c]} {vth_names[vth_c]}"
        rare = "  ← rare, merged" if c < MIN_STRATUM_SIZE else ""
        print(f"  {tag}: {c:5d} ({100*c/N:.1f}%){rare}")

    # Merge rare strata
    rare_ids = unique[counts < MIN_STRATUM_SIZE]
    strata = raw_strata.copy()
    strata[np.isin(raw_strata, rare_ids)] = -1

    # ── Sequential stratified splits ─────────────────────────────────────────
    idx = np.arange(N)

    # 1. Peel off steer_eval
    idx_rest, idx_steer = train_test_split(
        idx, test_size=steer_frac, stratify=strata, random_state=args.seed
    )

    # 2. Peel off steer_probe
    probe_frac_of_rest = args.probe_frac / (1.0 - steer_frac)
    idx_tv, idx_probe = train_test_split(
        idx_rest, test_size=probe_frac_of_rest,
        stratify=strata[idx_rest], random_state=args.seed
    )

    # 3. Split train / val
    val_frac_of_tv = args.val_frac / (args.train_frac + args.val_frac)
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=val_frac_of_tv,
        stratify=strata[idx_tv], random_state=args.seed
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "train_idx.npy"),       idx_train)
    np.save(os.path.join(args.output_dir, "val_idx.npy"),         idx_val)
    np.save(os.path.join(args.output_dir, "steer_probe_idx.npy"), idx_probe)
    np.save(os.path.join(args.output_dir, "steer_eval_idx.npy"),  idx_steer)

    print(f"\nSplit summary (total {N}):")
    for split_name, arr in [("train",       idx_train),
                             ("val",         idx_val),
                             ("steer_probe", idx_probe),
                             ("steer_eval",  idx_steer)]:
        src_counts = {n: int((source[arr] == i).sum())
                      for i, n in enumerate(SOURCE_NAMES)}
        src_str = "  ".join(f"{n}={c}" for n, c in src_counts.items())
        print(f"  {split_name:12s}: {len(arr):5d} ({100*len(arr)/N:.1f}%)   {src_str}")

    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
