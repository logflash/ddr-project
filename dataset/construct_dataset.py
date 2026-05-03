"""
construct_dataset.py

Downloads five lekiwi datasets from HuggingFace and saves them as .npz files,
keeping continuous action values (vx, vy, theta) without any discretization or
filtering. Images are resized from the native 480x640 to IMAGE_SIZE x IMAGE_SIZE
(uint8 RGB) for practical storage.

Output files (saved to --output-dir):
  dataset_strafe.npz          <- TagAggDann/lekiwi_strafe
  dataset_rotation.npz        <- TagAggDann/lekiwi_rotation
  dataset_strafe_rotation.npz <- TagAggDann/lekiwi_combined  (HF source with both strafe+rotation)
  dataset_straight.npz        <- TagAggDann/lekiwi_straight
  dataset_CW.npz              <- TagAggDann/lekiwi_CW
  dataset_full.npz            <- all 5 sources concatenated

Each .npz contains arrays:
  img    (N, IMAGE_SIZE, IMAGE_SIZE, 3)  uint8   RGB frames
  vx     (N,)                            float32 linear x velocity (m/s)
  vy     (N,)                            float32 lateral y velocity (m/s)
  vtheta (N,)                            float32 angular velocity (deg/s)

dataset_full.npz also contains:
  source (N,)                            uint8   source dataset index (see DATASETS order)

Usage:
    python dataset/construct_dataset.py
    python dataset/construct_dataset.py --skip-download
    python dataset/construct_dataset.py --output-dir ./data --raw-dir ./data/raw
"""

import argparse
import glob
import os

import cv2
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

IMAGE_SIZE = 224  # resize to this for ViT compatibility (native res is 480x640)

DATASETS: dict[str, str] = {
    "strafe":          "TagAggDann/lekiwi_strafe",
    "rotation":        "TagAggDann/lekiwi_rotation",
    "strafe_rotation": "TagAggDann/lekiwi_combined",
    "straight":        "TagAggDann/lekiwi_straight",
    "CW":              "TagAggDann/lekiwi_CW",
}


def load_parquet_labels(raw_dir: str) -> pd.DataFrame:
    """Read all episode parquet files and return a DataFrame sorted by global index."""
    pattern = os.path.join(raw_dir, "data", "chunk-*", "file-*.parquet")
    parquet_files = sorted(glob.glob(pattern))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found matching {pattern}. "
            "Did the download complete successfully?"
        )

    dfs = []
    for path in parquet_files:
        df = pd.read_parquet(path, columns=["index", "episode_index", "frame_index", "action"])
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("index").reset_index(drop=True)
    return df


def decode_video(video_path: str) -> list[np.ndarray]:
    """Decode all frames from an MP4 as uint8 RGB arrays resized to IMAGE_SIZE x IMAGE_SIZE."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        frames.append(frame)

    cap.release()
    return frames


def build_arrays(raw_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build raw arrays from one downloaded dataset directory.

    Returns:
        img    (N, IMAGE_SIZE, IMAGE_SIZE, 3) uint8
        vx     (N,) float32
        vy     (N,) float32
        vtheta (N,) float32
    """
    df = load_parquet_labels(raw_dir)
    N = len(df)

    video_pattern = os.path.join(
        raw_dir, "videos", "observation.images.front", "chunk-*", "file-*.mp4"
    )
    video_files = sorted(glob.glob(video_pattern))
    if not video_files:
        raise FileNotFoundError(
            f"No video files found matching {video_pattern}. "
            "Did the download complete successfully?"
        )

    all_frames: list[np.ndarray] = []
    for vf in video_files:
        print(f"  decoding {vf} ...")
        all_frames.extend(decode_video(vf))

    n_frames = len(all_frames)
    print(f"  total frames decoded: {n_frames}  |  parquet rows: {N}")

    if n_frames != N:
        print(f"  [warn] frame/row count mismatch — trimming to minimum ({min(n_frames, N)})")
        n = min(n_frames, N)
        all_frames = all_frames[:n]
        df = df.iloc[:n]

    frame_indices = df["index"].to_numpy()
    max_idx = int(frame_indices.max())
    if max_idx >= len(all_frames):
        raise ValueError(
            f"Parquet references frame index {max_idx} but only {len(all_frames)} frames decoded."
        )

    img = np.stack([all_frames[i] for i in frame_indices], axis=0)  # (N, H, W, 3)

    # action column stores [x.vel, y.vel, theta.vel]
    actions = np.array(df["action"].tolist(), dtype=np.float32)  # (N, 3)
    vx     = actions[:, 0]
    vy     = actions[:, 1]
    vtheta = actions[:, 2]

    for ep_idx, ep_df in df.groupby("episode_index"):
        print(f"  episode {ep_idx:3d}: {len(ep_df)} frames")

    assert img.shape[0] == vx.shape[0], "Frame/label count mismatch"
    return img, vx, vy, vtheta


def print_stats(name: str, img: np.ndarray, vx: np.ndarray, vy: np.ndarray, vtheta: np.ndarray) -> None:
    print(f"  img:    {img.shape}  {img.dtype}")
    print(f"  vx:     {vx.shape}  range [{vx.min():.4f}, {vx.max():.4f}]")
    print(f"  vy:     {vy.shape}  range [{vy.min():.4f}, {vy.max():.4f}]")
    print(f"  vtheta: {vtheta.shape}  range [{vtheta.min():.4f}, {vtheta.max():.4f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct lekiwi mechint datasets")
    parser.add_argument("--output-dir", default="./data",
                        help="Directory to save .npz files (default: ./data)")
    parser.add_argument("--raw-dir", default="./data/raw",
                        help="Base directory for raw HuggingFace downloads (default: ./data/raw)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading and use existing files in --raw-dir")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_imgs:   list[np.ndarray] = []
    all_vx:     list[np.ndarray] = []
    all_vy:     list[np.ndarray] = []
    all_vtheta: list[np.ndarray] = []
    all_source: list[np.ndarray] = []

    for src_id, (name, repo_id) in enumerate(DATASETS.items()):
        raw_dir  = os.path.join(args.raw_dir, name)
        out_path = os.path.join(args.output_dir, f"dataset_{name}.npz")

        print(f"\n{'='*60}")
        print(f"  {name}  ({repo_id})")
        print(f"{'='*60}")

        if not args.skip_download:
            print(f"Downloading to {raw_dir} ...")
            snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=raw_dir)
            print("Download complete.")
        else:
            print(f"Using existing files in {raw_dir}")

        print("\nProcessing ...")
        img, vx, vy, vtheta = build_arrays(raw_dir)
        print("\nAssembled:")
        print_stats(name, img, vx, vy, vtheta)

        print(f"\nSaving to {out_path} ...")
        np.savez_compressed(out_path, img=img, vx=vx, vy=vy, vtheta=vtheta)
        print(f"Saved ({os.path.getsize(out_path) / 1e6:.1f} MB).")

        all_imgs.append(img)
        all_vx.append(vx)
        all_vy.append(vy)
        all_vtheta.append(vtheta)
        all_source.append(np.full(len(vx), src_id, dtype=np.uint8))

    # ── Full file (all 5 sources) ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Building dataset_full.npz (all 5 sources)")
    print(f"{'='*60}")

    full_img    = np.concatenate(all_imgs,   axis=0)
    full_vx     = np.concatenate(all_vx,     axis=0)
    full_vy     = np.concatenate(all_vy,     axis=0)
    full_vtheta = np.concatenate(all_vtheta, axis=0)
    full_source = np.concatenate(all_source, axis=0)

    print("\nFull:")
    print_stats("full", full_img, full_vx, full_vy, full_vtheta)
    src_names = list(DATASETS.keys())
    print("  source distribution:")
    for i, sname in enumerate(src_names):
        count = int((full_source == i).sum())
        print(f"    {i} ({sname}): {count}")

    full_path = os.path.join(args.output_dir, "dataset_full.npz")
    print(f"\nSaving to {full_path} ...")
    np.savez_compressed(full_path,
                        img=full_img, vx=full_vx,
                        vy=full_vy,   vtheta=full_vtheta,
                        source=full_source)
    print(f"Saved ({os.path.getsize(full_path) / 1e6:.1f} MB).")


if __name__ == "__main__":
    main()
