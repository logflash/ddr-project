#!/usr/bin/env python
"""
run_vit_policy.py

Deploy a trained ViTContinuous policy on the LeKiwi robot.

The policy takes a 224x224 RGB image, runs through a ViT backbone,
and predicts continuous velocity commands (vx, vy, vtheta) sent at 30 FPS.

Usage:
    python run_vit_policy.py --model checkpoints/vit_continuous_tiny_best.pth --remote-ip <ip>
    REMOTE_IP=192.168.1.14 python run_vit_policy.py --model checkpoints/vit_continuous_tiny_best.pth

Controls:
    SPACE   pause / resume
    ESC     emergency stop
    Ctrl+C  quit
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
# Make models/ importable regardless of working directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from models.vit_policy import IMAGENET_MEAN, IMAGENET_STD, ViTContinuous, make_steer_hook

from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# ── Constants ─────────────────────────────────────────────────────────────────
FPS = 30
CAMERA = "front"

# Load .env from the script's directory if present (no extra dependency needed)
_ENV_FILE = os.path.join(_SCRIPT_DIR, ".env")
if os.path.exists(_ENV_FILE):
    with open(_ENV_FILE) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

REMOTE_IP = os.environ.get("REMOTE_IP")
STOP_ACTION = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

# Pre-compute normalization tensors (shape 3,1,1 for broadcasting over CHW)
_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD  = torch.tensor(IMAGENET_STD).view(3, 1, 1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def preprocess(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    """BGR uint8 HWC → (1, 3, 224, 224) float32 ImageNet-normalised tensor."""
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb  = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    t    = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
    t    = (t - _MEAN) / _STD
    return t.unsqueeze(0).to(device)


_STEER_LAYER = 9
_DEFAULT_STEER_VECS = os.path.join(
    _SCRIPT_DIR, "checkpoints", "vit_continuous_tiny_best_steer_vecs_layer9.pth"
)


def load_steer_vecs(pth_path: str, device: torch.device) -> dict[str, torch.Tensor]:
    """Load all steering vectors from the .pth bundle; return dict keyed by axis name."""
    bundle = torch.load(pth_path, map_location="cpu", weights_only=False)
    return {
        k: torch.from_numpy(v).to(device)
        for k, v in bundle.items()
        if isinstance(v, np.ndarray)
    }


def load_model(ckpt_path: str, device: torch.device) -> ViTContinuous:
    ckpt     = torch.load(ckpt_path, map_location=device)
    backbone = ckpt.get("backbone", "vit_tiny_patch16_224")
    model    = ViTContinuous(pretrained=False, backbone=backbone)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    print(
        f"Loaded  : {backbone}  (epoch {ckpt.get('epoch', '?')}, "
        f"val_loss={ckpt.get('val_loss', float('nan')):.4f})"
    )
    if ckpt.get("val_metrics"):
        m = ckpt["val_metrics"]
        print(
            f"Val metrics: "
            f"acc_vx={m.get('acc_vx', 0):.3f}  "
            f"acc_vy={m.get('acc_vy', 0):.3f}  "
            f"acc_vtheta={m.get('acc_vtheta', 0):.3f}"
        )
    return model


def init_stop_listener():
    stopped = {"value": False}
    paused  = {"value": False}
    try:
        from pynput import keyboard

        def on_press(key):
            if key == keyboard.Key.esc:
                print("\n[ESC] Emergency stop!")
                stopped["value"] = True
            elif key == keyboard.Key.space:
                paused["value"] = not paused["value"]
                state = "PAUSED" if paused["value"] else "RESUMED"
                print(f"\n[SPACE] {state}")

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        return listener, stopped, paused
    except Exception:
        print("Warning: pynput not available — use Ctrl+C to stop.")
        return None, stopped, paused


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run ViT continuous policy on LeKiwi")
    parser.add_argument("--model",     required=True, help="Path to .pth checkpoint")
    parser.add_argument("--camera",    default=CAMERA, help="Camera key in observation (default: front)")
    parser.add_argument("--remote-ip", default=REMOTE_IP, help="Robot IP (or set REMOTE_IP env var)")
    parser.add_argument("--fps",       type=int, default=FPS, help=f"Control loop frequency (default: {FPS})")
    parser.add_argument("--cpu",       action="store_true", help="Force CPU inference")
    parser.add_argument("--reversal",  action="store_true", help="Negate vx: robot drives backward instead of forward.")
    parser.add_argument(
        "--steer-vy", type=float, default=None, metavar="ALPHA",
        help=(
            "Strafe intervention at layer 9. "
            "Negative alpha → strafe right (−vy); positive alpha → strafe left (+vy). "
            "(LeKiwi convention: positive y.vel = left, negative y.vel = right.) "
            "Suggested interventions: --steer-vy -8 to strafe right, "
            "--steer-vy +8 to strafe left. "
            "Scale up to ±12 for stronger intervention."
        ),
    )
    parser.add_argument(
        "--steer-vtheta", type=float, default=None, metavar="ALPHA",
        help=(
            "Steer angular velocity at layer 9. "
            "Negative alpha → rotate CW (−vtheta); positive alpha → rotate CCW (+vtheta). "
            "Suggested interventions: --steer-vtheta -8 if robot spins CW, "
            "--steer-vtheta +8 if robot spins CCW. "
            "Scale up to ±12 for stronger intervention."
        ),
    )
    parser.add_argument(
        "--steer-vecs", default=_DEFAULT_STEER_VECS,
        help="Path to steering-vectors .pth bundle (default: checkpoints/vit_continuous_tiny_best_steer_vecs_layer9.pth)",
    )
    args = parser.parse_args()

    if not args.remote_ip:
        sys.exit("Error: set REMOTE_IP env var or pass --remote-ip <ip>")
    if not os.path.exists(args.model):
        sys.exit(f"Error: checkpoint not found: {args.model}")

    device = (
        torch.device("cpu") if args.cpu
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Device  : {device}")

    model = load_model(args.model, device)

    # Optional steering hooks (vy and/or vtheta)
    steer_hooks = []
    if args.steer_vy is not None or args.steer_vtheta is not None:
        if not os.path.exists(args.steer_vecs):
            sys.exit(f"Error: steering vectors not found: {args.steer_vecs}")
        vecs = load_steer_vecs(args.steer_vecs, device)
        if args.steer_vy is not None:
            steer_hooks.append(
                model.backbone.blocks[_STEER_LAYER].register_forward_hook(
                    make_steer_hook(vecs["vy"], alpha=args.steer_vy)
                )
            )
            direction = "left (+vy)" if args.steer_vy >= 0 else "right (−vy)"
            print(f"Steering : vy      alpha={args.steer_vy:+.1f}  layer={_STEER_LAYER}  → {direction}")
        if args.steer_vtheta is not None:
            steer_hooks.append(
                model.backbone.blocks[_STEER_LAYER].register_forward_hook(
                    make_steer_hook(vecs["vtheta"], alpha=args.steer_vtheta)
                )
            )
            direction = "CCW (+vtheta)" if args.steer_vtheta >= 0 else "CW (−vtheta)"
            print(f"Steering : vtheta  alpha={args.steer_vtheta:+.1f}  layer={_STEER_LAYER}  → {direction}")

    robot_config = LeKiwiClientConfig(remote_ip=args.remote_ip, id="lekiwi")
    robot        = LeKiwiClient(robot_config)
    robot.connect()
    if not robot.is_connected:
        sys.exit("Error: robot not connected.")
    print("Robot   : connected")

    init_rerun(session_name="lekiwi_vit_policy")
    listener, stopped, paused = init_stop_listener()

    print(f"Running policy at {args.fps} FPS")
    print("Controls: SPACE=pause/resume  ESC=emergency stop  Ctrl+C=quit")
    print("=" * 60)

    try:
        while not stopped["value"]:
            t0 = time.perf_counter()

            if paused["value"]:
                robot.send_action(STOP_ACTION)
                print("\r[PAUSED] Press SPACE to resume…          ", end="", flush=True)
                precise_sleep(1.0 / args.fps)
                continue

            obs   = robot.get_observation()
            frame = obs.get(args.camera)

            if frame is None:
                print("Warning: no camera frame received, skipping.")
                precise_sleep(1.0 / args.fps)
                continue

            img  = preprocess(frame, device)
            pred = model.predict(img)   # physical units: m/s, m/s, deg/s

            action = {
                "x.vel":     float(-pred["vx"] if args.reversal else pred["vx"]),
                "y.vel":     float(pred["vy"]),
                "theta.vel": float(pred["vtheta"]),
            }
            robot.send_action(action)

            tags = []
            if args.reversal:                tags.append("REVERSAL")
            if args.steer_vy     is not None: tags.append(f"steer_vy α={args.steer_vy:+.1f}")
            if args.steer_vtheta is not None: tags.append(f"steer_vθ α={args.steer_vtheta:+.1f}")
            steer_tag = f"  [{', '.join(tags)}]" if tags else ""
            print(
                f"\rvx={action['x.vel']:+.3f} m/s  "
                f"vy={action['y.vel']:+.3f} m/s  "
                f"vθ={action['theta.vel']:+5.1f} °/s{steer_tag}   ",
                end="", flush=True,
            )

            log_rerun_data(observation=obs, action=action)

            dt_s = time.perf_counter() - t0
            precise_sleep(max(1.0 / args.fps - dt_s, 0.0))

    except KeyboardInterrupt:
        print("\nCtrl+C received.")
        robot.send_action(STOP_ACTION)
    finally:
        for h in steer_hooks:
            h.remove()
        robot.send_action(STOP_ACTION)
        robot.disconnect()
        if listener:
            listener.stop()
        print("Done.")


if __name__ == "__main__":
    main()
