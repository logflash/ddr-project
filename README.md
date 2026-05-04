# Mechanistic Interpretability of a ViT Behaviour-Cloning Policy

We train a ViT-based behavior-cloning (BC) policy on the LeKiwi omnidirectional robot, then apply mechanistic interpretability techniques — linear probes, contrastive steering, activation patching, and sparse autoencoders — to understand how the network encodes and produces velocity commands.

## Robot & action space

LeKiwi omnidirectional robot, controlled via velocity commands sent over ZMQ:

| Axis | Range | Convention |
|------|-------|------------|
| `x.vel` (vx) | [0, 0.3] m/s | forward |
| `y.vel` (vy) | [−0.3, 0.3] m/s | **positive = left, negative = right** |
| `theta.vel` (vtheta) | [−45, 45] deg/s | positive = CCW, negative = CW |

## Environment

```bash
source ../lerobot/.venv/bin/activate   # timm, torch, sklearn, tqdm, av all installed here
```

Copy `.env.example` to `.env` and set `REMOTE_IP` to the robot's IP address.

## Repository structure

```
models/
  vit_policy.py        # ViTContinuous, ViTCategorical, collect_cls_streams, make_steer_hook
  vit_train.py         # training loop, LekiwiDataset, loss/metric functions

dataset/
  construct_dataset.py # download HuggingFace datasets, decode AV1 video, build .npz
  prepare_splits.py    # stratified train/val/steer_probe/steer_eval index splits

steering/
  steer.py             # contrastive steering directions, linear probe R², dose-response
  activation_patch.py  # causal localization via mean CLS patching per layer
  probe_epochs.py      # track representation evolution across training epochs
  results/             # saved plots and text logs

sae/
  train_sae.py         # TopK sparse autoencoder on layer-9 CLS activations

checkpoints/           # model weights and steering vectors (not in git for .pth > threshold)
data/                  # dataset_full.npz and splits/ (gitignored)

run_vit_policy.py      # deploy policy on real robot
debug_policy.py        # offline sanity-check: preprocessing, inference, accuracy
```

## Pipeline

### 1. Build dataset

```bash
python dataset/construct_dataset.py          # downloads ~5 HuggingFace datasets, builds data/dataset_full.npz
python dataset/prepare_splits.py             # writes data/splits/{train,val,steer_probe,steer_eval}_idx.npy
```

### 2. Train

```bash
cd models
python vit_train.py --mode continuous --backbone vit_tiny_patch16_224
# checkpoint saved to checkpoints/vit_continuous_tiny_best.pth
```

### 3. Debug / validate

```bash
python debug_policy.py --model checkpoints/vit_continuous_tiny_best.pth
# checks preprocessing consistency, inference accuracy, output variance
```

### 4. Interpretability

```bash
# Linear probes + contrastive steering directions at every layer
python steering/steer.py --ckpt checkpoints/vit_continuous_tiny_best.pth \
    --save-vecs checkpoints/steering_vecs_layer9.pth

# Causal localization via activation patching
python steering/activation_patch.py --ckpt checkpoints/vit_continuous_tiny_best.pth

# SAE on layer-9 CLS activations
python sae/train_sae.py --ckpt checkpoints/vit_continuous_tiny_best.pth

# Representation evolution across training epochs (requires --save-every checkpoints)
python steering/probe_epochs.py --mode continuous --size tiny
```

### 5. Deploy on robot

```bash
# Basic deployment
python run_vit_policy.py --model checkpoints/vit_continuous_tiny_best.pth

# With strafe intervention
python run_vit_policy.py --model checkpoints/vit_continuous_tiny_best.pth \
    --steer-vy -8      # -8 strafe right; +8 strafe left

# With angular steering intervention
python run_vit_policy.py --model checkpoints/vit_continuous_tiny_best.pth \
    --steer-vtheta -8  # -8 steers CW; +8 steers CCW

# Drive backwards
python run_vit_policy.py --model checkpoints/vit_continuous_tiny_best.pth --reversal
```

Controls: `SPACE` pause/resume · `ESC` emergency stop · `Ctrl+C` quit

## Key checkpoints

| File | Description |
|------|-------------|
| `vit_continuous_tiny_best.pth` | **Use this for deployment.** ViTContinuous, vit_tiny_patch16_224, pretrained ImageNet backbone, epoch 21, val_loss=0.0209. Generalises to new environments. |
| `vit_continuous_tiny_best_steer_vecs_layer9.pth` | Unit-norm contrastive steering vectors for vy and vtheta at layer 9, used by `run_vit_policy.py`. |

## Key findings

- **Action steerability emerges at block 10/12.** Activation patching shows a sharp causal jump at layer 9 (0-indexed) — the 10th of 12 blocks — for all three axes. Layers 0–8 have negligible patching effect; layers 9–11 are causally decisive.
- **Linear decodability rises gradually** (R² increases from ~0.17 at layer 0 to ~0.89 at layer 11), distinct from causal steerability which concentrates at the layer-9 transition.
- **Contrastive steering works at layer 9.** Adding `alpha × steering_vec` to the CLS token achieves ~100% conditional success for vy at |α|=7.5 on held-out data.
