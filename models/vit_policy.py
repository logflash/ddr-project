"""
vit_policy.py

ViT-based behavior-cloning policy for the LeKiwi robot.

Two variants share the same vit_small_patch16_224 backbone (pretrained
ImageNet) and expose the CLS-token residual stream for SAE / steering work.

  ViTCategorical  –  8 logits: 2 for vx  +  3 for vy  +  3 for vtheta
                     separate cross-entropy loss per axis
  ViTContinuous   –  3 continuous outputs: vx, vy, vtheta (normalised to [-1,1])

Residual-stream utilities
  collect_cls_streams(model, x)     ->  (B, L, D)   CLS at every block output
  steer_hook(vec, alpha)            ->  forward hook that adds alpha*vec to CLS
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm
from typing import Callable

# ── Action constants ──────────────────────────────────────────────────────────
VX_MAX     = 0.3    # m/s
VY_MAX     = 0.3    # m/s
VTHETA_MAX = 45.0   # deg/s

# Class-index → continuous value lookup tables
VX_VALUES     = torch.tensor([0.0,        VX_MAX                ])  # 2 classes
VY_VALUES     = torch.tensor([-VY_MAX,    0.0,       VY_MAX     ])  # 3 classes
VTHETA_VALUES = torch.tensor([-VTHETA_MAX, 0.0,      VTHETA_MAX ])  # 3 classes

# ImageNet normalization (applied in the DataLoader, not here)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DEFAULT_BACKBONE = "vit_small_patch16_224"

# Embed dim per backbone (used by external code to size SAE inputs etc.)
EMBED_DIMS = {
    "vit_tiny_patch16_224":  192,
    "vit_small_patch16_224": 384,
    "vit_base_patch16_224":  768,
}
NUM_LAYERS = 12


# ── Shared backbone ───────────────────────────────────────────────────────────

class _ViTBase(nn.Module):
    """ViT backbone with num_classes=0 (raw CLS output)."""

    def __init__(self, pretrained: bool = True, backbone: str = DEFAULT_BACKBONE):
        super().__init__()
        self.backbone_name = backbone
        self.embed_dim = EMBED_DIMS.get(backbone, 384)
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)

    def _cls_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Run full backbone and return the final CLS embedding (B, D)."""
        return self.backbone(x)


# ── Categorical model ─────────────────────────────────────────────────────────

class ViTCategorical(_ViTBase):
    """
    Predicts three independent softmax distributions:
      vx    (2 classes): [0, +VX_MAX]
      vy    (3 classes): [-VY_MAX, 0, +VY_MAX]
      vtheta(3 classes): [-VTHETA_MAX, 0, +VTHETA_MAX]
    """

    def __init__(self, pretrained: bool = True, backbone: str = DEFAULT_BACKBONE):
        super().__init__(pretrained, backbone)
        self.head_vx     = nn.Linear(self.embed_dim, 2)
        self.head_vy     = nn.Linear(self.embed_dim, 3)
        self.head_vtheta = nn.Linear(self.embed_dim, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, 224, 224) normalised float32
        Returns:
            {'vx': (B,2), 'vy': (B,3), 'vtheta': (B,3)}  – raw logits
        """
        cls = self._cls_embedding(x)
        return {
            "vx":     self.head_vx(cls),
            "vy":     self.head_vy(cls),
            "vtheta": self.head_vtheta(cls),
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Argmax → continuous values via lookup tables."""
        logits = self.forward(x)
        dev = x.device
        return {
            "vx":     VX_VALUES    .to(dev)[logits["vx"]    .argmax(dim=-1)],
            "vy":     VY_VALUES    .to(dev)[logits["vy"]    .argmax(dim=-1)],
            "vtheta": VTHETA_VALUES.to(dev)[logits["vtheta"].argmax(dim=-1)],
        }


# ── Continuous model ──────────────────────────────────────────────────────────

class ViTContinuous(_ViTBase):
    """
    Predicts three normalised continuous values stacked as (B, 3):
      [:, 0]  vx     normalised by VX_MAX     → range [0,  1]
      [:, 1]  vy     normalised by VY_MAX     → range [-1, 1]
      [:, 2]  vtheta normalised by VTHETA_MAX → range [-1, 1]

    Multiply by the respective MAX to recover physical units.
    """

    def __init__(self, pretrained: bool = True, backbone: str = DEFAULT_BACKBONE):
        super().__init__(pretrained, backbone)
        self.head = nn.Linear(self.embed_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) normalised float32
        Returns:
            (B, 3) normalised predictions [vx_n, vy_n, vtheta_n]
        """
        return self.head(self._cls_embedding(x))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns predictions in physical units."""
        out = self.forward(x)
        return {
            "vx":     out[:, 0] * VX_MAX,
            "vy":     out[:, 1] * VY_MAX,
            "vtheta": out[:, 2] * VTHETA_MAX,
        }


# ── Residual-stream utilities ─────────────────────────────────────────────────

@torch.no_grad()
def collect_cls_streams(
    model: _ViTBase,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Extract the CLS-token representation at the output of every transformer
    block (i.e. the residual stream seen by the action head).

    Args:
        model: ViTCategorical or ViTContinuous
        x:     (B, 3, 224, 224) normalised input batch

    Returns:
        streams: (B, NUM_LAYERS, EMBED_DIM) on CPU
                 streams[:, l, :] is the CLS vector after block l
    """
    captured: list[torch.Tensor] = []
    hooks: list[torch.utils.hooks.RemovableHook] = []

    for block in model.backbone.blocks:
        def _hook(module, inp, out, _buf=captured):
            _buf.append(out[:, 0, :].detach().cpu())  # CLS is token 0
        hooks.append(block.register_forward_hook(_hook))

    try:
        model(x)
    finally:
        for h in hooks:
            h.remove()

    return torch.stack(captured, dim=1)  # (B, L, D)


def make_steer_hook(
    steering_vec: torch.Tensor,
    alpha: float = 1.0,
) -> Callable:
    """
    Returns a forward hook that adds alpha * steering_vec to the CLS token.
    Attach it to model.backbone.blocks[layer_idx] before a forward pass,
    then remove it afterward.

    Example:
        hook = model.backbone.blocks[8].register_forward_hook(
                   make_steer_hook(vec, alpha=3.0))
        with torch.no_grad():
            logits = model(x)
        hook.remove()
    """
    def _hook(module, inp, out):
        out = out.clone()
        out[:, 0, :] = out[:, 0, :] + alpha * steering_vec.to(out.device)
        return out
    return _hook
