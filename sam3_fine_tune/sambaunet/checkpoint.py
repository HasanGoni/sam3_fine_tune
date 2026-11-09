"""Model construction from a Hydra config + trainable-only checkpoints.

Only parameters with ``requires_grad`` are stored (the frozen SAM ViT-B is re-downloaded from the Hub on load, like the
SAM3 flow re-loads its base checkpoint), which keeps checkpoints ~4x smaller.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from .model import SAMBAUNet, tiny_sam_config


def build_model(model_cfg: DictConfig, n_classes: int) -> SAMBAUNet:
    return SAMBAUNet(
        n_classes=n_classes,
        dims=tuple(model_cfg.dims),
        rank=model_cfg.rank,
        sam_name=model_cfg.sam_name,
        sam_config=tiny_sam_config() if model_cfg.sam_tiny else None,
        grad_checkpoint=model_cfg.get("grad_checkpoint", False),
    )


def trainable_state(model: torch.nn.Module) -> dict:
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names}


def _plain(obj):
    """Round-trip through JSON so the payload holds only builtin types (numpy scalars would break weights_only=True)."""
    return json.loads(
        json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    )


def save_checkpoint(
    path: Path, model, cfg: DictConfig, category_names: list[str], val=None, **extra
) -> None:
    """Checkpoints contain only tensors and builtin containers, so they load with ``weights_only=True`` (no pickle code execution)."""
    torch.save(
        {
            "model": trainable_state(model),
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "category_names": list(category_names),
            "val": _plain(val),
            **extra,
        },
        path,
    )


def load_model(path, device="cpu"):
    """Rebuild the model recorded in the checkpoint and load its trainable weights. Returns (model, ckpt)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = OmegaConf.create(ckpt["cfg"])
    model = build_model(cfg.model, len(ckpt["category_names"]))
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    bad = [
        k for k in missing if k not in frozen and k not in dict(model.named_buffers())
    ]
    if bad or unexpected:
        raise RuntimeError(
            f"checkpoint/model mismatch: missing={bad[:5]} unexpected={list(unexpected)[:5]}"
        )
    return model.to(device).eval(), ckpt
