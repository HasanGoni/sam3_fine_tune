"""Hydra evaluation CLI:  python -m sam3_fine_tune.sambaunet.evaluate checkpoint=... data.root=...

Predicts at the training resolution, upsamples the logits to each image's native size and scores against the original
COCO masks. Writes ``multiclass_foreground_iou_summary.json`` in the SAM3 pipeline's layout plus a ``meta`` block
(params, latency) for the SAM3-vs-SAMBA-UNet comparison.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image

from .checkpoint import load_model
from .data import CocoSemanticDataset
from .metrics import SUMMARY_FILENAME, CategoryScores
from .train import resolve_device

log = logging.getLogger(__name__)
PALETTE = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180]], np.uint8)


def overlay(image_path: Path, pred: np.ndarray, out_path: Path) -> None:
    img = np.asarray(Image.open(image_path).convert("RGB")).astype(np.float32)
    for k, m in enumerate(pred.astype(bool)):
        img[m] = 0.5 * img[m] + 0.5 * PALETTE[k % len(PALETTE)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img.astype(np.uint8)).save(out_path)


@torch.no_grad()
def evaluate(cfg: DictConfig, out_dir: Path) -> dict:
    device = resolve_device(cfg.device)
    model, ck = load_model(cfg.checkpoint, device)
    size = ck["cfg"]["data"]["size"]
    ds = CocoSemanticDataset(Path(cfg.data.root) / cfg.data.split, size, False, ck["category_names"], cfg.data.ann_file)
    scores, times = CategoryScores(ds.category_names), []
    for i in range(len(ds)):
        item = ds[i]
        gt = ds.original_mask(i)
        x = item["image"].unsqueeze(0).to(device)
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(x)
        if device.type == "cuda": torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        logits = F.interpolate(logits, gt.shape[-2:], mode="bilinear", align_corners=False)
        pred = (torch.sigmoid(logits)[0] > cfg.threshold).cpu().numpy()
        scores.update(pred, gt, ds.image_path(i).name)
        if cfg.save_overlays:
            overlay(ds.image_path(i), pred, out_dir / "overlays" / f"{ds.image_path(i).stem}.png")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result = {
        "summary": scores.summary(),
        "per_category_details": scores.details(),
        "meta": {"model": "sambaunet", "checkpoint": str(cfg.checkpoint), "trained_epoch": ck.get("epoch"),
                 "trainable_params": n_train, "total_params": sum(p.numel() for p in model.parameters()),
                 "input_size": size, "threshold": cfg.threshold, "device": str(device),
                 "latency_ms_per_image": round(1000 * float(np.median(times)), 2) if times else None},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SUMMARY_FILENAME).write_text(json.dumps(result, indent=2))
    for name, c in result["summary"]["categories"].items():
        log.info("%-20s overall FG IoU %s | P %s | R %s", name, c.get("overall_foreground_iou"), c.get("pixel_precision"), c.get("pixel_recall"))
    log.info("ALL categories FG IoU %s -> %s", result["summary"]["overall"]["foreground_iou"], out_dir / SUMMARY_FILENAME)
    return result


@hydra.main(config_path="conf", config_name="eval", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from hydra.core.hydra_config import HydraConfig
    evaluate(cfg, Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()
