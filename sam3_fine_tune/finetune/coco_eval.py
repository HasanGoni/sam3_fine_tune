"""Score a SAM3 checkpoint on a COCO test split: prompt = ``category_name``, ground truth = rasterised COCO masks.

Writes ``multiclass_foreground_iou_summary.json`` in the layout of ``evaluation/multi_eval_pipe.py`` (shared with the
SAMBA-UNet evaluation through ``CategoryScores``), so the two baselines are comparable key-by-key.
The model call is an injectable ``predict(image_path, prompt) -> bool mask (H, W)`` so the scoring is testable offline.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np

from sam3_fine_tune.sambaunet.data import CocoSemanticDataset
from sam3_fine_tune.sambaunet.metrics import CategoryScores, SUMMARY_FILENAME

Predict = Callable[[Path, str], np.ndarray]


def evaluate_coco(
    split_dir,
    ann_file: str,
    predict: Predict,
    out_dir,
    max_images: int | None = None,
    meta: dict | None = None,
) -> dict:
    ds = CocoSemanticDataset(split_dir, ann_file=ann_file)
    n = len(ds) if max_images is None else min(max_images, len(ds))
    scores, times = CategoryScores(ds.category_names), []
    for i in range(n):
        gt = ds.original_mask(i)
        t0 = time.perf_counter()
        pred = np.stack(
            [
                np.asarray(predict(ds.image_path(i), name)).astype(bool)
                for name in ds.category_names
            ]
        )
        times.append(time.perf_counter() - t0)
        scores.update(pred, gt, ds.image_path(i).name)
    return write_result(
        scores,
        num_images=n,
        latency=float(np.median(times)) if times else None,
        meta=meta,
        out_dir=out_dir,
    )


def write_result(
    scores: CategoryScores,
    *,
    num_images: int,
    latency: float | None,
    meta: dict | None,
    out_dir,
) -> dict:
    """Build and write ``multiclass_foreground_iou_summary.json`` (shared by ``sam3-finetune`` and ``sam3-infer``)."""
    result = {
        "summary": scores.summary(),
        "per_category_details": scores.details(),
        "meta": {
            "model": "sam3",
            "num_images": num_images,
            "categories": scores.names,
            "latency_s_per_image": None if latency is None else round(latency, 3),
            **(meta or {}),
        },
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SUMMARY_FILENAME).write_text(json.dumps(result, indent=2))
    return result


def make_sam3_predictor(
    checkpoint, bpe_path, device: str, confidence_threshold: float = 0.5
) -> Predict:
    """Load the fine-tuned weights (same recipe as ``multi_eval_pipe.load_model``) and return a ``Predict``."""
    import torch
    from PIL import Image

    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        device=device,
        eval_mode=True,
        checkpoint_path=None,
        load_from_HF=False,
    )
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(weights["model"])
    model.eval()
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
    cache: dict = {}

    @torch.no_grad()
    def predict(image_path: Path, prompt: str) -> np.ndarray:
        if (
            cache.get("path") != image_path
        ):  # the image backbone runs once per image, not once per category
            image = Image.open(image_path).convert("RGB")
            cache.update(
                path=image_path,
                size=(image.height, image.width),
                state=processor.set_image(image),
            )
        state = cache["state"]
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt=prompt)
        out = np.zeros(cache["size"], bool)
        for m in state.get("masks", []):
            m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
            out |= m.squeeze() > 0.5
        return out

    return predict
