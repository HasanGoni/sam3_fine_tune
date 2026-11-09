"""Run SAM3 (base or fine-tuned) with one text prompt on a folder of images and write masks, overlays and a summary.

The model call is an injectable ``predict(image_path, prompt) -> (masks, scores)`` (``masks``: list of bool ``(H, W)``
arrays, one per detected instance) so the output logic is testable without weights; ``make_predictor`` is the real one.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
DEVICES = ("auto", "cpu", "cuda")
SUMMARY_NAME = "summary.json"
Predict = Callable[[Path, str], "tuple[Sequence[np.ndarray], Sequence[float]]"]


def resolve_device(device: str, cuda_available: bool) -> str:
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {device!r}")
    if device == "cuda" and not cuda_available:
        raise ValueError("device=cuda requested but no CUDA GPU is visible; use device=cpu or device=auto")
    return ("cuda" if cuda_available else "cpu") if device == "auto" else device


def list_images(folder) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"images folder not found: {folder}")
    found = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not found:
        raise ValueError(f"no images ({', '.join(sorted(IMAGE_EXTENSIONS))}) in {folder}")
    stems = [p.stem for p in found]
    dup = sorted({s for s in stems if stems.count(s) > 1})
    if dup:
        raise ValueError(f"images share a file stem and would overwrite each other's outputs: {dup}")
    return found


def validate_inputs(*, images, prompt, sam3_checkpoint, bpe_path, checkpoint, threshold) -> list[Path]:
    """Fail early (before any model is loaded) with a message that says what to fix. Returns the image list."""
    if not str(prompt).strip():
        raise ValueError("prompt must be a non-empty text prompt")
    if not 0 < float(threshold) <= 1:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    for label, path in (("sam3_checkpoint (base sam3.pt)", sam3_checkpoint), ("bpe_path (tokenizer vocabulary)", bpe_path)):
        if not Path(path).is_file():
            raise ValueError(f"{label} not found: {path}")
    if checkpoint is not None and not Path(checkpoint).is_file():
        raise ValueError(f"checkpoint (fine-tuned model) not found: {checkpoint}")
    return list_images(images)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.5) -> np.ndarray:
    out = image.astype(np.float32).copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.asarray(color, np.float32)
    return out.round().astype(np.uint8)


def run_inference(image_paths: Sequence[Path], prompt: str, out, predict: Predict, meta: dict | None = None) -> dict:
    """Write ``out/masks/<stem>.png`` (0/255, union of instances), ``out/overlays/<stem>.png`` and ``out/summary.json``."""
    out = Path(out)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    rows = []
    for path in image_paths:
        path = Path(path)
        rgb = np.asarray(Image.open(path).convert("RGB"))
        masks, scores = predict(path, prompt)
        union = np.zeros(rgb.shape[:2], bool)
        for m in masks:
            m = np.asarray(m).astype(bool)
            if m.shape != union.shape:
                raise ValueError(f"{path.name}: predicted mask {m.shape} does not match image {union.shape}")
            union |= m
        Image.fromarray(union.astype(np.uint8) * 255).save(out / "masks" / f"{path.stem}.png")
        Image.fromarray(overlay_mask(rgb, union)).save(out / "overlays" / f"{path.stem}.png")
        rows.append({"image": path.name, "instances": len(masks), "foreground_pixels": int(union.sum()),
                     "scores": [round(float(s), 4) for s in scores]})
    summary = {**(meta or {}), "prompt": prompt, "num_images": len(rows), "images": rows}
    (out / SUMMARY_NAME).write_text(json.dumps(summary, indent=2))
    return summary


def make_predictor(sam3_checkpoint, bpe_path, device: str, threshold: float, checkpoint=None) -> Predict:
    """Base SAM3 from ``sam3_checkpoint``; if ``checkpoint`` is given its ``["model"]`` weights replace them."""
    import torch

    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(bpe_path=str(bpe_path), device=device, eval_mode=True,
                                   checkpoint_path=str(sam3_checkpoint), load_from_HF=False)
    if checkpoint is not None:
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(weights["model"])
        model.eval()
    processor = Sam3Processor(model, device=device, confidence_threshold=threshold)

    @torch.no_grad()
    def predict(image_path: Path, prompt: str):
        state = processor.set_image(Image.open(image_path).convert("RGB"))
        state = processor.set_text_prompt(state=state, prompt=prompt)
        masks = [(m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)).squeeze() > 0.5 for m in state.get("masks", [])]
        scores = [float(s) for s in state.get("scores", [])]
        return masks, scores

    return predict
