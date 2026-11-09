"""Run SAM3 (base or fine-tuned) with one or several prompts on a folder of images; write masks, overlays, a summary.

The model is an injectable ``Predictor``: ``encode(image_path)`` runs the image backbone once per image and returns an
opaque handle; ``segment(handle, spec)`` is then called once per prompt and returns ``(masks, scores)`` (``masks``: list
of bool ``(H, W)`` arrays, one per detected instance). This keeps the output logic testable without weights;
``make_predictor`` is the real one.

Prompt entries are the single extension seam for new prompt kinds: each is ``{type, text, threshold}`` and ``type``
selects the handling (``PROMPT_TYPES``). Adding a type must not change how existing ``type: text`` entries look.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
DEVICES = ("auto", "cpu", "cuda")
SUMMARY_NAME = "summary.json"
PROMPT_TYPES = (
    "text",
)  # extension point: add a kind here, then dispatch on spec.type in make_predictor's segment()
PROMPT_KEYS = {"type", "text", "threshold"}
PALETTE = (
    (255, 0, 0),
    (0, 200, 0),
    (0, 100, 255),
    (255, 200, 0),
    (200, 0, 255),
    (0, 220, 220),
    (255, 120, 0),
    (255, 0, 160),
)


@dataclass(frozen=True)
class PromptSpec:
    text: str
    threshold: float
    type: str = "text"

    @property
    def slug(self) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", self.text).strip("_").lower() or "prompt"


class Predictor(NamedTuple):
    encode: Callable[[Path], Any]
    segment: Callable[[Any, PromptSpec], "tuple[Sequence[np.ndarray], Sequence[float]]"]


def resolve_device(device: str, cuda_available: bool) -> str:
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {device!r}")
    if device == "cuda" and not cuda_available:
        raise ValueError(
            "device=cuda requested but no CUDA GPU is visible; use device=cpu or device=auto"
        )
    return ("cuda" if cuda_available else "cpu") if device == "auto" else device


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"images folder not found: {folder}")
    found = sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not found:
        raise ValueError(
            f"no images ({', '.join(sorted(IMAGE_EXTENSIONS))}) in {folder}"
        )
    dup = sorted(s for s, n in Counter(p.stem for p in found).items() if n > 1)
    if dup:
        raise ValueError(
            f"images share a file stem and would overwrite each other's outputs: {dup}"
        )
    return found


def parse_prompts(
    prompt: str | None,
    prompts: Sequence[Any] | None,
    default_threshold: float,
) -> list[PromptSpec]:
    """Normalise ``prompt`` (one string) or ``prompts`` (list of ``{text, type?, threshold?}``) into ``PromptSpec``s.

    Raises ``ValueError`` naming the offending entry; entries without a threshold use ``default_threshold``.
    """
    default_threshold = _check_threshold(default_threshold, "threshold")
    has_prompt = prompt is not None and str(prompt).strip() != ""
    if prompt is not None and not has_prompt:
        raise ValueError("prompt must be a non-empty text prompt")
    if has_prompt and prompts:
        raise ValueError("set either prompt=<text> or prompts=[...], not both")
    if not has_prompt and not prompts:
        raise ValueError("no prompts: set prompt=<text> or prompts=[{text: ...}, ...]")
    entries: list[Any] = [{"text": prompt}] if has_prompt else list(prompts or [])
    specs = []
    for i, entry in enumerate(entries):
        where = f"prompts[{i}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: expected a mapping with 'text', got {entry!r}")
        unknown = sorted(set(entry) - PROMPT_KEYS)
        if unknown:
            raise ValueError(
                f"{where}: unknown key(s) {unknown}; allowed: {sorted(PROMPT_KEYS)}"
            )
        kind = entry.get("type", "text")
        if kind not in PROMPT_TYPES:
            raise ValueError(
                f"{where}: unsupported type {kind!r}; supported: {PROMPT_TYPES}"
            )
        text = entry.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{where}: text must be a non-empty string")
        thr = entry.get("threshold")
        thr = (
            default_threshold
            if thr is None
            else _check_threshold(thr, f"{where} ({text!r}) threshold")
        )
        specs.append(PromptSpec(text=text, threshold=thr, type=kind))
    for field in ("text", "slug"):
        dup = sorted(
            v for v, n in Counter(getattr(s, field) for s in specs).items() if n > 1
        )
        if dup:
            raise ValueError(
                f"duplicate prompt {field}s (each prompt needs its own output): {dup}"
            )
    return specs


def _check_threshold(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number in (0, 1], got {value!r}") from None
    if not 0 < number <= 1:
        raise ValueError(f"{label} must be in (0, 1], got {number}")
    return number


def validate_inputs(
    *,
    images: str | Path,
    prompt: str | None = None,
    prompts: Sequence[Any] | None = None,
    sam3_checkpoint: str | Path,
    bpe_path: str | Path,
    checkpoint: str | Path | None,
    threshold: float,
) -> "tuple[list[Path], list[PromptSpec]]":
    """Fail early (before any model is loaded) with a message that says what to fix. Returns (images, prompt specs)."""
    specs = parse_prompts(prompt, prompts, threshold)
    for label, path in (
        ("sam3_checkpoint (base sam3.pt)", sam3_checkpoint),
        ("bpe_path (tokenizer vocabulary)", bpe_path),
    ):
        if not Path(path).is_file():
            raise ValueError(f"{label} not found: {path}")
    if checkpoint is not None and not Path(checkpoint).is_file():
        raise ValueError(f"checkpoint (fine-tuned model) not found: {checkpoint}")
    return list_images(images), specs


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Sequence[int] = (255, 0, 0),
    alpha: float = 0.5,
) -> np.ndarray:
    out = image.astype(np.float32).copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.asarray(color, np.float32)
    return out.round().astype(np.uint8)


def run_inference(
    image_paths: Sequence[Path],
    specs: Sequence[PromptSpec],
    out: str | Path,
    predictor: Predictor,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write per-prompt masks, one multi-colour overlay per image and ``out/summary.json``.

    One prompt: ``masks/<stem>.png``. Several: ``masks/<prompt-slug>/<stem>.png``. Masks are 0/255 (union of instances).
    """
    out = Path(out)
    multi = len(specs) > 1
    mask_dirs = {
        s.slug: out / "masks" / s.slug if multi else out / "masks" for s in specs
    }
    for d in (*mask_dirs.values(), out / "overlays"):
        d.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in image_paths:
        path = Path(path)
        rgb = np.asarray(Image.open(path).convert("RGB"))
        handle = predictor.encode(
            path
        )  # image backbone: once per image, shared by all prompts
        overlay, per_prompt = rgb, []
        for idx, spec in enumerate(specs):
            masks, scores = predictor.segment(handle, spec)
            union = np.zeros(rgb.shape[:2], bool)
            for m in masks:
                m = np.asarray(m).astype(bool)
                if m.shape != union.shape:
                    raise ValueError(
                        f"{path.name}: predicted mask {m.shape} does not match image {union.shape}"
                    )
                union |= m
            Image.fromarray(union.astype(np.uint8) * 255).save(
                mask_dirs[spec.slug] / f"{path.stem}.png"
            )
            overlay = overlay_mask(overlay, union, PALETTE[idx % len(PALETTE)])
            per_prompt.append(
                {
                    "prompt": spec.text,
                    "threshold": spec.threshold,
                    "instances": len(masks),
                    "foreground_pixels": int(union.sum()),
                    "scores": [round(float(s), 4) for s in scores],
                }
            )
        Image.fromarray(overlay).save(out / "overlays" / f"{path.stem}.png")
        row: dict[str, Any] = {"image": path.name}
        if multi:
            row["prompts"] = per_prompt
        else:  # single-prompt layout from #13 is unchanged
            row.update({key: v for key, v in per_prompt[0].items() if key != "prompt"})
        rows.append(row)
    summary = {
        **(meta or {}),
        "prompts": [
            {"type": s.type, "text": s.text, "threshold": s.threshold} for s in specs
        ],
        "num_images": len(rows),
        "images": rows,
    }
    if not multi:
        summary["prompt"] = specs[0].text
        summary["threshold"] = specs[0].threshold
    (out / SUMMARY_NAME).write_text(json.dumps(summary, indent=2))
    return summary


def make_predictor(
    sam3_checkpoint: str | Path,
    bpe_path: str | Path,
    device: str,
    checkpoint: str | Path | None = None,
) -> Predictor:
    """Base SAM3 from ``sam3_checkpoint``, or the full fine-tuned weights from ``checkpoint`` when given."""
    import torch

    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    # a fine-tuned checkpoint holds every weight, so the base file is only read when no checkpoint is given
    model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        device=device,
        eval_mode=True,
        load_from_HF=False,
        checkpoint_path=None if checkpoint is not None else str(sam3_checkpoint),
    )
    if checkpoint is not None:
        try:
            weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
            state = weights["model"]
            model.load_state_dict(state)
        except Exception as e:  # unpickling, missing "model" key, or key/shape mismatch
            raise ValueError(
                f"checkpoint {checkpoint} is not a compatible fine-tuned SAM3 checkpoint: "
                f"{type(e).__name__}: {str(e)[:300]}"
            ) from None
    model.eval()
    processor = Sam3Processor(model, device=device)

    @torch.no_grad()
    def encode(image_path: Path) -> Any:
        return processor.set_image(Image.open(image_path).convert("RGB"))

    @torch.no_grad()
    def segment(state: Any, spec: PromptSpec) -> "tuple[list[np.ndarray], list[float]]":
        # text features replace the previous prompt's; image features in ``state`` are reused
        processor.confidence_threshold = spec.threshold
        out = processor.set_text_prompt(state=state, prompt=spec.text)
        masks = []
        for m in out.get("masks", []):  # (N, 1, H, W) bool -> N x (H, W)
            m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
            masks.append(m[0].astype(bool) if m.ndim == 3 else m.astype(bool))
        scores = [float(s) for s in out.get("scores", [])]
        return masks, scores

    return Predictor(encode, segment)
