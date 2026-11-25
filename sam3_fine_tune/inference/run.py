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


def _thr_dir(threshold: float) -> str:
    return f"thr_{threshold:g}"


@dataclass(frozen=True)
class PromptSpec:
    text: str
    thresholds: tuple[float, ...]  # ascending, unique; more than one means a sweep
    type: str = "text"

    @property
    def threshold(self) -> float:
        """The lowest threshold: the model runs once at this, higher ones only re-filter by score."""
        return self.thresholds[0]

    @property
    def sweep(self) -> bool:
        return len(self.thresholds) > 1

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
        label = f"{where} ({text!r}) threshold"
        if thr is None:
            thresholds: tuple[float, ...] = (default_threshold,)
        elif isinstance(thr, (list, tuple)):
            if not thr:
                raise ValueError(f"{label} list must not be empty")
            values = [_check_threshold(v, label) for v in thr]
            if len({_thr_dir(v) for v in values}) != len(values):
                raise ValueError(
                    f"{label} list has duplicate values (compared as {_thr_dir(0.5)[:4]}<t:g>, "
                    f"the output folder name): {list(thr)}"
                )
            thresholds = tuple(sorted(values))
        else:
            thresholds = (_check_threshold(thr, label),)
        specs.append(PromptSpec(text=text, thresholds=thresholds, type=kind))
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
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a number in (0, 1], got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number in (0, 1], got {value!r}") from None
    if not 0 < number <= 1:
        raise ValueError(f"{label} must be in (0, 1], got {number}")
    return number


def normalize_checkpoints(checkpoint: Any) -> "list[Path | None]":
    """``checkpoint`` may be null (base model), a path, or a list of those where null means the base model."""
    items = list(checkpoint) if isinstance(checkpoint, (list, tuple)) else [checkpoint]
    if not items:
        raise ValueError(
            "checkpoint list must not be empty (use null for the base model)"
        )
    out: list[Path | None] = [None if c is None else Path(str(c)) for c in items]
    for c in out:
        if c is not None and not c.is_file():
            raise ValueError(f"checkpoint (fine-tuned model) not found: {c}")
    labels = [model_label(c) for c in out]
    dup = sorted(label for label, n in Counter(labels).items() if n > 1)
    if dup:
        raise ValueError(
            f"checkpoints would share an output name {dup}: list each model once"
        )
    return out


def model_label(checkpoint: Path | None) -> str:
    """Folder/table name of a model: ``base``, or ``<experiment>_<stem>`` for ``<experiment>/checkpoints/<stem>.pt``."""
    if checkpoint is None:
        return "base"
    if checkpoint.parent.name == "checkpoints" and checkpoint.parent.parent.name:
        return f"{checkpoint.parent.parent.name}_{checkpoint.stem}"
    return checkpoint.stem


def validate_inputs(
    *,
    images: str | Path | None,
    prompt: str | None = None,
    prompts: Sequence[Any] | None = None,
    sam3_checkpoint: str | Path,
    bpe_path: str | Path,
    checkpoint: Any,
    threshold: float,
) -> "tuple[list[Path], list[PromptSpec]]":
    """Fail early (before any model is loaded) with a message that says what to fix. Returns (images, prompt specs).

    ``images=None`` skips the folder check (images then come from a COCO split) and returns an empty list.
    """
    specs = parse_prompts(prompt, prompts, threshold)
    for label, path in (
        ("sam3_checkpoint (base sam3.pt)", sam3_checkpoint),
        ("bpe_path (tokenizer vocabulary)", bpe_path),
    ):
        if not Path(path).is_file():
            raise ValueError(f"{label} not found: {path}")
    normalize_checkpoints(checkpoint)
    return ([] if images is None else list_images(images)), specs


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Sequence[int] = (255, 0, 0),
    alpha: float = 0.5,
) -> np.ndarray:
    out = image.astype(np.float32).copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.asarray(color, np.float32)
    return out.round().astype(np.uint8)


def _union(masks: Sequence[Any], shape: tuple[int, ...], name: str) -> np.ndarray:
    union = np.zeros(shape, bool)
    for m in masks:
        m = np.asarray(m).astype(bool)
        if m.shape != union.shape:
            raise ValueError(
                f"{name}: predicted mask {m.shape} does not match image {union.shape}"
            )
        union |= m
    return union


def _stats(masks: Sequence[Any], scores: Sequence[float], union: np.ndarray) -> dict:
    return {
        "instances": len(masks),
        "foreground_pixels": int(union.sum()),
        "scores": [round(float(s), 4) for s in scores],
    }


def run_inference(
    image_paths: Sequence[Path],
    specs: Sequence[PromptSpec],
    out: str | Path,
    predictor: Predictor,
    meta: dict[str, Any] | None = None,
    on_union: Callable[[Path, PromptSpec, np.ndarray], None] | None = None,
) -> dict[str, Any]:
    """Write per-prompt masks, overlays and ``out/summary.json``.

    Masks (0/255, union of instances): ``masks/<stem>.png`` for one prompt, ``masks/<prompt-slug>/<stem>.png`` for
    several. A prompt with several thresholds (a sweep) gets one extra ``thr_<t>/`` level per threshold, in masks and in
    ``overlays/[<slug>/]thr_<t>/<stem>.png``; ``overlays/<stem>.png`` shows every prompt at its lowest threshold.
    The model runs once per image and prompt, at the lowest threshold; higher ones re-filter by score (``> t``, as the
    processor does). ``on_union(image, prompt, mask)`` receives each prompt's combined mask at its (single) threshold,
    e.g. for scoring.
    """
    out = Path(out)
    multi = len(specs) > 1

    def sub(root: str, spec: PromptSpec, threshold: float | None = None) -> Path:
        d = out / root / spec.slug if multi else out / root
        return d / _thr_dir(threshold) if threshold is not None else d

    for spec in specs:
        for t in spec.thresholds if spec.sweep else (None,):
            sub("masks", spec, t).mkdir(parents=True, exist_ok=True)
            if spec.sweep:
                sub("overlays", spec, t).mkdir(parents=True, exist_ok=True)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    rows = []
    for path in image_paths:
        path = Path(path)
        rgb = np.asarray(Image.open(path).convert("RGB"))
        handle = predictor.encode(
            path
        )  # backbone: once per image, shared by all prompts
        overlay, per_prompt = rgb, []
        for idx, spec in enumerate(specs):
            color = PALETTE[idx % len(PALETTE)]
            masks, scores = predictor.segment(handle, spec)  # decoder: once per prompt
            if len(masks) != len(scores):
                raise ValueError(
                    f"{path.name}: {len(masks)} masks but {len(scores)} scores for {spec.text!r}"
                )
            levels = []
            unions = []
            for t in spec.thresholds:
                keep = (
                    [k for k, sc in enumerate(scores) if np.float32(sc) > np.float32(t)]
                    if spec.sweep
                    else None
                )
                m_t = masks if keep is None else [masks[k] for k in keep]
                s_t = scores if keep is None else [scores[k] for k in keep]
                union = _union(m_t, rgb.shape[:2], path.name)
                unions.append(union)
                mask_dir = sub("masks", spec, t if spec.sweep else None)
                Image.fromarray(union.astype(np.uint8) * 255).save(
                    mask_dir / f"{path.stem}.png"
                )
                if spec.sweep:
                    Image.fromarray(overlay_mask(rgb, union, color)).save(
                        sub("overlays", spec, t) / f"{path.stem}.png"
                    )
                levels.append({"threshold": t, **_stats(m_t, s_t, union)})
            if on_union is not None and not spec.sweep:
                on_union(path, spec, unions[0])
            # thresholds ascend: unions[0] is the lowest
            overlay = overlay_mask(overlay, unions[0], color)
            entry: dict[str, Any] = {"prompt": spec.text}
            if spec.sweep:
                entry["sweep"] = levels
            else:
                entry.update(levels[0])
            per_prompt.append(entry)
        Image.fromarray(overlay).save(out / "overlays" / f"{path.stem}.png")
        row: dict[str, Any] = {"image": path.name}
        if multi:
            row["prompts"] = per_prompt
        else:  # single-prompt layout from #13 is unchanged
            row.update({k: v for k, v in per_prompt[0].items() if k != "prompt"})
        rows.append(row)
    summary: dict[str, Any] = {
        **(meta or {}),
        "prompts": [_prompt_meta(s) for s in specs],
        "num_images": len(rows),
        "images": rows,
    }
    if not multi:
        summary["prompt"] = specs[0].text
        if not specs[0].sweep:
            summary["threshold"] = specs[0].threshold
    (out / SUMMARY_NAME).write_text(json.dumps(summary, indent=2))
    return summary


def _prompt_meta(spec: PromptSpec) -> dict[str, Any]:
    meta: dict[str, Any] = {"type": spec.type, "text": spec.text}
    if spec.sweep:
        meta["thresholds"] = list(spec.thresholds)
    else:
        meta["threshold"] = spec.threshold
    return meta


def format_sweep_table(summary: dict[str, Any]) -> str:
    """Compact instances-vs-threshold table per sweep prompt (totals over all images); '' when nothing swept."""
    totals: dict[str, dict[float, int]] = {}
    for row in summary["images"]:
        entries = (
            row["prompts"]
            if "prompts" in row
            else [{"prompt": summary["prompt"], **row}]
        )
        for e in entries:
            for lvl in e.get("sweep", []):
                by_t = totals.setdefault(e["prompt"], {})
                by_t[lvl["threshold"]] = (
                    by_t.get(lvl["threshold"], 0) + lvl["instances"]
                )
    blocks = []
    for prompt, by_t in totals.items():
        ts = sorted(by_t)
        width = max(len(f"{t:g}") for t in ts + [0.0]) + 2
        blocks.append(
            f"{prompt!r}: instances vs threshold ({summary['num_images']} image(s))\n"
            + "  threshold  "
            + "".join(f"{t:g}".rjust(width) for t in ts)
            + "\n"
            + "  instances  "
            + "".join(str(by_t[t]).rjust(width) for t in ts)
        )
    return "\n".join(blocks)


def make_predictor(
    sam3_checkpoint: str | Path,
    bpe_path: str | Path,
    device: str,
    checkpoint: str | Path | None = None,
    trusted: bool = False,
) -> Predictor:
    """Base SAM3 from ``sam3_checkpoint``, or the full fine-tuned weights from ``checkpoint`` when given.

    ``trusted=True`` unpickles ``checkpoint`` fully (needed for training checkpoints, which also hold optimizer state
    and other Python objects); only pass it for files you made yourself. The default stays weights-only.
    """
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
            weights = torch.load(
                checkpoint, map_location="cpu", weights_only=not trusted
            )
            state = weights["model"]
            model.load_state_dict(state)
        except Exception as e:  # unpickling, missing "model" key, or key/shape mismatch
            hint = (
                ""
                if trusted
                else ". A full training checkpoint cannot load weights-only: for your own local checkpoint set "
                "trust_checkpoint=true (sam3-compare), or extract the 'model' state_dict "
                "(torch.save(ckpt['model'], path)) and use that file"
            )
            raise ValueError(
                f"checkpoint {checkpoint} is not a compatible fine-tuned SAM3 checkpoint: "
                f"{type(e).__name__}: {str(e)[:300]}{hint}"
            ) from None
    model.eval()
    processor = Sam3Processor(model, device=device)

    @torch.no_grad()
    def encode(image_path: Path) -> Any:
        return processor.set_image(Image.open(image_path).convert("RGB"))

    @torch.no_grad()
    def segment(state: Any, spec: PromptSpec) -> "tuple[list[np.ndarray], list[float]]":
        # drop the previous prompt's text features and results (same as sam3-finetune's evaluation); the image
        # features in ``state`` stay, so the backbone runs once per image
        processor.reset_all_prompts(state)
        processor.confidence_threshold = spec.threshold
        out = processor.set_text_prompt(state=state, prompt=spec.text)
        masks = []
        for m in out.get("masks", []):  # (N, 1, H, W) bool -> N x (H, W)
            m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
            masks.append(m[0].astype(bool) if m.ndim == 3 else m.astype(bool))
        scores = [float(s) for s in out.get("scores", [])]
        return masks, scores

    return Predictor(encode, segment)
