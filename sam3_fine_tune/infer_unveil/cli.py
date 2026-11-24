"""Hydra CLI:  sam3-infer-unveil images=... prompt=...|prompts=[...] out=... sam3_checkpoint=... bpe_path=... [gate=false]  (see conf/infer_unveil.yaml)"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from ..inference.run import (
    format_sweep_table,
    Predictor,
    PromptSpec,
    resolve_device,
    run_inference,
    validate_inputs,
)
from ..inference.score import CocoScorer, format_comparison
from ..train_unveil.checkpoint import cutoffs

PATHS = ("unveil", "sam3")
DEFAULT_THRESHOLD = 0.5


def _to_shape(concept_map: Any, height: int, width: int) -> np.ndarray:
    """The concept map at the image's size (bilinear, before the threshold), so masks match the image and its ground truth."""
    m = np.asarray(concept_map, np.float32)
    if m.shape == (height, width):
        return m
    resized = Image.fromarray(m, mode="F").resize((width, height), Image.BILINEAR)
    return np.asarray(resized, np.float32)


def make_predictor(
    model: Any, specs: list[PromptSpec], gate: bool, path: str
) -> Predictor:
    """Adapt UNVEIL's ``predict`` (a concept map per phrase) to the ``Predictor`` that ``run_inference`` writes from.

    The backbone is run once per image for all prompts (``encode``); a prompt's mask is its map above the prompt's
    threshold, scored with the presence probability. An all-empty mask is returned as "no instance".
    """
    texts = [s.text for s in specs]

    def encode(image_path: Path) -> Any:
        maps, presence = model.predict(str(image_path), texts, gate=gate, path=path)
        with Image.open(image_path) as im:
            width, height = im.size
        maps = [_to_shape(m, height, width) for m in maps]
        return dict(zip(texts, zip(maps, presence)))

    def segment(state: Any, spec: PromptSpec) -> "tuple[list[Any], list[float]]":
        concept_map, presence = state[spec.text]
        mask = np.asarray(concept_map) > spec.threshold
        return ([mask], [float(presence)]) if mask.any() else ([], [])

    return Predictor(encode, segment)


def load_unveil(cfg: DictConfig) -> Any:
    """The real UNVEIL model, on ``cfg.device`` (already resolved: ``cuda`` or ``cpu``; default ``cuda``). Imports are
    lazy so the CLI loads (and its tests run) without UNVEIL installed."""
    import torch

    device = cfg.get("device", "cuda")
    if device == "cpu" or not torch.cuda.is_available():
        raise ValueError(
            f"UNVEIL needs a CUDA GPU (device resolved to {device!r}): run on a GPU machine or the HPC GPU queue"
        )
    try:
        import unveil
    except ImportError:
        raise ValueError(
            "the `unveil` package is not installed: install it from the UNVEIL repository (pip install -e <clone>)"
        ) from None
    from sam3 import build_sam3_image_model

    sam3 = build_sam3_image_model(
        bpe_path=str(cfg.bpe_path),
        device="cpu",
        eval_mode=True,
        load_from_HF=False,
        checkpoint_path=str(cfg.sam3_checkpoint),
    )
    sam3 = sam3.to(device).eval().requires_grad_(False)
    ck = cfg.get("checkpoint")
    return unveil.load(
        weights=None if ck is None else str(ck), device=device, sam3_model=sam3
    )


def run(cfg: DictConfig, model: Any = None) -> dict[str, Any]:
    """Validate, then run UNVEIL. ``model`` (an object with UNVEIL's ``predict``) is injectable for tests."""
    try:
        prompts = cfg.get("prompts")
        coco_split = cfg.get("coco_split")
        if coco_split is not None and cfg.images is not None:
            raise ValueError(
                "set images or coco_split, not both (with coco_split the images come from the split)"
            )
        if coco_split is None and cfg.images is None:
            raise ValueError("set images=<folder> (or coco_split=<split folder>)")
        if cfg.path not in PATHS:
            raise ValueError(f"path must be one of {PATHS}, got {cfg.path!r}")
        checkpoint = _plain(cfg.checkpoint)
        if isinstance(checkpoint, (list, tuple)):
            raise ValueError("checkpoint takes one UNVEIL weights file per run")
        paths, specs = validate_inputs(
            images=cfg.images,
            prompt=cfg.get("prompt"),
            prompts=None
            if prompts is None
            else cast(list[Any], OmegaConf.to_container(prompts)),
            sam3_checkpoint=cfg.sam3_checkpoint,
            bpe_path=cfg.bpe_path,
            checkpoint=checkpoint,
            threshold=DEFAULT_THRESHOLD if cfg.threshold is None else cfg.threshold,
        )
        cut = None
        if (
            cfg.threshold is None
        ):  # no threshold given: the checkpoint's own binarize, else the default
            listed = (
                []
                if prompts is None
                else cast(list[Any], OmegaConf.to_container(prompts))
            )
            own = {
                p["text"] for p in listed if isinstance(p, dict) and "threshold" in p
            }
            cut = cutoffs(
                [s.text for s in specs if s.text not in own],
                checkpoint,
                DEFAULT_THRESHOLD,
            )
            specs = [
                replace(s, thresholds=(cut[s.text]["value"],))
                if s.text in cut and cut[s.text]["source"] != "threshold"
                else s
                for s in specs
            ]
        if any(s.sweep for s in specs):
            raise ValueError(
                "a threshold sweep is not supported (UNVEIL gives a concept map, not scored instances): "
                "give each prompt one threshold"
            )
        scorer = (
            CocoScorer(coco_split, cfg.get("coco_ann", "annotation.json"), specs)
            if coco_split is not None
            else None
        )
        if scorer is not None:
            paths = scorer.image_paths
            _report_mismatch(scorer.mismatch())
        if model is None:
            import torch

            device = resolve_device(cfg.device, torch.cuda.is_available())
            model = load_unveil(
                cast(DictConfig, OmegaConf.merge(cfg, {"device": device}))
            )
    except ValueError as e:
        raise SystemExit(f"sam3-infer-unveil: {e}") from None

    meta = {
        "model": "unveil",
        "checkpoint": checkpoint and str(checkpoint),
        "sam3_checkpoint": str(cfg.sam3_checkpoint),
        "default_threshold": DEFAULT_THRESHOLD
        if cfg.threshold is None
        else cfg.threshold,
        "cutoffs": cut,
        "device": cfg.device,
        "gate": bool(cfg.gate),
        "path": cfg.path,
    }
    out = Path(cfg.out)
    summary = run_inference(
        paths,
        specs,
        out,
        make_predictor(model, specs, bool(cfg.gate), cfg.path),
        meta,
        on_union=scorer.collect if scorer else None,
    )
    if scorer is not None:
        try:
            result = scorer.finish(
                out,
                {
                    "variant": "unveil",
                    "checkpoint": meta["checkpoint"],
                    "model_label": "unveil",
                    "thresholds": {s.text: s.threshold for s in specs},
                },
            )
        except ValueError as e:
            raise SystemExit(f"sam3-infer-unveil: {e}") from None
        summary["evaluation"] = result["summary"]
        summary["coco"] = scorer.mismatch()
    return summary


def _plain(value: Any) -> Any:
    return OmegaConf.to_container(value) if OmegaConf.is_config(value) else value


def _report_mismatch(mismatch: dict[str, list[str]]) -> None:
    for key, what in (
        ("prompts_without_category", "prompt(s) with no COCO category (not scored)"),
        ("categories_without_prompt", "COCO categor(ies) with no prompt (not scored)"),
    ):
        if mismatch[key]:
            print(
                f"sam3-infer-unveil: warning: {what}: {mismatch[key]}", file=sys.stderr
            )


@hydra.main(config_path="conf", config_name="infer_unveil", version_base="1.3")
def main(cfg: DictConfig) -> None:
    summary = run(cfg)
    texts = [p["text"] for p in summary["prompts"]]
    print(
        f"unveil ({summary['path']} path, gate={'on' if summary['gate'] else 'off'}), prompt(s) {texts}: "
        f"{summary['num_images']} image(s) -> {cfg.out}"
    )
    table = format_sweep_table(summary)
    if table:
        print(table)
    if "evaluation" in summary:
        print("\nforeground IoU / precision / recall (%) vs COCO ground truth")
        print(format_comparison({"unveil": summary["evaluation"]}))


if __name__ == "__main__":
    main()
