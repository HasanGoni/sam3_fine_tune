"""Hydra CLI:  sam3-infer images=... prompt=...|prompts=[...] out=... sam3_checkpoint=... bpe_path=... [checkpoint=...]  (see conf/infer.yaml)"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, cast, Union

import hydra
from omegaconf import DictConfig, OmegaConf

from .run import (
    format_sweep_table,
    make_predictor,
    model_label,
    normalize_checkpoints,
    Predictor,
    resolve_device,
    run_inference,
    validate_inputs,
)
from .score import CocoScorer, format_comparison

# a ready Predictor (used for every model) or a factory ``checkpoint-or-None -> Predictor`` (tests, custom backends)
PredictorSource = Union[Predictor, Callable[[Path | None], Predictor]]


def run(cfg: DictConfig, predictor: PredictorSource | None = None) -> dict[str, Any]:
    """Validate, then run every model. ``predictor`` is injectable; without it the real SAM3 is loaded.

    One model: returns its summary (layout unchanged). Several (``checkpoint`` is a list, ``null`` = base):
    returns ``{"models": {label: summary}}`` with each model's outputs under ``out/<label>/``.
    With ``coco_split`` each summary also carries ``evaluation`` (the scores) and ``coco`` (prompt/category mismatch).
    """
    try:
        prompts = cfg.get("prompts")
        coco_split = cfg.get("coco_split")
        if coco_split is not None and cfg.images is not None:
            raise ValueError(
                "set images or coco_split, not both (with coco_split the images come from the split)"
            )
        if coco_split is None and cfg.images is None:
            raise ValueError("set images=<folder> (or coco_split=<split folder>)")
        paths, specs = validate_inputs(
            images=cfg.images,
            prompt=cfg.get("prompt"),
            prompts=None
            if prompts is None
            else cast(list[Any], OmegaConf.to_container(prompts)),
            sam3_checkpoint=cfg.sam3_checkpoint,
            bpe_path=cfg.bpe_path,
            checkpoint=_plain(cfg.checkpoint),
            threshold=cfg.threshold,
        )
        checkpoints = normalize_checkpoints(_plain(cfg.checkpoint))
        scorer = (
            CocoScorer(coco_split, cfg.get("coco_ann", "annotation.json"), specs)
            if coco_split is not None
            else None
        )
        if scorer is not None:
            paths = scorer.image_paths
            _report_mismatch(scorer.mismatch())
        device = cfg.device
        if predictor is None:
            import torch

            device = resolve_device(cfg.device, torch.cuda.is_available())
    except ValueError as e:
        raise SystemExit(f"sam3-infer: {e}") from None

    summaries: dict[str, dict[str, Any]] = {}
    for ck in checkpoints:
        label = model_label(ck)
        try:
            if predictor is None:
                pred = make_predictor(cfg.sam3_checkpoint, cfg.bpe_path, device, ck)
            elif isinstance(predictor, Predictor):
                pred = predictor
            else:
                pred = predictor(ck)
        except ValueError as e:
            raise SystemExit(f"sam3-infer: {e}") from None
        meta = {
            "model": "fine-tuned" if ck else "base",
            "checkpoint": ck and str(ck),
            "sam3_checkpoint": str(cfg.sam3_checkpoint),
            "default_threshold": cfg.threshold,
            "device": device,
        }
        out = Path(cfg.out) / label if len(checkpoints) > 1 else Path(cfg.out)
        summary = run_inference(
            paths,
            specs,
            out,
            pred,
            meta,
            on_union=scorer.collect if scorer else None,
        )
        if scorer is not None:
            try:
                result = scorer.finish(
                    out,
                    {
                        "variant": meta["model"],
                        "checkpoint": meta["checkpoint"],
                        "model_label": label,
                        "thresholds": {s.text: s.threshold for s in specs},
                    },
                )
            except ValueError as e:
                raise SystemExit(f"sam3-infer: {e}") from None
            summary["evaluation"] = result["summary"]
            summary["coco"] = scorer.mismatch()
        summaries[label] = summary
    return (
        summaries[model_label(checkpoints[0])]
        if len(checkpoints) == 1
        else {"models": summaries}
    )


def _plain(value: Any) -> Any:
    return OmegaConf.to_container(value) if OmegaConf.is_config(value) else value


def _report_mismatch(mismatch: dict[str, list[str]]) -> None:
    for key, what in (
        ("prompts_without_category", "prompt(s) with no COCO category (not scored)"),
        ("categories_without_prompt", "COCO categor(ies) with no prompt (not scored)"),
    ):
        if mismatch[key]:
            print(f"sam3-infer: warning: {what}: {mismatch[key]}", file=sys.stderr)


@hydra.main(config_path="conf", config_name="infer", version_base="1.3")
def main(cfg: DictConfig) -> None:
    result = run(cfg)
    models = result["models"] if "models" in result else {"": result}
    for label, summary in models.items():
        texts = [p["text"] for p in summary["prompts"]]
        print(
            f"{summary['model']} model{f' [{label}]' if label else ''}, prompt(s) {texts}: "
            f"{summary['num_images']} image(s) -> {cfg.out}"
        )
        table = format_sweep_table(summary)
        if table:
            print(table)
    evaluations = {l: s["evaluation"] for l, s in models.items() if "evaluation" in s}
    if evaluations:
        print("\nforeground IoU / precision / recall (%) vs COCO ground truth")
        print(format_comparison(evaluations))
        for key, what in (
            ("prompts_without_category", "prompts with no COCO category (not scored)"),
            (
                "categories_without_prompt",
                "COCO categories with no prompt (not scored)",
            ),
        ):
            missing = next(iter(models.values()))["coco"][key]
            if missing:
                print(f"{what}: {missing}")


if __name__ == "__main__":
    main()
