"""Hydra CLI:  sam3-infer images=... prompt=... out=... sam3_checkpoint=... bpe_path=... [checkpoint=...]  (see conf/infer.yaml)"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from .run import make_predictor, resolve_device, run_inference, validate_inputs


def run(cfg: DictConfig, predict=None) -> dict:
    """Validate, then run. ``predict`` is injectable; without it the real SAM3 is loaded."""
    try:
        paths = validate_inputs(images=cfg.images, prompt=cfg.prompt, sam3_checkpoint=cfg.sam3_checkpoint,
                                bpe_path=cfg.bpe_path, checkpoint=cfg.checkpoint, threshold=cfg.threshold)
        if predict is None:
            import torch
            device = resolve_device(cfg.device, torch.cuda.is_available())
            predict = make_predictor(cfg.sam3_checkpoint, cfg.bpe_path, device, cfg.threshold, cfg.checkpoint)
        else:
            device = cfg.device
    except ValueError as e:
        raise SystemExit(f"sam3-infer: {e}") from None
    meta = {"model": "fine-tuned" if cfg.checkpoint else "base", "checkpoint": cfg.checkpoint and str(cfg.checkpoint),
            "sam3_checkpoint": str(cfg.sam3_checkpoint), "threshold": cfg.threshold, "device": device}
    return run_inference(paths, cfg.prompt, cfg.out, predict, meta)


@hydra.main(config_path="conf", config_name="infer", version_base="1.3")
def main(cfg: DictConfig) -> None:
    summary = run(cfg)
    print(f"{summary['model']} model, prompt {summary['prompt']!r}: {summary['num_images']} image(s) -> {cfg.out}")


if __name__ == "__main__":
    main()
