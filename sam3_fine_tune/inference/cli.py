"""Hydra CLI:  sam3-infer images=... prompt=...|prompts=[...] out=... sam3_checkpoint=... bpe_path=... [checkpoint=...]  (see conf/infer.yaml)"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from .run import make_predictor, resolve_device, run_inference, validate_inputs


def run(cfg: DictConfig, predictor=None) -> dict:
    """Validate, then run. ``predictor`` (a ``Predictor``) is injectable; without it the real SAM3 is loaded."""
    try:
        prompts = cfg.get("prompts")
        paths, specs = validate_inputs(
            images=cfg.images,
            prompt=cfg.get("prompt"),
            prompts=None if prompts is None else OmegaConf.to_container(prompts),
            sam3_checkpoint=cfg.sam3_checkpoint,
            bpe_path=cfg.bpe_path,
            checkpoint=cfg.checkpoint,
            threshold=cfg.threshold,
        )
        if predictor is None:
            import torch

            device = resolve_device(cfg.device, torch.cuda.is_available())
            predictor = make_predictor(
                cfg.sam3_checkpoint, cfg.bpe_path, device, cfg.checkpoint
            )
        else:
            device = cfg.device
    except ValueError as e:
        raise SystemExit(f"sam3-infer: {e}") from None
    meta = {
        "model": "fine-tuned" if cfg.checkpoint else "base",
        "checkpoint": cfg.checkpoint and str(cfg.checkpoint),
        "sam3_checkpoint": str(cfg.sam3_checkpoint),
        "default_threshold": cfg.threshold,
        "device": device,
    }
    return run_inference(paths, specs, cfg.out, predictor, meta)


@hydra.main(config_path="conf", config_name="infer", version_base="1.3")
def main(cfg: DictConfig) -> None:
    summary = run(cfg)
    texts = [p["text"] for p in summary["prompts"]]
    print(
        f"{summary['model']} model, prompt(s) {texts}: {summary['num_images']} image(s) -> {cfg.out}"
    )


if __name__ == "__main__":
    main()
