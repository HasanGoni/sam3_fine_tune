"""Hydra CLI:  sam3-train-unveil root=... [recipe=FILE] [validate_only=false trial_steps=N]  (see conf/train_unveil.yaml)"""

from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from .ops import epochs_problem, RecipeError, validate_recipe
from .streams import batch_composition, build_streams, load_index
from .trainer import run_trial


def _plain(value: object) -> object:
    return OmegaConf.to_container(value) if OmegaConf.is_config(value) else value


@hydra.main(config_path="conf", config_name="train_unveil", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if cfg.epochs is not None and (msg := epochs_problem(_plain(cfg.epochs))):
        raise SystemExit(msg)
    try:
        plan = validate_recipe(cfg.root, cfg.recipe)
        index = load_index(cfg.root, plan)
    except (RecipeError, ValueError) as e:
        raise SystemExit(str(e))
    plan["train_images"] = {n: len(r) for n, r in index.items()}
    shown = {
        "categories": plan["categories"],
        "kinds": plan["kinds"],
        "per_gpu": plan["per_gpu"],
        "train_images": plan["train_images"],
        "negatives": plan["negatives"],
        "joint": plan["joint"],
        "streams": {
            s.name: {
                "kind": s.kind,
                "prompt": s.prompts[0],
                "per_gpu": s.per_gpu,
                "images": len(index[s.name]),
            }
            for s in build_streams(plan)
        },
    }
    if cfg.show_batches:
        shown["batches"] = batch_composition(plan, index, cfg.show_batches)
    if not cfg.validate_only:
        device = cfg.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        plan["sam3_checkpoint"] = cfg.sam3_checkpoint
    if not cfg.validate_only:
        try:
            shown["trial"] = run_trial(
                plan,
                index,
                cfg.root,
                cfg.trial_steps,
                cfg.model,
                device,
                cfg.trial_size,
            )
        except ValueError as e:
            raise SystemExit(str(e))
    print(OmegaConf.to_yaml(OmegaConf.create(shown)))


if __name__ == "__main__":
    main()
