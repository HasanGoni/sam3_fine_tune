"""Hydra CLI:  sam3-train-unveil root=... [recipe=FILE] [validate_only=false trial_steps=N | train=true out=DIR epochs=E|steps=N]  (see conf/train_unveil.yaml)"""

from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from .calibrate import load_val

from .ops import epochs_problem, RecipeError, validate_recipe
from .streams import batch_composition, build_streams, load_index
from .trainer import run_training, run_trial, steps_for


def train(cfg: DictConfig, plan: dict, index: dict, device: str) -> dict:
    """The training run: steps from ``steps`` or ``epochs`` (rule 5), thresholds on val, a checkpoint in ``out``."""
    if not cfg.out:
        raise SystemExit(
            "train=true needs `out=<folder>` for the checkpoint and the log"
        )
    try:
        has_val = any(load_val(cfg.root, plan).values())
    except ValueError as e:
        raise SystemExit(str(e))
    if not has_val:
        raise SystemExit(
            "the prepared view has no val images, and thresholds are fitted on val only (never on test): "
            f"run `sam3-prepare root={cfg.root} val_fraction=0.1` (or more) and train again"
        )
    epochs = cfg.epochs if cfg.epochs is not None else plan["epochs"]
    accumulate = plan["accumulate"] if cfg.accumulate is None else cfg.accumulate
    if isinstance(accumulate, bool) or accumulate < 1 or accumulate != int(accumulate):
        raise SystemExit(
            f"accumulate is {accumulate!r}; use a whole number of 1 or more (`accumulate=N` or `accumulate:` in recipe.yaml)"
        )
    accumulate = int(accumulate)
    if cfg.steps is not None:
        steps = int(cfg.steps)
    elif epochs is not None:
        steps = steps_for(plan, index, epochs, accumulate)
    else:
        raise SystemExit(
            "no training length: set `steps=N` or `epochs=E` on the command line, or `epochs:` in recipe.yaml"
        )
    try:
        result = run_training(
            plan,
            index,
            cfg.root,
            cfg.out,
            steps,
            cfg.model,
            device,
            cfg.image_size,
            cfg.seed,
            accumulate,
        )
    except ValueError as e:
        raise SystemExit(str(e))
    return {**result, "epochs": epochs, "accumulate": accumulate}


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
    if not cfg.validate_only and cfg.train:
        shown["train"] = train(cfg, plan, index, device)
    elif not cfg.validate_only:
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
