"""Hydra CLI:  sam3-prepare root=... | mode=ingest images=... +masks.<category>=DIR into=...  (see conf/prepare.yaml)"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from .ingest import ingest_dataset
from .ops import prepare_dataset


def _ingest(cfg: DictConfig) -> None:
    masks = {str(k): str(v) for k, v in cfg.masks.items()}
    try:
        report = ingest_dataset(
            cfg.images,
            masks,
            cfg.into,
            test_fraction=cfg.test_fraction,
            seed=cfg.seed,
            dry_run=cfg.dry_run,
        )
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(str(e))
    print(OmegaConf.to_yaml(OmegaConf.create(report)))
    if report["problems"]:
        raise SystemExit(
            f"ingest found {len(report['problems'])} problem(s), first: {report['problems'][0]}"
        )
    for err in report["image_errors"]:
        print(f"warning: {err['image']} left out: {err['error']}", file=sys.stderr)
    if (
        report["image_errors"]
        and not sum(report["images_added"].values())
        and not report["already_annotated"]
    ):
        raise SystemExit("ingest wrote nothing: every picture was left out")
    if not cfg.dry_run and not (Path(cfg.into) / "test" / "annotation.json").exists():
        print("warning: the dataset has no test split yet", file=sys.stderr)


def _prepare(cfg: DictConfig) -> None:
    try:
        report = prepare_dataset(
            cfg.root,
            val_fraction=cfg.val_fraction,
            seed=cfg.seed,
            on_conflict=cfg.on_conflict,
            dry_run=cfg.dry_run,
        )
    except ValueError as e:
        raise SystemExit(str(e))
    print(OmegaConf.to_yaml(OmegaConf.create(report)))
    if report["problems"]:
        raise SystemExit(
            f"prepare found {len(report['problems'])} problem(s), first: {report['problems'][0]}"
        )
    if report["missing_images"]:
        print(
            f"warning: {len(report['missing_images'])} image file(s) not found and left out, "
            f"first: {report['missing_images'][0]}",
            file=sys.stderr,
        )
    for group in report["similar_category_names"]:
        print(
            f"warning: similar category names, probably one category: {group}",
            file=sys.stderr,
        )
    if report["recipe_missing_categories"]:
        print(
            "warning: recipe.yaml (kept, not overwritten) lacks categories "
            f"{report['recipe_missing_categories']}; add them before training",
            file=sys.stderr,
        )
    if report["images"]["val"] == 0:
        print("warning: the val split is empty", file=sys.stderr)


@hydra.main(config_path="conf", config_name="prepare", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if cfg.mode == "ingest":
        _ingest(cfg)
    elif cfg.mode == "prepare":
        _prepare(cfg)
    else:
        raise SystemExit(f"mode must be 'ingest' or 'prepare', got {cfg.mode!r}")


if __name__ == "__main__":
    main()
