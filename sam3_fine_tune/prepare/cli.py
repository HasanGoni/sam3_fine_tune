"""Hydra CLI:  sam3-prepare root=... [key=value ...]  (see conf/prepare.yaml)"""

from __future__ import annotations

import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from .ops import prepare_dataset


@hydra.main(config_path="conf", config_name="prepare", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if cfg.mode != "prepare":
        raise SystemExit(f"mode must be 'prepare', got {cfg.mode!r}")
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
    if report["images"]["val"] == 0:
        print("warning: the val split is empty", file=sys.stderr)


if __name__ == "__main__":
    main()
