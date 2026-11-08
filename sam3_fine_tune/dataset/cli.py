"""Hydra CLI:  sam3-dataset images=... masks=... category=... into=... [key=value ...]  (see conf/dataset.yaml)"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from .ops import add_category_data


@hydra.main(config_path="conf", config_name="dataset", version_base="1.3")
def main(cfg: DictConfig) -> None:
    report = add_category_data(cfg.images, cfg.masks, cfg.category, cfg.into, mode=cfg.mode,
                               test_fraction=cfg.test_fraction, seed=cfg.seed, prefix=cfg.prefix, dry_run=cfg.dry_run)
    print(OmegaConf.to_yaml(OmegaConf.create(report)))
    if report["problems"]:
        raise SystemExit(f"dataset validation found {len(report['problems'])} problem(s), first: {report['problems'][0]}")


if __name__ == "__main__":
    main()
