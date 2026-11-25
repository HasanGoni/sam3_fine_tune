"""Hydra CLI:  sam3-compare root=... models=[...] [out=compare.json subset=N seed=0]  (see conf/compare.yaml)"""

from __future__ import annotations

from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from .ops import compare, format_table, write_result


@hydra.main(config_path="conf", config_name="compare", version_base="1.3")
def main(cfg: DictConfig) -> None:
    try:
        result = compare(
            root=cfg.root,
            models=cast(list[Any], OmegaConf.to_container(cfg.models)),
            subset=cfg.subset,
            seed=cfg.seed,
            threshold=cfg.threshold,
            sam3_checkpoint=cfg.sam3_checkpoint,
            bpe_path=cfg.bpe_path,
            device=cfg.device,
            trust_checkpoint=cfg.trust_checkpoint,
            suspect_below=cfg.suspect_below,
        )
    except ValueError as e:
        raise SystemExit(f"sam3-compare: {e}") from None
    write_result(result, cfg.out)
    print(format_table(result))
    print(f"wrote {cfg.out}")


if __name__ == "__main__":
    main()
