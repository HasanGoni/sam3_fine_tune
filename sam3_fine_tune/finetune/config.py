"""Build a training config from ``full_train_config.yaml`` plus a few CLI overrides, and locate the run's artefacts.

``sam3/train/train.py`` has no ``key=value`` overrides, so the CLI patches a copy of the YAML instead. The base config
keeps its interpolations (``${paths.pin_dir_root}`` ...), so changing ``paths.*`` is enough for most settings.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "sam3" / "train" / "configs"
BASE_CONFIG = CONFIGS_DIR / "full_train_config.yaml"
BEST_CKPT_GLOB = "*val_*segm_AP.pt"  # e.g. val_pin_detection_coco_eval_segm_AP.pt  # kept by the trainer whenever validation AP improves
MERGED_NAME = "annotation_merged.json"  # what the shipped configs read (see CLAUDE.md)


def _set(cfg: DictConfig, key: str, value) -> None:
    if value is not None:
        OmegaConf.update(cfg, key, value, merge=False)


def build_train_config(
    base: Path,
    *,
    data_root,
    experiment_dir,
    sam3_checkpoint,
    bpe_path,
    epochs=None,
    train_batch_size=None,
    val_batch_size=None,
    category_chunk_size=None,
    grad_accum=None,
    accelerator=None,
    num_gpus=None,
) -> DictConfig:
    """Load ``base`` (interpolations preserved) and apply the overrides; ``None`` keeps the base value."""
    cfg = OmegaConf.load(base)
    exp = str(experiment_dir)
    _set(cfg, "paths.pin_dir_root", str(data_root))
    _set(cfg, "paths.experiment_log_dir", exp)
    _set(cfg, "launcher.experiment_log_dir", exp)
    _set(cfg, "paths.bpe_path", str(bpe_path))
    _set(cfg, "trainer.model.checkpoint_path", str(sam3_checkpoint))
    _set(cfg, "trainer.max_epochs", epochs)
    _set(cfg, "trainer.accelerator", accelerator)
    _set(cfg, "launcher.gpus_per_node", num_gpus)
    _set(cfg, "scratch.train_batch_size", train_batch_size)
    _set(cfg, "scratch.val_batch_size", val_batch_size)
    _set(cfg, "scratch.gradient_accumulation_steps", grad_accum)
    for split in (
        "train",
        "val",
    ):  # a GPU-memory lever, see CLAUDE.md "Training config gotchas"
        _set(
            cfg,
            f"trainer.data.{split}.dataset.coco_json_loader.category_chunk_size",
            category_chunk_size,
        )
    if (
        epochs is not None
    ):  # the trainer validates every val_epoch_freq epochs (+ once at the end if it does not divide)
        cfg.trainer.val_epoch_freq = min(int(cfg.trainer.val_epoch_freq), int(epochs))
    return cfg


def validate_dataset(root) -> None:
    """Fail early, before any GPU time is spent, if the dataset is not in the standard layout."""
    root = Path(root)
    for rel in (
        "train/images",
        "test/images",
        "train/annotation.json",
        "test/annotation.json",
    ):
        if not (root / rel).exists():
            raise FileNotFoundError(
                f"dataset layout is <root>/{{train,test}}/{{images,annotation.json}}: missing {root / rel}"
            )


def ensure_merged_annotations(root) -> None:
    """The shipped configs read ``annotation_merged.json``; create it as a copy of ``annotation.json`` if absent."""
    for split in ("train", "test"):
        merged = Path(root) / split / MERGED_NAME
        if not merged.exists():
            shutil.copyfile(Path(root) / split / "annotation.json", merged)


def find_best_checkpoint(experiment_dir) -> Path:
    """Best validation-AP checkpoint if the trainer kept one, otherwise the latest ``checkpoint.pt``."""
    ck = Path(experiment_dir) / "checkpoints"
    best = sorted(ck.glob(BEST_CKPT_GLOB))
    if best:
        return best[0]
    if (ck / "checkpoint.pt").exists():
        return ck / "checkpoint.pt"
    raise FileNotFoundError(f"no checkpoint in {ck} (training did not finish?)")
