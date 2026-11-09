from pathlib import Path

import pytest
from omegaconf import OmegaConf

from sam3.train.utils.train_utils import register_omegaconf_resolvers

from sam3_fine_tune.finetune.config import (
    BASE_CONFIG,
    build_train_config,
    ensure_merged_annotations,
    find_best_checkpoint,
    validate_dataset,
)

register_omegaconf_resolvers()  # same as train.py, needed to resolve ${times:...} in the base config

KW = dict(sam3_checkpoint="/w/sam3.pt", bpe_path="/w/bpe.txt.gz")


def test_overrides_are_applied_everywhere_they_are_read(tmp_path):
    cfg = build_train_config(
        BASE_CONFIG,
        data_root="/data/ds",
        experiment_dir=tmp_path / "run",
        epochs=3,
        train_batch_size=1,
        val_batch_size=2,
        category_chunk_size=4,
        grad_accum=1,
        accelerator="cpu",
        num_gpus=1,
        **KW,
    )
    r = OmegaConf.to_container(cfg, resolve=True)
    assert r["paths"]["pin_dir_root"] == "/data/ds"
    assert (
        r["launcher"]["experiment_log_dir"]
        == str(tmp_path / "run")
        == r["paths"]["experiment_log_dir"]
    )
    assert (
        r["trainer"]["model"]["checkpoint_path"] == "/w/sam3.pt"
        and r["paths"]["bpe_path"] == "/w/bpe.txt.gz"
    )
    assert r["trainer"]["max_epochs"] == 3 and r["trainer"]["accelerator"] == "cpu"
    assert (
        r["trainer"]["data"]["train"]["batch_size"] == 1
        and r["trainer"]["data"]["val"]["batch_size"] == 2
    )
    assert r["trainer"]["gradient_accumulation_steps"] == 1
    for split in ("train", "val"):
        assert (
            r["trainer"]["data"][split]["dataset"]["coco_json_loader"][
                "category_chunk_size"
            ]
            == 4
        )
    assert (
        r["trainer"]["data"]["train"]["dataset"]["ann_file"]
        == "/data/ds/train/annotation_merged.json"
    )


def test_unset_options_keep_the_base_values(tmp_path):
    base = OmegaConf.to_container(OmegaConf.load(BASE_CONFIG), resolve=False)
    cfg = build_train_config(BASE_CONFIG, data_root="/d", experiment_dir=tmp_path, **KW)
    assert cfg.trainer.max_epochs == base["trainer"]["max_epochs"]
    assert cfg.scratch.resolution == 1008  # fixed by the ViT RoPE, never touched


def test_val_runs_at_least_once_for_short_trainings(tmp_path):
    cfg = build_train_config(
        BASE_CONFIG, data_root="/d", experiment_dir=tmp_path, epochs=2, **KW
    )
    assert cfg.trainer.val_epoch_freq <= 2


def test_validate_dataset_reports_what_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="train/images"):
        validate_dataset(tmp_path)


def test_validate_dataset_accepts_the_standard_layout(coco_root):
    validate_dataset(coco_root)


def test_merged_annotation_is_created_but_never_overwritten(coco_root):
    ensure_merged_annotations(coco_root)
    merged = coco_root / "train" / "annotation_merged.json"
    assert merged.read_text() == (coco_root / "train" / "annotation.json").read_text()
    merged.write_text("{}")
    ensure_merged_annotations(coco_root)
    assert merged.read_text() == "{}"


def test_best_checkpoint_prefers_the_best_ap_then_falls_back(tmp_path):
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    with pytest.raises(FileNotFoundError):
        find_best_checkpoint(tmp_path)
    (ck / "checkpoint.pt").write_text("x")
    assert find_best_checkpoint(tmp_path).name == "checkpoint.pt"
    (ck / "val_pin_detection_coco_eval_segm_AP.pt").write_text(
        "x"
    )  # name the trainer really writes
    assert (
        find_best_checkpoint(tmp_path).name == "val_pin_detection_coco_eval_segm_AP.pt"
    )
