"""One command for SAM3 fine-tuning: train, then evaluate the best checkpoint on the test split.

    python -m sam3_fine_tune.finetune --data-root /path/to/dataset --sam3-checkpoint sam3.pt --bpe-path bpe.txt.gz \
        --epochs 10 [--skip-eval | --eval-only] [--accelerator auto|cuda|cpu]

``--data-root`` is the COCO dataset produced by the conversion/merge steps (``<root>/{train,test}/{images,annotation.json}``).
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

from .config import (
    BASE_CONFIG,
    build_train_config,
    CONFIGS_DIR,
    ensure_merged_annotations,
    find_best_checkpoint,
    MERGED_NAME,
    REPO_ROOT,
    validate_dataset,
)


def run_training(cfg_path: Path, num_gpus: int, accelerator: str) -> None:
    """Run the existing trainer. It composes its config by *name* from ``sam3/train/configs``, hence ``cfg_path`` lives there."""
    cmd = [
        sys.executable,
        "train.py",
        "-c",
        f"configs/{Path(cfg_path).name}",
        "--num-gpus",
        str(num_gpus),
        "--num-nodes",
        "1",
        "--accelerator",
        accelerator,
    ]
    env = {
        **os.environ,
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        ),
    }
    subprocess.run(cmd, cwd=REPO_ROOT / "sam3" / "train", env=env, check=True)


def run_evaluation(
    *,
    checkpoint: Path,
    split_dir: Path,
    bpe_path: Path,
    out_dir: Path,
    accelerator: str,
    threshold: float,
    max_images: int | None,
) -> dict:
    import torch

    from .coco_eval import evaluate_coco, make_sam3_predictor

    device = (
        "cuda"
        if accelerator == "auto" and torch.cuda.is_available()
        else "cpu"
        if accelerator == "auto"
        else accelerator
    )
    predict = make_sam3_predictor(checkpoint, bpe_path, device, threshold)
    return evaluate_coco(
        split_dir,
        MERGED_NAME,
        predict,
        out_dir,
        max_images,
        meta={
            "checkpoint": Path(checkpoint).name,
            "threshold": threshold,
            "device": device,
        },
    )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m sam3_fine_tune.finetune", description=__doc__.split("\n\n")[0]
    )
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument(
        "--sam3-checkpoint",
        required=True,
        type=Path,
        help="base sam3.pt to fine-tune from",
    )
    p.add_argument(
        "--bpe-path", required=True, type=Path, help="bpe_simple_vocab_16e6.txt.gz"
    )
    p.add_argument(
        "--experiment-dir", type=Path, help="default: experiments/sam3_<timestamp>"
    )
    p.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    p.add_argument("--epochs", type=int)
    p.add_argument("--train-batch-size", type=int)
    p.add_argument("--val-batch-size", type=int)
    p.add_argument(
        "--category-chunk-size",
        type=int,
        help="GPU-memory lever; see CLAUDE.md before raising it",
    )
    p.add_argument(
        "--grad-accum",
        type=int,
        help="num_gpus x train_batch_size x grad_accum must divide the train set size",
    )
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--accelerator", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="confidence threshold for evaluation",
    )
    p.add_argument("--max-eval-images", type=int)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--skip-eval", action="store_true", help="train only")
    mode.add_argument(
        "--eval-only",
        action="store_true",
        help="evaluate an existing --experiment-dir, no training",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    validate_dataset(a.data_root)
    exp = (
        a.experiment_dir
        or REPO_ROOT / "experiments" / f"sam3_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    )
    exp = Path(exp).resolve()

    if not a.eval_only:
        ensure_merged_annotations(a.data_root)
        cfg = build_train_config(
            a.base_config,
            data_root=a.data_root.resolve(),
            experiment_dir=exp,
            sam3_checkpoint=a.sam3_checkpoint.resolve(),
            bpe_path=a.bpe_path.resolve(),
            epochs=a.epochs,
            train_batch_size=a.train_batch_size,
            val_batch_size=a.val_batch_size,
            category_chunk_size=a.category_chunk_size,
            grad_accum=a.grad_accum,
            accelerator=a.accelerator,
            num_gpus=a.num_gpus,
        )
        exp.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, exp / "config_used.yaml")
        gen = CONFIGS_DIR / f"_gen_{exp.name}.yaml"
        OmegaConf.save(cfg, gen)
        try:
            run_training(gen, a.num_gpus, a.accelerator)
        finally:
            gen.unlink(missing_ok=True)

    if a.skip_eval:
        return 0
    ensure_merged_annotations(a.data_root)
    ckpt = find_best_checkpoint(exp)
    result = run_evaluation(
        checkpoint=ckpt,
        split_dir=a.data_root / "test",
        bpe_path=a.bpe_path,
        out_dir=exp / "eval",
        accelerator=a.accelerator,
        threshold=a.threshold,
        max_images=a.max_eval_images,
    )
    print(
        f"test foreground IoU (all categories): {result['summary']['overall']['foreground_iou']} -> {exp / 'eval'}"
    )
    return 0
