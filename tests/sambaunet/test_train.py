import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from sam3_fine_tune.sambaunet.checkpoint import load_model  # noqa: E402
from sam3_fine_tune.sambaunet.metrics import CategoryScores  # noqa: E402

TINY = [
    "model.sam_tiny=true",
    "model.dims=[8,16,32,64]",
    "data.size=64",
    "data.batch_size=2",
    "data.num_workers=0",
    "trainer.device=cpu",
    "trainer.log_every=1",
]


def run_cli(coco_root, run_dir, *extra):
    cmd = [
        sys.executable,
        "-m",
        "sam3_fine_tune.sambaunet.train",
        f"data.root={coco_root}",
        f"hydra.run.dir={run_dir}",
        *TINY,
        *extra,
    ]
    return subprocess.run(
        cmd, capture_output=True, text=True, cwd=Path(__file__).parents[2]
    )


def test_metrics_match_the_sam3_definition():
    s = CategoryScores(["a", "b"])
    gt = np.zeros((2, 4, 4), bool)
    gt[0, :2] = 1
    pred = np.zeros((2, 4, 4), bool)
    pred[0, :1] = 1  # tp=4 fn=4 -> iou .5 ; category b empty in both -> 1.0
    s.update(pred, gt)
    summ = s.summary()
    r = summ["categories"]
    assert (
        r["a"]["overall_foreground_iou"] == 50.0
        and r["a"]["pixel_precision"] == 100.0
        and r["a"]["pixel_recall"] == 50.0
    )
    assert (
        r["b"]["overall_foreground_iou"] == 0.0
        and r["b"]["mean_foreground_iou"] == 100.0
    )  # empty==empty: image scores 1, pixels 0
    assert r["a"]["total_TP_pixels"] == 4 and r["a"]["total_TN_pixels"] == 8
    assert (
        summ["overall"]["foreground_iou"] == 50.0
        and summ["macro_foreground_iou"] == 25.0
    )
    assert summ["overall"].keys() >= {"precision", "recall", "total_TP_pixels"}


def test_metrics_agree_with_sam3_multi_eval_compute_foreground_iou():
    ev = pytest.importorskip("sam3_fine_tune.evaluation.multi_eval_pipe")
    rng = np.random.default_rng(0)
    pred, gt = rng.random((1, 16, 16)) > 0.5, rng.random((1, 16, 16)) > 0.5
    s = CategoryScores(["a"])
    s.update(pred, gt)
    ref = ev.compute_foreground_iou(
        pred[0].astype(np.uint8) * 255, gt[0].astype(np.uint8) * 255
    )
    assert s.details()["a"]["per_image_results"][0]["foreground_iou"] == round(
        ref["foreground_iou"], 4
    )


def test_cli_trains_checkpoints_and_resumes(coco_root, tmp_path):
    run = tmp_path / "run"
    r = run_cli(coco_root, run, "trainer.epochs=2")
    assert r.returncode == 0, r.stderr[-2000:]
    for f in ("best.pt", "last.pt"):
        assert (run / "checkpoints" / f).exists()
    assert (
        run / ".hydra" / "config.yaml"
    ).exists()  # resolved config saved next to the run

    model, ck = load_model(run / "checkpoints" / "best.pt")
    assert ck["category_names"] == ["pin", "bond wire"]
    assert model(torch.rand(1, 3, 64, 64)).shape == (1, 2, 64, 64)
    assert "model" in ck and not any(
        k.startswith("sam.enc.")
        and "down_" not in k
        and "up_" not in k
        and "adapter" not in k
        for k in ck["model"]
    )

    run2 = tmp_path / "run2"
    r = run_cli(
        coco_root,
        run2,
        "trainer.epochs=3",
        f"trainer.resume={run / 'checkpoints' / 'last.pt'}",
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert torch.load(run2 / "checkpoints" / "last.pt", weights_only=True)["epoch"] == 2


def test_cli_override_changes_config(coco_root, tmp_path):
    run = tmp_path / "run"
    assert (
        run_cli(coco_root, run, "trainer.epochs=1", "optim.lr=0.0005").returncode == 0
    )
    assert "lr: 0.0005" in (run / ".hydra" / "config.yaml").read_text()
