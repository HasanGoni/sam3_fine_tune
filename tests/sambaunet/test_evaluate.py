import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")

from .test_train import run_cli  # noqa: E402


def test_eval_cli_writes_sam3_comparable_summary(coco_root, tmp_path):
    run = tmp_path / "train"
    assert run_cli(coco_root, run, "trainer.epochs=1").returncode == 0
    out = tmp_path / "eval"
    cmd = [
        sys.executable,
        "-m",
        "sam3_fine_tune.sambaunet.evaluate",
        f"checkpoint={run / 'checkpoints' / 'best.pt'}",
        f"data.root={coco_root}",
        f"hydra.run.dir={out}",
        "device=cpu",
        "save_overlays=true",
    ]
    r = subprocess.run(
        cmd, capture_output=True, text=True, cwd=Path(__file__).parents[2]
    )
    assert r.returncode == 0, r.stderr[-2000:]

    data = json.loads((out / "multiclass_foreground_iou_summary.json").read_text())
    cats = data["summary"]["categories"]
    assert list(cats) == ["pin", "bond wire"]
    assert cats["pin"]["total_images"] == 2  # test split has 2 images
    assert {
        "overall_foreground_iou",
        "mean_foreground_iou",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
        "best_sample",
        "worst_sample",
    } <= cats["pin"].keys()
    assert {"foreground_iou", "precision", "recall"} <= data["summary"][
        "overall"
    ].keys()
    assert len(data["per_category_details"]["pin"]["per_image_results"]) == 2
    assert data["meta"]["model"] == "sambaunet" and data["meta"]["trainable_params"] > 0
    assert len(list((out / "overlays").glob("*.png"))) == 2
