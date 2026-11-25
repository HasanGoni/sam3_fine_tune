"""Second review round of `sam3-compare`: untrusted split files, lazy sources."""

import json
import subprocess
import sys

import numpy as np
import pytest

from sam3_fine_tune.compare.ops import compare, format_table
from sam3_fine_tune.inference import run as inference_run
from sam3_fine_tune.inference.run import Predictor

from .test_cli import prepared, REPO  # noqa: F401


def edit_split(prepared, edit):
    f = prepared / "prepared/sam3/test/annotation_merged.json"
    coco = json.loads(f.read_text())
    edit(coco)
    f.write_text(json.dumps(coco))


def run_models(prepared, tmp_path):
    spec = f"[{{name: m, kind: predictions, path: {tmp_path}}}]"
    (tmp_path / "masks").mkdir(exist_ok=True)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "sam3_fine_tune.compare",
            f"root={prepared}",
            f"models={spec}",
            f"out={tmp_path / 'r.json'}",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


@pytest.mark.parametrize("bad", ["/etc/hostname", "../x.png"])
def test_a_file_name_outside_the_images_folder_is_rejected(
    prepared,
    tmp_path,
    bad,  # noqa: F811
):
    edit_split(prepared, lambda c: c["images"][0].update(file_name=bad))
    p = run_models(prepared, tmp_path)
    assert p.returncode != 0 and "Traceback" not in p.stderr
    assert bad in p.stderr and "annotation_merged.json" in p.stderr


def test_malformed_records_are_listed_as_problems(prepared, tmp_path):  # noqa: F811
    def edit(c):
        del c["images"][0]["file_name"]
        del c["annotations"][0]["category_id"]

    edit_split(prepared, edit)
    p = run_models(prepared, tmp_path)
    assert p.returncode != 0 and "Traceback" not in p.stderr
    assert "KeyError" not in p.stderr
    assert "file_name" in p.stderr and "category_id" in p.stderr


def test_model_source_segments_only_the_requested_category(
    prepared,
    monkeypatch,  # noqa: F811
):
    asked = []

    def segment(handle, spec):
        asked.append(spec.text)
        return [np.ones((64, 64), bool)], [0.9]

    monkeypatch.setattr(
        inference_run,
        "make_predictor",
        lambda *a, **k: Predictor(lambda p: p, segment),
    )
    result = compare(
        prepared,
        [{"name": "b", "kind": "sam3-base"}],
        sam3_checkpoint="x.pt",
        bpe_path="x.gz",
        device="cpu",
    )
    # bond wire: 3 images, pin: 2 images, crack: none; one segment call per scored (image, category)
    assert len(asked) == 5 and "crack" not in asked
    assert "cut-off" not in format_table(result)
