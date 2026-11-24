"""Second review round of `sam3-compare`: checkpoint cut-offs, untrusted split files, lazy sources."""

import json
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from sam3_fine_tune.compare.ops import compare, format_table
from sam3_fine_tune.infer_unveil.cli import run as run_infer_unveil
from sam3_fine_tune.inference import run as inference_run
from sam3_fine_tune.inference.run import Predictor
from sam3_fine_tune.train_unveil.checkpoint import make_meta, save_checkpoint
from tests.infer_unveil.test_infer_unveil import (  # noqa: F401
    cfg as infer_cfg,
    env as infer_env,
    StubUnveil as InferStub,
)

from .test_cli import prepared, REPO  # noqa: F401
from .test_unveil import run_cli


def write_checkpoint(path, binarize):
    save_checkpoint(path, {}, make_meta(list(binarize), {}, binarize, {}))
    return path


def test_unveil_columns_use_the_checkpoint_binarize_and_say_so(
    prepared,
    tmp_path,
    monkeypatch,
    capsys,  # noqa: F811
):
    ck = write_checkpoint(tmp_path / "u.safetensors", {"bond wire": 250, "pin": 100})
    out = tmp_path / "r.json"
    models = f"[{{name: u, kind: unveil, gate: false, checkpoint: {ck}}}]"
    stdout = run_cli(monkeypatch, capsys, prepared, models, out)
    result = json.loads(out.read_text())
    m = result["models"]["u"]
    assert m["bond wire"]["iou"] == 0.0  # the stub map is 204/255, under t=250
    assert m["pin"]["iou"] == pytest.approx(1.0)  # over t=100
    cut = result["cutoffs"]["u"]
    assert cut["bond wire"]["source"] == "checkpoint binarize t=250"
    assert cut["crack"]["source"] == "threshold" and cut["crack"]["value"] == 0.5
    assert "cut-off" in stdout and "t=250" in stdout


def test_without_a_checkpoint_the_fixed_threshold_is_used(
    prepared,
    tmp_path,
    monkeypatch,
    capsys,  # noqa: F811
):
    out = tmp_path / "r.json"
    run_cli(monkeypatch, capsys, prepared, "[{name: u, kind: unveil}]", out)
    cut = json.loads(out.read_text())["cutoffs"]["u"]
    assert {c["source"] for c in cut.values()} == {"threshold"}


def test_sam3_infer_unveil_uses_the_checkpoint_binarize_when_no_threshold_is_given(
    infer_env,  # noqa: F811
):
    ck = write_checkpoint(infer_env / "u.safetensors", {"pin": 250})
    s = run_infer_unveil(
        infer_cfg(infer_env, checkpoint=str(ck), threshold=None), model=InferStub()
    )
    assert np.asarray(Image.open(infer_env / "out" / "masks" / "a.png")).sum() == 0
    assert s["cutoffs"]["pin"]["source"] == "checkpoint binarize t=250"
    s = run_infer_unveil(
        infer_cfg(infer_env, checkpoint=str(ck), threshold=0.5), model=InferStub()
    )
    assert s["default_threshold"] == 0.5
    mask = np.asarray(Image.open(infer_env / "out" / "masks" / "a.png"))
    assert (mask > 0).sum() == 64


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
