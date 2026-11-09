import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

from sam3_fine_tune.inference.cli import run
from sam3_fine_tune.inference.run import resolve_device, run_inference, validate_inputs

REPO = Path(__file__).parents[2]


@pytest.fixture()
def env(tmp_path):
    imgs = tmp_path / "imgs"; imgs.mkdir()
    Image.fromarray(np.full((40, 50, 3), 100, np.uint8)).save(imgs / "a.png")
    Image.fromarray(np.full((30, 30, 3), 50, np.uint8)).save(imgs / "b.jpg")
    (tmp_path / "sam3.pt").write_bytes(b"x"); (tmp_path / "bpe.txt.gz").write_bytes(b"x")
    return tmp_path


def fake_predict(path, prompt):
    h, w = Image.open(path).size[::-1]
    m1 = np.zeros((h, w), bool); m1[2:10, 2:10] = True
    m2 = np.zeros((h, w), bool); m2[5:15, 5:15] = True
    return ([m1, m2], [0.9, 0.7]) if path.name == "a.png" else ([], [])


def cfg(env, **kw):
    base = dict(images=str(env / "imgs"), prompt="pin", out=str(env / "out"), sam3_checkpoint=str(env / "sam3.pt"),
                bpe_path=str(env / "bpe.txt.gz"), checkpoint=None, threshold=0.5, device="cpu")
    return OmegaConf.create({**base, **kw})


def test_writes_masks_overlays_and_summary(env):
    s = run(cfg(env), predict=fake_predict)
    out = env / "out"
    assert sorted(p.name for p in (out / "masks").iterdir()) == ["a.png", "b.png"]
    assert sorted(p.name for p in (out / "overlays").iterdir()) == ["a.png", "b.png"]
    mask = np.asarray(Image.open(out / "masks" / "a.png"))
    assert mask.shape == (40, 50) and set(np.unique(mask)) == {0, 255}
    assert (mask > 0).sum() == 64 + 100 - 25  # union of the two overlapping instances
    assert np.asarray(Image.open(out / "overlays" / "a.png")).shape == (40, 50, 3)
    a, b = json.loads((out / "summary.json").read_text())["images"]
    assert (a["image"], a["instances"], a["foreground_pixels"], a["scores"]) == ("a.png", 2, 139, [0.9, 0.7])
    assert (b["instances"], b["foreground_pixels"]) == (0, 0)
    assert s["model"] == "base" and s["prompt"] == "pin" and s["threshold"] == 0.5 and s["device"] == "cpu"


def test_finetuned_run_records_checkpoint_and_rerun_overwrites(env):
    ck = env / "ft.pt"; ck.write_bytes(b"x")
    run(cfg(env), predict=fake_predict)
    s = run(cfg(env, checkpoint=str(ck), prompt="crack"), predict=fake_predict)
    on_disk = json.loads((env / "out" / "summary.json").read_text())
    assert on_disk["model"] == "fine-tuned" and on_disk["checkpoint"] == str(ck) and on_disk["prompt"] == "crack"
    assert s == on_disk and len(list((env / "out" / "masks").iterdir())) == 2


@pytest.mark.parametrize("kw,msg", [
    (dict(images="nope"), "images folder not found"),
    (dict(sam3_checkpoint="nope.pt"), "sam3_checkpoint"),
    (dict(bpe_path="nope.gz"), "bpe_path"),
    (dict(checkpoint="nope.pt"), r"checkpoint \(fine-tuned model\) not found"),
    (dict(prompt="  "), "prompt"),
    (dict(threshold=0), "threshold"),
])
def test_bad_config_fails_early_with_clear_message(env, kw, msg):
    with pytest.raises(SystemExit, match=msg):
        run(cfg(env, **kw), predict=fake_predict)
    assert not (env / "out").exists()


def test_empty_folder_and_stem_collision(env, tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no images"):
        validate_inputs(images=tmp_path / "empty", prompt="p", sam3_checkpoint=env / "sam3.pt",
                        bpe_path=env / "bpe.txt.gz", checkpoint=None, threshold=0.5)
    Image.fromarray(np.zeros((5, 5, 3), np.uint8)).save(env / "imgs" / "a.jpg")
    with pytest.raises(SystemExit, match="share a file stem"):
        run(cfg(env), predict=fake_predict)


def test_mask_size_mismatch_is_an_error(env, tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        run_inference([env / "imgs" / "a.png"], "p", tmp_path / "o", lambda p, t: ([np.zeros((3, 3), bool)], [1.0]))


def test_resolve_device():
    assert resolve_device("auto", True) == "cuda" and resolve_device("auto", False) == "cpu"
    assert resolve_device("cpu", True) == "cpu"
    with pytest.raises(ValueError, match="no CUDA"):
        resolve_device("cuda", False)
    with pytest.raises(ValueError, match="device must be"):
        resolve_device("tpu", False)


def test_cli_missing_folder_exits_cleanly_without_hydra_outputs(env):
    p = subprocess.run([sys.executable, "-m", "sam3_fine_tune.inference", "images=nope", "prompt=pin",
                        f"out={env / 'out'}", f"sam3_checkpoint={env / 'sam3.pt'}", f"bpe_path={env / 'bpe.txt.gz'}"],
                       capture_output=True, text=True, cwd=env, env={**os.environ, "PYTHONPATH": str(REPO)})
    assert p.returncode != 0 and "images folder not found" in p.stderr
    assert not (env / "outputs").exists() and not (env / ".hydra").exists()


def test_console_script_is_registered_in_pyproject():
    scripts = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["sam3-infer"] == "sam3_fine_tune.inference.cli:main"
