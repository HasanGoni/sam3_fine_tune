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
from sam3_fine_tune.inference.run import (
    parse_prompts,
    Predictor,
    PromptSpec,
    resolve_device,
    run_inference,
    validate_inputs,
)

REPO = Path(__file__).parents[2]


@pytest.fixture()
def env(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    Image.fromarray(np.full((40, 50, 3), 100, np.uint8)).save(imgs / "a.png")
    Image.fromarray(np.full((30, 30, 3), 50, np.uint8)).save(imgs / "b.jpg")
    (tmp_path / "sam3.pt").write_bytes(b"x")
    (tmp_path / "bpe.txt.gz").write_bytes(b"x")
    return tmp_path


class CountingPredictor(Predictor):
    """Stub model that counts backbone (encode) runs and records the threshold used per prompt."""

    def __new__(cls, segment=None):
        calls = {"encode": 0, "segment": []}

        def encode(path):
            calls["encode"] += 1
            return path

        def seg(path, spec):
            calls["segment"].append((path.name, spec.text, spec.threshold))
            return (segment or _segment)(path, spec.text)

        self = super().__new__(cls, encode, seg)
        self.calls = calls
        return self


def _segment(path, prompt):
    h, w = Image.open(path).size[::-1]
    m1 = np.zeros((h, w), bool)
    m1[2:10, 2:10] = True
    m2 = np.zeros((h, w), bool)
    m2[5:15, 5:15] = True
    if prompt == "crack":  # a different region, so per-prompt masks differ
        m3 = np.zeros((h, w), bool)
        m3[20:25, 20:25] = True
        return ([m3], [0.8]) if path.name == "a.png" else ([], [])
    return ([m1, m2], [0.9, 0.7]) if path.name == "a.png" else ([], [])


fake_predict = CountingPredictor()


def cfg(env, **kw):
    base = dict(
        images=str(env / "imgs"),
        prompt="pin",
        prompts=None,
        out=str(env / "out"),
        sam3_checkpoint=str(env / "sam3.pt"),
        bpe_path=str(env / "bpe.txt.gz"),
        checkpoint=None,
        threshold=0.5,
        device="cpu",
    )
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
    assert (a["image"], a["instances"], a["foreground_pixels"], a["scores"]) == (
        "a.png",
        2,
        139,
        [0.9, 0.7],
    )
    assert (b["instances"], b["foreground_pixels"]) == (0, 0)
    assert (
        s["model"] == "base"
        and s["prompt"] == "pin"
        and s["threshold"] == 0.5
        and s["device"] == "cpu"
    )


def test_finetuned_run_records_checkpoint_and_rerun_overwrites(env):
    ck = env / "ft.pt"
    ck.write_bytes(b"x")
    run(cfg(env), predict=fake_predict)
    s = run(cfg(env, checkpoint=str(ck), prompt="crack"), predict=fake_predict)
    on_disk = json.loads((env / "out" / "summary.json").read_text())
    assert (
        on_disk["model"] == "fine-tuned"
        and on_disk["checkpoint"] == str(ck)
        and on_disk["prompt"] == "crack"
    )
    assert s == on_disk and len(list((env / "out" / "masks").iterdir())) == 2


@pytest.mark.parametrize(
    "kw,msg",
    [
        (dict(images="nope"), "images folder not found"),
        (dict(sam3_checkpoint="nope.pt"), "sam3_checkpoint"),
        (dict(bpe_path="nope.gz"), "bpe_path"),
        (dict(checkpoint="nope.pt"), r"checkpoint \(fine-tuned model\) not found"),
        (dict(prompt="  "), "prompt"),
        (dict(threshold=0), "threshold"),
    ],
)
def test_bad_config_fails_early_with_clear_message(env, kw, msg):
    with pytest.raises(SystemExit, match=msg):
        run(cfg(env, **kw), predict=fake_predict)
    assert not (env / "out").exists()


def test_empty_folder_and_stem_collision(env, tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no images"):
        validate_inputs(
            images=tmp_path / "empty",
            prompt="p",
            prompts=None,
            sam3_checkpoint=env / "sam3.pt",
            bpe_path=env / "bpe.txt.gz",
            checkpoint=None,
            threshold=0.5,
        )
    Image.fromarray(np.zeros((5, 5, 3), np.uint8)).save(env / "imgs" / "a.jpg")
    with pytest.raises(SystemExit, match="share a file stem"):
        run(cfg(env), predict=fake_predict)


def test_mask_size_mismatch_is_an_error(env, tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        run_inference(
            [env / "imgs" / "a.png"],
            [PromptSpec("p", 0.5)],
            tmp_path / "o",
            CountingPredictor(lambda p, t: ([np.zeros((3, 3), bool)], [1.0])),
        )


def test_resolve_device():
    assert (
        resolve_device("auto", True) == "cuda"
        and resolve_device("auto", False) == "cpu"
    )
    assert resolve_device("cpu", True) == "cpu"
    with pytest.raises(ValueError, match="no CUDA"):
        resolve_device("cuda", False)
    with pytest.raises(ValueError, match="device must be"):
        resolve_device("tpu", False)


def test_cli_missing_folder_exits_cleanly_without_hydra_outputs(env):
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "sam3_fine_tune.inference",
            "images=nope",
            "prompt=pin",
            f"out={env / 'out'}",
            f"sam3_checkpoint={env / 'sam3.pt'}",
            f"bpe_path={env / 'bpe.txt.gz'}",
        ],
        capture_output=True,
        text=True,
        cwd=env,
        env={**os.environ, "PYTHONPATH": str(REPO)},
    )
    assert p.returncode != 0 and "images folder not found" in p.stderr
    assert not (env / "outputs").exists() and not (env / ".hydra").exists()


def test_console_script_is_registered_in_pyproject():
    scripts = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["sam3-infer"] == "sam3_fine_tune.inference.cli:main"


PROMPTS = [
    {"text": "pin", "threshold": 0.3},
    {"type": "text", "text": "crack", "threshold": 0.8},
    {"text": "bond wire"},
]


def test_several_prompts_write_per_prompt_masks_overlay_and_summary(env):
    pred = CountingPredictor()
    s = run(cfg(env, prompt=None, prompts=PROMPTS, threshold=0.5), predict=pred)
    out = env / "out"
    assert sorted(p.name for p in (out / "masks").iterdir()) == [
        "bond_wire",
        "crack",
        "pin",
    ]
    pin = np.asarray(Image.open(out / "masks" / "pin" / "a.png"))
    crack = np.asarray(Image.open(out / "masks" / "crack" / "a.png"))
    assert (pin > 0).sum() == 139 and (crack > 0).sum() == 25
    # overlay carries one colour per prompt: crack (only at 22,22) is green; pin+bond wire overlap at 3,3 (red then blue)
    ov = np.asarray(Image.open(out / "overlays" / "a.png")).astype(int)
    assert ov[22, 22, 1] > 100 and ov[22, 22, 0] == 50 and ov[22, 22, 2] == 50
    assert (
        ov[3, 3, 0] > 50 and ov[3, 3, 2] > ov[3, 3, 1]
    )  # red and blue both blended in
    a, _ = s["images"]
    assert [(p["prompt"], p["threshold"], p["instances"]) for p in a["prompts"]] == [
        ("pin", 0.3, 2),
        ("crack", 0.8, 1),
        ("bond wire", 0.5, 2),
    ]
    assert a["prompts"][0]["foreground_pixels"] == 139
    assert [p["type"] for p in s["prompts"]] == ["text"] * 3
    assert json.loads((out / "summary.json").read_text()) == s


def test_per_prompt_threshold_overrides_default_and_unset_uses_default(env):
    pred = CountingPredictor()
    run(cfg(env, prompt=None, prompts=PROMPTS, threshold=0.5), predict=pred)
    used = {(t, thr) for n, t, thr in pred.calls["segment"] if n == "a.png"}
    assert used == {("pin", 0.3), ("crack", 0.8), ("bond wire", 0.5)}


def test_image_backbone_runs_once_per_image_regardless_of_prompt_count(env):
    pred = CountingPredictor()
    run(cfg(env, prompt=None, prompts=PROMPTS), predict=pred)
    assert pred.calls["encode"] == 2  # 2 images, 3 prompts
    assert len(pred.calls["segment"]) == 6


@pytest.mark.parametrize(
    "prompts,msg",
    [
        ([], "no prompts"),
        (
            [{"text": "pin"}, {"text": "pin", "threshold": 0.9}],
            "duplicate prompt texts",
        ),
        (
            [{"text": "pin"}, {"text": "crack", "threshold": 1.5}],
            r"prompts\[1\] \('crack'\) threshold",
        ),
        ([{"text": "pin", "threshold": 0}], r"prompts\[0\].*threshold"),
        ([{"text": "pin"}, {"text": " "}], r"prompts\[1\]: text"),
        ([{"text": "pin", "type": "box"}], r"prompts\[0\]: unsupported type 'box'"),
        ([{"text": "pin", "thresh": 0.2}], r"prompts\[0\]: unknown key"),
        ([{"text": "a b"}, {"text": "a-b"}], "duplicate prompt slugs"),
    ],
)
def test_invalid_prompt_config_fails_before_model_load_naming_the_entry(
    env, prompts, msg
):
    # predict=None would load the real model; validation must raise before that
    with pytest.raises(SystemExit, match=msg):
        run(cfg(env, prompt=None, prompts=prompts))
    assert not (env / "out").exists()


def test_prompt_and_prompts_together_is_an_error(env):
    with pytest.raises(SystemExit, match="not both"):
        run(cfg(env, prompts=PROMPTS))


def test_text_type_is_the_default_and_existing_shape_is_unchanged():
    spec = parse_prompts(None, [{"text": "pin"}], 0.4)[0]
    assert (spec.type, spec.text, spec.threshold) == ("text", "pin", 0.4)
    assert parse_prompts("pin", None, 0.4) == [spec]
