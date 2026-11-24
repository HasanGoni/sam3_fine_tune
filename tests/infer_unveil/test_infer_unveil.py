"""`sam3-infer-unveil` on CPU with a stub UNVEIL model: output layout, the gate option and the SAM3 semantic path."""

import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

from sam3_fine_tune.infer_unveil.cli import run
from sam3_fine_tune.inference.cli import run as run_sam3_infer
from sam3_fine_tune.inference.run import Predictor

PRESENCE = {
    "a.png": 0.9,
    "b.jpg": 0.2,
}  # b has a concept map but the model says "absent"


class StubUnveil:
    """Same ``predict`` contract as ``unveil.model.unveil.UNVEIL.predict`` for a list of phrases."""

    def __init__(self):
        self.calls = []

    def predict(self, image, prompt, gate=True, path="unveil"):
        self.calls.append((Path(image).name, list(prompt), gate, path))
        size = Image.open(image).size[::-1]
        maps, pres = [], []
        for _ in prompt:
            m = np.zeros(size, np.float32)
            m[2:10, 2:10] = 0.8
            p = PRESENCE[Path(image).name]
            if gate and path == "unveil" and p < 0.5:
                m[:] = 0
            maps.append(m)
            pres.append(p)
        return maps, pres


@pytest.fixture()
def env(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    Image.fromarray(np.full((40, 50, 3), 100, np.uint8)).save(imgs / "a.png")
    Image.fromarray(np.full((30, 30, 3), 50, np.uint8)).save(imgs / "b.jpg")
    (tmp_path / "sam3.pt").write_bytes(b"x")
    (tmp_path / "bpe.txt.gz").write_bytes(b"x")
    return tmp_path


def cfg(env, **kw):
    base = dict(
        images=str(env / "imgs"),
        prompt="pin",
        prompts=None,
        out=str(env / "out"),
        sam3_checkpoint=str(env / "sam3.pt"),
        bpe_path=str(env / "bpe.txt.gz"),
        checkpoint=None,
        coco_split=None,
        coco_ann="annotation.json",
        threshold=0.5,
        device="cpu",
        gate=True,
        path="unveil",
    )
    return OmegaConf.create({**base, **kw})


def tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def test_output_layout_equals_sam3_infer(env):
    prompts = [{"text": "pin"}, {"text": "crack"}]
    reference = Predictor(
        lambda p: p,
        lambda p, s: ([np.zeros(Image.open(p).size[::-1], bool)], [0.9]),
    )
    ref_cfg = cfg(env, out=str(env / "ref"), prompt=None, prompts=prompts)
    run_sam3_infer(ref_cfg, predictor=reference)

    summary = run(cfg(env, prompt=None, prompts=prompts), model=StubUnveil())

    assert tree(env / "out") == tree(env / "ref")
    mask = np.asarray(Image.open(env / "out" / "masks" / "pin" / "a.png"))
    assert mask.shape == (40, 50) and (mask > 0).sum() == 64
    assert json.loads((env / "out" / "summary.json").read_text()) == summary
    assert summary["model"] == "unveil" and summary["gate"] is True


def test_gate_false_returns_the_concept_map_without_the_presence_check(env):
    run(cfg(env, gate=True), model=StubUnveil())
    gated = np.asarray(Image.open(env / "out" / "masks" / "b.png"))
    assert gated.sum() == 0
    s = run(cfg(env, gate=False), model=StubUnveil())
    ungated = np.asarray(Image.open(env / "out" / "masks" / "b.png"))
    assert (ungated > 0).sum() == 64 and s["gate"] is False


def test_sam3_semantic_path_is_passed_to_unveil_and_recorded(env):
    stub = StubUnveil()
    s = run(cfg(env, path="sam3"), model=stub)
    assert {c[3] for c in stub.calls} == {"sam3"} and s["path"] == "sam3"


def test_threshold_binarizes_the_concept_map(env):
    run(cfg(env, threshold=0.7), model=StubUnveil())  # a's map is 0.8
    assert (np.asarray(Image.open(env / "out" / "masks" / "a.png")) > 0).sum() == 64
    run(cfg(env, threshold=0.9), model=StubUnveil())
    assert (np.asarray(Image.open(env / "out" / "masks" / "a.png")) > 0).sum() == 0


@pytest.mark.parametrize(
    "kw,msg",
    [
        (dict(images="nope"), "images folder not found"),
        (dict(bpe_path="nope.gz"), "bpe_path"),
        (dict(path="x"), "path must be"),
        (
            dict(prompt=None, prompts=[{"text": "pin", "threshold": [0.3, 0.5]}]),
            "sweep",
        ),
    ],
)
def test_bad_config_fails_early_with_clear_message(env, kw, msg):
    with pytest.raises(SystemExit, match=msg):
        run(cfg(env, **kw), model=StubUnveil())


def test_without_cuda_the_real_model_is_refused_with_the_fix(env, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="CUDA"):
        run(cfg(env, device="cpu"))


def test_the_resolved_device_reaches_the_model_loader(env, monkeypatch):
    import torch

    from sam3_fine_tune.infer_unveil import cli

    seen = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        cli, "load_unveil", lambda c: seen.append(c.device) or StubUnveil()
    )
    run(cfg(env, device="auto"))
    assert seen == ["cpu"]


def test_a_concept_map_of_another_size_is_resized_to_the_image(env):
    class Small(StubUnveil):
        def predict(self, image, prompt, gate=True, path="unveil"):
            return [np.full((5, 5), 0.8, np.float32) for _ in prompt], [0.9] * len(
                prompt
            )

    from sam3_fine_tune.infer_unveil.cli import make_predictor
    from sam3_fine_tune.inference.run import PromptSpec

    spec = PromptSpec("pin", (0.5,))
    predictor = make_predictor(Small(), [spec], True, "unveil")
    masks, _ = predictor.segment(predictor.encode(env / "imgs" / "a.png"), spec)
    assert masks[0].shape == (40, 50)
    assert masks[0].all()
