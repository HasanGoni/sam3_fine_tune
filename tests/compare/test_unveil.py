"""`sam3-compare` UNVEIL columns: the CLI run on a prepared root with a stub UNVEIL model (CPU, stored concept maps)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from sam3_fine_tune.compare.cli import main
from sam3_fine_tune.infer_unveil import cli as infer_unveil

from .test_cli import box, prepared  # noqa: F401  (prepared is a fixture)

# image stem -> presence probability of the stub model (b: the model says "absent" for every category)
PRESENCE = {"a": 0.9, "b": 0.2, "c": 0.9, "d": 0.9}


class StubUnveil:
    """UNVEIL's ``predict`` contract. The concept map is the stored one; the gate empties it when presence < 0.5.
    Bond wire map: exact on a and b, half of d. Pin: exact on a and c. The SAM3 path has no gate."""

    MAPS = {
        ("a", "bond wire"): box(10, 10, 30, 30),
        ("b", "bond wire"): box(0, 0, 20, 20),
        ("d", "bond wire"): box(30, 30, 50, 40),
        ("a", "pin"): box(40, 40, 50, 50),
        ("c", "pin"): box(10, 10, 20, 20),
    }

    def predict(self, image, prompt, gate=True, path="unveil"):
        stem = Path(image).stem
        maps = []
        for text in prompt:
            m = self.MAPS.get((stem, text), box(0, 0, 0, 0)).astype(np.float32) * 0.8
            if gate and path == "unveil" and PRESENCE[stem] < 0.5:
                m[:] = 0
            maps.append(m)
        return maps, [PRESENCE[stem]] * len(prompt)


def run_cli(monkeypatch, capsys, root, models, out):
    monkeypatch.setattr(infer_unveil, "load_unveil", lambda cfg: StubUnveil())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sam3-compare",
            f"root={root}",
            f"models={models}",
            f"out={out}",
            "sam3_checkpoint=x.pt",
            "bpe_path=x.gz",
        ],
    )
    main()
    return capsys.readouterr().out


def test_gated_ungated_and_sam3_path_columns_say_their_gate_setting(
    prepared,
    tmp_path,
    monkeypatch,
    capsys,  # noqa: F811
):
    out = tmp_path / "r.json"
    models = (
        "[{name: gated, kind: unveil, gate: true},"
        " {name: ungated, kind: unveil, gate: false},"
        " {name: sam3path, kind: unveil-sam3-path}]"
    )
    stdout = run_cli(monkeypatch, capsys, prepared, models, out)
    result = json.loads(out.read_text())
    iou = {n: {c: v["iou"] for c, v in m.items()} for n, m in result["models"].items()}
    # bond wire is scored on a, b, d: a exact (400/400), b exact only without the gate, d half (200/400)
    assert iou["ungated"]["bond wire"] == pytest.approx(1000 / 1200)
    assert iou["gated"]["bond wire"] == pytest.approx(600 / 1200)
    assert iou["sam3path"]["bond wire"] == pytest.approx(1000 / 1200)
    # pin is scored on a, c: exact on both, unaffected by the gate
    assert iou["gated"]["pin"] == pytest.approx(1.0)
    assert iou["gated"]["crack"] is None
    assert {n: s["gate"] for n, s in result["settings"].items()} == {
        "gated": "on",
        "ungated": "off",
        "sam3path": "none",
    }
    gate_row = [line for line in stdout.splitlines() if line.startswith("gate")][0]
    assert gate_row.split()[1:] == ["on", "off", "none"]
    assert "gated=on" in stdout and "ungated=off" in stdout


class Resized(StubUnveil):
    """Concept maps at twice the image size, and only the trained spelling ``bond_wire`` is understood."""

    def __init__(self):
        self.prompts = []

    def predict(self, image, prompt, gate=True, path="unveil"):
        self.prompts += list(prompt)
        maps, pres = super().predict(
            image, [t.replace("_", " ") for t in prompt], gate, path
        )
        maps = [
            np.kron(m, np.ones((2, 2), np.float32)) if "bond_wire" in prompt else m * 0
            for m in maps
        ]
        return maps, pres


def test_unveil_is_asked_for_the_trained_spelling_and_maps_are_resized(
    prepared,
    tmp_path,
    monkeypatch,
    capsys,  # noqa: F811
):
    stub = Resized()
    monkeypatch.setattr(infer_unveil, "load_unveil", lambda cfg: stub)
    out = tmp_path / "r.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sam3-compare",
            f"root={prepared}",
            "models=[{name: u, kind: unveil, gate: false, trained_categories: [bond_wire, pin]}]",
            f"out={out}",
            "sam3_checkpoint=x.pt",
            "bpe_path=x.gz",
        ],
    )
    main()
    assert "bond_wire" in stub.prompts and "bond wire" not in stub.prompts
    result = json.loads(out.read_text())
    assert result["models"]["u"]["bond wire"]["iou"] == pytest.approx(1000 / 1200)
