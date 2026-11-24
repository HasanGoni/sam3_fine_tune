"""A full training checkpoint (optimizer state, non-tensor objects) loads for `sam3-compare` without extraction."""

import sys
import types

import pytest
import torch

from sam3_fine_tune.inference.run import make_predictor

from tests.compare.test_cli import prepared  # noqa: F401  (fixture)


class RunState:  # a non-tensor object, as trainers store in checkpoints (weights-only unpickling rejects it)
    epoch = 3


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)


@pytest.fixture()
def fake_sam3(monkeypatch):
    model = TinyModel()
    sam3 = types.ModuleType("sam3")
    sam3.build_sam3_image_model = lambda **kw: model
    proc = types.ModuleType("sam3.model.sam3_image_processor")
    proc.Sam3Processor = lambda m, device: object()
    for name, mod in {
        "sam3": sam3,
        "sam3.model": types.ModuleType("sam3.model"),
        "sam3.model.sam3_image_processor": proc,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return model


@pytest.fixture()
def full_checkpoint(tmp_path):
    src = TinyModel()
    opt = torch.optim.AdamW(src.parameters())
    src.layer(torch.ones(1, 2)).sum().backward()
    opt.step()
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {"model": src.state_dict(), "optimizer": opt.state_dict(), "run": RunState()},
        path,
    )
    return path, src


def test_full_checkpoint_loads_when_trusted_and_not_by_default(
    fake_sam3, full_checkpoint, tmp_path
):
    path, src = full_checkpoint
    with pytest.raises(ValueError, match="not a compatible"):
        make_predictor(tmp_path / "sam3.pt", tmp_path / "bpe.gz", "cpu", path)
    make_predictor(tmp_path / "sam3.pt", tmp_path / "bpe.gz", "cpu", path, trusted=True)
    assert torch.equal(fake_sam3.layer.weight, src.layer.weight)


def test_untrusted_load_error_names_the_fix(fake_sam3, full_checkpoint, tmp_path):
    path, _ = full_checkpoint
    with pytest.raises(ValueError) as e:
        make_predictor(tmp_path / "sam3.pt", tmp_path / "bpe.gz", "cpu", path)
    assert "trust_checkpoint=true" in str(e.value)
    assert "state_dict" in str(e.value)


@pytest.fixture()
def seen_trusted(monkeypatch):
    """Replace make_predictor so the CLI can run without SAM3; records the `trusted` it was given."""
    from sam3_fine_tune.inference import run

    seen = []

    def fake(sam3_checkpoint, bpe_path, device, checkpoint=None, trusted=False):
        seen.append(trusted)
        return run.Predictor(lambda image: None, lambda state, spec: ([], []))

    monkeypatch.setattr(run, "make_predictor", fake)
    return seen


def run_cli(monkeypatch, root, tmp_path, *overrides):
    from sam3_fine_tune.compare.cli import main

    models = f"[{{name: ft, kind: sam3-finetuned, checkpoint: {tmp_path / 'c.pt'}}}]"
    argv = [
        "sam3-compare",
        f"root={root}",
        f"models={models}",
        f"out={tmp_path / 'out.json'}",
        f"sam3_checkpoint={tmp_path / 'sam3.pt'}",
        f"bpe_path={tmp_path / 'bpe.gz'}",
        "device=cpu",
        *overrides,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()


def test_cli_does_not_trust_checkpoints_by_default(
    seen_trusted, prepared, tmp_path, monkeypatch, capsys
):
    run_cli(monkeypatch, prepared, tmp_path)
    assert seen_trusted == [False]
    assert "trust_checkpoint" not in capsys.readouterr().err


def test_cli_trust_checkpoint_true_trusts_and_warns_on_stderr(
    seen_trusted, prepared, tmp_path, monkeypatch, capsys
):
    run_cli(monkeypatch, prepared, tmp_path, "trust_checkpoint=true")
    assert seen_trusted == [True]
    assert "arbitrary code" in capsys.readouterr().err
