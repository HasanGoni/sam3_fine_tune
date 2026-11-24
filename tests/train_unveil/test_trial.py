"""`sam3-train-unveil` trial run and joint module, on a tiny prepared root with the stub model."""

import pytest
import torch
import yaml

from tests.train_unveil.test_validate import (
    prepared_root,
    run_train_unveil,
    write_recipe,
)

TRIAL = (
    "validate_only=false",
    "model=stub",
    "trial_steps=6",
    "trial_size=64",
    "device=cpu",
)


def trial(path, *extra):
    p = run_train_unveil(path, *TRIAL, *extra)
    assert p.returncode == 0, p.stderr
    return yaml.safe_load(p.stdout)["trial"]


def test_the_trial_run_prints_step_time_and_peak_memory(tmp_path):
    path = prepared_root(tmp_path)
    write_recipe(path)
    t = trial(path)
    assert t["steps"] == 6
    assert t["step_ms"] > 0
    assert t["peak_memory_gb"] > 0
    assert t["device"] == "cpu"


def test_without_a_joint_pair_the_joint_module_is_skipped(tmp_path):
    path = prepared_root(tmp_path)
    write_recipe(path)
    assert trial(path)["joint"] is None


def test_a_configured_joint_pair_switches_the_module_on(tmp_path):
    path = prepared_root(tmp_path)
    write_recipe(path, joint=["chip", "substrate"])
    with_joint = trial(path)
    write_recipe(path)
    without = trial(path)
    assert with_joint["joint"] == ["chip", "substrate"]
    assert with_joint["trainable_params"] > without["trainable_params"]


def test_a_joint_pair_must_be_two_different_recipe_categories(tmp_path):
    path = prepared_root(tmp_path)
    for bad in (["chip"], ["chip", "chip"], ["chip", "ghost"]):
        write_recipe(path, joint=bad)
        p = run_train_unveil(path)
        assert p.returncode != 0
        assert "joint" in p.stderr and "recipe.yaml" in p.stderr


def test_a_calibration_option_is_rejected_because_nothing_uses_it(tmp_path):
    path = prepared_root(tmp_path)
    write_recipe(path, calibration={"fraction": 0.5})
    p = run_train_unveil(path)
    assert p.returncode != 0
    assert "calibration" in p.stderr and "remove" in p.stderr


def test_no_train_image_is_held_out(tmp_path):
    path = prepared_root(tmp_path)
    write_recipe(path)
    out = yaml.safe_load(run_train_unveil(path).stdout)
    assert out["train_images"] == {"chip": 2, "substrate": 1, "wire": 2}
    assert "calibration" not in out


def test_a_missing_train_index_names_sam3_prepare(tmp_path):
    path = prepared_root(tmp_path)
    write_recipe(path)
    (path / "prepared/unveil/index/train/wire.jsonl").unlink()
    p = run_train_unveil(path)
    assert p.returncode != 0
    assert "sam3-prepare" in p.stderr and "wire" in p.stderr
    assert "FileNotFoundError" not in p.stderr


def test_trial_steps_must_be_positive(tmp_path):
    path = prepared_root(tmp_path)
    write_recipe(path)
    p = run_train_unveil(path, *TRIAL[:2], "trial_steps=0", *TRIAL[3:])
    assert p.returncode != 0
    assert "trial_steps" in p.stderr and "above 0" in p.stderr


@pytest.mark.parametrize("bad", ["0.5", "0", "-1", "1.5"])
def test_accumulate_must_be_a_positive_integer(tmp_path, bad):
    path = prepared_root(tmp_path)
    write_recipe(path, accumulate=float(bad))
    p = run_train_unveil(path)
    assert p.returncode != 0
    assert "accumulate" in p.stderr and "whole number" in p.stderr


def test_the_stub_joint_target_follows_the_configured_pair():
    from sam3_fine_tune.train_unveil.trainer import StubModel

    m = StubModel(["a", "b", "c"], ["a", "b"])
    x, y = torch.zeros(3, 8, 8), torch.ones(8, 8)
    batch = [
        (x, y, "region", 0, 1.0),
        (x, y, "region", 1, 1.0),
        (x, y, "region", 2, 1.0),
        (x, None, "region", 0, 0.0),
    ]
    target = m.joint_target(batch)
    assert target.tolist() == [[1, 0], [0, 1], [0, 0], [0, 0]]


def test_the_stub_joint_head_survives_a_batch_without_positives():
    from sam3_fine_tune.train_unveil.trainer import StubModel

    m = StubModel(["a", "b"], ["a", "b"])
    loss = m.step_loss([(torch.zeros(3, 8, 8), None, "region", 0, 0.0)])
    assert torch.isfinite(loss)
    loss.backward()
