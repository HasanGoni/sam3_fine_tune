"""Second review round of `sam3-train-unveil`: one prompt-id order, malformed recipes and records, epochs, devices."""

import json

import numpy as np
import pytest
import torch
import yaml

from sam3_fine_tune.train_unveil import calibrate as cal
from sam3_fine_tune.train_unveil.checkpoint import read_checkpoint
from sam3_fine_tune.train_unveil.trainer import StubModel
from tests.train_unveil.test_train import train, val_root
from tests.train_unveil.test_validate import (
    prepared_root,
    run_train_unveil,
    VALID,
    write_recipe,
)

REORDERED = {
    "categories": {
        "wire": {"kind": "balanced"},
        "chip": {"kind": "region"},
        "substrate": {"kind": "region"},
    },
    "families": {"wires": ["wire"], "chip_family": ["chip", "substrate"]},
}


def test_prompt_ids_follow_the_recipe_order_in_streams_and_checkpoint(tmp_path):
    path = val_root(tmp_path)  # prompts.json order is chip, substrate, wire
    recipe = {**VALID, "epochs": 1, **REORDERED}
    # sort_keys=False: the recipe order is the point
    (path / "prepared/unveil/recipe.yaml").write_text(
        yaml.safe_dump(recipe, sort_keys=False)
    )
    p = run_train_unveil(path)
    streams = yaml.safe_load(p.stdout)["streams"]
    assert {n: s["prompt"] for n, s in streams.items()} == {
        "wire": 0,
        "chip": 1,
        "substrate": 2,
    }
    train(path, tmp_path / "run")
    _, meta = read_checkpoint(tmp_path / "run" / "unveil.safetensors")
    assert meta["prompts"] == ["wire", "chip", "substrate"]


def test_a_subset_recipe_trains_with_ids_of_its_own_categories(tmp_path):
    path = val_root(tmp_path)
    write_recipe(
        path,
        epochs=1,
        categories={"wire": {"kind": "balanced"}},
        families={"w": ["wire"]},
    )
    train(path, tmp_path / "run")  # the old id 2 would be out of range for one category
    _, meta = read_checkpoint(tmp_path / "run" / "unveil.safetensors")
    assert meta["prompts"] == ["wire"]


@pytest.mark.parametrize(
    "changes, words",
    [
        ({"categories": {}}, "categories"),
        ({"negatives": {"chip": None}}, "negatives"),
        ({"negatives": None, "families": {"chip_family": None}}, "families"),
        ({"negatives": ["wire"]}, "negatives"),
    ],
)
def test_empty_or_odd_recipe_values_are_reported_as_problems(tmp_path, changes, words):
    path = prepared_root(tmp_path)
    write_recipe(path, **changes)
    p = run_train_unveil(path)
    assert p.returncode != 0
    assert "Traceback" not in p.stderr and words in p.stderr


@pytest.mark.parametrize("value", ["abc", "0", "-1", "nan", "inf", "true"])
def test_bad_epochs_on_the_command_line_name_the_fix(tmp_path, value):
    path = prepared_root(tmp_path)
    write_recipe(path)
    p = run_train_unveil(path, f"epochs={value}")
    assert p.returncode != 0 and "Traceback" not in p.stderr
    assert "epochs" in p.stderr and "above 0" in p.stderr


def test_nan_epochs_in_the_recipe_is_rejected(tmp_path):
    path = prepared_root(tmp_path)
    write_recipe(path, epochs=float("nan"))
    p = run_train_unveil(path)
    assert p.returncode != 0 and "epochs" in p.stderr


def test_stub_model_builds_its_tensors_on_the_features_device():
    model = StubModel(["a", "b"], ["a", "b"]).to("meta")
    x = torch.zeros(3, 8, 8, device="meta")
    y = torch.zeros(8, 8, device="meta")
    batch = [(x, y, "region", 0, 1.0), (x, None, "balanced", 1, 0.0)]
    assert model.step_loss(batch).device.type == "meta"


def edit_index(path, split, name, edit):
    f = path / "prepared/unveil/index" / split / f"{name}.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines() if x]
    f.write_text("\n".join(json.dumps(edit(dict(r))) for r in rows) + "\n")


@pytest.mark.parametrize(
    "edit, words",
    [
        (lambda r: {**r, "image": "/etc/hostname"}, "/etc/hostname"),
        (lambda r: {**r, "gt": "../../x.png"}, "../../x.png"),
        (lambda r: {k: v for k, v in r.items() if k != "gt"}, "gt"),
    ],
)
def test_train_index_paths_and_keys_are_checked(tmp_path, edit, words):
    path = prepared_root(tmp_path)
    write_recipe(path)
    edit_index(path, "train", "chip", edit)
    p = run_train_unveil(path)
    assert p.returncode != 0 and "Traceback" not in p.stderr
    assert words in p.stderr and "chip" in p.stderr


def test_val_index_paths_are_checked_before_training(tmp_path):
    path = val_root(tmp_path)
    edit_index(path, "val", "wire", lambda r: {**r, "image": "../outside.png"})
    p = run_train_unveil(
        path,
        "validate_only=false",
        "train=true",
        "model=stub",
        "device=cpu",
        "steps=2",
        f"out={tmp_path / 'run'}",
    )
    assert p.returncode != 0 and "Traceback" not in p.stderr
    assert "../outside.png" in p.stderr and "val" in p.stderr


class CountingModel:
    """A fixed map per image; counts predict calls."""

    def __init__(self):
        self.calls = 0

    def eval(self):
        pass

    def predict(self, x, prompt):
        self.calls += 1
        prob = np.zeros((4, 4), np.float32)
        prob[:2] = 0.8  # 204 of 255 on the top half, which is the ground truth
        return prob, 3.0  # presence logit: sigmoid(3) = 0.95


def test_calibrate_runs_each_image_once_and_fits_binarize(tmp_path, monkeypatch):
    path = val_root(tmp_path)
    plan = {
        "categories": ["chip"],
        "kinds": {"chip": "region"},
        "prompts": {"chip": 0},
        "family_of": {"chip": "f"},
        "negatives": {"chip": []},
    }
    gt = torch.zeros(4, 4)
    gt[:2] = 1
    monkeypatch.setattr(
        cal, "read_item", lambda root, rec, size: (torch.zeros(3, 4, 4), gt)
    )
    model = CountingModel()
    _, binarize, report = cal.calibrate(model, plan, path, 4, "cpu")
    n = len(cal.load_val(path, plan)["chip"])
    assert model.calls == n and report["binarize"]["chip"]["n"] == n
    assert 1 <= binarize["chip"] <= 204  # 204 on the ground truth, 0 elsewhere
