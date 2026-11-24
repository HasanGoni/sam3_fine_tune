"""Second review round of `sam3-train-unveil`: malformed recipes and records, epochs."""

import json

import pytest
import torch
import yaml

from sam3_fine_tune.train_unveil.trainer import StubModel
from tests.train_unveil.test_validate import (
    prepared_root,
    run_train_unveil,
    write_recipe,
)


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
