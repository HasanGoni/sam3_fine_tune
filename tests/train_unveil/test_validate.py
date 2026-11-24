"""`sam3-train-unveil` (validate only): the recipe and the prepared UNVEIL view are checked before any training."""

import json
import subprocess
import sys

import yaml

from tests.prepare.conftest import BOX, REPO, Root, run_prepare, square

VALID = {
    "categories": {
        "chip": {"kind": "region"},
        "substrate": {"kind": "region"},
        "wire": {"kind": "balanced"},
    },
    "families": {"chip_family": ["chip", "substrate"], "wires": ["wire"]},
    "exclude_unannotated": True,
}


def prepared_root(tmp_path):
    """chip + substrate share an image; wire is on its own; one prepared root with the UNVEIL view."""
    root = Root(tmp_path / "ds")
    root.add("train", "a.png", 1, {"chip": BOX, "substrate": square(4, 4, 60, 60)})
    root.add("train", "b.png", 2, {"wire": BOX})
    root.add("train", "c.png", 3, {"chip": BOX})
    root.add("train", "d.png", 4, {"wire": BOX})
    root.add("test", "t.png", 5, {"chip": BOX})
    path = root.write()
    assert run_prepare(path, "val_fraction=0").returncode == 0
    return path


def write_recipe(path, **changes):
    recipe = json.loads(json.dumps(VALID))
    recipe.update(changes)
    (path / "prepared" / "unveil" / "recipe.yaml").write_text(yaml.safe_dump(recipe))


def run_train_unveil(root, *overrides):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "sam3_fine_tune.train_unveil",
            f"root={root}",
            *overrides,
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


def test_a_valid_recipe_passes_and_reports_the_plan(tmp_path):
    path = prepared_root(tmp_path)
    write_recipe(path)
    p = run_train_unveil(path)
    assert p.returncode == 0, p.stderr
    out = yaml.safe_load(p.stdout)
    assert out["per_gpu"] == 8  # max(1, 24 // 3)
    assert out["train_images"] == {"chip": 2, "substrate": 1, "wire": 2}
    # chip images are verified absent for wire only; wire images for chip and substrate
    assert out["negatives"] == {
        "chip": ["wire"],
        "substrate": ["wire"],
        "wire": ["chip", "substrate"],
    }


def test_a_category_not_in_the_prepared_data_is_rejected(tmp_path):
    path = prepared_root(tmp_path)
    cats = dict(VALID["categories"], ghost={"kind": "region"})
    write_recipe(path, categories=cats)
    p = run_train_unveil(path)
    assert p.returncode != 0
    assert "ghost" in p.stderr and "prompts.json" in p.stderr
    assert "remove it from recipe.yaml" in p.stderr


def test_a_category_without_a_kind_is_rejected_by_name(tmp_path):
    path = prepared_root(tmp_path)
    cats = dict(VALID["categories"], wire={"kind": "???"})
    write_recipe(path, categories=cats)
    p = run_train_unveil(path)
    assert p.returncode != 0
    assert "wire" in p.stderr and "kind" in p.stderr
    assert "region" in p.stderr and "balanced" in p.stderr


def test_a_negative_from_the_same_family_is_rejected(tmp_path):
    path = prepared_root(tmp_path)
    write_recipe(path, negatives={"chip": ["substrate", "wire"]})
    p = run_train_unveil(path)
    assert p.returncode != 0
    assert "chip" in p.stderr and "substrate" in p.stderr
    assert "same family" in p.stderr


def test_unannotated_images_may_not_reach_the_training_index(tmp_path):
    path = prepared_root(tmp_path)
    write_recipe(path, exclude_unannotated=False)
    p = run_train_unveil(path)
    assert p.returncode != 0
    assert "exclude_unannotated" in p.stderr and "true" in p.stderr


def test_an_index_record_without_categories_is_rejected(tmp_path):
    path = prepared_root(tmp_path)
    write_recipe(path)
    f = path / "prepared" / "unveil" / "index" / "train" / "wire.jsonl"
    recs = [json.loads(x) for x in f.read_text().splitlines()]
    recs[0]["categories"] = []
    f.write_text("".join(json.dumps(r) + "\n" for r in recs))
    p = run_train_unveil(path)
    assert p.returncode != 0
    assert "not annotated" in p.stderr and "sam3-prepare" in p.stderr


def test_a_missing_prepared_view_names_the_fix(tmp_path):
    p = run_train_unveil(tmp_path)
    assert p.returncode != 0
    assert "sam3-prepare" in p.stderr


def test_an_empty_sha256_file_is_a_clear_error(tmp_path):
    import torch

    from sam3_fine_tune.train_unveil import checkpoint as C

    ck = tmp_path / "unveil.safetensors"
    C.save_checkpoint(ck, {"heads.w": torch.zeros(2)}, C.make_meta(["a"], {}, {}, {}))
    ck.with_name(ck.name + ".sha256").write_text("")
    import pytest

    with pytest.raises(ValueError, match="sha256"):
        C.read_checkpoint(ck)
