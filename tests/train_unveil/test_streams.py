"""`sam3-train-unveil show_batches=N`: one stream per recipe category, built from the prepared view."""

import json

import yaml

from tests.prepare.conftest import BOX, Root, run_prepare
from tests.train_unveil.test_validate import run_train_unveil, write_recipe


def prepared_with_unannotated(tmp_path):
    root = Root(tmp_path / "ds")
    for i in range(1, 5):
        root.add("train", f"chip{i}.png", i, {"chip": BOX})
    root.add("train", "wire.png", 10, {"wire": BOX})
    root.add("train", "empty.png", 11, {})  # annotated for nothing: never a stream item
    root.add("test", "t.png", 5, {"chip": BOX})
    path = root.write()
    assert run_prepare(path, "val_fraction=0").returncode == 0
    write_recipe(
        path,
        categories={"chip": {"kind": "region"}, "wire": {"kind": "balanced"}},
        families={"chip_family": ["chip"], "wires": ["wire"]},
    )
    return path


def test_one_stream_per_category_with_the_batch_share(tmp_path):
    path = prepared_with_unannotated(tmp_path)
    p = run_train_unveil(path, "show_batches=3")
    assert p.returncode == 0, p.stderr
    out = yaml.safe_load(p.stdout)
    assert list(out["streams"]) == ["chip", "wire"]
    assert out["streams"]["chip"] == {
        "kind": "region",
        "prompt": 0,
        "per_gpu": 12,  # max(1, 24 // 2)
        "images": 4,
    }
    assert out["streams"]["wire"]["kind"] == "balanced"
    assert out["streams"]["wire"]["prompt"] == 1
    assert len(out["batches"]) == 3
    for step in out["batches"]:
        assert len(step["chip"]) == 12 and len(step["wire"]) == 12  # composition


def test_no_unannotated_image_in_any_stream(tmp_path):
    path = prepared_with_unannotated(tmp_path)
    out = yaml.safe_load(run_train_unveil(path, "show_batches=5").stdout)
    index = path / "prepared" / "unveil" / "index" / "train"
    records = [
        json.loads(line)
        for f in index.glob("*.jsonl")
        for line in f.read_text().splitlines()
    ]
    assert not any("empty" in r["image"] for r in records)
    seen = {i for step in out["batches"] for ids in step.values() for i in ids}
    assert seen and seen <= {r["id"] for r in records}
