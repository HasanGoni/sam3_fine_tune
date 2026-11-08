import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sam3_fine_tune.dataset import ops
from sam3_fine_tune.sambaunet.data import CocoSemanticDataset


def run(raw, into, **kw):
    imgs, masks = raw
    return ops.add_category_data(imgs, masks, "crack", into, **{"test_fraction": 0.25, "seed": 0, **kw})


def load(into, split="train", name="annotation.json"):
    return json.loads((Path(into) / split / name).read_text())


def test_pairs_are_matched_by_stem_across_extensions(raw):
    imgs, masks = raw
    Image.open(imgs / "s0.png").convert("RGB").save(imgs / "s0.jpg"); (imgs / "s0.png").unlink()
    (masks / "s1.png").unlink()                                   # image without mask
    Image.fromarray(np.zeros((5, 5), np.uint8)).save(masks / "extra.png")  # mask without image
    pairs, unmatched = ops.find_pairs(imgs, masks)
    assert sorted(p[0].name for p in pairs) == ["s0.jpg"] + [f"s{i}.png" for i in range(2, 8)]
    assert unmatched == {"images_without_mask": ["s1.png"], "masks_without_image": ["extra.png"]}


def test_fresh_dataset_is_created_split_and_loadable(raw, tmp_path):
    into = tmp_path / "ds"
    rep = run(raw, into)
    assert rep["mode"] == "new" and rep["added"] == {"train": 6, "test": 2}
    tr = load(into)
    assert tr["categories"] == [{"id": 1, "name": "crack"}] and len(tr["images"]) == 6
    assert all((into / "train" / "images" / i["file_name"]).exists() for i in tr["images"])
    assert all(i["file_name"].startswith("crack_") for i in tr["images"])
    ds = CocoSemanticDataset(into / "train")
    assert ds.category_names == ["crack"] and ds.original_mask(0).shape[0] == 1


def test_split_is_deterministic_per_seed(raw, tmp_path):
    a, b, c = (tmp_path / n for n in "abc")
    run(raw, a, seed=1); run(raw, b, seed=1); run(raw, c, seed=2)
    names = lambda p: sorted(i["file_name"] for i in load(p, "test")["images"])
    assert names(a) == names(b) and names(a) != names(c)


def test_test_fraction_zero_puts_everything_in_train(raw, tmp_path):
    rep = run(raw, tmp_path / "ds", test_fraction=0)
    assert rep["added"] == {"train": 8, "test": 0}


def test_new_category_gets_one_id_across_splits_and_keeps_existing_data(raw, coco_root):
    before = {s: load(coco_root, s) for s in ("train", "test")}
    rep = run(raw, coco_root)
    assert rep["mode"] == "new"
    for s in ("train", "test"):
        after = load(coco_root, s)
        assert {c["name"]: c["id"] for c in after["categories"]}["crack"] == 8  # max existing id (7) + 1, same in both
        assert after["images"][: len(before[s]["images"])] == before[s]["images"]
        assert after["annotations"][: len(before[s]["annotations"])] == before[s]["annotations"]
        ids = [a["id"] for a in after["annotations"]]
        assert len(ids) == len(set(ids))


def test_extend_reuses_the_existing_category_id(raw, coco_root):
    imgs, masks = raw
    rep = ops.add_category_data(imgs, masks, "pin", coco_root, test_fraction=0.25, seed=0)
    assert rep["mode"] == "extend"
    cats = load(coco_root)["categories"]
    assert len(cats) == 2 and {c["name"]: c["id"] for c in cats}["pin"] == 3
    assert any(a["category_id"] == 3 and a["image_id"] > 3 for a in load(coco_root)["annotations"])


def test_explicit_mode_conflicts_are_errors(raw, coco_root):
    imgs, masks = raw
    with pytest.raises(ValueError, match="already exists"):
        ops.add_category_data(imgs, masks, "pin", coco_root, mode="new")
    with pytest.raises(ValueError, match="not found"):
        ops.add_category_data(imgs, masks, "crack", coco_root, mode="extend")


def test_rerunning_is_idempotent(raw, tmp_path):
    into = tmp_path / "ds"
    run(raw, into)
    snapshot = load(into)
    rep = run(raw, into)
    assert rep["added"] == {"train": 0, "test": 0} and rep["skipped_existing"] > 0
    assert load(into) == snapshot


def test_dry_run_writes_nothing(raw, tmp_path):
    rep = run(raw, tmp_path / "ds", dry_run=True)
    assert rep["added"] == {"train": 6, "test": 2} and not (tmp_path / "ds").exists()


def test_merged_annotation_is_created_and_kept_in_sync(raw, coco_root):
    from sam3_fine_tune.finetune.config import ensure_merged_annotations
    ensure_merged_annotations(coco_root)
    merged = coco_root / "train" / "annotation_merged.json"
    data = json.loads(merged.read_text()); data["info"] = {"consolidated": True}; merged.write_text(json.dumps(data))
    run(raw, coco_root)
    after = load(coco_root, "train", "annotation_merged.json")
    assert after["info"] == {"consolidated": True}            # a hand-consolidated file is extended, not replaced
    assert any(c["name"] == "crack" for c in after["categories"])
    assert (coco_root / "test" / "annotation_merged.json").exists()


def test_fresh_dataset_gets_a_merged_annotation(raw, tmp_path):
    run(raw, tmp_path / "ds")
    assert load(tmp_path / "ds", "train", "annotation_merged.json") == load(tmp_path / "ds")


def test_mask_and_image_sizes_must_match(raw, tmp_path):
    imgs, masks = raw
    Image.fromarray(np.zeros((10, 10), np.uint8)).save(masks / "s0.png")
    with pytest.raises(ValueError, match="s0.png.*size"):
        run(raw, tmp_path / "ds")
    assert not (tmp_path / "ds").exists()  # validated before anything is written


def test_validate_flags_missing_images_and_duplicate_ids(raw, tmp_path):
    into = tmp_path / "ds"; run(raw, into)
    assert ops.validate_dataset(into) == []
    (into / "train" / "images" / load(into)["images"][0]["file_name"]).unlink()
    assert any("missing image" in p for p in ops.validate_dataset(into))


def test_empty_mask_image_is_kept_without_annotations(raw, tmp_path):
    run(raw, tmp_path / "ds", test_fraction=0)
    d = load(tmp_path / "ds")
    empty = next(i for i in d["images"] if i["file_name"].endswith("s7.png"))
    assert not [a for a in d["annotations"] if a["image_id"] == empty["id"]]


def test_missing_input_folders_are_reported_clearly(raw, tmp_path):
    imgs, masks = raw
    with pytest.raises(FileNotFoundError, match="masks folder not found"):
        ops.add_category_data(imgs, tmp_path / "nope", "crack", tmp_path / "ds")
    with pytest.raises(FileNotFoundError, match="images folder not found"):
        ops.add_category_data(tmp_path / "nope", masks, "crack", tmp_path / "ds")
