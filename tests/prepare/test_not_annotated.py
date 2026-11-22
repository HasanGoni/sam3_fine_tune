"""Fair negatives for SAM3: `prepare` flags same-family, not-annotated categories; the loader makes no query for them."""

import json
from pathlib import Path

import yaml

from sam3.train.data.coco_json_loaders import COCO_FROM_JSON

from .conftest import BOX, run_prepare

FAMILIES = {"chip": ["matrix_chip", "gray_void"], "wires": ["thin_wire", "thick_wire"]}


def coco_file(tmp_path, flagged):
    """One 64x64 image annotated for category 1, categories 1-3, `flagged` ids listed as not annotated."""
    image = {"id": 1, "file_name": "a.png", "width": 64, "height": 64}
    if flagged is not None:
        image["not_annotated_category_ids"] = flagged
    d = {
        "images": [image],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [BOX],
                "area": 900.0,
                "bbox": [10, 10, 30, 30],
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": i, "name": n} for i, n in ((1, "a"), (2, "b"), (3, "c"))],
    }
    f = tmp_path / "ann.json"
    f.write_text(json.dumps(d))
    return f


def query_texts(f):
    queries, _ = COCO_FROM_JSON(str(f)).loadQueriesAndAnnotationsFromDatapoint(0)
    return [q["query_text"] for q in queries]


def test_a_flagged_category_gets_no_query_and_the_others_still_do(tmp_path):
    assert query_texts(coco_file(tmp_path, [2])) == ["a", "c"]


def test_without_the_field_every_category_is_queried_as_before(tmp_path):
    assert query_texts(coco_file(tmp_path, None)) == ["a", "b", "c"]


def _prepared_entries(path):
    d = json.loads(
        (Path(path) / "prepared/sam3/train/annotation_merged.json").read_text()
    )
    ids = {c["id"]: c["name"] for c in d["categories"]}
    return {
        i["file_name"]: sorted(ids[c] for c in i.get("not_annotated_category_ids", []))
        for i in d["images"]
    }


def _dataset(root):
    root.add("train", "chip.png", 1, {"matrix_chip": BOX})
    root.add("train", "void.png", 2, {"gray_void": BOX, "matrix_chip": BOX})
    root.add("train", "wire.png", 3, {"thin_wire": BOX})
    root.add("train", "thick.png", 5, {"thick_wire": BOX})
    root.add("train", "pin.png", 4, {"et_pin": BOX})
    return root.write()


def test_without_families_the_sam3_view_has_no_flags(root):
    path = _dataset(root)
    assert run_prepare(path, "val_fraction=0").returncode == 0
    assert set(map(tuple, _prepared_entries(path).values())) == {()}


def test_prepare_flags_same_family_categories_the_image_is_not_annotated_for(root):
    path = _dataset(root)
    assert run_prepare(path, "val_fraction=0").returncode == 0
    recipe = path / "prepared/unveil/recipe.yaml"
    r = yaml.safe_load(recipe.read_text())
    r["families"] = FAMILIES
    recipe.write_text(yaml.safe_dump(r))
    p = run_prepare(path, "val_fraction=0")
    assert p.returncode == 0, p.stderr
    assert _prepared_entries(path) == {
        "chip.png": ["gray_void"],  # same family, not annotated: no query
        "void.png": [],  # annotated for both
        "wire.png": ["thick_wire"],
        "thick.png": ["thin_wire"],
        "pin.png": [],  # in no family: other families are real negatives, not flagged
    }


def test_a_family_naming_an_unknown_category_is_a_problem(root):
    path = _dataset(root)
    assert run_prepare(path, "val_fraction=0").returncode == 0
    recipe = path / "prepared/unveil/recipe.yaml"
    r = yaml.safe_load(recipe.read_text())
    r["families"] = {"chip": ["matrix_chip", "typo_chip"]}
    recipe.write_text(yaml.safe_dump(r))
    p = run_prepare(path, "val_fraction=0")
    assert p.returncode != 0 and "typo_chip" in p.stderr


def test_a_stale_flag_in_the_source_is_not_carried_over(root):
    path = _dataset(root)
    f = path / "train" / "annotation.json"
    d = json.loads(f.read_text())
    d["images"][0]["not_annotated_category_ids"] = [1, 2, 3]
    f.write_text(json.dumps(d))
    assert run_prepare(path, "val_fraction=0").returncode == 0
    assert set(map(tuple, _prepared_entries(path).values())) == {()}


def test_the_report_counts_flagged_images_and_the_manifest_records_the_families(root):
    import yaml as _y

    from .conftest import report

    path = _dataset(root)
    assert run_prepare(path, "val_fraction=0").returncode == 0
    recipe = path / "prepared/unveil/recipe.yaml"
    r = _y.safe_load(recipe.read_text())
    r["families"] = FAMILIES
    recipe.write_text(_y.safe_dump(r))
    p = run_prepare(path, "val_fraction=0")
    assert report(p)["images_with_not_annotated_flags"] == 3
    m = json.loads((path / "prepared/manifest.json").read_text())
    assert m["families"] == FAMILIES


def test_a_family_given_as_a_bare_string_is_a_clear_problem(root):
    path = _dataset(root)
    assert run_prepare(path, "val_fraction=0").returncode == 0
    recipe = path / "prepared/unveil/recipe.yaml"
    r = yaml.safe_load(recipe.read_text())
    r["families"] = {"chip": "matrix_chip"}
    recipe.write_text(yaml.safe_dump(r))
    p = run_prepare(path, "val_fraction=0")
    assert p.returncode != 0 and "must be a list" in p.stderr


def test_a_chunk_whose_categories_are_all_flagged_is_not_a_datapoint(tmp_path):
    f = coco_file(tmp_path, [2])
    api = COCO_FROM_JSON(str(f), category_chunk_size=1)
    ids = api.getDatapointIds()
    texts = [
        [q["query_text"] for q in api.loadQueriesAndAnnotationsFromDatapoint(i)[0]]
        for i in ids
    ]
    assert texts == [["a"], ["c"]]
    assert api.loadImagesFromDatapoint(ids[1])[0]["file_name"] == "a.png"
