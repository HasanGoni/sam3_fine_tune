"""`sam3-prepare` also writes the UNVEIL view: per-(image, category) masks, per-split indices, prompts, recipe skeleton."""

import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from .conftest import BOX, run_prepare, square, view

PATH_FIELDS = ("image", "gt")  # the fields UNVEIL's catalog joins with the data root


def prepared(path):
    return Path(path) / "prepared"


def records(path, split, category_slug):
    f = prepared(path) / "unveil" / "index" / split / f"{category_slug}.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()]


def unveil_open(rec, root):
    """What UNVEIL's `catalog._join` + `transforms.open_item` + `region_label` do with one index record."""
    rec = dict(rec)
    for k in PATH_FIELDS:
        rec[k] = str(Path(root) / rec[k])
    img = Image.open(rec["image"]).convert("RGB")
    mask = Image.open(rec["gt"])
    label = ((np.asarray(mask.convert("L")) >= 128) * 255).astype(np.uint8)
    return img, label


def nested(root):
    root.add(
        "train",
        "a.png",
        1,
        {
            "substrate": square(4, 4, 60, 60),
            "chip": square(10, 10, 50, 50),
            "lifted_chip": square(20, 20, 40, 40),
        },
    )
    root.add("train", "b.png", 2, {"chip": BOX})
    root.add("train", "c.png", 3, {"substrate": square(4, 4, 60, 60)})
    root.add("test", "t.png", 4, {"chip": BOX})
    return root.write()


def test_index_records_open_with_unveils_loader_code(root):
    path = nested(root)
    assert run_prepare(path, "val_fraction=0").returncode == 0
    seen = 0
    for split in ("train", "test"):
        for f in (prepared(path) / "unveil" / "index" / split).glob("*.jsonl"):
            for rec in records(path, split, f.stem):
                img, label = unveil_open(rec, path)
                assert (img.width, img.height) == (rec["w"], rec["h"])
                assert label.shape == (rec["h"], rec["w"])
                assert set(np.unique(label)) <= {0, 255}
                seen += 1
    assert seen == 6


def test_nested_categories_keep_their_full_extent(root):
    path = nested(root)
    assert run_prepare(path, "val_fraction=0").returncode == 0
    polygons = {
        "substrate": square(4, 4, 60, 60),
        "chip": square(10, 10, 50, 50),
        "lifted_chip": square(20, 20, 40, 40),
    }
    for cat, poly in polygons.items():
        rec = next(
            r
            for r in records(path, "train", cat)
            if r["id"] is not None
            and cat in r["categories"]
            and len(r["categories"]) == 3
        )
        _, label = unveil_open(rec, path)
        want = Image.new("L", (64, 64), 0)
        ImageDraw.Draw(want).polygon(poly, fill=255)
        a, b = label > 0, np.asarray(want) > 0
        assert (a & b).sum() / (a | b).sum() >= 0.99, cat


def test_the_view_lists_exactly_the_images_of_the_sam3_view(root):
    for i in range(1, 11):
        root.add(
            "train", f"p{i}.png", i, {"chip": BOX} | ({"crack": BOX} if i % 2 else {})
        )
    root.add("test", "t.png", 50, {"crack": BOX})
    path = root.write()
    assert run_prepare(path, "val_fraction=0.2").returncode == 0
    for split in ("train", "val", "test"):
        listed = set()
        for f in (prepared(path) / "unveil" / "index" / split).glob("*.jsonl"):
            listed |= {Path(r["image"]).name for r in records(path, split, f.stem)}
        assert listed == set(view(path, split)), split


def test_index_records_carry_the_images_annotated_categories(root):
    path = nested(root)
    assert run_prepare(path, "val_fraction=0").returncode == 0
    by_name = {Path(r["image"]).name: r for r in records(path, "train", "chip")}
    assert by_name["a.png"]["categories"] == ["chip", "lifted_chip", "substrate"]
    assert by_name["b.png"]["categories"] == ["chip"]
    assert (
        "c.png" not in by_name
    )  # not annotated for chip: no positive, and no negative either


def test_prompts_are_the_category_names_in_id_order(root):
    path = nested(root)
    assert run_prepare(path, "val_fraction=0").returncode == 0
    prompts = json.loads((prepared(path) / "unveil" / "prompts.json").read_text())
    assert prompts["task"] == ["chip", "lifted_chip", "substrate"]


def test_category_names_with_odd_characters_stay_inside_the_view(root):
    root.add("train", "a.png", 1, {"thin/wire ../HD": BOX})
    path = root.write()
    assert run_prepare(path, "val_fraction=0").returncode == 0
    prompts = json.loads((prepared(path) / "unveil" / "prompts.json").read_text())
    slug = prompts["slugs"]["thin/wire ../HD"]
    assert "/" not in slug and ".." not in slug
    assert records(path, "train", slug)


def test_recipe_skeleton_is_written_and_never_overwritten(root):
    path = nested(root)
    assert run_prepare(path, "val_fraction=0").returncode == 0
    f = prepared(path) / "unveil" / "recipe.yaml"
    recipe = yaml.safe_load(f.read_text())
    assert recipe["categories"] == {
        c: {"kind": "???"} for c in ("chip", "lifted_chip", "substrate")
    }
    assert recipe["families"] == {} and recipe["exclude_unannotated"] is True
    f.write_text(f.read_text() + "\n# my edit\n")
    edited = f.read_text()
    assert run_prepare(path, "val_fraction=0").returncode == 0
    assert f.read_text() == edited


def test_dry_run_writes_no_unveil_files(root):
    path = nested(root)
    assert run_prepare(path, "dry_run=true").returncode == 0
    assert not prepared(path).exists()


def test_a_kept_recipe_missing_a_new_category_is_warned_about(root):
    path = nested(root)
    assert run_prepare(path, "val_fraction=0").returncode == 0
    root.add("train", "d.png", 9, {"newcat": BOX})
    root.write()
    p = run_prepare(path, "val_fraction=0")
    assert p.returncode == 0, p.stderr
    assert "newcat" in p.stderr and "recipe.yaml" in p.stderr


def test_categories_whose_file_names_would_collide_get_distinct_folders(root):
    root.add("train", "a.png", 1, {"a!": BOX, "a": BOX, "A": BOX, "a_2": BOX})
    path = root.write()
    assert run_prepare(path, "val_fraction=0").returncode == 0
    slugs = json.loads((prepared(path) / "unveil" / "prompts.json").read_text())[
        "slugs"
    ]
    assert len({s.lower() for s in slugs.values()}) == 4


def test_an_rle_annotation_is_a_problem_not_a_crash(root):
    root.add("train", "a.png", 1, {"chip": BOX})
    path = root.write()
    f = path / "train" / "annotation.json"
    d = json.loads(f.read_text())
    d["annotations"][0]["segmentation"] = {"size": [64, 64], "counts": "abc"}
    f.write_text(json.dumps(d))
    p = run_prepare(path, "val_fraction=0")
    assert p.returncode != 0 and "RLE" in p.stderr
