"""`sam3-prepare mode=ingest`: images plus one mask folder per category -> canonical COCO in the dataset root."""

import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

from .conftest import (
    coco,
    entries,
    make_images,
    make_masks,
    REPO,
    report,
    run_ingest,
    run_prepare,
)

STEMS = [f"im{i}" for i in range(10)]
BIG, MID, SMALL = (4, 4, 60, 60), (10, 10, 50, 50), (20, 20, 40, 40)


def two_categories(tmp_path):
    images = make_images(tmp_path / "imgs", STEMS)
    substrate = make_masks(tmp_path / "m_substrate", {s: BIG for s in STEMS})
    chip = make_masks(tmp_path / "m_chip", {s: MID for s in STEMS[:8]})
    return images, {"substrate": substrate, "chip": chip}


def test_two_categories_give_one_entry_per_picture(tmp_path):
    images, masks = two_categories(tmp_path)
    into = tmp_path / "ds"
    p = run_ingest(images, into, masks)
    assert p.returncode == 0, p.stderr
    per, where = entries(into)
    assert sorted(per) == sorted(f"{s}.png" for s in STEMS)  # no `{category}_` copies
    assert per["im0.png"] == ["chip", "substrate"]
    assert per["im9.png"] == ["substrate"]  # no chip mask: not annotated for chip
    assert set(where.values()) == {"train", "test"}


def test_result_passes_prepare_without_merging(tmp_path):
    images, masks = two_categories(tmp_path)
    into = tmp_path / "ds"
    assert run_ingest(images, into, masks).returncode == 0
    p = run_prepare(into)
    assert p.returncode == 0, p.stderr
    r = report(p)
    assert r["merged_duplicates"] == 0
    assert r["dropped_unannotated"] == {"train": 0, "test": 0}
    assert sum(r["images"].values()) == 10


def test_any_image_extension_matches_by_stem(tmp_path):
    images = make_images(tmp_path / "imgs", ["a", "b"], ext=".jpg")
    masks = {"chip": make_masks(tmp_path / "m", {"a": MID, "b": MID})}
    p = run_ingest(images, tmp_path / "ds", masks, "test_fraction=0")
    assert p.returncode == 0, p.stderr
    assert sorted(entries(tmp_path / "ds")[0]) == ["a.jpg", "b.jpg"]


def test_unmatched_files_are_reported(tmp_path):
    images = make_images(tmp_path / "imgs", ["a", "b", "c"])
    masks = {"chip": make_masks(tmp_path / "m", {"a": MID, "b": MID, "ghost": MID})}
    p = run_ingest(images, tmp_path / "ds", masks)
    assert p.returncode == 0, p.stderr
    r = report(p)
    assert r["images_without_mask"] == ["c.png"]
    assert r["masks_without_image"] == {"chip": ["ghost.png"]}
    assert sorted(entries(tmp_path / "ds")[0]) == ["a.png", "b.png"]


def test_size_mismatch_fails_only_that_image(tmp_path):
    images = make_images(tmp_path / "imgs", ["a", "b", "c"])
    m = make_masks(tmp_path / "m", {"a": MID, "c": MID})
    make_masks(m, {"b": (1, 1, 5, 5)}, size=(32, 32))
    p = run_ingest(images, tmp_path / "ds", {"chip": m})
    assert p.returncode == 0, p.stderr
    r = report(p)
    assert [e["image"] for e in r["image_errors"]] == ["b.png"]
    assert "size" in r["image_errors"][0]["error"]
    assert sorted(entries(tmp_path / "ds")[0]) == ["a.png", "c.png"]
    assert "b.png" in p.stderr


def test_nested_masks_keep_their_full_extent(tmp_path):
    images = make_images(tmp_path / "imgs", ["a"])
    boxes = {"substrate": BIG, "chip": MID, "lifted_chip": SMALL}
    masks = {c: make_masks(tmp_path / f"m_{c}", {"a": b}) for c, b in boxes.items()}
    into = tmp_path / "ds"
    assert run_ingest(images, into, masks, "test_fraction=0").returncode == 0
    d = coco(into, "train")
    cats = {c["id"]: c["name"] for c in d["categories"]}
    for cat, (x0, y0, x1, y1) in boxes.items():
        got = Image.new("L", (64, 64), 0)
        for a in d["annotations"]:
            if cats[a["category_id"]] == cat:
                for poly in a["segmentation"]:
                    ImageDraw.Draw(got).polygon(poly, fill=255)
        want = np.zeros((64, 64), bool)
        want[y0:y1, x0:x1] = True
        have = np.asarray(got) > 0
        assert (have & want).sum() / (have | want).sum() >= 0.9, cat


def test_second_category_extends_the_same_pictures(tmp_path):
    images = make_images(tmp_path / "imgs", STEMS)
    into = tmp_path / "ds"
    sub = {"substrate": make_masks(tmp_path / "m_s", {s: BIG for s in STEMS})}
    assert run_ingest(images, into, sub).returncode == 0
    before = entries(into)
    chip = {"chip": make_masks(tmp_path / "m_c", {s: MID for s in STEMS[:8]})}
    p = run_ingest(images, into, chip)
    assert p.returncode == 0, p.stderr
    per, where = entries(into)
    assert where == before[1]  # nobody moved across the split
    assert sorted(per) == sorted(before[0])  # no new images
    assert per["im0.png"] == ["chip", "substrate"]
    assert per["im9.png"] == ["substrate"]
    assert report(p)["images_added"] == {"train": 0, "test": 0}


def test_new_images_only_add(tmp_path):
    into = tmp_path / "ds"
    first = make_images(tmp_path / "i1", STEMS)
    masks = {"chip": make_masks(tmp_path / "m1", {s: MID for s in STEMS})}
    assert run_ingest(first, into, masks).returncode == 0
    before = entries(into)
    more = [f"new{i}" for i in range(6)]
    second = make_images(tmp_path / "i2", more, start=50)
    masks2 = {"chip": make_masks(tmp_path / "m2", {s: MID for s in more})}
    p = run_ingest(second, into, masks2)
    assert p.returncode == 0, p.stderr
    per, where = entries(into)
    for name, split in before[1].items():
        assert where[name] == split
        assert per[name] == before[0][name]
    assert len(per) == 16
    assert run_prepare(into).returncode == 0


def test_rerun_changes_nothing(tmp_path):
    images, masks = two_categories(tmp_path)
    into = tmp_path / "ds"
    assert run_ingest(images, into, masks).returncode == 0
    snapshot = {
        s: (into / s / "annotation.json").read_text() for s in ("train", "test")
    }
    p = run_ingest(images, into, masks)
    assert p.returncode == 0, p.stderr
    assert {
        s: (into / s / "annotation.json").read_text() for s in ("train", "test")
    } == snapshot
    assert report(p)["already_annotated"] == 18


def test_different_pictures_with_the_same_name_are_an_error_for_that_image(tmp_path):
    into = tmp_path / "ds"
    a = make_images(tmp_path / "i1", ["x"])
    assert (
        run_ingest(
            a,
            into,
            {"chip": make_masks(tmp_path / "m1", {"x": MID})},
            "test_fraction=0",
        ).returncode
        == 0
    )
    b = make_images(tmp_path / "i2", ["x", "y"], start=1)  # x is now another picture
    p = run_ingest(
        b,
        into,
        {"chip": make_masks(tmp_path / "m2", {"y": MID, "x": MID})},
        "test_fraction=0",
    )
    assert p.returncode == 0, p.stderr
    assert [e["image"] for e in report(p)["image_errors"]] == ["x.png"]


def test_dry_run_writes_nothing(tmp_path):
    images, masks = two_categories(tmp_path)
    into = tmp_path / "ds"
    p = run_ingest(images, into, masks, "dry_run=true")
    assert p.returncode == 0, p.stderr
    assert not into.exists()
    assert sum(report(p)["images_added"].values()) == 10


def test_missing_folder_is_a_clear_error(tmp_path):
    p = run_ingest(tmp_path / "nope", tmp_path / "ds", {"chip": tmp_path})
    assert p.returncode != 0
    assert "nope" in p.stderr


def test_no_mask_folders_is_a_clear_error(tmp_path):
    images = make_images(tmp_path / "imgs", ["a"])
    p = run_ingest(images, tmp_path / "ds", {})
    assert p.returncode != 0
    assert "masks" in p.stderr


def test_corrupt_file_fails_only_that_image(tmp_path):
    images = make_images(tmp_path / "imgs", ["a", "b"])
    (images / "b.png").write_bytes(b"not a png")
    masks = {"chip": make_masks(tmp_path / "m", {"a": MID, "b": MID})}
    p = run_ingest(images, tmp_path / "ds", masks, "test_fraction=0")
    assert p.returncode == 0, p.stderr
    assert [e["image"] for e in report(p)["image_errors"]] == ["b.png"]
    assert sorted(entries(tmp_path / "ds")[0]) == ["a.png"]


def test_zero_one_masks_are_an_error_not_silent_negatives(tmp_path):
    images = make_images(tmp_path / "imgs", ["a", "b"])
    m = make_masks(tmp_path / "m", {"a": MID})
    Image.fromarray(np.ones((64, 64), np.uint8)).save(m / "b.png")
    p = run_ingest(images, tmp_path / "ds", {"chip": m}, "test_fraction=0")
    assert [e["image"] for e in report(p)["image_errors"]] == ["b.png"]
    assert "0/255" in report(p)["image_errors"][0]["error"]


def test_missing_arguments_are_a_clear_error(tmp_path):
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "sam3_fine_tune.prepare",
            "mode=ingest",
            "+masks.chip=x",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert p.returncode != 0 and "required" in p.stderr


def test_same_stem_with_two_extensions_is_refused(tmp_path):
    images = make_images(tmp_path / "imgs", ["x"])
    make_images(images, ["x"], ext=".jpg")
    p = run_ingest(
        images, tmp_path / "ds", {"chip": make_masks(tmp_path / "m", {"x": MID})}
    )
    assert p.returncode != 0 and "share the stem" in p.stderr


def test_duplicate_picture_in_input_is_reported(tmp_path):
    images = make_images(tmp_path / "imgs", ["a"])
    (images / "a_copy.png").write_bytes((images / "a.png").read_bytes())
    masks = {"chip": make_masks(tmp_path / "m", {"a": MID, "a_copy": MID})}
    p = run_ingest(images, tmp_path / "ds", masks, "test_fraction=0")
    assert [e["image"] for e in report(p)["image_errors"]] == ["a_copy.png"]


def test_orphan_copy_from_an_interrupted_run_does_not_block(tmp_path):
    images = make_images(tmp_path / "imgs", ["a"])
    into = tmp_path / "ds"
    (into / "train" / "images").mkdir(parents=True)
    (into / "train" / "images" / "a.png").write_bytes((images / "a.png").read_bytes())
    p = run_ingest(
        images,
        into,
        {"chip": make_masks(tmp_path / "m", {"a": MID})},
        "test_fraction=0",
    )
    assert p.returncode == 0 and report(p)["image_errors"] == []
    assert "a.png" in entries(into)[0]


def test_every_picture_rejected_exits_nonzero(tmp_path):
    images = make_images(tmp_path / "imgs", ["a"])
    m = make_masks(tmp_path / "m", {"a": MID}, size=(32, 32))
    p = run_ingest(images, tmp_path / "ds", {"chip": m})
    assert p.returncode != 0 and "wrote nothing" in p.stderr
