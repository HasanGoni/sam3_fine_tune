import json
import shutil

from .conftest import BOX, report, run_prepare, square, view


def test_duplicate_entries_of_one_picture_become_one_image_with_all_categories(root):
    # a dataset built the old way: the same picture added once per category under another file name
    root.add("train", "substrate_a.png", 1, {"substrate": square(5, 5, 55, 55)})
    root.add("train", "chip_a.png", 1, {"chip": BOX})
    root.add("train", "substrate_b.png", 2, {"substrate": square(5, 5, 55, 55)})
    path = root.write()

    p = run_prepare(path, "val_fraction=0")

    assert p.returncode == 0, p.stderr[-2000:]
    assert sorted(view(path, "train").values()) == [
        ["chip", "substrate"],
        ["substrate"],
    ]


def _leaky(root):
    # the same picture (2) is in train under one name and in test under another
    root.add("train", "a.png", 1, {"chip": BOX}).add(
        "train", "leak_train.png", 2, {"chip": BOX}
    )
    root.add("test", "b.png", 3, {"chip": BOX}).add(
        "test", "leak_test.png", 2, {"chip": BOX}
    )
    return root.write()


def test_a_picture_in_both_splits_stops_with_a_list_and_writes_nothing(root):
    path = _leaky(root)

    p = run_prepare(path, "val_fraction=0")

    assert p.returncode != 0
    assert "leak_train.png" in p.stdout and "leak_test.png" in p.stdout
    assert not (path / "prepared").exists()


def test_on_conflict_test_keeps_the_picture_in_test_only(root):
    path = _leaky(root)

    p = run_prepare(path, "val_fraction=0", "on_conflict=test")

    assert p.returncode == 0, p.stderr[-2000:]
    assert sorted(view(path, "train")) == ["a.png"]
    assert sorted(view(path, "test")) == ["b.png", "leak_test.png"]


def _ten(root):
    for i in range(10):
        root.add("train", f"p{i}.png", i, {"chip": BOX})
    return root.write()


def test_val_is_carved_from_train_by_picture_and_is_reproducible(root):
    path = _ten(root)

    first = run_prepare(path, "val_fraction=0.2")
    val, train = sorted(view(path, "val")), sorted(view(path, "train"))
    second = run_prepare(path, "val_fraction=0.2")

    assert first.returncode == 0, first.stderr[-2000:]
    assert len(val) == 2 and len(train) == 8 and not set(val) & set(train)
    assert sorted(view(path, "val")) == val and second.returncode == 0


def test_images_without_any_annotation_are_dropped_and_counted(root):
    root.add("train", "labelled.png", 1, {"chip": BOX})
    root.add("train", "empty.png", 2, {})
    path = root.write()

    p = run_prepare(path, "val_fraction=0")

    assert report(p)["dropped_unannotated"] == {"train": 1, "val": 0, "test": 0}
    assert list(view(path, "train")) == ["labelled.png"]


def _snapshot(path):
    return {
        p.relative_to(path).as_posix(): p.read_bytes()
        for split in ("train", "test")
        for p in sorted((path / split).rglob("*"))
        if p.is_file()
    }


def test_dry_run_prints_the_report_and_writes_nothing(root):
    path = _ten(root)

    p = run_prepare(path, "dry_run=true")

    assert (
        p.returncode == 0
        and report(p)["images"]["train"] + report(p)["images"]["val"] == 10
    )
    assert not (path / "prepared").exists()


def test_the_source_dataset_is_never_modified(root):
    path = _leaky(root)
    before = _snapshot(path)

    run_prepare(path, "on_conflict=test")

    assert _snapshot(path) == before


def test_a_missing_image_file_is_reported_and_not_fatal(root):
    root.add("train", "here.png", 1, {"chip": BOX})
    root.add("train", "gone.png", 2, {"chip": BOX}, write_file=False)
    path = root.write()

    p = run_prepare(path, "val_fraction=0")

    assert p.returncode == 0, p.stderr[-2000:]
    assert report(p)["missing_images"] == ["train/gone.png"]
    assert list(view(path, "train")) == ["here.png"]


def test_one_category_with_two_different_polygons_on_one_picture_is_an_error(root):
    root.add("train", "x1.png", 1, {"chip": BOX}).add(
        "train", "x2.png", 1, {"chip": square(0, 0, 20, 20)}
    )
    path = root.write()

    p = run_prepare(path, "val_fraction=0")

    assert p.returncode != 0 and "chip" in p.stdout and not (path / "prepared").exists()


def test_the_same_polygon_annotated_twice_is_kept_once(root):
    root.add("train", "x1.png", 1, {"chip": BOX}).add(
        "train", "x2.png", 1, {"chip": BOX}
    )
    path = root.write()

    p = run_prepare(path, "val_fraction=0")

    d = json.loads((path / "prepared/sam3/train/annotation_merged.json").read_text())
    assert (
        p.returncode == 0
        and len(d["annotations"]) == 1
        and report(p)["merged_duplicates"] == 1
    )


def test_two_pictures_with_the_same_file_name_in_one_view_keep_both(root):
    # train/a.png (picture 1) is also in test as z.png, so it moves to test, where a different a.png already lives
    root.add("train", "a.png", 1, {"chip": BOX})
    root.add("test", "a.png", 2, {"chip": BOX}).add("test", "z.png", 1, {"chip": BOX})
    path = root.write()

    p = run_prepare(path, "val_fraction=0", "on_conflict=test")

    names = list(view(path, "test"))
    assert p.returncode == 0 and len(names) == 2 and len(set(names)) == 2
    assert all((path / "prepared/sam3/test/images" / n).exists() for n in names)


def test_the_manifest_records_each_picture_once_with_its_split(root):
    path = _ten(root)

    run_prepare(path, "val_fraction=0.2")

    m = json.loads((path / "prepared/manifest.json").read_text())
    splits = [v["split"] for v in m["images"].values()]
    assert len(m["images"]) == 10 and sorted(set(splits)) == ["train", "val"]
    assert splits.count("val") == 2 and m["categories"] == {"chip": 1}
    assert all(
        (path / "prepared/sam3" / v["split"] / "images" / v["file_name"]).exists()
        for v in m["images"].values()
    )


def test_file_names_with_a_subfolder_are_kept(root):
    root.add("train", "lot1/a.png", 1, {"chip": BOX})
    path = root.write()

    p = run_prepare(path, "val_fraction=0")

    assert p.returncode == 0, p.stderr[-2000:]
    assert (path / "prepared/sam3/train/images/lot1/a.png").exists()


def test_the_view_survives_moving_the_dataset_root(root, tmp_path):
    path = _ten(root)
    run_prepare(path, "val_fraction=0.2")

    moved = tmp_path / "moved"
    shutil.move(str(path), str(moved))

    assert all(
        p.exists() for p in (moved / "prepared/sam3").rglob("*.png")
    ), "symlinks must be relative"


def test_a_split_folder_without_annotations_is_reported(root):
    path = _ten(root)
    (path / "test" / "images").mkdir(parents=True)

    p = run_prepare(path, "val_fraction=0")

    assert p.returncode == 0 and report(p)["splits_without_annotations"] == ["test"]


def test_an_annotation_with_an_unknown_category_id_is_a_problem_not_a_crash(root):
    path = _ten(root)
    f = path / "train" / "annotation.json"
    d = json.loads(f.read_text())
    d["annotations"][0]["category_id"] = 99
    f.write_text(json.dumps(d))

    p = run_prepare(path, "val_fraction=0")

    assert p.returncode != 0 and "99" in p.stdout and "train" in p.stdout
    assert "Traceback" not in p.stderr


def test_val_fraction_must_be_between_zero_and_one(root):
    path = _ten(root)

    for bad in ("1.0", "-0.1"):
        p = run_prepare(path, f"val_fraction={bad}")
        assert p.returncode != 0 and "val_fraction" in p.stderr
        assert "Traceback" not in p.stderr
