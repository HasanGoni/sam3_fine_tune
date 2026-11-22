"""Re-running `sam3-prepare` after adding data moves nothing."""

import json
import os
from pathlib import Path

from .conftest import BOX, report, Root, run_prepare


def _dataset(path, extra=()):
    r = Root(path)
    for i in range(1, 11):
        r.add("train", f"p{i}.png", i, {"chip": BOX})
    r.add("test", "t1.png", 50, {"chip": BOX})
    for split, name, picture, cats in extra:
        r.add(split, name, picture, cats)
    return r.write()


def _manifest(path):
    return json.loads((Path(path) / "prepared" / "manifest.json").read_text())


def _snapshot(path):
    out = {}
    for p in sorted((Path(path) / "prepared").rglob("*")):
        if p.is_symlink():
            out[str(p)] = os.readlink(p)
        elif p.is_file():
            out[str(p)] = p.read_bytes()
    return out


def test_rerun_with_no_new_data_changes_no_output_file(tmp_path):
    path = _dataset(tmp_path / "ds")
    assert run_prepare(path, "val_fraction=0.2").returncode == 0
    first = _snapshot(path)
    assert run_prepare(path, "val_fraction=0.2").returncode == 0
    assert _snapshot(path) == first


def test_adding_images_and_a_category_keeps_earlier_assignments_and_ids(tmp_path):
    path = _dataset(tmp_path / "ds")
    assert run_prepare(path, "val_fraction=0.2").returncode == 0
    before = _manifest(path)
    extra = [
        ("train", f"n{i}.png", 100 + i, {"chip": BOX, "aaa_new": BOX})
        for i in range(10)
    ] + [("test", "t2.png", 60, {"chip": BOX})]
    _dataset(path, extra)
    assert run_prepare(path, "val_fraction=0.2").returncode == 0
    after = _manifest(path)
    for h, rec in before["images"].items():
        assert after["images"][h]["split"] == rec["split"]
        assert after["images"][h]["id"] == rec["id"]
    for name, cid in before["categories"].items():
        assert after["categories"][name] == cid
    assert after["categories"]["aaa_new"] not in before["categories"].values()
    assert len(after["images"]) == len(before["images"]) + 11


def test_a_copy_under_a_new_file_name_inherits_the_old_split(tmp_path):
    path = _dataset(tmp_path / "ds")
    assert run_prepare(path, "val_fraction=0.2").returncode == 0
    before = _manifest(path)
    in_val = next(
        rec["file_name"] for rec in before["images"].values() if rec["split"] == "val"
    )
    picture = int(in_val.removeprefix("p").removesuffix(".png"))
    _dataset(path, [("train", "again.png", picture, {"chip": BOX})])
    p = run_prepare(path, "val_fraction=0.2")
    assert p.returncode == 0, p.stderr
    after = _manifest(path)
    h = next(h for h, r in before["images"].items() if r["file_name"] == in_val)
    assert after["images"][h]["split"] == "val"
    assert "again.png" in " ".join(after["images"][h]["files"])
    assert len(after["images"]) == len(before["images"])


def test_category_names_that_differ_only_in_case_or_space_are_reported(tmp_path):
    r = Root(tmp_path / "ds")
    r.add("train", "a.png", 1, {"Thin Wire": BOX})
    r.add("train", "b.png", 2, {"thin wire": BOX})
    r.add("train", "c.png", 3, {"thinwire": BOX, "crack": BOX})
    path = r.write()
    p = run_prepare(path, "val_fraction=0")
    assert p.returncode == 0, p.stderr
    assert report(p)["similar_category_names"] == [
        ["Thin Wire", "thin wire", "thinwire"]
    ]
    assert "Thin Wire" in _manifest(path)["categories"]  # names stay exactly as given
    assert "similar" in p.stderr


def test_a_picture_moved_between_train_and_test_is_a_problem(tmp_path):
    path = _dataset(tmp_path / "ds")
    assert run_prepare(path, "val_fraction=0.2").returncode == 0
    r = Root(path)  # picture 1 is now only in the test split
    for i in range(2, 11):
        r.add("train", f"p{i}.png", i, {"chip": BOX})
    r.add("test", "t1.png", 50, {"chip": BOX}).add("test", "p1.png", 1, {"chip": BOX})
    (path / "train" / "images" / "p1.png").unlink()
    r.write()
    p = run_prepare(path, "val_fraction=0.2")
    assert p.returncode != 0 and "moved between train and test" in p.stderr


def test_a_small_addition_does_not_inflate_val(tmp_path):
    r = Root(tmp_path / "ds")
    for i in range(1, 41):
        r.add("train", f"p{i}.png", i, {"chip": BOX})
    path = r.write()
    assert run_prepare(path, "val_fraction=0.1").returncode == 0
    for i in (101, 102):
        r.add("train", f"n{i}.png", i, {"chip": BOX})
    r.write()
    assert run_prepare(path, "val_fraction=0.1").returncode == 0
    splits = [v["split"] for v in _manifest(path)["images"].values()]
    assert splits.count("val") == 4  # round(0.1 * 42), not one per batch


def test_an_unreadable_manifest_is_a_clear_error(tmp_path):
    path = _dataset(tmp_path / "ds")
    assert run_prepare(path).returncode == 0
    (path / "prepared" / "manifest.json").write_text("{not json")
    p = run_prepare(path)
    assert p.returncode != 0 and "manifest.json is unreadable" in p.stderr


def test_an_interrupted_earlier_run_is_not_mistaken_for_a_new_dataset(tmp_path):
    path = _dataset(tmp_path / "ds")
    assert run_prepare(path).returncode == 0
    (path / "prepared").rename(path / "prepared.old")
    p = run_prepare(path)
    assert p.returncode != 0 and "interrupted" in p.stderr
