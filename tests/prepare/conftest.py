"""Synthetic dataset roots for the `sam3-prepare` tests: tiny images, polygon annotations, any split layout."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).parents[2]
BOX = [10, 10, 40, 10, 40, 40, 10, 40]  # a square polygon (flat x,y list)


def square(x0, y0, x1, y1):
    return [x0, y0, x1, y0, x1, y1, x0, y1]


class Root:
    """Builds `<root>/{train,test}/{images,annotation.json}` from (split, file_name, picture, {category: polygon})."""

    def __init__(self, path):
        self.path = Path(path)
        self.entries = {"train": [], "test": []}

    def add(self, split, file_name, picture, cats, write_file=True):
        """picture: an int, equal ints = identical pixels (the same physical image under another name)."""
        self.entries[split].append((file_name, picture, cats, write_file))
        return self

    def write(self):
        for split, entries in self.entries.items():
            if not entries:
                continue
            images = self.path / split / "images"
            images.mkdir(parents=True, exist_ok=True)
            coco = {"images": [], "annotations": [], "categories": []}
            cat_ids = {}
            for i, (name, picture, cats, write_file) in enumerate(entries, 1):
                if write_file:
                    (images / name).parent.mkdir(parents=True, exist_ok=True)
                    pixels = np.full((64, 64, 3), 10 * picture + 5, np.uint8)
                    Image.fromarray(pixels).save(images / name)
                coco["images"].append(
                    {"id": i, "file_name": name, "width": 64, "height": 64}
                )
                for cat, polygon in cats.items():
                    cid = cat_ids.setdefault(cat, len(cat_ids) + 1)
                    coco["annotations"].append(
                        {
                            "id": len(coco["annotations"]) + 1,
                            "image_id": i,
                            "category_id": cid,
                            "segmentation": [polygon],
                            "area": 900.0,
                            "bbox": [10, 10, 30, 30],
                            "iscrowd": 0,
                        }
                    )
            coco["categories"] = [{"id": v, "name": k} for k, v in cat_ids.items()]
            (self.path / split / "annotation.json").write_text(json.dumps(coco))
        return self.path


@pytest.fixture()
def root(tmp_path):
    return Root(tmp_path / "ds")


def run_prepare(root_path, *overrides):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "sam3_fine_tune.prepare",
            f"root={root_path}",
            *overrides,
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


def view(root_path, split, name="annotation_merged.json"):
    """The SAM3 view of one split as {file_name: sorted category names}."""
    d = json.loads((Path(root_path) / "prepared" / "sam3" / split / name).read_text())
    cats = {c["id"]: c["name"] for c in d["categories"]}
    per = {i["id"]: set() for i in d["images"]}
    for a in d["annotations"]:
        per[a["image_id"]].add(cats[a["category_id"]])
    return {i["file_name"]: sorted(per[i["id"]]) for i in d["images"]}


def report(process):
    import yaml

    return yaml.safe_load(process.stdout)


def make_images(folder, stems, ext=".png", size=(64, 64), start=0):
    """Write one distinct picture per stem (pixels depend on `start` + the stem's position) and return the folder."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for i, stem in enumerate(stems):
        pixels = np.full((size[1], size[0], 3), 7 * (start + i) + 3, np.uint8)
        Image.fromarray(pixels).save(folder / f"{stem}{ext}")
    return folder


def make_masks(folder, boxes, size=(64, 64), ext=".png"):
    """boxes: {stem: (x0, y0, x1, y1)} -> one binary mask file per stem."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for stem, (x0, y0, x1, y1) in boxes.items():
        m = np.zeros((size[1], size[0]), np.uint8)
        m[y0:y1, x0:x1] = 255
        Image.fromarray(m).save(folder / f"{stem}{ext}")
    return folder


def run_ingest(images, into, masks, *overrides):
    """`sam3-prepare mode=ingest`: masks is {category: folder}."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "sam3_fine_tune.prepare",
            "mode=ingest",
            f"images={images}",
            f"into={into}",
            *[f"+masks.{c}={d}" for c, d in masks.items()],
            *overrides,
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


def coco(root_path, split):
    return json.loads((Path(root_path) / split / "annotation.json").read_text())


def entries(root_path):
    """{file_name: sorted category names} over train and test of the ingested root, plus the split of each file."""
    per, where = {}, {}
    for split in ("train", "test"):
        if not (Path(root_path) / split / "annotation.json").exists():
            continue
        d = coco(root_path, split)
        cats = {c["id"]: c["name"] for c in d["categories"]}
        by_id = {i["id"]: i["file_name"] for i in d["images"]}
        for name in by_id.values():
            per[name] = set()
            where[name] = split
        for a in d["annotations"]:
            per[by_id[a["image_id"]]].add(cats[a["category_id"]])
    return {k: sorted(v) for k, v in per.items()}, where
