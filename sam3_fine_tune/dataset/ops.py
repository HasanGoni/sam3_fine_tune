"""Add images + binary masks of one category to a COCO fine-tuning dataset (new dataset, new category or extension).

Dataset layout: ``<into>/{train,test}/{images/, annotation.json[, annotation_merged.json]}``. The SAM3 configs read
``annotation_merged.json``; when it exists (possibly hand-consolidated) it receives the same additions as
``annotation.json`` instead of being overwritten, otherwise it is created as a copy.
"""
from __future__ import annotations

import json
import random
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from sam3_fine_tune.preprocessing.coco_format import mask_to_coco

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
ANN, MERGED = "annotation.json", "annotation_merged.json"
SPLITS = ("train", "test")


def find_pairs(images_dir, masks_dir):
    """Match images and masks by file stem. Returns ``(pairs, unmatched)`` with pairs sorted by name."""
    imgs = {p.stem: p for p in sorted(Path(images_dir).iterdir()) if p.suffix.lower() in IMAGE_EXTS}
    masks = {p.stem: p for p in sorted(Path(masks_dir).iterdir()) if p.suffix.lower() in IMAGE_EXTS}
    pairs = [(imgs[s], masks[s]) for s in sorted(imgs.keys() & masks.keys())]
    unmatched = {"images_without_mask": [imgs[s].name for s in sorted(imgs.keys() - masks.keys())],
                 "masks_without_image": [masks[s].name for s in sorted(masks.keys() - imgs.keys())]}
    return pairs, unmatched


def split_pairs(pairs, test_fraction: float, seed: int):
    """Seeded shuffle; at least one test pair whenever ``test_fraction > 0`` and there are 2+ pairs."""
    order = list(pairs)
    random.Random(seed).shuffle(order)
    n_test = 0 if test_fraction <= 0 else min(len(order) - 1, max(1, round(len(order) * test_fraction))) if len(order) > 1 else 0
    return {"train": sorted(order[n_test:]), "test": sorted(order[:n_test])}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "cat"


def _read_mask(path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L")) > 127).astype(np.uint8) * 255


def _check_sizes(pairs) -> None:
    for img, mask in pairs:
        with Image.open(img) as a, Image.open(mask) as b:
            if a.size != b.size:
                raise ValueError(f"{img.name}: mask size {b.size} != image size {a.size} ({mask.name})")


def _load(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"info": {"description": "SAM3 fine-tuning dataset"}, "images": [], "annotations": [], "categories": []}


def _save(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def _category_state(files: list[Path], category: str):
    """(existing id or None, next free id) over every annotation file that will be touched."""
    ids, top = set(), 0
    for f in files:
        for c in _load(f)["categories"]:
            top = max(top, c["id"])
            if c["name"] == category:
                ids.add(c["id"])
    if len(ids) > 1:
        raise ValueError(f"category '{category}' has different ids across annotation files: {sorted(ids)}")
    return (ids.pop() if ids else None), top + 1


def add_category_data(images, masks, category: str, into, *, mode: str = "auto", test_fraction: float = 0.2,
                      seed: int = 0, prefix: str | None = None, dry_run: bool = False) -> dict:
    if mode not in ("auto", "new", "extend"):
        raise ValueError(f"mode must be auto|new|extend, got {mode!r}")
    for label, folder in (("images", images), ("masks", masks)):
        if not Path(folder).is_dir():
            raise FileNotFoundError(f"{label} folder not found: {folder}")
    into, prefix = Path(into), prefix or _slug(category)
    pairs, unmatched = find_pairs(images, masks)
    if not pairs:
        raise ValueError(f"no image/mask pairs matched by file stem in {images} and {masks}")
    _check_sizes(pairs)  # validate everything before writing anything
    plan = split_pairs(pairs, test_fraction, seed)

    ann_files = {s: [into / s / ANN] + ([into / s / MERGED] if (into / s / MERGED).exists() else []) for s in SPLITS}
    all_files = [f for fs in ann_files.values() for f in fs]
    existing_id, next_id = _category_state(all_files, category)
    if mode == "new" and existing_id is not None:
        raise ValueError(f"category '{category}' already exists (id {existing_id}); use mode=extend or mode=auto")
    if mode == "extend" and existing_id is None:
        raise ValueError(f"category '{category}' not found in {into}; use mode=new or mode=auto")
    resolved = "extend" if existing_id is not None else "new"
    cat_id = existing_id if existing_id is not None else next_id
    all_cats = {c["id"]: c["name"] for f in all_files for c in _load(f)["categories"]}
    all_cats[cat_id] = category

    report = {"mode": resolved, "category": category, "category_id": cat_id, "prefix": prefix,
              "added": {s: 0 for s in SPLITS}, "annotations_added": 0, "skipped_existing": 0, "unmatched": unmatched,
              "dry_run": dry_run}
    for split in SPLITS:
        split_pairs_ = plan[split]
        exists = any(f.exists() for f in ann_files[split])
        if not split_pairs_ and not exists:
            continue  # nothing to add and nothing there yet: do not invent an empty split
        files = list(ann_files[split])
        written, copied = [], set()
        for f in files:
            data = _load(f)
            names = {c["name"]: c for c in data["categories"]}
            if category not in names:
                data["categories"].append({"id": cat_id, "name": category})
            have = {i["file_name"] for i in data["images"]}
            img_id = max((i["id"] for i in data["images"]), default=0) + 1
            ann_id = max((a["id"] for a in data["annotations"]), default=0) + 1
            for img_path, mask_path in split_pairs_:
                fname = f"{prefix}_{img_path.name}"
                if fname in have:
                    if f == files[0]:
                        report["skipped_existing"] += 1
                    continue
                with Image.open(img_path) as im:
                    w, h = im.size
                data["images"].append({"id": img_id, "file_name": fname, "width": w, "height": h})
                anns = mask_to_coco(_read_mask(mask_path), img_id, cat_id, ann_id)
                data["annotations"].extend(anns)
                if f == files[0]:
                    report["added"][split] += 1
                    report["annotations_added"] += len(anns)
                img_id += 1
                ann_id += len(anns)
                copied.add((img_path, fname))
            written.append((f, data))
        if dry_run:
            continue
        (into / split / "images").mkdir(parents=True, exist_ok=True)
        for img_path, fname in copied:
            shutil.copy2(img_path, into / split / "images" / fname)
        for f, data in written:
            _save(f, data)
        if len(files) == 1:  # no merged file yet: the SAM3 configs need one
            shutil.copyfile(files[0], into / split / MERGED)
    report["problems"] = [] if dry_run else validate_dataset(into)
    return report


def validate_dataset(into) -> list[str]:
    """Cheap structural checks on every annotation file; returns human-readable problems (empty = fine)."""
    problems = []
    for split in SPLITS:
        for name in (ANN, MERGED):
            f = Path(into) / split / name
            if not f.exists():
                continue
            d, tag = json.loads(f.read_text()), f"{split}/{name}"
            img_ids = [i["id"] for i in d["images"]]
            ann_ids = [a["id"] for a in d["annotations"]]
            cat_ids = {c["id"] for c in d["categories"]}
            if len(img_ids) != len(set(img_ids)):
                problems.append(f"{tag}: duplicate image ids")
            if len(ann_ids) != len(set(ann_ids)):
                problems.append(f"{tag}: duplicate annotation ids")
            for i in d["images"]:
                if not (Path(into) / split / "images" / i["file_name"]).exists():
                    problems.append(f"{tag}: missing image {i['file_name']}")
            known = set(img_ids)
            for a in d["annotations"]:
                if a["image_id"] not in known:
                    problems.append(f"{tag}: annotation {a['id']} points to unknown image {a['image_id']}")
                if a["category_id"] not in cat_ids:
                    problems.append(f"{tag}: annotation {a['id']} has unknown category {a['category_id']}")
                if not any(len(p) >= 6 for p in a.get("segmentation", [])):
                    problems.append(f"{tag}: annotation {a['id']} has an empty polygon")
    return problems
