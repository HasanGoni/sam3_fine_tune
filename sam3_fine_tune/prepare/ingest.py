"""Turn an images folder plus one mask folder per category into canonical COCO in a dataset root.

Output: ``<into>/{train,test}/{images/, annotation.json}``, one COCO image entry per picture carrying all its
categories. A picture is identified by the sha256 of its file bytes, so a picture already in the root keeps its split
and entry, and a later run adds only the categories it does not have yet. A picture without a mask file for a category
is *not annotated* for it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from sam3_fine_tune.preprocessing.coco_format import mask_to_coco

from .ops import _scan, PREPARED, sha256_file, SPLITS

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
ANN, MERGED = "annotation.json", "annotation_merged.json"


def _by_stem(folder: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            if p.stem in found:
                raise ValueError(
                    f"{folder}: '{found[p.stem].name}' and '{p.name}' share the stem '{p.stem}'; "
                    "rename one so images and masks can be matched"
                )
            found[p.stem] = p
    return found


def _polygons(mask_path: Path) -> list[dict]:
    """The mask's COCO polygons (ids are filled in when the annotations are written)."""
    gray = np.asarray(Image.open(mask_path).convert("L"))
    if gray.max() > 0 and gray.max() <= 127:
        raise ValueError(
            f"mask {mask_path.name} has values up to {gray.max()}; masks must be 0/255 (above 127 = object)"
        )
    polygons: list[dict] = mask_to_coco((gray > 127).astype(np.uint8) * 255, 0, 0, 0)
    return polygons


def _load(path: Path) -> dict:
    if path.exists():
        data: dict = json.loads(path.read_text())
        return data
    return {
        "info": {"description": "SAM3 fine-tuning dataset"},
        "images": [],
        "annotations": [],
        "categories": [],
    }


def _apply(
    data: dict, additions: list[dict], category_ids: dict[str, int]
) -> tuple[int, int]:
    """Add each picture's polygons to ``data``: a new image entry, or, for a picture already there, the categories it
    does not have. Returns (images added, (image, category) pairs already annotated)."""
    for name, cid in category_ids.items():
        if not any(c["name"] == name for c in data["categories"]):
            data["categories"].append({"id": cid, "name": name})
    cat_id = {c["name"]: c["id"] for c in data["categories"]}
    entry_of = {i["file_name"]: i for i in data["images"]}
    annotated: dict[int, set[int]] = {}
    for a in data["annotations"]:
        annotated.setdefault(a["image_id"], set()).add(a["category_id"])
    next_image = max((i["id"] for i in data["images"]), default=0) + 1
    next_ann = max((a["id"] for a in data["annotations"]), default=0) + 1
    added = skipped = 0
    for add in additions:
        entry = entry_of.get(add["file_name"])
        if entry is None:
            entry = {
                "id": next_image,
                "file_name": add["file_name"],
                "width": add["width"],
                "height": add["height"],
            }
            data["images"].append(entry)
            next_image += 1
            added += 1
        for cat, polygons in add["polygons"].items():
            cid = cat_id[cat]
            if cid in annotated.get(entry["id"], set()):
                skipped += 1
                continue
            for p in polygons:
                data["annotations"].append(
                    {**p, "id": next_ann, "image_id": entry["id"], "category_id": cid}
                )
                next_ann += 1
    return added, skipped


def ingest_dataset(
    images: str | Path,
    masks: dict[str, str],
    into: str | Path,
    *,
    test_fraction: float = 0.2,
    seed: int = 0,
    dry_run: bool = False,
) -> dict:
    """Write the pictures in ``images`` and their masks (``masks[category]`` = folder) into the dataset root ``into``.

    Returns the report. Pictures that cannot be used (size mismatch, a different picture already under that file
    name) are listed in ``image_errors`` and left out; the others are written. ``report["problems"]`` is not empty
    only when the existing root is itself unreadable, and then nothing is written.
    """
    if images is None or into is None:
        raise ValueError("images=DIR and into=ROOT are required")
    if not masks:
        raise ValueError(
            "masks is empty: pass one folder per category, e.g. +masks.chip=DIR"
        )
    if not 0 <= test_fraction < 1:
        raise ValueError(f"test_fraction must be in [0, 1), got {test_fraction}")
    for label, folder in [
        ("images", images),
        *[(f"masks.{c}", d) for c, d in masks.items()],
    ]:
        if not Path(folder).is_dir():
            raise FileNotFoundError(f"{label} folder not found: {folder}")
    into = Path(into)
    imgs = _by_stem(Path(images))
    mask_files = {c: _by_stem(Path(d)) for c, d in masks.items()}
    report: dict = {
        "problems": [],
        "images_without_mask": sorted(
            imgs[s].name for s in imgs.keys() - set().union(*mask_files.values())
        ),
        "masks_without_image": {
            c: sorted(m[s].name for s in m.keys() - imgs.keys())
            for c, m in mask_files.items()
            if m.keys() - imgs.keys()
        },
        "image_errors": [],
        "images_added": {s: 0 for s in SPLITS},
        "already_annotated": 0,
        "dry_run": dry_run,
    }

    existing, found = (
        _scan(into) if into.is_dir() else ([], {"problems": [], "missing": []})
    )
    if found["problems"]:
        return {"problems": found["problems"]}
    known = {r["hash"]: r for r in existing}

    # per picture: its polygons per category, or why it cannot be used
    pictures: dict[str, dict] = {}
    for stem, path in imgs.items():
        have = {c: m[stem] for c, m in mask_files.items() if stem in m}
        if not have:
            continue
        digest = sha256_file(path)
        if (
            digest in pictures
        ):  # the same picture twice in the input: the first one wins
            report["image_errors"].append(
                {
                    "image": path.name,
                    "error": f"same picture as {pictures[digest]['file_name']}; left out",
                }
            )
            continue
        try:
            with Image.open(path) as im:
                size = im.size
            polygons = {}
            for cat, mpath in have.items():
                with Image.open(mpath) as m:
                    if m.size != size:
                        raise ValueError(
                            f"mask size {m.size} != image size {size} ({cat}: {mpath.name})"
                        )
                polygons[cat] = _polygons(mpath)
        except (ValueError, OSError, Image.DecompressionBombError) as e:
            report["image_errors"].append({"image": path.name, "error": str(e)})
            continue
        pictures[digest] = {
            "file_name": path.name,
            "path": path,
            "width": size[0],
            "height": size[1],
            "polygons": polygons,
        }

    plan: dict[str, list[dict]] = {s: [] for s in SPLITS}
    new: list[str] = []
    for digest, pic in sorted(pictures.items()):
        if digest in known:
            r = known[digest]
            plan[r["split"]].append({**pic, "file_name": r["entry"]["file_name"]})
            continue
        on_disk = [into / s / "images" / pic["file_name"] for s in SPLITS]
        stale = any(m.split("/", 1)[1] == pic["file_name"] for m in found["missing"])
        if stale or any(
            f.exists() and sha256_file(f) != digest for f in on_disk
        ):  # an orphan copy of this same picture (an interrupted run) is not a clash
            report["image_errors"].append(
                {
                    "image": pic["file_name"],
                    "error": "a different picture already has this file name in the dataset",
                }
            )
            continue
        new.append(digest)
    new.sort(key=lambda h: hashlib.sha256(f"{seed}:{h}".encode()).hexdigest())
    n_test = 0
    if test_fraction > 0 and len(new) > 1:
        n_test = min(len(new) - 1, max(1, round(len(new) * test_fraction)))
    for i, digest in enumerate(new):
        plan["test" if i < n_test else "train"].append(pictures[digest])

    category_ids: dict[str, int] = {}
    files: dict[str, list[Path]] = {}
    for split in SPLITS:
        files[split] = [
            f for f in (into / split / ANN, into / split / MERGED) if f.exists()
        ] or [into / split / ANN]
        for f in files[split]:
            for c in _load(f)["categories"]:
                category_ids.setdefault(c["name"], c["id"])
    top = max(category_ids.values(), default=0)
    for cat in masks:
        if cat not in category_ids:
            top += 1
            category_ids[cat] = top

    for split in SPLITS:
        additions = plan[split]
        if not additions and not any(f.exists() for f in files[split]):
            continue  # do not invent an empty split
        written = []
        for f in files[split]:
            data = _load(f)
            added, skipped = _apply(data, additions, category_ids)
            if f == files[split][0]:
                report["images_added"][split] = added
                report["already_annotated"] += skipped
            written.append((f, data))
        if dry_run:
            continue
        (into / split / "images").mkdir(parents=True, exist_ok=True)
        for add in additions:
            target = into / split / "images" / add["file_name"]
            if not target.exists():
                shutil.copy2(add["path"], target)
        for f, data in written:
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(f)
    report["categories"] = sorted(category_ids)
    report["note"] = f"run `sam3-prepare root={into}` to build {PREPARED}/"
    return report
