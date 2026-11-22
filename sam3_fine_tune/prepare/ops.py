"""Derive the SAM3 view of a COCO dataset root: merge duplicate entries of one physical image, assign each physical
image to train / val / test once, and write ``<root>/prepared/``. The source dataset is only read.

Source layout: ``<root>/{train,test}/{images/, annotation_merged.json | annotation.json}`` (the merged file, which the
SAM3 configs read, wins when both exist). Output: ``<root>/prepared/manifest.json`` and
``<root>/prepared/sam3/{train,val,test}/{images/, annotation_merged.json}``; images are relative symlinks to the source,
so the whole root can be moved. A physical image is identified by the sha256 of its file bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath

SPLITS = ("train", "test")
VIEW_SPLITS = ("train", "val", "test")
PREPARED = "prepared"


def _annotation_file(split_dir: Path) -> Path | None:
    for name in ("annotation_merged.json", "annotation.json"):
        if (split_dir / name).exists():
            return split_dir / name
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _scan(root: Path) -> tuple[list[dict], dict]:
    """One record per COCO image entry that has its file (hash, origin split, entry, annotations by category name),
    plus ``{"missing": [...], "no_annotations": [...], "problems": [...]}`` for what could not be read."""
    records: list[dict] = []
    found: dict = {
        "missing": [],
        "no_annotations": [],
        "problems": [],
        "categories": set(),
    }
    for split in SPLITS:
        ann = _annotation_file(root / split)
        if ann is None:
            if (root / split).is_dir():
                found["no_annotations"].append(split)
            continue
        coco = json.loads(ann.read_text())
        names = {c["id"]: c["name"] for c in coco["categories"]}
        found["categories"].update(names.values())
        by_image: dict[int, list[dict]] = {}
        for a in coco["annotations"]:
            absent = [
                k for k in ("image_id", "category_id", "segmentation") if k not in a
            ]
            if absent:
                found["problems"].append(
                    f"{split}/{ann.name}: annotation {a.get('id')} has no {absent[0]!r}"
                )
                continue
            if a["category_id"] not in names:
                found["problems"].append(
                    f"{split}/{ann.name}: annotation {a['id']} has unknown category id {a['category_id']}"
                )
                continue
            by_image.setdefault(a["image_id"], []).append(a)
        for im in coco["images"]:
            parts = PurePosixPath(im["file_name"]).parts
            if (
                not parts
                or PurePosixPath(im["file_name"]).is_absolute()
                or ".." in parts
            ):
                found["problems"].append(
                    f"{split}/{ann.name}: file_name {im['file_name']!r} is not inside the images folder"
                )
                continue
            path = root / split / "images" / im["file_name"]
            if not path.is_file():
                found["missing"].append(f"{split}/{im['file_name']}")
                continue
            records.append(
                {
                    "hash": _sha256(path),
                    "split": split,
                    "entry": im,
                    "path": path,
                    "annotations": [
                        {**a, "category": names[a["category_id"]]}
                        for a in by_image.get(im["id"], [])
                    ],
                }
            )
    return records, found


def _merge(records: list[dict]) -> dict[str, dict]:
    """Physical images keyed by hash, each with its entries (one per source split) and its source files."""
    images: dict[str, dict] = {}
    for r in records:
        img = images.setdefault(
            r["hash"], {"entries": {}, "splits": set(), "files": [], "by_entry": []}
        )
        img["entries"].setdefault(r["split"], (r["entry"], r["path"]))
        img["splits"].add(r["split"])
        img["files"].append(f"{r['split']}/{r['entry']['file_name']}")
        img["by_entry"].append((img["files"][-1], r["annotations"]))
    return images


def _union(img: dict) -> tuple[list[dict], list[str]]:
    """The picture's annotations over all its entries. A category annotated in several entries must carry the same
    polygons in each (then it is kept once); different polygons are a problem."""
    per_cat: dict[str, list[tuple[str, list[dict]]]] = {}
    for file, anns in img["by_entry"]:
        for cat in dict.fromkeys(a["category"] for a in anns):
            per_cat.setdefault(cat, []).append(
                (file, [a for a in anns if a["category"] == cat])
            )
    merged, problems = [], []
    for cat, versions in per_cat.items():
        shapes = [
            sorted(json.dumps(a["segmentation"]) for a in anns) for _, anns in versions
        ]
        if any(s != shapes[0] for s in shapes[1:]):
            problems.append(
                f"category '{cat}' has different polygons on one picture: "
                + ", ".join(f for f, _ in versions)
            )
        merged.extend(versions[0][1])
    return merged, problems


def _source(img: dict, split: str) -> tuple[dict, Path]:
    """The entry and file to show in `split`: the one from that source split, else the picture's train entry
    (val is carved from train, so it has no source split of its own)."""
    key = split if split in img["entries"] else "train"
    entry: tuple[dict, Path] = img["entries"][key]
    return entry


def prepare_dataset(
    root: str | Path,
    *,
    val_fraction: float = 0.1,
    seed: int = 0,
    on_conflict: str = "error",
    dry_run: bool = False,
) -> dict:
    """Prepare ``root`` and return the report. ``report["problems"]`` is empty on success; when it is not, nothing was
    written and only ``problems`` is meaningful. ``dry_run`` computes the same report without writing.

    on_conflict: a picture found in both train and test is a problem (``error``) or stays in test (``test``).
    val_fraction: share of the train pictures that becomes val, in [0, 1).
    """
    if on_conflict not in ("error", "test"):
        raise ValueError(f"on_conflict must be error|test, got {on_conflict!r}")
    if not 0 <= val_fraction < 1:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    root = Path(root)
    records, found = _scan(root)
    images = _merge(records)
    problems: list[str] = list(found["problems"])
    assigned: dict[str, str] = {}
    for img in images.values():
        img["annotations"], conflicts = _union(img)
        problems.extend(conflicts)
    for h, img in images.items():
        if len(img["splits"]) > 1:
            if on_conflict == "error":
                problems.append(
                    "one picture is in both train and test: " + ", ".join(img["files"])
                )
            assigned[h] = "test"
        else:
            assigned[h] = next(iter(img["splits"]))
    if problems:
        return {"problems": problems}

    dropped = {s: 0 for s in SPLITS}
    for h in sorted(images):
        if not images[h]["annotations"]:
            dropped[assigned.pop(h)] += 1
    images = {h: img for h, img in images.items() if h in assigned}
    train = sorted(
        (h for h, s in assigned.items() if s == "train"),
        key=lambda h: hashlib.sha256(f"{seed}:{h}".encode()).hexdigest(),
    )
    n_val = round(val_fraction * len(train))
    if val_fraction > 0 and len(train) > 1:
        n_val = max(1, n_val)
    n_val = min(n_val, max(len(train) - 1, 0))  # train never ends up empty
    for h in train[:n_val]:
        assigned[h] = "val"

    # every category the source declares, even one with no kept annotation (SAM3 queries all of them); ids by
    # name, so they do not depend on which pictures are present
    categories = {n: i for i, n in enumerate(sorted(found["categories"]), 1)}
    ids = {h: i for i, h in enumerate(sorted(images), 1)}
    views: dict[str, dict] = {
        s: {
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": n} for n, i in categories.items()],
        }
        for s in VIEW_SPLITS
    }
    sources: dict[str, list[tuple[str, Path]]] = {s: [] for s in VIEW_SPLITS}
    for h in sorted(images):
        img, split = images[h], assigned[h]
        view = views[split]
        entry, src = _source(img, split)
        view["images"].append({**entry, "id": ids[h]})
        sources[split].append((entry["file_name"], src))
        for a in img["annotations"]:
            view["annotations"].append(
                {k: v for k, v in a.items() if k != "category"}
                | {
                    "id": len(view["annotations"]) + 1,
                    "image_id": ids[h],
                    "category_id": categories[a["category"]],
                }
            )
    manifest = {
        "version": 1,
        "seed": seed,
        "val_fraction": val_fraction,
        "categories": categories,
        "images": {
            h: {
                "split": assigned[h],
                "id": ids[h],
                "file_name": _source(images[h], assigned[h])[0]["file_name"],
                "files": images[h]["files"],
            }
            for h in sorted(images)
        },
    }
    if not dry_run:
        _write(root, views, sources, manifest)
    return {
        "problems": [],
        "images": {s: len(v["images"]) for s, v in views.items()},
        "dropped_unannotated": dropped,
        "merged_duplicates": sum(len(i["by_entry"]) > 1 for i in images.values()),
        "missing_images": found["missing"],
        "splits_without_annotations": found["no_annotations"],
        "categories": categories,
    }


def _write(root: Path, views: dict, sources: dict, manifest: dict) -> None:
    """Build the whole view beside the old one and swap it in, so a failure never leaves a half-written ``prepared/``."""
    final, tmp = root / PREPARED, root / f"{PREPARED}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1))
        for split, view in views.items():
            d = tmp / "sam3" / split
            (d / "images").mkdir(parents=True)
            (d / "annotation_merged.json").write_text(json.dumps(view, indent=2))
            for file_name, src in sources[split]:
                link = d / "images" / file_name
                if not link.resolve().is_relative_to(tmp.resolve()):
                    raise ValueError(f"refusing to write outside {tmp}: {file_name!r}")
                link.parent.mkdir(parents=True, exist_ok=True)
                # relative to where the link will live after the swap, so the root can be moved as a whole
                where = final / "sam3" / split / "images" / file_name
                os.symlink(
                    os.path.relpath(
                        os.path.abspath(src), os.path.abspath(where.parent)
                    ),
                    link,
                )
        old = root / f"{PREPARED}.old"
        shutil.rmtree(old, ignore_errors=True)
        if final.exists():
            os.replace(
                final, old
            )  # the previous view stays recoverable until the new one is in place
        os.replace(tmp, final)
        shutil.rmtree(old, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
