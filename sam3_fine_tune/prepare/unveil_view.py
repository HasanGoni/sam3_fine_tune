"""The UNVEIL view of a prepared dataset, written under ``<prepared>/unveil/``:

- ``masks/<slug>/<image id>.png``: one 0/255 mask per (image, category), rasterized from the COCO polygons on its own,
  so nested or overlapping categories never compete for a pixel;
- ``index/<split>/<slug>.jsonl``: one record per image annotated for the category, in UNVEIL's loader format
  (``id``, ``image``, ``gt``, ``w``, ``h``), plus ``categories``: every category the image is annotated for. Paths are
  relative to the dataset root, as in UNVEIL's own manifests (its loader joins them with the data root);
- ``prompts.json``: ``task`` = category names in id order (prompt id = position), ``slugs`` = name -> file-system name;
- ``recipe.yaml``: a skeleton the user fills in; an existing one is carried over untouched.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

PREPARED = "prepared"  # the folder name; ops.py imports it from here


def slugs(categories: dict[str, int]) -> dict[str, str]:
    """File-system names for the categories: letters, digits and underscores; made unique by the category id."""
    out: dict[str, str] = {}
    used: set[str] = (
        set()
    )  # lower case: two names must differ on case-insensitive file systems too
    for name, cid in categories.items():
        slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") or "category"
        while slug.lower() in used:
            slug = f"{slug}_{cid}"
        used.add(slug.lower())
        out[name] = slug
    return out


def _mask(annotations: list[dict], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), np.uint8)
    for a in annotations:
        for poly in a["segmentation"]:
            n = len(poly) // 2 * 2  # a stray last coordinate is dropped
            if n >= 6:
                pts = np.round(np.array(poly[:n], dtype=np.float64).reshape(-1, 2))
                cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    return mask


def _skeleton(names: list[str]) -> str:
    lines = [
        "# Fill in `kind` for every category: region (BCE + IoU) or balanced (class-balanced BCE, for thin structures).",
        "# families: group categories. UNVEIL takes presence negatives only from images of other families, and sam3-prepare",
        "# makes no SAM3 query for a same-family category an image is not annotated for. Run sam3-prepare again after editing.",
        "# Rules: docs/unveil-recipe.md. This file is never overwritten by sam3-prepare.",
        "categories:",
        *[f"  {json.dumps(n)}:\n    kind: ???" for n in names],
        "families: {}",
        "exclude_unannotated: true",
        "replay: false",
        "",
    ]
    return "\n".join(lines)


def _recipe_gaps(recipe: Path, names: list[str]) -> list[str]:
    """Categories missing from a recipe carried over from an earlier run (all of them when it cannot be read)."""
    try:
        listed = yaml.safe_load(recipe.read_text())["categories"]
        return [n for n in names if n not in listed]
    except (OSError, yaml.YAMLError, KeyError, TypeError):
        return list(names)


def write_unveil_view(
    tmp: Path, views: dict[str, dict], categories: dict[str, int], previous: Path
) -> list[str]:
    """Write the view into ``tmp`` (the ``prepared`` folder being built). ``previous`` is the current ``prepared``
    folder, whose ``unveil/recipe.yaml`` is kept when it exists. Returns the categories that kept recipe lacks."""
    out = tmp / "unveil"
    out.mkdir(parents=True)
    slug = slugs(categories)
    by_id = {cid: name for name, cid in categories.items()}
    ordered = [n for n, _ in sorted(categories.items(), key=lambda kv: kv[1])]
    (out / "prompts.json").write_text(
        json.dumps({"task": ordered, "slugs": slug}, indent=1)
    )
    old_recipe = previous / "unveil" / "recipe.yaml"
    gaps: list[str] = []
    if old_recipe.is_file():
        shutil.copyfile(old_recipe, out / "recipe.yaml")
        gaps = _recipe_gaps(old_recipe, ordered)
    else:
        (out / "recipe.yaml").write_text(_skeleton(ordered))

    for split, view in views.items():
        per_image: dict[int, list[dict]] = {}
        for a in view["annotations"]:
            per_image.setdefault(a["image_id"], []).append(a)
        index: dict[str, list[dict]] = {}
        for im in view["images"]:
            width, height = im.get("width"), im.get("height")
            if not width or not height:
                with Image.open(tmp / "sam3" / split / "images" / im["file_name"]) as f:
                    width, height = f.size
            by_cat: dict[str, list[dict]] = {}
            for a in per_image.get(im["id"], []):
                by_cat.setdefault(by_id[a["category_id"]], []).append(a)
            for name, anns in by_cat.items():
                mask_dir = out / "masks" / slug[name]
                mask_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(_mask(anns, width, height)).save(
                    mask_dir / f"{im['id']}.png"
                )
                index.setdefault(name, []).append(
                    {
                        "id": im["id"],
                        "image": f"{PREPARED}/sam3/{split}/images/{im['file_name']}",
                        "gt": f"{PREPARED}/unveil/masks/{slug[name]}/{im['id']}.png",
                        "w": width,
                        "h": height,
                        "categories": sorted(by_cat),
                    }
                )
        folder = out / "index" / split
        folder.mkdir(parents=True)
        for name in ordered:
            (folder / f"{slug[name]}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in index.get(name, []))
            )
    return gaps
