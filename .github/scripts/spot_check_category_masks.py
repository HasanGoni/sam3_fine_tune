#!/usr/bin/env python3
"""(Ha) Added 2026-07-19: one-off visual spot-check of GT mask quality for the
newly-added bondwire categories (thick_wire/thin_wire/gate_pad), to rule in/out
a Phase-1 mask-splitting data bug before assuming the low per-category AP
(0.19-0.37) is purely a modeling/geometry issue. See CHANGELOG.md 2026-07-19
and the per-category eval findings.

For each requested category, picks up to N sample images that have at least
one annotation of that category, and renders the GT polygon mask overlay +
bbox on top of the source image so a human can visually confirm the mask
boundary matches the actual object in the photo.

Usage:
    python spot_check_category_masks.py --ann <annotation.json> --img-dir <images/> \
        --categories thick_wire thin_wire gate_pad --num-samples 3 --out-dir <out/>
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
]


def draw_annotation(draw: ImageDraw.ImageDraw, ann: dict, color: tuple):
    """Draw a polygon segmentation outline + bbox for one annotation."""
    seg = ann["segmentation"]
    if isinstance(seg, list):
        for poly in seg:
            pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
            if len(pts) >= 2:
                draw.line(pts + [pts[0]], fill=color, width=3)
    x, y, w, h = ann["bbox"]
    draw.rectangle([x, y, x + w, y + h], outline=color, width=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann", required=True, help="Path to COCO annotation.json")
    parser.add_argument("--img-dir", required=True, help="Path to the images/ folder")
    parser.add_argument("--categories", nargs="+", required=True, help="Category names to spot-check")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of sample images per category")
    parser.add_argument("--out-dir", required=True, help="Directory to write overlay PNGs to")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.ann) as f:
        data = json.load(f)

    cat_name_to_id = {c["name"]: c["id"] for c in data["categories"]}
    img_id_to_info = {i["id"]: i for i in data["images"]}

    anns_by_image = defaultdict(list)
    for a in data["annotations"]:
        anns_by_image[a["image_id"]].append(a)

    for cat_name in args.categories:
        cat_id = cat_name_to_id.get(cat_name)
        if cat_id is None:
            print(f"WARNING: category '{cat_name}' not found in annotation file, skipping")
            continue

        image_ids_with_cat = sorted({a["image_id"] for a in data["annotations"] if a["category_id"] == cat_id})
        if not image_ids_with_cat:
            print(f"WARNING: no annotations found for '{cat_name}'")
            continue

        sample_ids = random.sample(image_ids_with_cat, min(args.num_samples, len(image_ids_with_cat)))
        print(f"{cat_name}: {len(image_ids_with_cat)} images total, spot-checking {len(sample_ids)}")

        for img_id in sample_ids:
            img_info = img_id_to_info[img_id]
            img_path = Path(args.img_dir) / img_info["file_name"]
            if not img_path.exists():
                print(f"  MISSING image file: {img_path}")
                continue

            image = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(image)

            all_anns_this_image = anns_by_image[img_id]
            cat_id_to_color = {c["id"]: COLORS[i % len(COLORS)] for i, c in enumerate(data["categories"])}
            for ann in all_anns_this_image:
                draw_annotation(draw, ann, cat_id_to_color[ann["category_id"]])

            out_name = f"{cat_name}_{img_info['file_name']}"
            image.save(out_dir / out_name)
            n_this_cat = sum(1 for a in all_anns_this_image if a["category_id"] == cat_id)
            print(f"  {out_name}  ({n_this_cat} '{cat_name}' instance(s), {len(all_anns_this_image)} total anns in image)")

    print(f"\nWrote overlays to {out_dir}")


if __name__ == "__main__":
    main()
