#!/usr/bin/env python3
"""Create a poster of COCO image + mask + overlay visualizations.

Used as a default quality-check step after merge-category-dataset.
Outputs one composite panel per sampled image: original image, single-color
flattened mask, and blended overlay with category legend.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO


def palette(n):
    rng = np.random.default_rng(42)
    colors = rng.integers(64, 255, size=(n, 3), dtype=np.uint8)
    return [tuple(c.tolist()) for c in colors]


def draw_legend(draw, cats, colors, x, y, font):
    for cat_id, cat_name in cats.items():
        color = colors[cat_id % len(colors)]
        draw.rectangle([x, y, x + 14, y + 14], fill=color, outline=(0, 0, 0))
        draw.text((x + 20, y), f"{cat_id}: {cat_name}", fill=(255, 255, 255), font=font)
        y += 20
    return y


def build_panel(coco, img_id, img_root, colors, cats, alpha=0.55):
    img = coco.loadImgs(img_id)[0]
    img_path = img_root / img["file_name"]
    if not img_path.exists():
        return None

    image = Image.open(img_path).convert("RGB")
    arr = np.array(image, dtype=np.uint8)
    mask_canvas = np.zeros_like(arr)
    overlay = arr.copy()

    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    present_cats = set()
    for ann in anns:
        cat_id = ann["category_id"]
        present_cats.add(cat_id)
        color = np.array(colors[cat_id % len(colors)], dtype=np.uint8)
        mask = coco.annToMask(ann).astype(bool)
        mask_canvas[mask] = color
        overlay[mask] = (overlay[mask] * (1 - alpha) + color * alpha).astype(np.uint8)

    blended = (arr * 0.25 + overlay * 0.75).astype(np.uint8)

    # resize to consistent height while keeping aspect
    target_h = 512
    def resize(im):
        w, h = im.size
        return im.resize((int(w * target_h / h), target_h), Image.Resampling.LANCZOS)

    panels = [resize(Image.fromarray(x)) for x in (arr, mask_canvas, blended)]
    total_w = sum(p.size[0] for p in panels) + 40
    poster = Image.new("RGB", (total_w, target_h + 80), (30, 30, 30))
    x = 10
    for p in panels:
        poster.paste(p, (x, 10))
        x += p.size[0] + 10

    draw = ImageDraw.Draw(poster)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
        small = font

    labels = ["image", "mask", "overlay"]
    x = 10
    for p, label in zip(panels, labels):
        draw.text((x, target_h + 15), label, fill=(255, 255, 255), font=font)
        x += p.size[0] + 10

    draw.text((10, target_h + 40), img["file_name"], fill=(200, 200, 200), font=small)
    legend_y = draw_legend(
        draw,
        {k: cats[k] for k in sorted(present_cats) if k in cats},
        colors,
        total_w - 250,
        target_h + 15,
        small,
    )
    return poster


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, help="Root with train/ and test/ splits")
    parser.add_argument("--out-dir", required=True, help="Where to write poster PNGs")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    root = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_path = root / args.split / "annotation_merged.json"
    img_root = root / args.split / "images"
    coco = COCO(str(ann_path))

    cats = {c["id"]: c["name"] for c in coco.dataset["categories"]}
    colors = palette(len(cats) + 1)

    img_ids = list(coco.imgs.keys())
    sample = random.sample(img_ids, min(args.num_samples, len(img_ids)))

    for i, img_id in enumerate(sample):
        panel = build_panel(coco, img_id, img_root, colors, cats)
        if panel is None:
            continue
        panel.save(out_dir / f"poster_{i:03d}_img{img_id}.png")

    print(f"Saved {len(list(out_dir.glob('*.png')))} posters to {out_dir}")


if __name__ == "__main__":
    main()
