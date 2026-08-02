#!/usr/bin/env python3
"""Save a poster of images with COCO mask overlays to verify COCO conversion."""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend for HPC
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from pycocotools.coco import COCO


COLORS = {
    "thick_wire": (1.0, 0.0, 0.0),      # red
    "thin_wire": (0.0, 1.0, 0.0),       # green
    "gate_pad": (1.0, 1.0, 0.0),        # yellow
}

DEFAULT_CMAP = matplotlib.colormaps["tab10"]


def _color_for_name(name: str):
    if name in COLORS:
        return np.array(COLORS[name])
    # deterministic fallback
    rng = random.Random(abs(hash(name)) % (2**31))
    return np.array(DEFAULT_CMAP(rng.random())[:3])


def save_overlay_poster(
    coco_json: Path,
    image_dir: Path | None = None,
    output_path: Path | None = None,
    n_samples: int = 4,
    seed: int = 42,
    show_boxes: bool = False,
    show_labels: bool = False,
    alpha: float = 0.45,
    dpi: int = 200,
    category: str | None = None,
    one_per_category: bool = False,
    categories: list[str] | None = None,
):
    """Save a JPEG poster with 3 panels per image: original, overlay, mask.

    If `one_per_category` is set, ignores `n_samples` and instead picks one
    random image per category (restricted to `categories` if given, else
    every category in the COCO file), so every category is guaranteed a row
    in the resulting single poster image.
    """
    coco_json = Path(coco_json)
    if image_dir is None:
        image_dir = coco_json.parent / "images"
    image_dir = Path(image_dir)

    if output_path is None:
        output_path = coco_json.parent / "overlay_poster.jpg"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coco = COCO(str(coco_json))
    cat_id_to_name = {c["id"]: c["name"] for c in coco.dataset["categories"]}
    name_to_cat_id = {c["name"]: c["id"] for c in coco.dataset["categories"]}

    if category is not None and category not in name_to_cat_id:
        raise ValueError(
            f"Category '{category}' not found. Available: {sorted(name_to_cat_id)}"
        )

    rng = random.Random(seed)

    row_categories: list[str | None]
    if one_per_category:
        cat_names = categories if categories else [c["name"] for c in coco.dataset["categories"]]
        chosen = []
        row_categories = []
        for cname in cat_names:
            if cname not in name_to_cat_id:
                raise ValueError(
                    f"Category '{cname}' not found. Available: {sorted(name_to_cat_id)}"
                )
            img_ids_for_cat = sorted(set(coco.getImgIds(catIds=[name_to_cat_id[cname]])))
            if not img_ids_for_cat:
                print(f"  [warn] no images found for category '{cname}', skipping its row")
                continue
            chosen.append(rng.choice(img_ids_for_cat))
            row_categories.append(cname)
        n_samples = len(chosen)
    else:
        img_ids = coco.getImgIds()
        chosen = rng.sample(img_ids, min(n_samples, len(img_ids)))
        row_categories = [category] * len(chosen)

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    axes = np.atleast_2d(axes)

    for row, img_id in enumerate(chosen):
        row_category = row_categories[row]
        img_info = coco.loadImgs(img_id)[0]
        img_path = image_dir / img_info["file_name"]
        img = np.array(Image.open(img_path).convert("RGB"))

        ax_orig, ax_overlay, ax_mask = axes[row]

        # 1) Original image
        ax_orig.imshow(img)
        ax_orig.set_title("original" if row_category is None else f"original\n[{row_category}]")
        ax_orig.axis("off")

        # 2) Overlay image
        ax_overlay.imshow(img)

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        if row_category is not None:
            anns = [ann for ann in anns if cat_id_to_name[ann["category_id"]] == row_category]
        title_parts = defaultdict(int)

        mask_canvas = np.zeros((*img.shape[:2], 3), dtype=np.uint8)
        for ann_idx, ann in enumerate(anns):
            name = cat_id_to_name[ann["category_id"]]
            title_parts[name] += 1
            color = (_color_for_name(name) * 255).astype(np.uint8)
            mask = coco.annToMask(ann).astype(bool)
            mask_canvas[mask] = color

            if show_boxes:
                x, y, w, h = ann["bbox"]
                ax_overlay.add_patch(
                    patches.Rectangle(
                        (x, y), w, h, linewidth=1.5, edgecolor=color / 255.0, facecolor="none"
                    )
                )

            if show_labels:
                ys, xs = np.where(mask)
                if len(xs) > 0:
                    ax_overlay.text(
                        xs[0],
                        ys[0],
                        f"{ann_idx + 1}",
                        color=color / 255.0,
                        fontsize=8,
                        bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=1),
                    )

        ax_overlay.imshow(mask_canvas, alpha=alpha)
        title = ", ".join(f"{k}={v}" for k, v in sorted(title_parts.items())) or "no annotations"
        overlay_title = f"overlay\n{title}" if row_category is None else f"overlay [{row_category}]\n{title}"
        ax_overlay.set_title(overlay_title)
        ax_overlay.axis("off")

        # 3) Mask only
        ax_mask.imshow(mask_canvas)
        ax_mask.set_title("mask only")
        ax_mask.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, format="jpeg", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved poster to: {output_path}")
    print(f"  images: {n_samples}, source: {coco_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Save a JPEG poster of images with COCO mask overlays."
    )
    parser.add_argument(
        "--coco-json",
        type=Path,
        default=Path("/home/hpc_storage/data/projects/userB/sam3_data/bondwire_clean_merged/train/annotation.json"),
        help="Path to COCO annotation JSON.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory containing COCO images. Defaults to <coco-json parent>/images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JPEG path. Defaults to <coco-json parent>/overlay_poster.jpg.",
    )
    parser.add_argument("--n-samples", type=int, default=4, help="Number of images in the poster.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for image selection.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Mask overlay alpha.")
    parser.add_argument("--dpi", type=int, default=200, help="JPEG resolution.")
    parser.add_argument("--show-boxes", action="store_true", help="Also draw bounding boxes.")
    parser.add_argument("--show-labels", action="store_true", help="Also draw annotation indices.")
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter to one category name (e.g. thin_wire). If omitted, all categories are shown.",
    )
    parser.add_argument(
        "--one-per-category",
        action="store_true",
        help=(
            "Ignore --n-samples and instead put exactly one random image per "
            "category (restricted to --categories if given, else every "
            "category in the file) as one row each, in a single poster."
        ),
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated list of category names to use with --one-per-category, in row order.",
    )
    args = parser.parse_args()

    save_overlay_poster(
        coco_json=args.coco_json,
        image_dir=args.image_dir,
        output_path=args.output,
        n_samples=args.n_samples,
        seed=args.seed,
        alpha=args.alpha,
        dpi=args.dpi,
        show_boxes=args.show_boxes,
        show_labels=args.show_labels,
        category=args.category,
        one_per_category=args.one_per_category,
        categories=args.categories.split(",") if args.categories else None,
    )


if __name__ == "__main__":
    main()
