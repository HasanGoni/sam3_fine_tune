#!/usr/bin/env python3
"""Save a large grid poster of COCO category images with mask overlays.

Single category: every image containing the category is rendered in a square-ish
grid (good for checking whether all instances are labeled).

Multiple categories: one poster with one labeled row per category, showing up to
`--max-per-category` images per row (good for comparing categories side-by-side).
"""

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend for HPC
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools.coco import COCO


DEFAULT_COLORS = [
    (0.0, 1.0, 0.0),   # green
    (1.0, 0.0, 0.0),   # red
    (0.0, 0.0, 1.0),   # blue
    (1.0, 1.0, 0.0),   # yellow
    (1.0, 0.0, 1.0),   # magenta
    (0.0, 1.0, 1.0),   # cyan
    (1.0, 0.5, 0.0),   # orange
    (0.5, 0.0, 1.0),   # purple
    (0.0, 0.5, 0.0),   # dark green
]


def _render_cell(ax, coco, img_id, cat_id, panel_size, alpha, color):
    """Render one grid cell: image + mask overlay for a single category."""
    img_info = coco.loadImgs(img_id)[0]
    img_path = coco.img_dir / img_info["file_name"]
    img = np.array(Image.open(img_path).convert("RGB"))

    ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[cat_id])
    anns = coco.loadAnns(ann_ids)
    mask_canvas = np.zeros_like(img)
    for ann in anns:
        mask = coco.annToMask(ann).astype(bool)
        if mask.shape[:2] != img.shape[:2]:
            mask = np.array(
                Image.fromarray(mask.astype(np.uint8)).resize(
                    (img.shape[1], img.shape[0]), Image.NEAREST
                )
            ).astype(bool)
        mask_canvas[mask] = color

    pil_img = Image.fromarray(img)
    pil_mask = Image.fromarray(mask_canvas)
    pil_img = pil_img.resize((panel_size, panel_size), Image.LANCZOS)
    pil_mask = pil_mask.resize((panel_size, panel_size), Image.NEAREST)
    img_resized = np.array(pil_img)
    mask_resized = np.array(pil_mask)

    ax.imshow(img_resized)
    ax.imshow(mask_resized, alpha=alpha)
    ax.axis("off")


def save_category_grid_poster(
    coco_json: Path,
    categories: list[str],
    output_path: Path,
    image_dir: Path | None = None,
    panel_size: int = 256,
    n_cols: int | None = None,
    max_per_category: int = 0,
    alpha: float = 0.5,
    mask_colors: list[tuple[float, float, float]] | None = None,
    dpi: int = 100,
    jpeg_quality: int = 90,
):
    """Render grid poster for one or more COCO categories.

    Args:
        coco_json: Path to COCO annotation JSON.
        categories: List of category names to render.
        output_path: Output JPEG path.
        image_dir: Directory containing COCO images. Defaults to
            `<coco-json parent>/images`.
        panel_size: Size of each square panel in pixels.
        n_cols: Number of columns. For single category defaults to ~sqrt(n_images).
            For multiple categories defaults to 16.
        max_per_category: Maximum images per category when multiple categories are
            given. 0 means use all images (not recommended for many categories).
        alpha: Mask overlay alpha.
        mask_colors: List of RGB tuples for mask colors. Cycles if fewer than
            categories.
        dpi: JPEG DPI.
        jpeg_quality: JPEG quality.
    """
    coco_json = Path(coco_json)
    if image_dir is None:
        image_dir = coco_json.parent / "images"
    image_dir = Path(image_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coco = COCO(str(coco_json))
    coco.img_dir = image_dir
    cat_name_to_id = {c["name"]: c["id"] for c in coco.dataset["categories"]}

    for cat in categories:
        if cat not in cat_name_to_id:
            raise ValueError(
                f"Category '{cat}' not found. Available: {sorted(cat_name_to_id)}"
            )

    colors = mask_colors or DEFAULT_COLORS

    if len(categories) == 1:
        # Single category: render ALL images in a square-ish grid
        cat_name = categories[0]
        cat_id = cat_name_to_id[cat_name]
        color = (np.array(colors[0]) * 255).astype(np.uint8)
        img_ids = sorted(set(coco.getImgIds(catIds=[cat_id])))
        n_images = len(img_ids)
        if n_images == 0:
            raise ValueError(f"No images found for category '{cat_name}'")

        print(f"Building single-category poster for '{cat_name}': {n_images} images")
        n_cols = n_cols or math.ceil(math.sqrt(n_images))
        n_rows = math.ceil(n_images / n_cols)
        print(f"Grid: {n_rows} rows x {n_cols} cols, panel size: {panel_size}px")

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * panel_size / dpi, n_rows * panel_size / dpi),
            dpi=dpi,
        )
        axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

        for idx, img_id in enumerate(img_ids):
            row = idx // n_cols
            col = idx % n_cols
            _render_cell(axes[row, col], coco, img_id, cat_id, panel_size, alpha, color)

        for idx in range(n_images, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis("off")

        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02, hspace=0.02)

    else:
        # Multiple categories: one row per category, labeled, with image grids
        n_cols = n_cols or 16
        rows = []
        for cat_name in categories:
            cat_id = cat_name_to_id[cat_name]
            img_ids = sorted(set(coco.getImgIds(catIds=[cat_id])))
            if max_per_category > 0:
                img_ids = img_ids[:max_per_category]
            rows.append((cat_name, cat_id, img_ids))

        n_rows = len(rows)
        print(
            f"Building multi-category poster: {n_rows} categories, "
            f"up to {n_cols} columns, panel size: {panel_size}px"
        )

        # One extra column on the left for category labels
        fig, axes = plt.subplots(
            n_rows,
            n_cols + 1,
            figsize=((n_cols + 1) * panel_size / dpi, n_rows * panel_size / dpi),
            dpi=dpi,
            gridspec_kw={"width_ratios": [0.4] + [1.0] * n_cols},
        )
        axes = np.atleast_1d(axes).reshape(n_rows, n_cols + 1)

        for row_idx, (cat_name, cat_id, img_ids) in enumerate(rows):
            color = (np.array(colors[row_idx % len(colors)]) * 255).astype(np.uint8)

            # Label cell
            ax_label = axes[row_idx, 0]
            ax_label.text(
                0.5,
                0.5,
                cat_name,
                rotation=90,
                va="center",
                ha="center",
                fontsize=max(8, panel_size // 6),
                transform=ax_label.transAxes,
            )
            ax_label.axis("off")

            # Image cells
            for col_idx, img_id in enumerate(img_ids[:n_cols]):
                _render_cell(
                    axes[row_idx, col_idx + 1],
                    coco,
                    img_id,
                    cat_id,
                    panel_size,
                    alpha,
                    color,
                )

            for col_idx in range(len(img_ids), n_cols):
                axes[row_idx, col_idx + 1].axis("off")

        plt.subplots_adjust(left=0.02, right=1, bottom=0, top=1, wspace=0.02, hspace=0.02)

    fig.savefig(
        output_path,
        format="jpeg",
        dpi=dpi,
        pil_kwargs={"quality": jpeg_quality},
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(fig)

    print(f"Saved poster to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Save a grid poster of COCO category images with mask overlays."
    )
    parser.add_argument(
        "--coco-json",
        type=Path,
        required=True,
        help="Path to COCO annotation JSON.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        required=True,
        help="Comma-separated category names to render (e.g. 'thin wire panda,thick wire panda').",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JPEG path.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory containing COCO images. Defaults to <coco-json parent>/images.",
    )
    parser.add_argument(
        "--panel-size",
        type=int,
        default=256,
        help="Size of each square panel in pixels (default: 256).",
    )
    parser.add_argument(
        "--n-cols",
        type=int,
        default=None,
        help="Number of image columns. Single category defaults to ~sqrt(n_images); multi defaults to 16.",
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=0,
        help="For multi-category posters, max images per category. 0 = all (default: 0).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Mask overlay alpha (default: 0.5).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="JPEG DPI (default: 100).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality (default: 90).",
    )
    args = parser.parse_args()

    category_list = [c.strip() for c in args.categories.split(",") if c.strip()]
    if not category_list:
        raise ValueError("At least one category must be provided.")

    save_category_grid_poster(
        coco_json=args.coco_json,
        categories=category_list,
        output_path=args.output,
        image_dir=args.image_dir,
        panel_size=args.panel_size,
        n_cols=args.n_cols,
        max_per_category=args.max_per_category,
        alpha=args.alpha,
        dpi=args.dpi,
        jpeg_quality=args.jpeg_quality,
    )


if __name__ == "__main__":
    main()
