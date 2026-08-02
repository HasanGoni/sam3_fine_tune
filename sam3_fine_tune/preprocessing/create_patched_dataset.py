"""Create a patch-based COCO dataset from large source images and masks.

This script tiles large images (and their masks) into fixed-size patches with a
configurable overlap, filters out mostly-empty patches, and writes a COCO-format
annotation file compatible with the SAM3 training pipeline.

Usage:
    python create_patched_dataset.py --config configs/patch_dataset_example.yaml
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm.auto import tqdm

from coco_format import form_coco_dataset, save_ann_file, save_metadata_file


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def compute_patch_grid(image_h: int, image_w: int, patch_size: int, overlap: int):
    """Return (rows, cols) and a list of (y_start, x_start) for sliding patches.

    The stride is ``patch_size - overlap``.  The last patch in each dimension is
    aligned to the far edge, so adjacent final patches may have a smaller overlap
    than requested, but no patch exceeds the image bounds.
    """
    stride = max(1, patch_size - overlap)

    # Number of patches along each axis, ensuring the final patch touches the edge.
    n_y = max(1, int(np.ceil((image_h - patch_size) / stride)) + 1)
    n_x = max(1, int(np.ceil((image_w - patch_size) / stride)) + 1)

    ys = []
    xs = []
    for i in range(n_y):
        y = min(i * stride, image_h - patch_size)
        ys.append(y)
    for j in range(n_x):
        x = min(j * stride, image_w - patch_size)
        xs.append(x)

    return n_y, n_x, ys, xs


def process_split(
    split_cfg: dict,
    output_split_dir: Path,
    patch_size: int,
    overlap: int,
    min_mask_ratio: float,
    category: dict,
    keep_masks: bool = True,
) -> dict:
    """Tile one split (train/test) and return COCO dataset dict."""
    src_img_dir = Path(split_cfg["images"])
    src_mask_dir = Path(split_cfg["masks"])

    dst_img_dir = output_split_dir / "images"
    dst_mask_dir = output_split_dir / "masks" if keep_masks else None
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    if dst_mask_dir is not None:
        dst_mask_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(src_img_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {src_img_dir}")

    kept_patches = 0
    skipped_patches = 0

    for img_path in tqdm(image_paths, desc=f"Patching {output_split_dir.name}"):
        mask_path = src_mask_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {img_path}: {mask_path}")

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Could not read image {img_path}")

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read mask {mask_path}")

        h, w = img.shape[:2]
        n_y, n_x, ys, xs = compute_patch_grid(h, w, patch_size, overlap)

        for py, y in enumerate(ys):
            for px, x in enumerate(xs):
                patch_img = img[y : y + patch_size, x : x + patch_size]
                patch_mask = mask[y : y + patch_size, x : x + patch_size]

                # Pad if the image is smaller than the patch size.
                if patch_img.shape[0] != patch_size or patch_img.shape[1] != patch_size:
                    pad_h = patch_size - patch_img.shape[0]
                    pad_w = patch_size - patch_img.shape[1]
                    patch_img = cv2.copyMakeBorder(
                        patch_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
                    )
                    patch_mask = cv2.copyMakeBorder(
                        patch_mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
                    )

                valid_ratio = np.count_nonzero(patch_mask) / (patch_size * patch_size)
                if valid_ratio < min_mask_ratio:
                    skipped_patches += 1
                    continue

                out_name = f"{img_path.stem}_py{py}_px{px}_y{y}_x{x}.png"
                cv2.imwrite(str(dst_img_dir / out_name), patch_img)
                if dst_mask_dir is not None:
                    cv2.imwrite(str(dst_mask_dir / out_name), patch_mask)
                kept_patches += 1

    print(
        f"[{output_split_dir.name}] kept {kept_patches} patches, "
        f"skipped {skipped_patches} empty/low-content patches"
    )

    if dst_mask_dir is not None:
        coco_data = form_coco_dataset(dst_mask_dir, dst_img_dir, categories=[category])
    else:
        # Fallback: build COCO without annotations if masks were not kept.
        coco_data = {
            "info": {"description": "Patched Dataset", "version": "1.0", "year": 2025},
            "images": [],
            "annotations": [],
            "categories": [category],
        }
        for img_id, img_path in enumerate(sorted(dst_img_dir.glob("*.png")), start=1):
            h, w = cv2.imread(str(img_path)).shape[:2]
            coco_data["images"].append(
                {"id": img_id, "file_name": img_path.name, "width": w, "height": h}
            )

    return coco_data


def main():
    parser = argparse.ArgumentParser(
        description="Create a patch-based COCO dataset from large images and masks."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument(
        "--keep-masks",
        action="store_true",
        default=True,
        help="Keep the per-patch binary masks on disk (default: True)",
    )
    parser.add_argument(
        "--no-keep-masks",
        action="store_false",
        dest="keep_masks",
        help="Do not keep the per-patch binary masks on disk",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    patch_cfg = cfg["patch"]
    patch_size = int(patch_cfg["size"])
    overlap = int(patch_cfg.get("overlap", 0))
    min_mask_ratio = float(patch_cfg.get("min_mask_ratio", 0.001))

    category = {
        "id": int(cfg["category"]["id"]),
        "name": cfg["category"]["name"],
    }

    output_root = Path(cfg["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)

    # Save a copy of the config used for reproducibility.
    shutil.copy(args.config, output_root / "patch_config.yaml")

    for split_name in cfg["splits"]:
        split_cfg = cfg["splits"][split_name]
        output_split_dir = output_root / split_name

        coco_data = process_split(
            split_cfg=split_cfg,
            output_split_dir=output_split_dir,
            patch_size=patch_size,
            overlap=overlap,
            min_mask_ratio=min_mask_ratio,
            category=category,
            keep_masks=args.keep_masks,
        )

        save_ann_file(output_split_dir, "annotation", coco_data)
        save_metadata_file(output_split_dir, "annotation", coco_data)
        print(
            f"Saved {split_name}: {len(coco_data['images'])} images, "
            f"{len(coco_data['annotations'])} annotations "
            f"-> {output_split_dir / 'annotation.json'}"
        )

    print(f"\nPatched dataset ready at: {output_root}")


if __name__ == "__main__":
    main()
