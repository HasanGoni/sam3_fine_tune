"""Patch selected categories in an existing COCO dataset.

Replaces source images that contain target categories with fixed-size overlapping
patches, preserving annotations from ALL categories that fall inside each patch.
Images without target categories are copied unchanged.

Usage:
    python patch_coco_categories.py --config configs/patch_coco_categories_example.yaml
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm.auto import tqdm

from coco_format import save_metadata_file


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_patch_grid(image_h: int, image_w: int, patch_size: int, overlap: int):
    """Return (n_y, n_x, list of y starts, list of x starts)."""
    stride = max(1, patch_size - overlap)
    n_y = max(1, int(np.ceil((image_h - patch_size) / stride)) + 1)
    n_x = max(1, int(np.ceil((image_w - patch_size) / stride)) + 1)
    ys = [min(i * stride, image_h - patch_size) for i in range(n_y)]
    xs = [min(j * stride, image_w - patch_size) for j in range(n_x)]
    return n_y, n_x, ys, xs


def clip_segmentation_to_patch(segmentation: list, x0: int, y0: int, patch_size: int):
    """Clip a COCO polygon segmentation to a patch and return new polygons.

    Each ``segmentation`` entry is a flat list of ``[x1, y1, x2, y2, ...]``.
    Coordinates are shifted by ``(-x0, -y0)`` and clipped to the patch bounds.
    Empty or degenerate polygons are dropped.
    """
    new_polys = []
    for poly in segmentation:
        pts = np.array(poly).reshape(-1, 2).astype(np.float32)
        pts[:, 0] -= x0
        pts[:, 1] -= y0

        # Drop points outside patch bounds.
        inside = (
            (pts[:, 0] >= 0)
            & (pts[:, 0] <= patch_size)
            & (pts[:, 1] >= 0)
            & (pts[:, 1] <= patch_size)
        )
        if not inside.any():
            continue

        # Simple clipping: keep only inside points.  For small patches this is
        # usually sufficient; complex polygons crossing the boundary twice may
        # need full Sutherland-Hodgman, but those are rare for instance masks.
        clipped = pts[inside]
        if len(clipped) < 3:
            continue

        # Round and drop duplicates.
        clipped = np.round(clipped).astype(np.int32)
        clipped = np.unique(clipped, axis=0)
        if len(clipped) < 3:
            continue

        new_polys.append(clipped.flatten().tolist())

    return new_polys


def bbox_area(bbox: list) -> float:
    return max(0.0, bbox[2]) * max(0.0, bbox[3])


def clip_bbox_to_patch(bbox: list, x0: int, y0: int, patch_size: int):
    """Shift and clip a COCO bbox ``[x, y, w, h]`` to a patch."""
    x, y, w, h = bbox
    x1 = max(0.0, x - x0)
    y1 = max(0.0, y - y0)
    x2 = min(patch_size, x + w - x0)
    y2 = min(patch_size, y + h - y0)
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def process_split(
    input_split_dir: Path,
    output_split_dir: Path,
    target_cat_ids: set,
    patch_size: int,
    overlap: int,
    min_target_area: float,
) -> dict:
    """Patch target categories in one split and return new COCO dict."""
    ann_path = input_split_dir / "annotation_merged.json"
    with open(ann_path) as f:
        coco = json.load(f)

    src_img_dir = input_split_dir / "images"
    dst_img_dir = output_split_dir / "images"
    dst_img_dir.mkdir(parents=True, exist_ok=True)

    # Build lookups.
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_img: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    target_img_ids = set()
    for ann in coco["annotations"]:
        if ann["category_id"] in target_cat_ids:
            target_img_ids.add(ann["image_id"])

    new_images = []
    new_annotations = []
    next_ann_id = 1
    next_img_id = 1

    for img in tqdm(coco["images"], desc=f"Processing {output_split_dir.name}"):
        img_id = img["id"]
        src_img_path = src_img_dir / img["file_name"]
        if not src_img_path.exists():
            raise FileNotFoundError(f"Image not found: {src_img_path}")

        if img_id not in target_img_ids:
            # Copy image unchanged.
            shutil.copy2(src_img_path, dst_img_dir / img["file_name"])
            new_img = dict(img)
            new_img["id"] = next_img_id
            new_images.append(new_img)

            for ann in anns_by_img.get(img_id, []):
                new_ann = dict(ann)
                new_ann["id"] = next_ann_id
                new_ann["image_id"] = next_img_id
                new_annotations.append(new_ann)
                next_ann_id += 1

            next_img_id += 1
            continue

        # Image contains target categories -> patch it.
        image = cv2.imread(str(src_img_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Could not read image {src_img_path}")
        h, w = image.shape[:2]

        n_y, n_x, ys, xs = compute_patch_grid(h, w, patch_size, overlap)

        for py, y0 in enumerate(ys):
            for px, x0 in enumerate(xs):
                patch_img = image[y0 : y0 + patch_size, x0 : x0 + patch_size]
                pad_h = patch_size - patch_img.shape[0]
                pad_w = patch_size - patch_img.shape[1]
                if pad_h > 0 or pad_w > 0:
                    patch_img = cv2.copyMakeBorder(
                        patch_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
                    )

                patch_name = f"{Path(img['file_name']).stem}_py{py}_px{px}_y{y0}_x{x0}.png"

                patch_anns = []
                has_target = False
                for ann in anns_by_img.get(img_id, []):
                    new_bbox = clip_bbox_to_patch(ann["bbox"], x0, y0, patch_size)
                    if bbox_area(new_bbox) < min_target_area:
                        continue

                    new_segs = clip_segmentation_to_patch(
                        ann.get("segmentation", []), x0, y0, patch_size
                    )
                    if not new_segs:
                        continue

                    new_ann = {
                        "id": next_ann_id,
                        "image_id": next_img_id,
                        "category_id": ann["category_id"],
                        "bbox": new_bbox,
                        "segmentation": new_segs,
                        "area": sum(cv2.contourArea(np.array(s).reshape(-1, 1, 2).astype(np.float32)) for s in new_segs),
                        "iscrowd": ann.get("iscrowd", 0),
                    }
                    patch_anns.append(new_ann)
                    next_ann_id += 1

                    if ann["category_id"] in target_cat_ids:
                        has_target = True

                if not has_target:
                    continue

                cv2.imwrite(str(dst_img_dir / patch_name), patch_img)
                new_images.append(
                    {
                        "id": next_img_id,
                        "file_name": patch_name,
                        "width": patch_size,
                        "height": patch_size,
                    }
                )
                new_annotations.extend(patch_anns)
                next_img_id += 1

    return {
        "info": coco.get("info", {}),
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco["categories"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    patch_cfg = cfg["patch"]
    patch_size = int(patch_cfg["size"])
    overlap = int(patch_cfg.get("overlap", 0))
    min_target_area = float(patch_cfg.get("min_target_area", 1.0))

    target_categories = cfg["target_categories"]
    input_root = Path(cfg["input"]["root"])
    output_root = Path(cfg["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, output_root / "patch_coco_config.yaml")

    for split in cfg["splits"]:
        input_split_dir = input_root / split
        output_split_dir = output_root / split
        output_split_dir.mkdir(parents=True, exist_ok=True)

        ann_path = input_split_dir / "annotation_merged.json"
        with open(ann_path) as f:
            coco = json.load(f)

        cat_name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
        target_cat_ids = set()
        for name in target_categories:
            if name not in cat_name_to_id:
                raise ValueError(f"Target category '{name}' not found in {ann_path}. "
                                 f"Available: {list(cat_name_to_id.keys())}")
            target_cat_ids.add(cat_name_to_id[name])

        new_coco = process_split(
            input_split_dir=input_split_dir,
            output_split_dir=output_split_dir,
            target_cat_ids=target_cat_ids,
            patch_size=patch_size,
            overlap=overlap,
            min_target_area=min_target_area,
        )

        out_ann = output_split_dir / "annotation_merged.json"
        with open(out_ann, "w") as f:
            json.dump(new_coco, f, indent=2)
        save_metadata_file(output_split_dir, "annotation_merged", new_coco)

        print(
            f"[{split}] images: {len(new_coco['images'])}, "
            f"annotations: {len(new_coco['annotations'])}, "
            f"categories: {[c['name'] for c in new_coco['categories']]}"
        )

    print(f"\nPatched dataset ready at: {output_root}")


if __name__ == "__main__":
    main()
