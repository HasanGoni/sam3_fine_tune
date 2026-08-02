"""
Thin wire panda label-quality analysis.

Splits category-9 predictions into two buckets for manual review:
1. "no_ground_truth_available" — images where the model predicts thin wire panda
   masks but the GT has no thin wire panda annotations at all.
2. "uncorrect_label" — images that DO have thin wire panda GT, but the model
   also predicts large continuous masks that do not overlap any GT mask
   (likely missing or incorrect GT labels).

For each bucket, raw images and a grid poster are produced.
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--category", type=int, default=9)
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument("--min-pred-area", type=int, default=300,
                        help="Minimum prediction mask area to be considered a 'big continuous mask'")
    parser.add_argument("--max-overlap-iou", type=float, default=0.1,
                        help="If a big prediction mask overlaps any GT mask less than this, flag as uncorrect label")
    parser.add_argument("--max-cases", type=int, default=64)
    parser.add_argument("--cols", type=int, default=8)
    return parser.parse_args()


def mask_from_ann(ann, h, w):
    seg = ann["segmentation"]
    if isinstance(seg, dict):
        m = maskUtils.decode(seg)
    else:
        m = maskUtils.decode(maskUtils.frPyObjects(seg, h, w))
    if m.ndim == 3:
        m = m.max(axis=2)
    return m


def add_header(img, text, color=(0, 0, 0)):
    header_h = 30
    header = np.full((header_h, img.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(header, text, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return np.vstack([header, img])


def build_poster(image_paths, titles, out_path, cols=8):
    if not image_paths:
        return False
    loaded = []
    for p in image_paths:
        img = np.array(Image.open(p).convert("RGB"))
        loaded.append(img)
    cell_h, cell_w = loaded[0].shape[:2]
    rows = (len(loaded) + cols - 1) // cols
    grid = np.full((rows * (cell_h + 30), cols * cell_w, 3), 255, dtype=np.uint8)
    for i, (img, title) in enumerate(zip(loaded, titles)):
        r, c = divmod(i, cols)
        labeled = add_header(img, title)
        grid[r * (cell_h + 30) : (r + 1) * (cell_h + 30), c * cell_w : (c + 1) * cell_w] = labeled
    Image.fromarray(grid).save(out_path, quality=92)
    return True


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = json.loads(Path(args.gt).read_text())
    preds = json.loads(Path(args.pred).read_text())

    img_by_id = {im["id"]: im for im in gt["images"]}

    gt_by_img = defaultdict(list)
    for ann in gt["annotations"]:
        if ann["category_id"] == args.category:
            gt_by_img[ann["image_id"]].append(ann)

    pred_by_img = defaultdict(list)
    for ann in preds:
        if ann["category_id"] == args.category and ann["score"] >= args.score_thresh:
            pred_by_img[ann["image_id"]].append(ann)

    no_gt_images = []
    uncorrect_label_images = []

    for img_id, pred_anns in pred_by_img.items():
        gt_anns = gt_by_img.get(img_id, [])
        img_meta = img_by_id[img_id]
        h, w = img_meta["height"], img_meta["width"]

        if not gt_anns:
            no_gt_images.append({
                "image_id": img_id,
                "file_name": img_meta["file_name"],
                "pred_count": len(pred_anns),
                "reason": "no thin wire panda GT in this image",
            })
            continue

        gt_masks = [mask_from_ann(a, h, w) for a in gt_anns]
        gt_union = np.any(np.stack(gt_masks, axis=0), axis=0)

        # Look for big continuous prediction masks with low overlap to GT
        big_bad_masks = 0
        for ann in pred_anns:
            pmask = mask_from_ann(ann, h, w)
            area = int(pmask.sum())
            if area < args.min_pred_area:
                continue
            overlap = float(np.logical_and(pmask, gt_union).sum()) / area
            if overlap < args.max_overlap_iou:
                big_bad_masks += 1

        if big_bad_masks > 0:
            uncorrect_label_images.append({
                "image_id": img_id,
                "file_name": img_meta["file_name"],
                "pred_count": len(pred_anns),
                "big_unmatched_masks": big_bad_masks,
                "gt_count": len(gt_anns),
                "reason": f"{big_bad_masks} large predicted masks overlap < {args.max_overlap_iou:.0%} with GT",
            })

    # Sort by number of suspicious masks / predictions
    no_gt_images.sort(key=lambda x: x["pred_count"], reverse=True)
    uncorrect_label_images.sort(key=lambda x: x["big_unmatched_masks"], reverse=True)

    print(f"Images with predictions but NO GT: {len(no_gt_images)}")
    print(f"Images with big unmatched predictions (uncorrect label): {len(uncorrect_label_images)}")

    def save_bucket(bucket, name):
        bucket_dir = out_dir / name
        bucket_dir.mkdir(exist_ok=True)
        image_paths = []
        titles = []
        for i, item in enumerate(bucket[: args.max_cases], start=1):
            src = Path(args.images) / item["file_name"]
            if not src.exists():
                print(f"Warning: missing image {src}")
                continue
            dst = bucket_dir / f"{i:03d}_id{item['image_id']}_{item['file_name']}"
            shutil.copy2(src, dst)
            image_paths.append(dst)
            titles.append(f"id{item['image_id']} p={item['pred_count']}")

        if image_paths:
            poster_path = out_dir / f"{name}_poster.jpg"
            build_poster(image_paths, titles, poster_path, cols=args.cols)
            print(f"Saved {len(image_paths)} images to {bucket_dir}; poster: {poster_path}")
        else:
            print(f"No images saved for {name}")

        summary_path = out_dir / f"{name}_summary.json"
        summary_path.write_text(json.dumps(bucket, indent=2))
        return bucket_dir

    save_bucket(no_gt_images, "no_ground_truth_available")
    save_bucket(uncorrect_label_images, "uncorrect_label")

    # Also write a combined summary
    combined = {
        "no_ground_truth_available": no_gt_images,
        "uncorrect_label": uncorrect_label_images,
    }
    (out_dir / "all_label_quality_cases.json").write_text(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
