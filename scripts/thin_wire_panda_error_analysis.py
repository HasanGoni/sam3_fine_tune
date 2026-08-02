"""
Thin wire panda error analysis.

Identifies the worst-performing test images for category 9 (thin wire panda)
from a COCO predictions file, copies visualizations of the bad cases into a
single folder, and creates a combined grid poster for quick inspection.

Usage (inside project venv on HPC):
    python scripts/thin_wire_panda_error_analysis.py \
        --gt /path/to/test/annotation_merged.json \
        --pred /path/to/coco_predictions_segm.json \
        --images /path/to/test/images \
        --out /path/to/output_dir \
        --category 9 \
        --max-cases 64
"""

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="Path to COCO-format GT JSON")
    parser.add_argument("--pred", required=True, help="Path to COCO-format predictions JSON")
    parser.add_argument("--images", required=True, help="Path to image directory")
    parser.add_argument("--out", required=True, help="Output directory for bad-case visualizations")
    parser.add_argument("--category", type=int, default=9, help="Category ID to analyze")
    parser.add_argument("--max-cases", type=int, default=64, help="Maximum bad cases to visualize")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for TP/FP/FN")
    parser.add_argument("--score-thresh", type=float, default=0.05, help="Score threshold for predictions")
    return parser.parse_args()


def mask_from_ann(ann, h, w):
    seg = ann["segmentation"]
    if isinstance(seg, dict):
        m = maskUtils.decode(seg)
    else:
        m = maskUtils.decode(maskUtils.frPyObjects(seg, h, w))
    if m.ndim == 3:
        # Multi-component polygon mask -> collapse to single 2D mask
        m = m.max(axis=2)
    return m


def compute_iou_matrix(gt_masks, pred_masks):
    """Return (#gt, #pred) IoU matrix."""
    if len(gt_masks) == 0 or len(pred_masks) == 0:
        return np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
    gt_stack = np.stack(gt_masks, axis=2)  # HxWxNgt
    pred_stack = np.stack(pred_masks, axis=2)  # HxWxNpred
    intersection = np.einsum("ijk,ijl->kl", gt_stack, pred_stack)  # Ngt x Npred
    gt_area = gt_stack.sum(axis=(0, 1))  # (Ngt,)
    pred_area = pred_stack.sum(axis=(0, 1))  # (Npred,)
    union = gt_area[:, None] + pred_area[None, :] - intersection
    union = np.maximum(union, 1e-8)
    return intersection / union


def make_overlay(img_path, gt_anns, pred_anns, img_h, img_w):
    img = np.array(Image.open(img_path).convert("RGB"))
    # Resize if needed
    if img.shape[0] != img_h or img.shape[1] != img_w:
        img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

    gt_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for ann in gt_anns:
        gt_mask |= mask_from_ann(ann, img_h, img_w)

    pred_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for ann in pred_anns:
        pred_mask |= mask_from_ann(ann, img_h, img_w)

    overlay = img.copy()
    # GT in green
    overlay[gt_mask > 0] = (overlay[gt_mask > 0] * 0.6 + np.array([0, 220, 0]) * 0.4).astype(np.uint8)
    # Predictions in red
    overlay[pred_mask > 0] = (overlay[pred_mask > 0] * 0.6 + np.array([255, 0, 0]) * 0.4).astype(np.uint8)

    # Side-by-side: original | overlay
    combined = np.hstack([img, overlay])
    return combined, gt_mask, pred_mask


def add_text_header(img, text, color=(0, 0, 0)):
    h, w = img.shape[:2]
    header_h = 30
    header = np.full((header_h, w, 3), 255, dtype=np.uint8)
    cv2.putText(header, text, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return np.vstack([header, img])


def make_grid(images, titles, cols=8):
    if len(images) == 0:
        return None
    rows = (len(images) + cols - 1) // cols
    labeled0 = add_text_header(images[0], titles[0])
    cell_h, cell_w = labeled0.shape[:2]
    grid = np.full((rows * cell_h, cols * cell_w, 3), 255, dtype=np.uint8)
    for i, (img, title) in enumerate(zip(images, titles)):
        r, c = divmod(i, cols)
        labeled = add_text_header(img, title)
        grid[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = labeled
    return grid


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.gt) as f:
        gt = json.load(f)
    with open(args.pred) as f:
        preds = json.load(f)

    cat_info = {c["id"]: c["name"] for c in gt["categories"]}
    cat_name = cat_info.get(args.category, f"cat_{args.category}")

    img_by_id = {im["id"]: im for im in gt["images"]}

    gt_by_img = defaultdict(list)
    for ann in gt["annotations"]:
        if ann["category_id"] == args.category:
            gt_by_img[ann["image_id"]].append(ann)

    pred_by_img = defaultdict(list)
    for ann in preds:
        if ann["category_id"] == args.category and ann["score"] >= args.score_thresh:
            pred_by_img[ann["image_id"]].append(ann)

    results = []
    for img_id, img_meta in img_by_id.items():
        gt_anns = gt_by_img.get(img_id, [])
        pred_anns = pred_by_img.get(img_id, [])
        if not gt_anns and not pred_anns:
            continue

        h, w = img_meta["height"], img_meta["width"]
        gt_masks = [mask_from_ann(a, h, w) for a in gt_anns]
        pred_masks = [mask_from_ann(a, h, w) for a in pred_anns]
        iou_mat = compute_iou_matrix(gt_masks, pred_masks)

        # Hungarian-ish greedy matching for TP/FP/FN
        tp = 0
        matched_gt = set()
        matched_pred = set()
        if iou_mat.size > 0:
            sorted_pairs = np.dstack(np.unravel_index(np.argsort(-iou_mat.ravel()), iou_mat.shape))[0]
            for g, p in sorted_pairs:
                g, p = int(g), int(p)
                if g in matched_gt or p in matched_pred:
                    continue
                if iou_mat[g, p] >= args.iou_thresh:
                    matched_gt.add(g)
                    matched_pred.add(p)
                    tp += 1

        fn = len(gt_anns) - tp
        fp = len(pred_anns) - tp
        if len(gt_anns) > 0 and len(pred_anns) > 0:
            mean_iou = float(iou_mat.max(axis=1).mean())
            worst_iou = float(iou_mat.max(axis=1).min())
        else:
            mean_iou = 0.0
            worst_iou = 0.0

        # Combined badness score: prioritize images with TP=0 and high FP
        badness = fp + 3 * fn + (1.0 - mean_iou) * 2
        results.append({
            "image_id": img_id,
            "file_name": img_meta["file_name"],
            "gt": len(gt_anns),
            "pred": len(pred_anns),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "mean_iou": mean_iou,
            "worst_iou": worst_iou,
            "badness": badness,
        })

    results.sort(key=lambda x: x["badness"], reverse=True)

    summary_path = out_dir / "bad_case_summary.json"
    summary_path.write_text(json.dumps(results[: args.max_cases], indent=2))
    print(f"Saved summary of top {len(results[:args.max_cases])} bad cases to {summary_path}")

    # Copy visualizations
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    images_for_grid = []
    titles_for_grid = []

    for rank, r in enumerate(results[: args.max_cases], start=1):
        img_id = r["image_id"]
        file_name = r["file_name"]
        img_path = Path(args.images) / file_name
        if not img_path.exists():
            print(f"Warning: image not found {img_path}")
            continue

        img_meta = img_by_id[img_id]
        h, w = img_meta["height"], img_meta["width"]
        gt_anns = gt_by_img.get(img_id, [])
        pred_anns = pred_by_img.get(img_id, [])

        overlay, _, _ = make_overlay(img_path, gt_anns, pred_anns, h, w)
        title = f"id{img_id} GT{r['gt']} P{r['pred']} TP{r['tp']} FP{r['fp']} FN{r['fn']}"
        labeled = add_text_header(overlay, title)

        out_path = vis_dir / f"{rank:03d}_id{img_id}_{file_name}"
        Image.fromarray(labeled).save(out_path)

        # Resize raw overlay for grid; make_grid will add the header itself
        grid_img = cv2.resize(overlay, (512, 256))
        images_for_grid.append(grid_img)
        titles_for_grid.append(title)

    if images_for_grid:
        grid = make_grid(images_for_grid, titles_for_grid, cols=8)
        poster_path = out_dir / "thin_wire_panda_bad_cases_poster.jpg"
        Image.fromarray(grid).save(poster_path, quality=92)
        print(f"Saved poster to {poster_path} ({len(images_for_grid)} cases)")
    else:
        print("No visualizations generated.")

    # Also copy raw bad-case images into a flat folder
    raw_dir = out_dir / "bad_case_raw_images"
    raw_dir.mkdir(exist_ok=True)
    for r in results[: args.max_cases]:
        src = Path(args.images) / r["file_name"]
        if src.exists():
            dst = raw_dir / f"id{r['image_id']}_{r['file_name']}"
            shutil.copy2(src, dst)
    print(f"Copied {len(results[:args.max_cases])} raw bad-case images to {raw_dir}")

    # Write a short text summary
    txt_path = out_dir / "summary.txt"
    with open(txt_path, "w") as f:
        f.write(f"Category: {cat_name} (ID {args.category})\n")
        f.write(f"Total images with GT or predictions: {len(results)}\n")
        f.write(f"Top {len(results[:args.max_cases])} worst cases by badness score\n\n")
        f.write(f"{'rank':>4} {'id':>8} {'gt':>4} {'pred':>4} {'tp':>4} {'fp':>4} {'fn':>4} {'mean_iou':>8} {'file_name'}\n")
        for i, r in enumerate(results[: args.max_cases], start=1):
            f.write(
                f"{i:>4} {r['image_id']:>8} {r['gt']:>4} {r['pred']:>4} "
                f"{r['tp']:>4} {r['fp']:>4} {r['fn']:>4} {r['mean_iou']:>8.3f} {r['file_name']}\n"
            )
    print(f"Saved text summary to {txt_path}")


if __name__ == "__main__":
    main()
