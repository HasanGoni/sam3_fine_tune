"""
Generate prediction / ground-truth / overlay posters for label-quality review.

Reads the JSON summaries produced by thin_wire_panda_label_quality_analysis.py
and creates a 4-panel figure for each case:
  1. original image
  2. ground truth mask
  3. prediction mask
  4. overlay (image + GT in green + predictions in red)

Output:
  <out>/uncorrect_label_poster.jpg
  <out>/no_ground_truth_available_poster.jpg
  <out>/uncorrect_label_visualizations/  (individual PNGs)
  <out>/no_ground_truth_available_visualizations/  (individual PNGs)
"""

import argparse
import json
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
    parser.add_argument("--summaries", required=True,
                        help="Folder containing uncorrect_label_summary.json and no_ground_truth_available_summary.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--category", type=int, default=9)
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument("--max-cases", type=int, default=64)
    parser.add_argument("--cols", type=int, default=4)
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


def make_quad_visualization(img_path, gt_anns, pred_anns, h, w):
    img = np.array(Image.open(img_path).convert("RGB"))
    if img.shape[0] != h or img.shape[1] != w:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    gt_mask = np.zeros((h, w), dtype=np.uint8)
    for ann in gt_anns:
        gt_mask |= mask_from_ann(ann, h, w)

    pred_mask = np.zeros((h, w), dtype=np.uint8)
    for ann in pred_anns:
        pred_mask |= mask_from_ann(ann, h, w)

    # Color masks
    gt_color = np.zeros_like(img)
    gt_color[gt_mask > 0] = [0, 220, 0]
    pred_color = np.zeros_like(img)
    pred_color[pred_mask > 0] = [255, 0, 0]

    # Overlay: original + GT + predictions
    overlay = img.copy().astype(np.float32)
    alpha = 0.4
    for color_mask in [gt_color, pred_color]:
        mask = (color_mask.sum(axis=2) > 0).astype(np.float32)
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask > 0,
                overlay[:, :, c] * (1 - alpha) + color_mask[:, :, c].astype(np.float32) * alpha,
                overlay[:, :, c]
            )
    overlay = overlay.astype(np.uint8)

    # Build 2x2 grid: original | GT mask | pred mask | overlay
    top = np.hstack([img, gt_color])
    bottom = np.hstack([pred_color, overlay])
    quad = np.vstack([top, bottom])
    return quad


def add_header(img, text, color=(255, 255, 255)):
    header_h = 32
    header = np.zeros((header_h, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, text, (5, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return np.vstack([header, img])


def build_poster(image_paths, titles, out_path, cols=4):
    if not image_paths:
        return False
    raw_imgs = [np.array(Image.open(p).convert("RGB")) for p in image_paths]
    # Add header to first image to determine cell size
    first_labeled = add_header(raw_imgs[0], titles[0])
    cell_h, cell_w = first_labeled.shape[:2]
    rows = (len(raw_imgs) + cols - 1) // cols
    grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for i, (img, title) in enumerate(zip(raw_imgs, titles)):
        r, c = divmod(i, cols)
        labeled = add_header(img, title)
        grid[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = labeled
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

    summaries_dir = Path(args.summaries)

    def process_bucket(name):
        summary_path = summaries_dir / f"{name}_summary.json"
        if not summary_path.exists():
            print(f"Summary not found: {summary_path}")
            return
        items = json.loads(summary_path.read_text())[: args.max_cases]

        vis_dir = out_dir / f"{name}_visualizations"
        vis_dir.mkdir(exist_ok=True)

        image_paths = []
        titles = []

        for i, item in enumerate(items, start=1):
            img_id = item["image_id"]
            file_name = item["file_name"]
            img_path = Path(args.images) / file_name
            if not img_path.exists():
                print(f"Warning: missing image {img_path}")
                continue

            img_meta = img_by_id[img_id]
            h, w = img_meta["height"], img_meta["width"]
            gt_anns = gt_by_img.get(img_id, [])
            pred_anns = pred_by_img.get(img_id, [])

            quad = make_quad_visualization(img_path, gt_anns, pred_anns, h, w)
            title = f"id{img_id} GT{len(gt_anns)} P{len(pred_anns)}"
            labeled = add_header(quad, title)

            out_path = vis_dir / f"{i:03d}_id{img_id}_{file_name}"
            Image.fromarray(labeled).save(out_path)

            image_paths.append(out_path)
            titles.append(title)

        if image_paths:
            poster_path = out_dir / f"{name}_poster.jpg"
            build_poster(image_paths, titles, poster_path, cols=args.cols)
            print(f"Saved {len(image_paths)} visualizations to {vis_dir}; poster: {poster_path}")
        else:
            print(f"No visualizations for {name}")

    process_bucket("uncorrect_label")
    process_bucket("no_ground_truth_available")


if __name__ == "__main__":
    main()
