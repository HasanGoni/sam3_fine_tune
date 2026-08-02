"""
Conservative cleaning of thin wire panda TRAINING annotations.

Training data is risky to auto-label from predictions, because bad labels will
be learned. This script therefore takes a conservative route:

- For images flagged as "no_ground_truth_available" it converts predictions to
  GT only when the predictions are high-confidence and sparse:
      score >= --min-score (default 0.5)
      prediction count <= --max-preds (default 5)
- Images that fail the conservative filter are kept as-is; their other-category
  annotations are preserved. Only unreliable thin-wire-panda predictions are
  ignored, not the whole image.
- Images flagged as "uncorrect_label" are also kept; no predictions are added
  to them.

This preserves the full training set while only adding high-confidence thin
wire panda annotations where they were missing.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--label-quality-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--category", type=int, default=9)
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument("--min-score", type=float, default=0.5,
                        help="Minimum prediction score to convert to GT")
    parser.add_argument("--max-preds", type=int, default=5,
                        help="Maximum number of predictions on a no-GT image to consider converting")
    return parser.parse_args()


def main():
    args = parse_args()

    gt = json.loads(Path(args.gt).read_text())
    preds = json.loads(Path(args.pred).read_text())

    lq_dir = Path(args.label_quality_dir)
    no_gt = json.loads((lq_dir / "no_ground_truth_available_summary.json").read_text())
    uncorrect = json.loads((lq_dir / "uncorrect_label_summary.json").read_text())

    uncorrect_ids = {x["image_id"] for x in uncorrect}
    no_gt_ids = {x["image_id"] for x in no_gt}

    # Count qualifying predictions per no-GT image
    pred_counts_by_image = defaultdict(int)
    max_score_by_image = defaultdict(float)
    for p in preds:
        if p["category_id"] == args.category and p["score"] >= args.score_thresh:
            if p["image_id"] in no_gt_ids:
                pred_counts_by_image[p["image_id"]] += 1
                max_score_by_image[p["image_id"]] = max(max_score_by_image[p["image_id"]], p["score"])

    # Decide which no-GT images to convert: sparse + high-confidence
    no_gt_convert_ids = set()
    no_gt_skip_ids = set()
    for img_id in no_gt_ids:
        n_preds = pred_counts_by_image.get(img_id, 0)
        max_score = max_score_by_image.get(img_id, 0.0)
        if n_preds <= args.max_preds and max_score >= args.min_score:
            no_gt_convert_ids.add(img_id)
        else:
            no_gt_skip_ids.add(img_id)

    print(f"no-GT images to convert: {len(no_gt_convert_ids)}")
    print(f"no-GT images skipped (too many/low-confidence preds): {len(no_gt_skip_ids)}")
    print(f"uncorrect_label images kept unchanged: {len(uncorrect_ids)}")

    # Convert selected no-GT predictions to annotations
    max_ann_id = max((a["id"] for a in gt["annotations"]), default=0)
    next_ann_id = max_ann_id + 1
    new_annotations = []
    for pred in preds:
        if pred["category_id"] != args.category:
            continue
        if pred["image_id"] not in no_gt_convert_ids:
            continue
        if pred["score"] < args.score_thresh:
            continue
        new_ann = {
            "id": next_ann_id,
            "image_id": pred["image_id"],
            "category_id": args.category,
            "segmentation": pred["segmentation"],
            "area": pred["area"],
            "bbox": pred["bbox"],
            "iscrowd": pred.get("iscrowd", 0),
            "score": pred["score"],
            "source": "converted_from_model_prediction_train",
        }
        new_annotations.append(new_ann)
        next_ann_id += 1

    print(f"Added {len(new_annotations)} new training annotations from converted predictions")

    # Build cleaned training set: keep ALL images and original annotations
    cleaned_images = list(gt["images"])
    cleaned_annotations = list(gt["annotations"]) + new_annotations
    cleaned_annotations.sort(key=lambda a: a["id"])

    cleaned = {
        "info": gt.get("info", {}),
        "licenses": gt.get("licenses", []),
        "images": cleaned_images,
        "annotations": cleaned_annotations,
        "categories": gt["categories"],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(cleaned, indent=2))

    record_path = Path(args.out).parent / "thin_wire_panda_train_cleaning_record.json"
    record = {
        "removed_image_ids": [],
        "removed_image_count": 0,
        "skipped_image_ids": sorted(no_gt_skip_ids),
        "skipped_image_count": len(no_gt_skip_ids),
        "no_gt_converted_image_ids": sorted(no_gt_convert_ids),
        "no_gt_converted_image_count": len(no_gt_convert_ids),
        "added_annotation_count": len(new_annotations),
        "min_score_threshold": args.min_score,
        "max_preds_threshold": args.max_preds,
    }
    record_path.write_text(json.dumps(record, indent=2))

    print(f"Original: {len(gt['images'])} images / {len(gt['annotations'])} annotations")
    print(f"Cleaned:  {len(cleaned_images)} images / {len(cleaned_annotations)} annotations")
    print(f"Saved cleaned training annotations to {args.out}")
    print(f"Saved cleaning record to {record_path}")


if __name__ == "__main__":
    main()
