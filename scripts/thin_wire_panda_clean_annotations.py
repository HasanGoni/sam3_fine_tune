"""
Clean thin wire panda test annotations based on model predictions.

1. For images flagged as "no_ground_truth_available" where the model
   predictions are deemed correct, convert those predictions into new
   ground-truth annotations.
2. Remove images flagged as "uncorrect_label" from the annotation file.

Outputs a new COCO annotation JSON without touching the original.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="Original COCO GT JSON")
    parser.add_argument("--pred", required=True, help="COCO predictions JSON")
    parser.add_argument("--label-quality-dir", required=True,
                        help="Folder containing *_summary.json files")
    parser.add_argument("--out", required=True, help="Output cleaned JSON path")
    parser.add_argument("--category", type=int, default=9)
    parser.add_argument("--score-thresh", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()

    gt = json.loads(Path(args.gt).read_text())
    preds = json.loads(Path(args.pred).read_text())

    lq_dir = Path(args.label_quality_dir)
    no_gt = json.loads((lq_dir / "no_ground_truth_available_summary.json").read_text())
    uncorrect = json.loads((lq_dir / "uncorrect_label_summary.json").read_text())

    no_gt_ids = {x["image_id"] for x in no_gt}
    uncorrect_ids = {x["image_id"] for x in uncorrect}

    # Existing annotations by image
    existing_anns_by_image = defaultdict(list)
    for ann in gt["annotations"]:
        existing_anns_by_image[ann["image_id"]].append(ann)

    # Max annotation id for new ids
    max_ann_id = max((a["id"] for a in gt["annotations"]), default=0)
    next_ann_id = max_ann_id + 1

    # Convert predictions for no-GT images into GT annotations
    new_annotations = []
    for pred in preds:
        if pred["category_id"] != args.category:
            continue
        if pred["image_id"] not in no_gt_ids:
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
            "score": pred["score"],  # keep for traceability
            "source": "converted_from_model_prediction",
        }
        new_annotations.append(new_ann)
        next_ann_id += 1

    print(f"Adding {len(new_annotations)} new annotations from no-GT predictions")

    # Build cleaned dataset: remove uncorrect label images and their annotations
    keep_image_ids = {im["id"] for im in gt["images"] if im["id"] not in uncorrect_ids}
    cleaned_images = [im for im in gt["images"] if im["id"] in keep_image_ids]
    cleaned_annotations = [
        ann for ann in gt["annotations"]
        if ann["image_id"] in keep_image_ids
    ] + new_annotations

    # Keep annotations sorted by id for cleanliness
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

    print(f"Removed {len(uncorrect_ids)} uncorrect_label images")
    print(f"Final images: {len(cleaned_images)}, final annotations: {len(cleaned_annotations)}")
    print(f"Saved cleaned annotations to {args.out}")

    # Also write a removal record
    record_path = Path(args.out).parent / "thin_wire_panda_cleaning_record.json"
    record = {
        "removed_image_ids": sorted(uncorrect_ids),
        "removed_image_count": len(uncorrect_ids),
        "added_annotation_count": len(new_annotations),
        "added_from_image_ids": sorted(no_gt_ids),
    }
    record_path.write_text(json.dumps(record, indent=2))
    print(f"Saved cleaning record to {record_path}")


if __name__ == "__main__":
    main()
