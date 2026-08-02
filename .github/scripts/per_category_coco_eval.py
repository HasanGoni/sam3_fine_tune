#!/usr/bin/env python3
"""(Ha) Added 2026-07-19: standalone per-category COCO AP breakdown.

The trainer's built-in validation meter (`val_pin/detection`) reports a single
AGGREGATE segm AP across ALL categories in the merged dataset -- it does not
break results down per category. This script reuses the predictions already
dumped by that same validation run (`dumps/pins/coco_predictions_segm.json`,
written by `sam3.eval.coco_writer.PredictionDumper` with `merge_predictions:
True`) plus the ground-truth test annotation file, and runs pycocotools
`COCOeval` once per category (by restricting `params.catIds`) to get a
per-category AP/AR breakdown, in addition to the overall combined number.

No new inference run is needed -- this is a pure re-analysis of predictions
that were already computed and saved during training's last validation epoch.

Usage:
    python per_category_coco_eval.py --gt <test/annotation.json> --pred <coco_predictions_segm.json> [--iou-type segm]
"""

import argparse
import io
import json
from contextlib import redirect_stdout

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def run_eval(coco_gt: COCO, coco_dt, cat_ids, iou_type: str) -> dict:
    """Run COCOeval restricted to the given category ids and return the summary metrics dict."""
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.params.catIds = cat_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    buf = io.StringIO()
    with redirect_stdout(buf):
        coco_eval.summarize()
    stats = coco_eval.stats
    metric_names = [
        "AP", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large",
        "AR_maxDets@1", "AR_maxDets@10", "AR_maxDets@100",
        "AR_small", "AR_medium", "AR_large",
    ]
    return dict(zip(metric_names, [round(float(s), 4) for s in stats]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", required=True, help="Path to ground-truth COCO annotation.json")
    parser.add_argument("--pred", required=True, help="Path to dumped COCO predictions json")
    parser.add_argument("--iou-type", default="segm", choices=["segm", "bbox"])
    parser.add_argument("--out", default=None, help="Optional path to write the JSON summary to")
    args = parser.parse_args()

    coco_gt = COCO(args.gt)
    coco_dt = coco_gt.loadRes(args.pred)

    cat_id_to_name = {c["id"]: c["name"] for c in coco_gt.dataset["categories"]}
    all_cat_ids = sorted(cat_id_to_name.keys())

    print(f"Ground truth: {args.gt}")
    print(f"Predictions:  {args.pred}")
    print(f"Categories:   {[cat_id_to_name[c] for c in all_cat_ids]}")
    print()

    summary = {"overall": None, "per_category": {}}

    print("=" * 70)
    print("OVERALL (all categories combined)")
    print("=" * 70)
    summary["overall"] = run_eval(coco_gt, coco_dt, all_cat_ids, args.iou_type)
    for k, v in summary["overall"].items():
        print(f"  {k:16s}: {v}")

    for cat_id in all_cat_ids:
        cat_name = cat_id_to_name[cat_id]
        n_gt_anns = len(coco_gt.getAnnIds(catIds=[cat_id]))
        print()
        print("=" * 70)
        print(f"CATEGORY: {cat_name} (id={cat_id}, {n_gt_anns} GT instances)")
        print("=" * 70)
        if n_gt_anns == 0:
            print("  No ground-truth instances for this category -- skipping.")
            continue
        metrics = run_eval(coco_gt, coco_dt, [cat_id], args.iou_type)
        summary["per_category"][cat_name] = {"num_gt_instances": n_gt_anns, **metrics}
        for k, v in metrics.items():
            print(f"  {k:16s}: {v}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote summary to {args.out}")


if __name__ == "__main__":
    main()
