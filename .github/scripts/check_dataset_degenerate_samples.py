#!/usr/bin/env python3
"""Fast, CPU-only, no-GPU audit of a merged COCO-format SAM3 dataset for
degenerate samples that could feed a zero-area box / empty mask into the
training loss (see .github/skills/loss-debugging/SKILL.md, Step 7).

This is meant to be run via the `hpc-cpu-submit` skill (a plain Python script,
no bsub-wrapper needed):

    ssh hpc_muc 'bash -s' < .github/scripts/hpc_repo_context.sh   # get REPO_ROOT
    bsub -Is -R "type=X64LIN && osrel>=70 && ui=pytest rusage[mem=50G]" -n 30 \
      "source ~/.bashrc && source <REPO_ROOT>/.venv/bin/activate && cd <REPO_ROOT> && \
       python .github/scripts/check_dataset_degenerate_samples.py <path/to/annotation.json>"

Checks performed (all cheap, purely from the COCO json + a listdir, no image
decoding unless --check-pixels is passed):
  1. Annotations with non-positive bbox area (`bbox[2] <= 0 or bbox[3] <= 0`,
     or COCO `area` field <= 0) -- these are exactly the degenerate/zero-area
     boxes that could reach the (now-guarded) GIoU math or the (previously
     unguarded, now-guarded) IABCEMdetr/Boxes loss terms.
  2. Annotations with an empty `segmentation` polygon list.
  3. Annotations referencing an `image_id` not present in `images`.
  4. Images listed in the json whose `file_name` does not exist on disk
     (only checked if --images-dir is given) -- catches the documented
     image/mask filename-stem mismatch gotcha for patch-style datasets.
  5. (Optional, --check-pixels, slower) Images/masks that fail to decode or
     contain NaN/Inf pixel values.

Exit code is 0 even if issues are found (this is a report tool) -- read the
printed summary. Non-zero exit code only on a hard error (e.g. json not
found).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict


def audit(annotation_path: str, images_dir: str | None, check_pixels: bool) -> dict:
    with open(annotation_path, "r") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = {c["id"]: c["name"] for c in coco.get("categories", [])}

    image_ids = {img["id"] for img in images}

    zero_area_boxes = []
    empty_segmentations = []
    orphan_annotations = []

    for ann in annotations:
        bbox = ann.get("bbox")
        area = ann.get("area")
        bad_bbox = bbox is not None and (bbox[2] <= 0 or bbox[3] <= 0)
        bad_area = area is not None and area <= 0
        if bad_bbox or bad_area:
            zero_area_boxes.append(
                {
                    "annotation_id": ann.get("id"),
                    "image_id": ann.get("image_id"),
                    "category": categories.get(ann.get("category_id")),
                    "bbox": bbox,
                    "area": area,
                }
            )

        seg = ann.get("segmentation")
        if not seg or (isinstance(seg, list) and len(seg) == 0):
            empty_segmentations.append(
                {
                    "annotation_id": ann.get("id"),
                    "image_id": ann.get("image_id"),
                    "category": categories.get(ann.get("category_id")),
                }
            )

        if ann.get("image_id") not in image_ids:
            orphan_annotations.append(
                {"annotation_id": ann.get("id"), "image_id": ann.get("image_id")}
            )

    missing_files = []
    if images_dir:
        for img in images:
            file_name = img.get("file_name")
            if not file_name:
                continue
            candidate = os.path.join(images_dir, file_name)
            if not os.path.isfile(candidate):
                missing_files.append({"image_id": img.get("id"), "file_name": file_name})

    pixel_issues = []
    if check_pixels and images_dir:
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            print("WARNING: --check-pixels requested but numpy/Pillow not importable; skipping.", file=sys.stderr)
        else:
            for img in images:
                file_name = img.get("file_name")
                if not file_name:
                    continue
                candidate = os.path.join(images_dir, file_name)
                if not os.path.isfile(candidate):
                    continue
                try:
                    arr = np.asarray(Image.open(candidate))
                except Exception as e:  # noqa: BLE001 - report and continue
                    pixel_issues.append({"file_name": file_name, "error": f"decode failed: {e}"})
                    continue
                if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
                    pixel_issues.append({"file_name": file_name, "error": "NaN/Inf pixel values"})

    per_category_ann_counts = defaultdict(int)
    for ann in annotations:
        per_category_ann_counts[categories.get(ann.get("category_id"), "?")] += 1

    return {
        "annotation_path": annotation_path,
        "num_images": len(images),
        "num_annotations": len(annotations),
        "per_category_annotation_counts": dict(per_category_ann_counts),
        "zero_area_boxes": zero_area_boxes,
        "empty_segmentations": empty_segmentations,
        "orphan_annotations": orphan_annotations,
        "missing_files": missing_files,
        "pixel_issues": pixel_issues,
    }


def print_report(report: dict) -> None:
    print(f"=== Degenerate-sample audit: {report['annotation_path']} ===")
    print(f"images: {report['num_images']}  annotations: {report['num_annotations']}")
    print(f"per-category annotation counts: {report['per_category_annotation_counts']}")

    def _section(title: str, items: list) -> None:
        print(f"\n--- {title}: {len(items)} found ---")
        for item in items[:20]:
            print(f"  {item}")
        if len(items) > 20:
            print(f"  ... and {len(items) - 20} more")

    _section("Zero/negative-area boxes", report["zero_area_boxes"])
    _section("Empty segmentation polygons", report["empty_segmentations"])
    _section("Orphan annotations (unknown image_id)", report["orphan_annotations"])
    _section("Missing image files on disk", report["missing_files"])
    _section("Pixel decode/NaN-Inf issues", report["pixel_issues"])

    total_issues = sum(
        len(report[k])
        for k in (
            "zero_area_boxes",
            "empty_segmentations",
            "orphan_annotations",
            "missing_files",
            "pixel_issues",
        )
    )
    print(f"\nTOTAL ISSUES FOUND: {total_issues}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("annotation_path", help="Path to a COCO-format annotation.json")
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Directory containing the images referenced by file_name (enables missing-file and pixel checks)",
    )
    parser.add_argument(
        "--check-pixels",
        action="store_true",
        help="Also decode each image and check for NaN/Inf pixel values (slower, requires --images-dir)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.annotation_path):
        print(f"ERROR: annotation file not found: {args.annotation_path}", file=sys.stderr)
        return 1

    report = audit(args.annotation_path, args.images_dir, args.check_pixels)
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
