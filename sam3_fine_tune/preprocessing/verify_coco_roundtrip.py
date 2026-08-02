"""Verify that a COCO annotation file round-trips back to the source composite mask.

This is intended to catch the "gate_pad splits a wire into two instances" bug:
cv2.findContours(RETR_EXTERNAL) emits two annotations for a single wire that
passes behind a gate_pad.  After bridging the gap in the binary mask, the COCO
file should contain one annotation per wire, and rasterizing all annotations
with the original priority rule should reproduce the source composite mask.

Usage:
    python verify_coco_roundtrip.py \
        --coco_json /path/to/annotation.json \
        --src_mask_dir /path/to/composite_masks \
        --label_map "1=thick_wire,2=thin_wire,3=gate_pad" \
        --priority "3,2,1" \
        --mask_suffix _mask \
        --output_report roundtrip_report.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from tqdm.auto import tqdm


def parse_label_map(label_map: str) -> dict[int, str]:
    mapping = {}
    for pair in label_map.split(","):
        pair = pair.strip()
        if not pair:
            continue
        value_str, name = pair.split("=", 1)
        mapping[int(value_str.strip())] = name.strip()
    return mapping


def rle_to_mask(ann, h: int, w: int) -> np.ndarray:
    if "segmentation" not in ann:
        return np.zeros((h, w), dtype=np.uint8)
    seg = ann["segmentation"]
    if isinstance(seg, dict):
        rle = seg
    else:
        rle = mask_utils.merge(mask_utils.frPyObjects(seg, h, w))
    return mask_utils.decode(rle).astype(np.uint8)


def reconstruct_composite(
    coco: COCO,
    img_id: int,
    h: int,
    w: int,
    priority_order: list[int],
    cat_id_to_label: dict[int, int],
) -> np.ndarray:
    """Rasterize annotations for an image with the given priority order."""
    composite = np.zeros((h, w), dtype=np.uint8)
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)

    # Group annotations by category
    by_cat: dict[int, list] = {}
    for ann in anns:
        by_cat.setdefault(ann["category_id"], []).append(ann)

    # Paint in priority order (last written wins)
    for label in priority_order:
        cat_id = None
        for cid, lbl in cat_id_to_label.items():
            if lbl == label:
                cat_id = cid
                break
        if cat_id is None or cat_id not in by_cat:
            continue
        canvas = np.zeros((h, w), dtype=np.uint8)
        for ann in by_cat[cat_id]:
            canvas |= rle_to_mask(ann, h, w)
        composite[canvas > 0] = label

    return composite


def _strip_prefix(filename: str, prefix: str | None) -> str:
    if prefix and filename.startswith(prefix):
        return filename[len(prefix):]
    return filename


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union > 0 else 1.0


def verify_roundtrip(
    coco_json: Path,
    src_mask_dir: Path,
    label_map: dict[int, str],
    priority_order: list[int] | None = None,
    mask_suffix: str = "_mask",
    filename_prefix: str | None = None,
    per_category: bool = False,
    output_report: Path | None = None,
) -> dict:
    coco = COCO(str(coco_json))
    cat_name_to_id = {c["name"]: c["id"] for c in coco.dataset["categories"]}
    cat_id_to_label = {cat_name_to_id[name]: label for label, name in label_map.items() if name in cat_name_to_id}

    images = coco.dataset["images"]
    per_image = []
    total_pixels = 0
    total_wrong = 0
    cat_iou_sum: dict[str, float] = {}
    cat_iou_count: dict[str, int] = {}

    for img in tqdm(images, desc="Verifying round-trip"):
        img_id = img["id"]
        h, w = img["height"], img["width"]

        base_name = _strip_prefix(img["file_name"], filename_prefix)
        src_mask_path = src_mask_dir / f"{Path(base_name).stem}{mask_suffix}.png"
        if not src_mask_path.exists():
            # Try without suffix replacement if file_name already ends with mask suffix
            src_mask_path = src_mask_dir / base_name
        if not src_mask_path.exists():
            per_image.append({
                "image_id": img_id,
                "file_name": img["file_name"],
                "status": "source_mask_not_found",
                "source_mask_path": str(src_mask_path),
            })
            continue

        source = cv2.imread(str(src_mask_path), cv2.IMREAD_GRAYSCALE)
        if source is None:
            per_image.append({
                "image_id": img_id,
                "file_name": img["file_name"],
                "status": "source_mask_unreadable",
                "source_mask_path": str(src_mask_path),
            })
            continue

        if source.shape != (h, w):
            per_image.append({
                "image_id": img_id,
                "file_name": img["file_name"],
                "status": "size_mismatch",
                "source_shape": source.shape,
                "coco_shape": (h, w),
            })
            continue

        if per_category:
            # Compare per-category masks separately
            wrong = 0
            for label, name in label_map.items():
                if name not in cat_name_to_id:
                    continue
                cid = cat_name_to_id[name]
                src_binary = (source == label).astype(np.uint8)
                ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[cid])
                anns = coco.loadAnns(ann_ids)
                rec_binary = np.zeros((h, w), dtype=np.uint8)
                for ann in anns:
                    rec_binary |= rle_to_mask(ann, h, w)

                wrong += int(np.sum(src_binary != rec_binary))
                iou = _iou(src_binary, rec_binary)
                cat_iou_sum[name] = cat_iou_sum.get(name, 0.0) + iou
                cat_iou_count[name] = cat_iou_count.get(name, 0) + 1

            total_pixels += h * w
            total_wrong += wrong
            accuracy = 1.0 - wrong / (h * w)

            source_counts = {}
            coco_counts = {}
            for label, name in label_map.items():
                if name not in cat_name_to_id:
                    continue
                src_binary = (source == label).astype(np.uint8)
                source_counts[name] = cv2.connectedComponents(src_binary, connectivity=8)[0] - 1
                coco_counts[name] = len(coco.getAnnIds(imgIds=[img_id], catIds=[cat_name_to_id[name]]))
        else:
            # Full composite comparison with priority paint order
            reconstructed = reconstruct_composite(coco, img_id, h, w, priority_order or [], cat_id_to_label)
            wrong = int(np.sum(source != reconstructed))
            total_pixels += h * w
            total_wrong += wrong
            accuracy = 1.0 - wrong / (h * w)

            source_counts = {}
            for label, name in label_map.items():
                if name not in cat_name_to_id:
                    continue
                src_binary = (source == label).astype(np.uint8)
                source_counts[name] = cv2.connectedComponents(src_binary, connectivity=8)[0] - 1

            coco_counts = {}
            for name, cid in cat_name_to_id.items():
                coco_counts[name] = len(coco.getAnnIds(imgIds=[img_id], catIds=[cid]))

        status = "ok" if wrong == 0 else "mismatch"
        per_image.append({
            "image_id": img_id,
            "file_name": img["file_name"],
            "status": status,
            "pixel_accuracy": round(accuracy, 6),
            "wrong_pixels": wrong,
            "source_instance_counts": source_counts,
            "coco_instance_counts": coco_counts,
        })

    summary = {
        "coco_json": str(coco_json),
        "source_mask_dir": str(src_mask_dir),
        "label_map": label_map,
        "priority_order": priority_order,
        "per_category_mode": per_category,
        "images_checked": len(images),
        "overall_pixel_accuracy": round(1.0 - total_wrong / max(total_pixels, 1), 6),
        "total_wrong_pixels": total_wrong,
        "mean_per_category_iou": {name: round(cat_iou_sum[name] / cat_iou_count[name], 6) for name in cat_iou_sum},
        "mismatched_images": [r for r in per_image if r.get("status") == "mismatch"],
        "missing_source_masks": [r for r in per_image if r.get("status") == "source_mask_not_found"],
        "per_image": per_image,
    }

    if output_report:
        output_report.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Report saved to: {output_report}")

    print("=== Round-trip verification summary ===")
    print(f"Images checked: {summary['images_checked']}")
    print(f"Overall pixel accuracy: {summary['overall_pixel_accuracy']}")
    print(f"Total wrong pixels: {summary['total_wrong_pixels']}")
    if per_category:
        print("Mean per-category IoU:")
        for name, iou in summary["mean_per_category_iou"].items():
            print(f"  {name}: {iou}")
    print(f"Mismatched images: {len(summary['mismatched_images'])}")
    print(f"Missing source masks: {len(summary['missing_source_masks'])}")
    if summary["mismatched_images"]:
        print("First 5 mismatches:")
        for r in summary["mismatched_images"][:5]:
            print(f"  img {r['image_id']} ({r['file_name']}): acc={r['pixel_accuracy']} wrong={r['wrong_pixels']}")
            print(f"    source counts: {r['source_instance_counts']}")
            print(f"    coco counts:   {r['coco_instance_counts']}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_json", type=str, required=True, help="Path to COCO annotation JSON")
    parser.add_argument("--src_mask_dir", type=str, required=True, help="Directory containing source composite masks")
    parser.add_argument("--label_map", type=str, required=True, help='e.g. "1=thick_wire,2=thin_wire,3=gate_pad"')
    parser.add_argument(
        "--priority",
        type=str,
        default=None,
        help='Comma-separated label values in paint order (last wins), e.g. "1,2,3" means gate_pad on top. Required unless --per_category is set.',
    )
    parser.add_argument("--mask_suffix", type=str, default="_mask", help="Suffix for source mask files (default: _mask)")
    parser.add_argument(
        "--filename_prefix",
        type=str,
        default=None,
        help='Prefix prepended to COCO file_names that is NOT part of the source mask filename (e.g. "bw_thick_")',
    )
    parser.add_argument(
        "--per_category",
        action="store_true",
        help="Compare each category mask separately instead of the full composite",
    )
    parser.add_argument("--output_report", type=str, default=None, help="Optional JSON path to write detailed report")
    args = parser.parse_args()

    label_map = parse_label_map(args.label_map)
    priority_order = None
    if args.priority:
        priority_order = [int(v.strip()) for v in args.priority.split(",") if v.strip()]
    elif not args.per_category:
        raise SystemExit("ERROR: --priority is required unless --per_category is set")

    verify_roundtrip(
        coco_json=Path(args.coco_json),
        src_mask_dir=Path(args.src_mask_dir),
        label_map=label_map,
        priority_order=priority_order,
        mask_suffix=args.mask_suffix,
        filename_prefix=args.filename_prefix,
        per_category=args.per_category,
        output_report=Path(args.output_report) if args.output_report else None,
    )


if __name__ == "__main__":
    main()
