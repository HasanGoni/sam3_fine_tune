"""Replace bondwire annotations in an existing merged COCO dataset with
freshly-generated (e.g. gate-pad-bridged) bondwire annotations.

This preserves the non-bondwire categories, image IDs, and filenames, and only
updates the annotations for the bondwire categories.  It is useful when the
bondwire source masks have been re-split/re-bridged but the other categories
should stay exactly as they were.

Usage:
    python patch_bondwire_annotations.py \
        --src_dataset /path/to/5_plus_bondwire_dataset \
        --dst_dataset /path/to/5_plus_bondwire_dataset_v2 \
        --bridged_staging /path/to/bondwire_staging_bridged3 \
        --bondwire_categories thick_wire,thin_wire,gate_pad \
        --bondwire_cat_ids 6,7,8 \
        --src_prefix g_ \
        --bridged_prefixes bw_thick_,bw_thin_,bw_gate_
"""

import argparse
import json
import shutil
from pathlib import Path


def patch_split(
    src_ann_path: Path,
    dst_ann_path: Path,
    bridged_staging: Path,
    bondwire_categories: list[str],
    bondwire_cat_ids: list[int],
    src_prefix: str,
    bridged_prefixes: list[str],
):
    with open(src_ann_path) as f:
        merged = json.load(f)

    cat_name_to_id = {c["name"]: c["id"] for c in merged["categories"]}
    bondwire_cat_ids = [int(cid) for cid in bondwire_cat_ids]

    # Map source image filename (without prefix) -> existing image_id
    src_stem_to_img_id = {}
    for img in merged["images"]:
        fname = img["file_name"]
        if fname.startswith(src_prefix):
            fname = fname[len(src_prefix):]
        src_stem_to_img_id[Path(fname).stem] = img["id"]

    # Remove existing bondwire annotations
    merged["annotations"] = [
        ann for ann in merged["annotations"] if ann["category_id"] not in bondwire_cat_ids
    ]

    next_ann_id = max((ann["id"] for ann in merged["annotations"]), default=0) + 1

    for cat_name, prefix in zip(bondwire_categories, bridged_prefixes):
        if cat_name not in cat_name_to_id:
            raise ValueError(f"Category {cat_name} not found in merged categories")
        cat_id = cat_name_to_id[cat_name]

        split_name = src_ann_path.parent.name  # train or test
        bridged_ann_path = bridged_staging / cat_name / split_name / "annotation.json"
        if not bridged_ann_path.exists():
            raise FileNotFoundError(f"Bridged annotation not found: {bridged_ann_path}")

        with open(bridged_ann_path) as f:
            data = json.load(f)

        # Map bridged image_id -> existing image_id
        old_to_new_img = {}
        for img in data["images"]:
            fname = img["file_name"]
            if fname.startswith(prefix):
                fname = fname[len(prefix):]
            stem = Path(fname).stem
            if stem in src_stem_to_img_id:
                old_to_new_img[img["id"]] = src_stem_to_img_id[stem]
            else:
                print(f"WARNING: no matching source image for {cat_name} {fname}")

        for ann in data["annotations"]:
            old_img_id = ann["image_id"]
            if old_img_id not in old_to_new_img:
                continue
            merged["annotations"].append({
                "id": next_ann_id,
                "image_id": old_to_new_img[old_img_id],
                "category_id": cat_id,
                "bbox": ann["bbox"],
                "segmentation": ann.get("segmentation", []),
                "area": ann["area"],
                "iscrowd": ann.get("iscrowd", 0),
            })
            next_ann_id += 1

    dst_ann_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_ann_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Patched {src_ann_path} -> {dst_ann_path}")
    print(f"  images: {len(merged['images'])}, annotations: {len(merged['annotations'])}")
    print(f"  categories: {[c['name'] for c in merged['categories']]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dataset", type=str, required=True, help="Source merged dataset root")
    parser.add_argument("--dst_dataset", type=str, required=True, help="Destination merged dataset root")
    parser.add_argument("--bridged_staging", type=str, required=True, help="Bridged per-category staging root")
    parser.add_argument("--bondwire_categories", type=str, default="thick_wire,thin_wire,gate_pad")
    parser.add_argument("--bondwire_cat_ids", type=str, default="6,7,8")
    parser.add_argument("--src_prefix", type=str, default="g_", help="Prefix used for bondwire images in the source merged dataset")
    parser.add_argument("--bridged_prefixes", type=str, default="bw_thick_,bw_thin_,bw_gate_", help="Prefixes used in bridged COCO files")
    args = parser.parse_args()

    src_dataset = Path(args.src_dataset)
    dst_dataset = Path(args.dst_dataset)
    bridged_staging = Path(args.bridged_staging)

    bondwire_categories = [c.strip() for c in args.bondwire_categories.split(",") if c.strip()]
    bondwire_cat_ids = [c.strip() for c in args.bondwire_cat_ids.split(",") if c.strip()]
    bridged_prefixes = [p.strip() for p in args.bridged_prefixes.split(",") if p.strip()]

    if len(bondwire_categories) != len(bondwire_cat_ids) or len(bondwire_categories) != len(bridged_prefixes):
        raise ValueError("bondwire_categories, bondwire_cat_ids, and bridged_prefixes must have the same length")

    # Copy image directory from source to destination
    for split in ["train", "test"]:
        src_img_dir = src_dataset / split / "images"
        dst_img_dir = dst_dataset / split / "images"
        if src_img_dir.exists():
            dst_img_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_img_dir, dst_img_dir, dirs_exist_ok=True)

    # Patch annotation_merged.json (used by full_train_config.yaml)
    for split in ["train", "test"]:
        src_ann = src_dataset / split / "annotation_merged.json"
        dst_ann = dst_dataset / split / "annotation_merged.json"
        if src_ann.exists():
            patch_split(
                src_ann,
                dst_ann,
                bridged_staging,
                bondwire_categories,
                bondwire_cat_ids,
                args.src_prefix,
                bridged_prefixes,
            )

    print(f"\nDestination dataset ready at: {dst_dataset}")


if __name__ == "__main__":
    main()
