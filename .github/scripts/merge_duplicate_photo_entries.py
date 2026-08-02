#!/usr/bin/env python3
"""(Ha) Added 2026-07-19: merge duplicate per-category photo entries in the
merged 8-category bondwire dataset.

ROOT CAUSE (see CHANGELOG.md 2026-07-19 and .github/skills/loss-debugging
adjacent per-category eval investigation): the source "Thin_multicat" dataset's
composite masks genuinely contain multiple category values per photo
(Thick=1, Thin=2, Gatepads=3 -- confirmed via metadata.json). Phase-1's
split_multiclass_masks.py correctly extracts a separate binary mask per
category per photo (this is by design, not a bug), but the downstream COCO
conversion + merge-append pipeline kept each category's copy of a shared photo
as a SEPARATE COCO "image" entry (distinguished only by a `g_`/`h_`/`i_`
filename prefix), each carrying ONLY that category's own annotations.

Verified at scale: ALL 662 source photos in this dataset appear as EXACTLY 3
separate image entries (thick_wire/thin_wire/gate_pad), 0 photos appear in only
1 or 2 entries. Every one of those 3 entries implicitly asserts "the other 2
categories are absent from this photo" via COCO_FROM_JSON's
`include_negatives=True` -- even though the composite source mask proves all 3
are genuinely present together. This creates systematically contradictory
negative supervision for these 3 categories across the entire dataset, and is
believed to be the primary driver of their poor per-category AP
(thick_wire=0.369, thin_wire=0.242, gate_pad=0.188) relative to the 5 original,
non-duplicated categories (AP 0.91-0.97).

This script merges each group of same-base-photo image entries into ONE image
entry carrying the union of all their annotations, for both train/ and test/
splits. This is idempotent -- running it on an already-merged file is a no-op.

Usage:
    python merge_duplicate_photo_entries.py --ann <annotation.json> --out <output.json> [--prefixes g_ h_ i_]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def merge_duplicates(data: dict, prefixes: list[str]) -> dict:
    images = data["images"]
    annotations = data["annotations"]

    def base_name(file_name: str) -> str:
        for p in prefixes:
            if file_name.startswith(p):
                return file_name[len(p):]
        return file_name

    groups = defaultdict(list)
    for img in images:
        groups[base_name(img["file_name"])].append(img)

    n_merged_groups = sum(1 for v in groups.values() if len(v) > 1)
    n_total_dropped_images = sum(len(v) - 1 for v in groups.values() if len(v) > 1)
    print(f"Total unique base photos: {len(groups)}")
    print(f"Groups with >1 duplicate image entry: {n_merged_groups}")
    print(f"Image entries that will be merged away: {n_total_dropped_images}")

    # image_id -> canonical (kept) image_id
    id_remap = {}
    kept_images = []
    for base, group in groups.items():
        canonical = group[0]
        kept_images.append(canonical)
        for dup in group[1:]:
            id_remap[dup["id"]] = canonical["id"]

    remapped_count = 0
    for ann in annotations:
        if ann["image_id"] in id_remap:
            ann["image_id"] = id_remap[ann["image_id"]]
            remapped_count += 1

    print(f"Annotations remapped to a canonical image_id: {remapped_count}")

    data["images"] = kept_images
    data["annotations"] = annotations
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann", required=True, help="Path to the input annotation.json")
    parser.add_argument("--out", required=True, help="Path to write the merged annotation.json")
    parser.add_argument("--prefixes", nargs="+", default=["g_", "h_", "i_"], help="Filename prefixes identifying duplicate-photo category copies")
    args = parser.parse_args()

    with open(args.ann) as f:
        data = json.load(f)

    print(f"Before merge: {len(data['images'])} images, {len(data['annotations'])} annotations")
    data = merge_duplicates(data, args.prefixes)
    print(f"After merge:  {len(data['images'])} images, {len(data['annotations'])} annotations")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f)

    print(f"Wrote merged annotation file to {args.out}")


if __name__ == "__main__":
    main()
