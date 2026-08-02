"""Split a multi-class label mask (one mask file per image, several pixel
values each representing a different category) into per-category binary
masks, staged into <output_root>/<category_slug>/{images,masks}/ folders
ready for sam3_fine_tune/preprocessing/coco_format.py.

Expected source layout (flat, no separate images/masks subfolders):
    <src_dir>/
    ├── <name>.png            # image
    ├── <name>_mask.png       # multi-class label mask (values = category ids)
    └── <name>_mask_vis.png   # optional visualization, ignored

Each output binary mask is scaled to {0, 255} (NOT {0, 1, label_id, ...}) -
required because coco_format.py's form_coco_dataset() does
`cv2.threshold(mask, 127, 255, THRESH_BINARY)` internally, which would
silently zero out any raw label value below 127.

Gate-pad gap bridging (optional):
    If a category passes underneath another category in the composite mask
    (e.g. a thin wire going behind a gate_pad), the binary mask will contain
    two disconnected pieces.  cv2.findContours(RETR_EXTERNAL) then emits two
    COCO instances for what should be one object.  Use --bridge_categories
    and --bridge_obstacle_label to connect those pieces before writing the
    binary masks; see bridge_category_gaps() for the exact heuristic.

Usage:
    python split_multiclass_masks.py \
        --src_dir /path/to/Thin_multicat \
        --output_root /path/to/staging/train \
        --label_map "1=Thick Wire,2=Thin Wire,3=Gate Pad" \
        --bridge_categories 1,2 \
        --bridge_obstacle_label 3 \
        --max_bridge_px 50
"""

import argparse
import re
import shutil
from pathlib import Path

import cv2
import numpy as np


def slugify(name: str) -> str:
    slug = name.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


def parse_label_map(label_map: str) -> dict[int, str]:
    mapping = {}
    for pair in label_map.split(","):
        pair = pair.strip()
        if not pair:
            continue
        value_str, name = pair.split("=", 1)
        mapping[int(value_str.strip())] = name.strip()
    return mapping


def _component_bbox(stats, label_id: int) -> tuple[int, int, int, int]:
    """Return (x, y, w, h) for a connected-component label id."""
    x = stats[label_id, cv2.CC_STAT_LEFT]
    y = stats[label_id, cv2.CC_STAT_TOP]
    w = stats[label_id, cv2.CC_STAT_WIDTH]
    h = stats[label_id, cv2.CC_STAT_HEIGHT]
    return x, y, w, h


def _closest_pair_between_masks(a: np.ndarray, b: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], float]:
    """Return ((ax, ay), (bx, by), distance) for the closest pixel pair."""
    ys_a, xs_a = np.where(a > 0)
    ys_b, xs_b = np.where(b > 0)
    pts_a = np.column_stack([xs_a, ys_a])
    pts_b = np.column_stack([xs_b, ys_b])
    from scipy.spatial import cKDTree

    tree = cKDTree(pts_a)
    dists, idx_a = tree.query(pts_b, k=1)
    idx_b = int(np.argmin(dists))
    idx_a = int(idx_a[idx_b])
    return (int(pts_a[idx_a, 0]), int(pts_a[idx_a, 1])), (int(pts_b[idx_b, 0]), int(pts_b[idx_b, 1])), float(dists[idx_b])


def bridge_category_gaps(
    composite_mask: np.ndarray,
    category_label: int,
    obstacle_label: int,
    max_bridge_px: int = 50,
) -> np.ndarray:
    """
    Connect disconnected components of `category_label` that are separated only
    by pixels of `obstacle_label` (e.g. a gate_pad drawn on top of a wire).

    Returns a binary mask for `category_label` where bridge pixels have been
    filled in.  This lets downstream `cv2.findContours(RETR_EXTERNAL)` produce
    one COCO instance per physical wire instead of two instances split by the
    obstacle.

    Algorithm:
    1. Build a "bridge" mask = wire_mask | obstacle_mask.
    2. Each connected component of the bridge mask represents one physical
       object region; any wire pixels inside it belong to the same instance.
    3. If a bridge component contains multiple disconnected wire pieces,
       iteratively connect the two closest pieces with a short line until only
       one connected component remains.  The line is drawn through the
       obstacle (gate_pad) pixels that caused the split, adding the minimum
       bridge area necessary for one contour.
    """
    binary = (composite_mask == category_label).astype(np.uint8)
    if not binary.any():
        return binary

    num_wire_labels, wire_labels, wire_stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_wire_labels <= 2:
        return (binary > 0).astype(np.uint8) * 255

    obstacle = (composite_mask == obstacle_label).astype(np.uint8)
    bridge = cv2.bitwise_or(binary, obstacle)
    num_bridge_labels, bridge_labels, _, _ = cv2.connectedComponentsWithStats(
        bridge, connectivity=8
    )

    out = binary.copy()

    for bridge_id in range(1, num_bridge_labels):
        bridge_mask = (bridge_labels == bridge_id).astype(np.uint8)

        # Find wire component ids that intersect this bridge component
        wire_ids_in_bridge = []
        for wire_id in range(1, num_wire_labels):
            wire_mask = (wire_labels == wire_id).astype(np.uint8)
            if cv2.bitwise_and(bridge_mask, wire_mask).any():
                wire_ids_in_bridge.append(wire_id)

        if len(wire_ids_in_bridge) <= 1:
            continue

        # Size guard: skip bridges that span too large an area in both axes
        bridge_bbox = cv2.boundingRect(bridge_mask)
        bx, by, bw, bh = bridge_bbox
        if bw > max_bridge_px * 3 and bh > max_bridge_px * 3:
            continue

        # Iteratively merge the closest pair of wire components in this bridge
        active_ids = list(wire_ids_in_bridge)
        while len(active_ids) > 1:
            best_pair = None
            best_dist = float("inf")
            best_points = None
            for i_idx, i in enumerate(active_ids):
                mask_i = (wire_labels == i).astype(np.uint8)
                for j in active_ids[i_idx + 1:]:
                    mask_j = (wire_labels == j).astype(np.uint8)
                    p_i, p_j, dist = _closest_pair_between_masks(mask_i, mask_j)
                    if dist < best_dist:
                        best_dist = dist
                        best_pair = (i, j)
                        best_points = (p_i, p_j)

            if best_pair is None or best_dist > max_bridge_px:
                break

            (x1, y1), (x2, y2) = best_points
            cv2.line(out, (x1, y1), (x2, y2), 1, thickness=3)

            # Re-label the two components as one
            i, j = best_pair
            wire_labels[wire_labels == j] = i
            active_ids.remove(j)

    return (out > 0).astype(np.uint8) * 255


def split_masks(
    src_dir: Path,
    output_root: Path,
    label_map: dict[int, str],
    mask_suffix: str = "_mask",
    split: str | None = None,
    bridge_categories: list[int] | None = None,
    bridge_obstacle_label: int | None = None,
    max_bridge_px: int = 50,
):
    mask_paths = sorted(src_dir.glob(f"*{mask_suffix}.png"))

    counts = {name: 0 for name in label_map.values()}
    bridge_counts = {name: 0 for name in label_map.values()}
    skipped_missing_image = []

    for mask_path in mask_paths:
        stem = mask_path.stem
        if not stem.endswith(mask_suffix):
            continue
        base_stem = stem[: -len(mask_suffix)]
        img_path = src_dir / f"{base_stem}.png"

        if not img_path.exists():
            skipped_missing_image.append(str(img_path))
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"WARNING: could not read mask {mask_path}, skipping")
            continue

        for label_value, category_name in label_map.items():
            binary = np.where(mask == label_value, 255, 0).astype(np.uint8)
            if not binary.any():
                continue  # this image doesn't contain this category

            if bridge_categories and label_value in bridge_categories and bridge_obstacle_label is not None:
                before = cv2.connectedComponents(binary, connectivity=8)[0]
                binary = bridge_category_gaps(
                    mask,
                    category_label=label_value,
                    obstacle_label=bridge_obstacle_label,
                    max_bridge_px=max_bridge_px,
                )
                after = cv2.connectedComponents((binary > 0).astype(np.uint8), connectivity=8)[0]
                if after < before:
                    bridge_counts[category_name] += 1

            slug = slugify(category_name)
            if split:
                out_img_dir = output_root / slug / split / "images"
                out_mask_dir = output_root / slug / split / "masks"
            else:
                out_img_dir = output_root / slug / "images"
                out_mask_dir = output_root / slug / "masks"
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_mask_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(img_path, out_img_dir / f"{base_stem}.png")
            cv2.imwrite(str(out_mask_dir / f"{base_stem}.png"), binary)
            counts[category_name] += 1

    print("=== Split summary ===")
    print(f"Total mask files found: {len(mask_paths)}")
    for name, count in counts.items():
        extra = f" (bridged in {bridge_counts[name]} images)" if bridge_counts[name] else ""
        print(f"  {name}: {count} images{extra}")
    if skipped_missing_image:
        print(f"WARNING: {len(skipped_missing_image)} masks had no matching image, e.g. {skipped_missing_image[:5]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True, help="Directory with <name>.png + <name>_mask.png pairs")
    parser.add_argument("--output_root", type=str, required=True, help="Root dir to stage per-category images/+masks/")
    parser.add_argument("--label_map", type=str, required=True, help='e.g. "1=Thick Wire,2=Thin Wire,3=Gate Pad"')
    parser.add_argument("--mask_suffix", type=str, default="_mask", help="Suffix identifying mask files (default: _mask)")
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help='If given, create <category>/<split>/{images,masks} subdirs (e.g. "train" or "test")',
    )
    parser.add_argument(
        "--bridge_categories",
        type=str,
        default=None,
        help='Comma-separated label values for categories whose gaps should be bridged, e.g. "1,2"',
    )
    parser.add_argument(
        "--bridge_obstacle_label",
        type=int,
        default=None,
        help='Label value of the obstacle that may sit on top of a wire and split it, e.g. 3',
    )
    parser.add_argument(
        "--max_bridge_px",
        type=int,
        default=50,
        help="Maximum gap width/height (pixels) to bridge (default: 50)",
    )
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    output_root = Path(args.output_root)
    label_map = parse_label_map(args.label_map)

    bridge_categories = None
    if args.bridge_categories:
        bridge_categories = [int(v.strip()) for v in args.bridge_categories.split(",") if v.strip()]

    if not src_dir.is_dir():
        raise SystemExit(f"ERROR: src_dir not found: {src_dir}")

    split_masks(
        src_dir,
        output_root,
        label_map,
        mask_suffix=args.mask_suffix,
        split=args.split,
        bridge_categories=bridge_categories,
        bridge_obstacle_label=args.bridge_obstacle_label,
        max_bridge_px=args.max_bridge_px,
    )


if __name__ == "__main__":
    main()
