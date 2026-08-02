#!/usr/bin/env python3
"""(Ha) Added, 2026-07-17: inspect a batch dumped by
`Trainer._maybe_dump_anomalous_batch()` (sam3/train/trainer.py) when a loss
went non-finite or exactly zero.

Prints a side-by-side comparison of the pre-augmentation ("raw") boxes/images
against the post-augmentation ("batched") boxes/images, so you can tell
whether the augmentation pipeline (random resize/crop/pad, box noise, ...)
produced a degenerate box, or whether the box/image was already degenerate
in the source annotation -- see the zero-loss collapse investigation in
.github/skills/loss-debugging/SKILL.md.

Usage (no GPU needed, run locally or via hpc-cpu-submit):

    python .github/scripts/inspect_anomalous_batch.py \
        experiments/5cat_matrix_chip_v2/dumps/anomalous_batches/train_pin_epoch9_step0_0.pt \
        [--save-images /path/to/output_dir]

Notes:
  - `raw_images` / `raw_boxes` will be None if the dataset class in use
    doesn't populate them (only Sam3ImageDataset does, as of 2026-07-17).
  - Box area is computed directly from the tensor's own coordinate
    convention (XYXY for raw boxes, whatever the collated `find_targets`
    boxes use -- typically normalized CxCyWH after ToTensorAPI/NormalizeAPI),
    so raw vs. post areas are NOT directly comparable in absolute units --
    compare whether either is exactly zero/degenerate, not their magnitudes.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch


def _box_area_xyxy(box: torch.Tensor) -> float:
    return float((box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0))


def _describe_raw(raw_images, raw_boxes) -> None:
    if raw_images is None and raw_boxes is None:
        print("No pre-augmentation (raw) data available in this dump "
              "(dataset class didn't populate raw_images/raw_boxes).")
        return

    print(f"\n=== PRE-augmentation (raw source) ===")
    if raw_images is not None:
        for i, img in enumerate(raw_images):
            size = getattr(img, "size", None)  # PIL Image (w, h)
            print(f"  image[{i}]: size={size}")
    if raw_boxes is not None:
        for i, per_image_boxes in enumerate(raw_boxes):
            for j, box in enumerate(per_image_boxes):
                area = _box_area_xyxy(box)
                flag = "  <-- DEGENERATE (area<=0)" if area <= 0 else ""
                print(f"  image[{i}] object[{j}] raw_box(xyxy)={box.tolist()} area={area:.4f}{flag}")


def _describe_model_output_debug(debug_info) -> None:
    """(Ha) Added, 2026-07-18: describe the raw model output tensors captured
    at the matched indices when a zero-loss event fired (see
    Sam3LossWrapper.last_zero_loss_debug in sam3_loss.py). Only present in
    dumps written after this was added -- older dumps won't have it.

    (Ha) Fixed, 2026-07-18: the fields are now pre-sliced to just the matched
    (batch, query) positions (a handful of rows) and already CPU tensors --
    the first version captured the FULL pred_boxes/pred_logits/pred_masks
    tensors (all queries) and kept them GPU-resident, which caused an OOM.
    See sam3_loss.py's comment at the capture site for the full story.

    Prints tensor stats for the matched rows, to directly check the "matched
    pairs point at a zero-valued padding slot on both sides" hypothesis: if
    matched_pred_boxes/matched_pred_logits are literally all-zero (or
    identical to target_boxes), that confirms it rather than 30+ independent
    loss terms coincidentally landing on 0.0.
    """
    print("\n=== Model output debug (captured at zero-loss time) ===")
    if debug_info is None:
        print("  Not present in this dump (written before 2026-07-18, or capture failed).")
        return

    pairs = debug_info.get("matched_pairs") or []
    print(f"  matched (batch_idx, query_idx) pairs: {pairs}")

    for name in ("matched_pred_boxes", "matched_pred_logits", "matched_pred_masks", "target_boxes"):
        t = debug_info.get(name)
        if t is None:
            print(f"  {name}: None")
            continue
        print(f"  {name}: shape={tuple(t.shape)} dtype={t.dtype} "
              f"min={t.min().item():.6g} max={t.max().item():.6g} mean={t.float().mean().item():.6g} "
              f"all_zero={bool((t == 0).all().item())}")

    matched_pred_boxes = debug_info.get("matched_pred_boxes")
    matched_pred_logits = debug_info.get("matched_pred_logits")
    if matched_pred_boxes is not None:
        for i, (b, q) in enumerate(pairs[: matched_pred_boxes.shape[0]]):
            box_val = matched_pred_boxes[i]
            logit_val = matched_pred_logits[i] if matched_pred_logits is not None else None
            print(f"  matched[{i}] (batch={b}, query={q}) pred_boxes={box_val.tolist()} "
                  f"pred_logits={logit_val.tolist() if logit_val is not None else None}")


def _describe_post(img_batch, find_targets) -> None:
    print(f"\n=== POST-augmentation (what the model saw) ===")
    print(f"  img_batch.shape = {tuple(img_batch.shape)}")
    for stage_id, target in enumerate(find_targets):
        boxes = target.get("boxes")
        num_boxes = target.get("num_boxes")
        segments = target.get("segments")
        print(f"  stage[{stage_id}]: num_boxes={num_boxes.tolist() if num_boxes is not None else None} "
              f"segments={'present' if segments is not None else 'not captured in this dump'}")
        if boxes is None:
            continue
        for k, box in enumerate(boxes):
            # Post-augmentation boxes are typically normalized cxcywh: [cx, cy, w, h] in [0, 1]
            # (fixed 2026-07-18: previous version had an operator-precedence bug --
            # `w, h = float(box[2]), float(box[3]) if box.numel() >= 4 else (None, None)`
            # parses as `w = float(box[2])` unconditionally, `h = (... if ... else ...)`,
            # so `box[2]` was accessed even when box.numel() < 4, causing an IndexError.
            # Also flatten first: a box can come through with an extra leading dim
            # (e.g. shape (1, 4)), where box[2] would index dim 0 -- size 1 -- not
            # the 4 coordinates.)
            box_flat = box.reshape(-1)
            if box_flat.numel() >= 4:
                w, h = float(box_flat[2]), float(box_flat[3])
            else:
                w, h = None, None
            area = w * h if w is not None and h is not None else None
            flag = "  <-- DEGENERATE (area<=0)" if (area is not None and area <= 0) else ""
            print(f"  stage[{stage_id}] box[{k}] (cxcywh, normalized)={box.tolist()} area={area}{flag}")


def _chunk_by_counts(flat_list, counts):
    """Split a flat per-object list into per-image groups using num_boxes counts
    (assumes num_boxes[i] objects belong to image i, in the same order the
    collator appended them -- true for boxes/segments within one stage).
    """
    out = []
    idx = 0
    for c in counts:
        c = int(c)
        out.append(list(flat_list[idx : idx + c]))
        idx += c
    return out


def _denormalize_image(img_chw, mean, std):
    """Reverse NormalizeAPI's (x - mean) / std back to a viewable [0,1] image,
    then to a uint8 HWC PIL Image. Mean/std must match
    scratch.train_norm_mean/train_norm_std in full_train_config.yaml.
    """
    from PIL import Image
    import numpy as np

    mean_t = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
    img = img_chw.float() * std_t + mean_t
    img = img.clamp(0, 1)
    arr = (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def _box_cxcywh_norm_to_xyxy_px(box, width, height):
    box_flat = box.reshape(-1)
    cx, cy, w, h = (float(v) for v in box_flat[:4])
    x1 = (cx - w / 2) * width
    y1 = (cy - h / 2) * height
    x2 = (cx + w / 2) * width
    y2 = (cy + h / 2) * height
    return x1, y1, x2, y2


def save_post_augmentation_jpegs(payload, out_dir, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), stage_id=0):
    """Save the post-augmentation images the model actually saw this step as
    JPEGs, with box outlines and (if captured in this dump) mask overlays.

    Older dumps (written before 2026-07-18) won't have `segments` in their
    find_targets -- those images are still saved, just without mask overlays.
    """
    from PIL import Image, ImageDraw

    os.makedirs(out_dir, exist_ok=True)
    img_batch = payload["img_batch"]  # (N, 3, H, W)
    find_targets = payload["find_targets"]
    if not find_targets:
        print("No find_targets in this dump -- nothing to overlay.")
        stage_id = None
    elif stage_id >= len(find_targets):
        stage_id = 0

    target = find_targets[stage_id] if stage_id is not None else {}
    boxes = target.get("boxes")
    num_boxes = target.get("num_boxes")
    segments = target.get("segments")

    n = img_batch.shape[0]
    counts = num_boxes.tolist() if num_boxes is not None else [0] * n
    # (Fixed 2026-07-18) `counts` has one entry per (image, category) query --
    # e.g. 32 entries for 4 images x category_chunk_size=8 -- NOT one entry
    # per image. Chunking the flat boxes/segments list directly by `counts`
    # (as the first version did) produces batch_size-many groups from the
    # WRONG slice boundaries, silently misattributing boxes/masks to the
    # wrong image (caught by inspecting a rendered JPEG: a box showed up in
    # a blank padding region of an image that had zero real annotations).
    # Fix: sum every len(counts)//n consecutive entries into one per-image
    # total first, THEN chunk the flat list by those correct per-image sums.
    if len(counts) % n != 0:
        print(f"WARNING: num_boxes length ({len(counts)}) is not a multiple of "
              f"batch size ({n}) -- per-image box/mask attribution may be wrong.")
    per_image_query_count = max(len(counts) // n, 1) if counts else 1
    image_totals = [
        sum(counts[i * per_image_query_count : (i + 1) * per_image_query_count])
        for i in range(n)
    ]
    boxes_per_image = _chunk_by_counts(list(boxes), image_totals) if boxes is not None else [[] for _ in range(n)]
    segs_per_image = _chunk_by_counts(segments, image_totals) if segments is not None else None

    palette = [
        (255, 0, 0, 90), (0, 200, 0, 90), (0, 0, 255, 90),
        (255, 255, 0, 90), (255, 0, 255, 90), (0, 255, 255, 90),
    ]

    saved = []
    for i in range(n):
        img = _denormalize_image(img_batch[i], mean, std)
        width, height = img.size
        draw_img = img.convert("RGBA")

        if segs_per_image is not None:
            overlay = Image.new("RGBA", draw_img.size, (0, 0, 0, 0))
            for k, mask in enumerate(segs_per_image[i]):
                mask_np = mask.numpy().astype(bool)
                if not mask_np.any():
                    continue
                color = palette[k % len(palette)]
                mask_img = Image.fromarray((mask_np * 255).astype("uint8"), mode="L")
                colored = Image.new("RGBA", draw_img.size, color)
                overlay.paste(colored, (0, 0), mask_img)
            draw_img = Image.alpha_composite(draw_img, overlay)

        draw_img = draw_img.convert("RGB")
        ddraw = ImageDraw.Draw(draw_img)
        for box in boxes_per_image[i]:
            if box.reshape(-1).numel() < 4:
                continue
            x1, y1, x2, y2 = _box_cxcywh_norm_to_xyxy_px(box, width, height)
            ddraw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=2)

        has_masks = segs_per_image is not None
        out_path = os.path.join(out_dir, f"post_image_{i}_stage{stage_id}.jpg")
        draw_img.save(out_path, "JPEG", quality=92)
        saved.append(out_path)
        print(f"Saved {out_path} (boxes={len(boxes_per_image[i])}, masks={'yes' if has_masks else 'not captured in this dump'})")

    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump_path", help="Path to a .pt file written by _maybe_dump_anomalous_batch()")
    parser.add_argument(
        "--save-images",
        default=None,
        help="If given, save the raw (pre-augmentation) PIL images to this directory for visual inspection",
    )
    parser.add_argument(
        "--save-post-images",
        default=None,
        help="If given, save the POST-augmentation images (what the model actually saw) as JPEGs to this "
        "directory, with box outlines and mask overlays (if this dump captured segments).",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=0,
        help="Which find_targets stage's boxes/masks to overlay onto --save-post-images (default: 0)",
    )
    parser.add_argument(
        "--norm-mean", type=float, nargs=3, default=(0.5, 0.5, 0.5),
        help="Must match scratch.train_norm_mean in full_train_config.yaml (default: 0.5 0.5 0.5)",
    )
    parser.add_argument(
        "--norm-std", type=float, nargs=3, default=(0.5, 0.5, 0.5),
        help="Must match scratch.train_norm_std in full_train_config.yaml (default: 0.5 0.5 0.5)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.dump_path):
        print(f"ERROR: dump file not found: {args.dump_path}", file=sys.stderr)
        return 1

    payload = torch.load(args.dump_path, map_location="cpu", weights_only=False)

    print(f"=== Anomalous batch dump: {args.dump_path} ===")
    print(f"phase={payload['phase']} key={payload['key']} epoch={payload['epoch']} "
          f"step={payload['step']} loss_value={payload['loss_value']}")

    _describe_raw(payload.get("raw_images"), payload.get("raw_boxes"))
    _describe_post(payload["img_batch"], payload["find_targets"])
    _describe_model_output_debug(payload.get("model_output_debug"))

    if args.save_images and payload.get("raw_images") is not None:
        os.makedirs(args.save_images, exist_ok=True)
        for i, img in enumerate(payload["raw_images"]):
            out_path = os.path.join(args.save_images, f"raw_image_{i}.png")
            img.save(out_path)
            print(f"Saved {out_path}")

    if args.save_post_images:
        save_post_augmentation_jpegs(
            payload,
            args.save_post_images,
            mean=tuple(args.norm_mean),
            std=tuple(args.norm_std),
            stage_id=args.stage,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
