---
name: inspect-mask-format
description: 'Auto-detect a mask image''s format (binary / multi-class label map / RGB color-coded) and its pixel values in one HPC round-trip, using .github/scripts/inspect_mask.sh. Use BEFORE coco-format-conversion or merge-category-dataset whenever a new dataset''s mask format/category encoding is not already known, instead of asking the user to describe it from scratch.'
argument-hint: '[--mask-path <path>]'
user-invocable: true
disable-model-invocation: false
---
# Inspect Mask Format

Read a single mask image on the Acme HPC and automatically report its
shape, dtype, and pixel values, then classify it so downstream steps
(splitting a multi-class mask, converting to COCO format) know exactly what
they're dealing with — without a multi-round-trip back-and-forth with the
user about folder layout, naming, or pixel encoding.

## When To Use

- Before `coco-format-conversion` or `merge-category-dataset`, whenever a new
  raw dataset's mask format isn't already documented/known.
- Whenever a user says a dataset has "masks" but hasn't specified whether
  they're plain binary masks, multi-class label maps (one mask file encoding
  several categories via different pixel values), or RGB/color-coded masks.
- Whenever `coco_format.py` produces an unexpectedly empty annotation (e.g.
  zero annotations for a category that should have content) — re-run this to
  check whether the mask values are being zeroed out by
  `coco_format.py`'s internal `cv2.threshold(mask, 127, 255, THRESH_BINARY)`
  (any raw label value below 127, e.g. `1`, `2`, `3`, will vanish).

## Prerequisites

Run the `configure-repo` skill first (or at least once recently) so
`~/.cache/hpc_repo_root/<project_name>` is populated — the inspector reuses
that cache to find the repo/venv without a second discovery round-trip.

## Procedure

### 1. Pick one representative mask file

Ask the user for (or find via `ls`) a single mask file path — you only need
ONE real example to determine the format; there's no need to sample many
files unless the first one looks ambiguous.

### 2. Run the inspector

```bash
ssh hpc_muc 'bash -s -- <mask_path> [project_name]' < .github/scripts/inspect_mask.sh
```

Example:

```bash
ssh hpc_muc 'bash -s -- /home/hpc_storage/data/projects/userB/data/bent-wire/Thin_multicat/0001_original_multicat_mask.png' \
  < .github/scripts/inspect_mask.sh
```

This prints:

```text
path: <mask_path>
shape: (H, W) or (H, W, C)
dtype: uint8 (or other)
channels: 1 or 3/4
unique values (<n>): <value>=<pixel_count>, ...
CLASSIFICATION: <binary mask | multi-class label map | RGB/multi-channel color-coded mask | unknown>
```

### 3. Act on the classification

- **binary mask**: proceed directly with `coco-format-conversion` /
  `merge-category-dataset` as normal — one category per mask file.
- **multi-class label map**: the script prints the sorted non-zero label
  values (e.g. `[1, 2, 3]`). Ask the user ONLY to confirm which category
  name each label number corresponds to — do not re-ask about folder
  layout or file naming, those are already known from step 1's path. Once
  confirmed, this dataset needs a mask-splitting step (per-label
  binarization, scaled to 0/255 — see "Critical constraint" below) before
  `coco_format.py` can be run, once per category.
- **RGB/multi-channel color-coded mask**: ask the user which color maps to
  which category (the script prints up to 10 distinct colors found). Same
  splitting requirement as multi-class, but split by color instead of
  scalar value.
- **unknown**: stop and ask the user to describe the mask format manually —
  don't guess a splitting strategy from an ambiguous result.

## Critical constraint (why 0/255, not 0/1/2/3)

`sam3_fine_tune/preprocessing/coco_format.py`'s `form_coco_dataset` does:

```python
mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
```

Any per-category binary mask produced from a multi-class label map (or any
other source) MUST be scaled so the "foreground" value is `255` (e.g.
`np.where(mask == label, 255, 0).astype(np.uint8)`), not left as `1`/`label`
— otherwise `cv2.threshold(..., 127, 255, ...)` silently zeroes it out,
producing an empty annotation set with no error.

## Decision Logic

- If the classification is ambiguous or `unknown`, don't guess — ask the
  user for a manual description instead of writing a splitting script blind.
- If multiple representative masks are needed (e.g. to confirm a rare label
  value appears somewhere), re-run against 2-3 files rather than assuming
  one file is fully representative — cheap, still a single round-trip each.

## Output Format

1. Shape/dtype/channel summary.
2. Unique pixel values (or colors) found.
3. Automatic classification.
4. For multi-class/RGB results: the exact follow-up question to ask the
   user (label-to-category-name mapping) — nothing else needs to be asked.

## Example prompts

- `Check the mask format for my new dataset before converting it.`
- `Is this a multi-class mask or one category per file?`
- `Why did my COCO conversion produce zero annotations for this category?`
