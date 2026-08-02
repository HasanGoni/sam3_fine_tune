---
name: merge-category-dataset
description: 'Add a new category''s data into an existing merged SAM3 dataset on the Acme HPC. Validates the new data in one round-trip, converts it to COCO format if needed, and appends it via append_dataset() (read from nbs/02_merge_dataset.ipynb without running nbdev-export). Does not touch training config or submit any training job — use finetune-merged-dataset for that.'
argument-hint: '[--new-data-dir <path>] [--category-name <name>] [--prefix <letter>] [--existing-dataset <path>]'
user-invocable: true
disable-model-invocation: false
---
# Merge Category Dataset

Add a brand-new category's data into an already-merged SAM3 dataset. This is
Step 3 Option B of the [README](../../../README.md) pipeline. Purely a data
step — it does not touch training config or submit any job. Follow up with
the `finetune-merged-dataset` skill once this completes.

## Common pitfalls

- **Duplicate photo entries from a shared composite mask.** If the source
  uses one composite label mask per photo with multiple category pixel
  values, do not stop at splitting it into per-category COCO files. Each
  per-category conversion will produce its own copy of the same source
  image; appending them naively creates duplicate image entries that
  contradict each other (each copy says the other categories are absent).
  Merge duplicate-photo entries into single multi-category records before
  appending or training.  See the dedicated section below.
- **Training config expects `annotation_merged.json`.** SAM3 configs look
  for `${paths.pin_dir_root}/train/annotation_merged.json` and the matching
  test file. Create these as copies/renames of `annotation.json` if they do
  not exist; otherwise training launches fail at dataset load time.

## When To Use

- You have a merged multi-category dataset already used for training (e.g.
  `et_pin`, `eberhard_pin`, `wuxi_pin`).
- You collected data for a **new category** (a text prompt that doesn't exist
  in the dataset yet) and want to add it into the merged dataset.
- You do **not** want to rebuild the merged dataset from scratch.

**Not for:** adding more images to an **existing** category (same
`category_name`). `append_dataset()` always assigns a new `category_id` — it
does not merge into a matching existing category. Use **Option C** below
(`append_images_to_existing_category`) for that case instead.

## Prerequisites

Run the `configure-repo` skill first to resolve `REPO_ROOT` in one
round-trip (it also verifies SSH connectivity as part of reaching HPC).

## Inputs

Ask for **both** of these up front, in the same message, before doing any
HPC exploration. Discovering `existing_dataset` is missing partway through
wastes a round-trip — always request it alongside `new_data_dir`.

- `new_data_dir`: root of the new category's raw images/masks (or already
  COCO-converted dataset), following the [README](../../../README.md#step-1-data-preparation)
  `train/{images,masks}` + `test/{images,masks}` layout.
- `category_name`: text prompt for the new category (must be unique in the
  merged dataset and must match what will be used at inference time).
- `prefix`: short string prepended to the new category's image filenames to
  avoid collisions with existing filenames (e.g. `e`). Pick the next unused
  letter — check the existing dataset's image filename prefixes if unsure.
- `existing_dataset`: root of the existing merged dataset to append into
  (contains `train/` and `test/` with `images/` + `annotation.json`). This is
  modified **in place**.

## Procedure

### 0. Pre-flight validation (single round-trip — do this first, always)

Before anything else, validate `new_data_dir` in **one** ssh call instead of
ad-hoc `ls`/`cat` exploration (each ssh round-trip is slow on this network
mount):

```bash
ssh hpc_muc 'bash -s -- <new_data_dir>' < .github/scripts/check_new_category_data.sh
```

This reports, per split (`train`/`test`):

- image/mask counts
- whether `annotation.json` already exists at the exact path the pipeline
  expects (`<new_data_dir>/<split>/annotation.json`)
- any *other* leftover `.json` files found under the split (e.g. from earlier
  `coco_format.py` test runs with a different `--category_name`), each with
  their own category name and counts

**Known gotcha:** repeated/experimental COCO conversions often leave multiple
annotation files with **different category names** in a `coco_output/`
subfolder (e.g. one run tagged `matrix_chip`, another `cegled_econo_pin`) with
no `annotation.json` at the expected top-level split path. If this happens:

- Stop and ask the user which category name / which specific json file is the
  correct one to use (don't guess). If multiple candidate files have the
  chosen category name, `md5sum` them — if identical, any one is fine to use.
- Once confirmed, copy the correct file to `<new_data_dir>/<split>/annotation.json`
  before proceeding to step 2 — don't invent a new `--category_name`/`--category_id`
  pair for step 1 if a usable one already exists.

If `test/annotation.json` (or `train/annotation.json`) is missing entirely
(no leftovers either), run step 1 for that split.

### 1. Confirm this is a new category, not more data for an existing one

Ask the user (if not already clear): "Is `category_name` a brand-new class, or
does it already exist in the merged dataset?" If it already exists, stop and
switch to **Option C** below instead of silently duplicating the category.

Also check whether `existing_dataset` is owned by another user (shared asset)
— if so, treat it as shared infrastructure: confirm with the user before
modifying it, and back up its `train/annotation.json` + `test/annotation.json`
first (cheap — just the two json files, not the whole dataset):

```bash
ssh hpc_muc '
EXIST=<existing_dataset>
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$EXIST/.annotation_backups"
cp "$EXIST/train/annotation.json" "$EXIST/.annotation_backups/train_annotation_${TS}.json"
cp "$EXIST/test/annotation.json" "$EXIST/.annotation_backups/test_annotation_${TS}.json"
'
```

### 2. Convert the new category's images/masks to COCO format (only if missing)

Use the `coco-format-conversion` skill, once per split that's missing
`annotation.json` after step 0:

```bash
ssh hpc_muc 'bash -s -- <new_data_dir>/train <category_name> <new_category_id>' \
  < .github/scripts/run_coco_conversion.sh
ssh hpc_muc 'bash -s -- <new_data_dir>/test <category_name> <new_category_id>' \
  < .github/scripts/run_coco_conversion.sh
```

`<new_category_id>` here is just a placeholder id used inside the
single-category annotation.json — it gets renumbered automatically in step 3,
so any positive integer works (e.g. `1`).

This produces `<new_data_dir>/{train,test}/annotation.json`.

### 3. Append the new category into the existing merged dataset

Do **not** run `nbdev-export` / `nbdev_export` for this — this repo's nbdev
config is still `settings.ini` (not migrated to `pyproject.toml`), so the real
`nbdev-export` CLI cannot run without a repo-wide config migration. Instead,
use the helper script, which reads only the `#| export`-tagged cells directly
out of `nbs/02_merge_dataset.ipynb` (stdlib `json`, no nbdev involved) and
calls `append_dataset()` from them in-memory:

```bash
ssh hpc_muc 'bash -s -- <new_data_dir> <category_name> <prefix> <existing_dataset>' \
  < .github/scripts/run_append_dataset.sh
```

Example:

```bash
ssh hpc_muc 'bash -s -- /home/hpc_storage/data/projects/userB/sam3_data/war_crack_coco war_crack e /home/hpc_storage/data/projects/userB/sam3_data/merged_dataset' \
  < .github/scripts/run_append_dataset.sh
```

This updates `<existing_dataset>/{train,test}/annotation.json` in place,
copying the new category's images in with the given prefix, and prints the
new totals and category list for both splits.

**Important:** `<existing_dataset>` is modified in place. If step 1 skipped
the backup (e.g. user's own dataset, not shared), it's still modified without
one — mention this before running if the user might want to preserve the
previous state.

## Option C: Add more images to an EXISTING category

Use this instead of steps 1-3 above when the user has **more images for a
category that's already in the merged dataset** (e.g. more `matrix_chip`
samples), not a brand-new class. `append_dataset()` cannot do this correctly
(see the caveat above) — use `append_images_to_existing_category()` instead,
which looks up the category by name and reuses its existing `category_id`
instead of minting a new one, and raises a clear error if the name isn't
found in the merged dataset (rather than silently duplicating it).

1. Run the same Step 0 pre-flight validation and the same backup step as
   above (this is still a shared/another-user's asset scenario if applicable).
2. Convert the new images to COCO format (Step 2 above), same
   `coco-format-conversion` procedure, using the **same `category_name`** as
   the existing category (a placeholder `category_id`, e.g. `1`, is fine —
   it's discarded and the existing id is reused).
   - **Known gotcha:** `coco_format.py` requires each mask file's filename
     stem to exactly match a same-named file in `images/` (it does
     `img_dir / f"{mask_path.stem}.png"`). If the raw images/masks use
     different suffixes for the same sample (e.g. `<id>_ch0.png` images vs
     `<id>_mask.png` masks — seen with patch-style datasets), `coco_format.py`
     will fail to find the image. Fix by building a small staging directory
     with **copies** (not symlinks — the conversion script's file counter
     uses `find -type f`, which excludes symlinks) renamed to a common stem
     for both `images/` and `masks/`, then run the conversion against the
     staging directory instead of the raw one.
   - If the new data uses a different split-naming convention than
     `train`/`test` (e.g. `train`/`val`), map it to `train`/`test` when
     building the staging directory — `val`/held-out data plays the role of
     `test` in this repo's convention (see the notebook's own
     "train and test(validation)" terminology).
3. Append into the existing merged dataset, reusing its category id. This
   submits `append_images_to_existing_category.py` as a **bsub job on the HPC
   CPU queue** (same pattern as `coco-format-conversion`) rather than copying
   hundreds/thousands of images synchronously in the login-node ssh session:

```bash
ssh hpc_muc 'bash -s -- <new_dataset_path> <category_name> <prefix> <existing_dataset>' \
  < .github/scripts/run_append_images_to_existing_category.sh
```

Example:

```bash
ssh hpc_muc 'bash -s -- /home/hpc_storage/data/projects/userB/data/cegled_econo/47834/curated8bit/train_patches/coco_staging matrix_chip f /home/hpc_storage/data/projects/userA/sam3_fine_tune/asset/dataset' \
  < .github/scripts/run_append_images_to_existing_category.sh
```

Pick a `prefix` not already used by any existing category's image filenames
in the target dataset (check `ls <existing_dataset>/train/images | sed -E
's/^([^_]*)_.*/\1/' | sort -u` first) — even though it's the same category,
a fresh prefix avoids any filename collisions with the images already merged
in from an earlier batch.

4. Validate: confirm the category **count is unchanged** (no duplicate entry)
   and the target category's image count increased by the new batch size —
   printed automatically by the script, but worth double-checking manually
   for a shared dataset.
5. Clean up any temporary staging directory created in step 2 once the merge
   is verified.

**Speed tip for next time:** when the user gives an ambiguous or partially
wrong path (username spelling, dataset location), batch all path-discovery
checks (`ls`/`find` for every candidate spelling, plus the new data dir's
structure) into a **single** `ssh hpc_muc` call instead of one round trip per
candidate — each round trip on this network mount is slow. Ask the user for
all path inputs up front (same principle as the existing "ask for both
inputs up front" note above) before doing any HPC exploration.

## Bondwire special case: gate-pad gap bridging

For bondwire categories (`thick_wire`, `thin_wire`) the composite label mask
uses pixel priority: `gate_pad` is painted on top of the wires.  When the mask
is split into per-category binary masks, a single physical wire that passes
behind a gate_pad becomes two disconnected pieces.  `cv2.findContours`
(`RETR_EXTERNAL`) then emits **two COCO instances** for what should be one
wire, which destroys thin_wire AP.

Fix this at split time using the `--bridge_categories` and
`--bridge_obstacle_label` options of
`sam3_fine_tune/preprocessing/split_multiclass_masks.py`:

```bash
ssh hpc_muc '
REPO=/home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune
SRC=/home/hpc_storage/data/projects/userB/data/bent-wire/Thin_multicat
OUT=/home/hpc_storage/data/projects/userB/sam3_data/bondwire_staging
LABEL_MAP="1=thick_wire,2=thin_wire,3=gate_pad"
cd "$REPO" && source .venv/bin/activate
python3 sam3_fine_tune/preprocessing/split_multiclass_masks.py \
    --src_dir "$SRC" \
    --output_root "$OUT" \
    --split train \
    --label_map "$LABEL_MAP" \
    --bridge_categories 1,2 \
    --bridge_obstacle_label 3 \
    --max_bridge_px 50
'
```

Then convert each split + category to COCO as usual and merge.

## Round-trip verification

After COCO conversion, verify that the annotations reconstruct the source
composite masks correctly.  This catches both the "split wire" bug and
unwanted over-merging.  Use
`sam3_fine_tune/preprocessing/verify_coco_roundtrip.py`:

```bash
ssh hpc_muc '
STAGE=/home/hpc_storage/data/projects/userB/sam3_data/bondwire_staging
SRC=/home/hpc_storage/data/projects/userB/data/bent-wire/Thin_multicat
REPO=/home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune
cd "$REPO" && source .venv/bin/activate

for cat in thick_wire thin_wire gate_pad; do
    PREFIX="bw_${cat}_"
    python3 sam3_fine_tune/preprocessing/verify_coco_roundtrip.py \
        --coco_json "$STAGE/$cat/train/annotation.json" \
        --src_mask_dir "$SRC" \
        --label_map "1=thick_wire,2=thin_wire,3=gate_pad" \
        --filename_prefix "$PREFIX" \
        --per_category \
        --output_report "$STAGE/$cat/train_roundtrip_report.json"
done
'
```

Check the reported **mean per-category IoU** and the **source vs. COCO
instance counts**.  For the bondwire fix you should see `thin_wire` source
counts drop from 2-5 components per image down to 1 (or the true number of
physical wires), while IoU stays high.

## Poster visualization (default quality check before training)

After any merge, generate a poster of image / mask / overlay panels from the
merged dataset as a final sanity check before training.  This is now a
default step — run it for every merged dataset and inspect a few samples.

**HPC-only workflow:** the user has asked to do everything on the HPC.  Do
not copy scripts or data to/from the local workstation.  Create scripts
directly on the HPC with `ssh hpc_muc 'cat > /tmp/script.py << "EOF" ... EOF'`
and run them there.  Inspect outputs on the HPC path; only fetch small
poster PNGs to the local workspace if the user explicitly asks to see them.

Use `.github/scripts/coco_visualization_poster.py`:

```bash
ssh hpc_muc '
REPO=/home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune
DATASET=/home/hpc_storage/data/projects/userB/sam3_data/<merged_dataset>
OUT=$DATASET/overlay_posters
source "$REPO/.venv/bin/activate"
python3 "$REPO/.github/scripts/coco_visualization_poster.py" \
    --dataset-dir "$DATASET" \
    --out-dir "$OUT" \
    --split test \
    --num-samples 30
'
```

The script produces one PNG per sampled image with three columns: original
image, flattened category mask, and blended overlay.  A legend lists the
categories present in that image.

Inspect the posters for:
- Missing or truncated masks
- Overlapping masks from different categories that shouldn't overlap
- Filename/source mismatches (e.g. `univ_` prefix on the wrong images)
- Duplicate-photo entries still present as separate rows

Only proceed to `finetune-merged-dataset` after the visualization looks
correct.

## Multi-class source masks: merge duplicate photo entries after per-category splitting

If a dataset has **one composite mask per photo** containing multiple
category pixel values (e.g. bondwire `Thin_multicat` with
`1=thick_wire,2=thin_wire,3=gate_pad`), a common workflow is to first split
it into **per-category binary masks** and then convert each category to COCO
separately.  A critical gotcha: because every category keeps a copy of the
same source photo, the resulting COCO files contain **duplicate image
entries** (e.g. `g_...png`, `h_...png`, `i_...png` all from the same photo),
and `append_dataset()` treats each entry as a separate image that only
contains that category.  This creates systematically contradictory negative
supervision — every copy implicitly asserts the other categories are absent.

Fix: after converting each category to COCO, merge duplicate-photo entries
into a single multi-category image record **before** appending to an existing
merged dataset (or before training).  The same fix applies if you build a
standalone merged dataset like `bentwire_separate`.

Python sketch:

```python
from collections import defaultdict
from pycocotools.coco import COCO

coco = COCO('/path/to/merged/annotation.json')
stem_groups = defaultdict(list)
for img in coco.dataset['images']:
    # strip per-category prefix (e.g. g_00163206_Image_... -> Image_...)
    parts = img['file_name'].split('_', 2)
    stem = parts[2] if len(parts) >= 3 and parts[1].isdigit() else img['file_name']
    stem_groups[stem].append(img)

merged = {'images': [], 'annotations': [], 'categories': coco.dataset['categories']}
img_id = 1
ann_id = 1
for stem, imgs in stem_groups.items():
    merged['images'].append({**imgs[0], 'id': img_id})
    for img in imgs:
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img['id']])):
            merged['annotations'].append({**ann, 'id': ann_id, 'image_id': img_id})
            ann_id += 1
    img_id += 1
# write merged annotation.json and keep only base image files
```

After merging, every unique source photo becomes one COCO image carrying the
union of all its categories' annotations.  The number of unique images
should drop to the number of source photos (502 for the bondwire case), and
every remaining image should have all categories present.

To apply the fixed bondwire annotations to an existing multi-category merged
dataset without touching the other categories, use
`sam3_fine_tune/preprocessing/patch_bondwire_annotations.py`:

```bash
ssh hpc_muc '
REPO=/home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune
cd "$REPO" && source .venv/bin/activate
python3 sam3_fine_tune/preprocessing/patch_bondwire_annotations.py \
    --src_dataset /home/hpc_storage/data/projects/userB/sam3_data/5_plus_bondwire_dataset \
    --dst_dataset /home/hpc_storage/data/projects/userB/sam3_data/5_plus_bondwire_dataset_v2 \
    --bridged_staging /home/hpc_storage/data/projects/userB/sam3_data/bondwire_staging
'
```

Then point the training config's `paths.pin_dir_root` at the new
`..._v2` directory.

## Decision Logic

- If `category_name` collides with an existing category name in the merged
  dataset, stop and clarify with the user (see step 1) — don't silently
  create a duplicate category id.
- If step 0 finds leftover `.json` files with conflicting category names and
  no `annotation.json` at the expected path, stop and ask which one is
  correct before doing anything else (see step 0's known gotcha).
- If the user only has raw images + binary masks (no `annotation.json` yet)
  for the new category, run step 2 for that split; if COCO-format data
  already exists (from step 0's discovery), skip straight to step 3.
- If `<existing_dataset>` should be preserved unmodified, copy it to a new
  output path before step 3 and operate on the copy instead.

## Output Format

1. Step 0 validation report (image/mask counts, existing/missing
   `annotation.json`, any leftover json files and their category names).
2. Confirmation of backup (if `existing_dataset` is shared) or explicit
   skip decision.
3. Path to the new category's COCO-converted `annotation.json` (train/test),
   if step 2 ran.
4. Updated category/image/annotation counts for the merged dataset (from
   step 3) — hand the total category count to `finetune-merged-dataset` next.

## After the merge: count categories and prepare the training config

Before handing off to `finetune-merged-dataset`, record the **total number of
categories** in the merged dataset. This number drives the training config's
`category_chunk_size` setting, so it must be checked explicitly every time a
category is added.

Get the category count in the same round-trip as the merge validation:

```bash
ssh hpc_muc '
DATASET=/home/hpc_storage/data/projects/userB/sam3_data/<merged_dataset>
python3 -c "
import json
for split in [\"train\", \"test\"]:
    d = json.load(open(f\"$DATASET/{split}/annotation.json\"))
    print(f\"{split}: {len(d[\"categories\"])} categories -> {[c[\"name\"] for c in d[\"categories\"]]}\")
"
'
```

Then **open `sam3/train/configs/full_train_config.yaml`** (do not edit yet,
just read) and check the current `category_chunk_size` values (train and val).
Decide the new value:

- If the new total category count is **≤ the current memory-safe
  `category_chunk_size`** (or ≤ the card memory allows), set
  `category_chunk_size` **equal to the total number of categories** so every
  training step sees all categories together.
- If the count is larger, keep the last validated memory-safe value and let
  categories split into multiple chunks (longer epochs, but no OOM). See the
  `finetune-merged-dataset` skill for the memory trade-offs.

**Do not skip this check.** Adding a category without updating
`category_chunk_size` can silently leave the new category in a separate chunk
from the others every epoch, changing the effective training mixture.

## Next Step

Once this completes, use the `finetune-merged-dataset` skill with
`existing_dataset` and the new total category count to update the training
config and submit the job.

## Example prompts

- `I have new war_crack data, add it to my merged dataset.`
- `Append this new category to my existing dataset.`
- `Combine my new pin data with the current dataset.`
