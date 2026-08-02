---
name: patch-dataset
description: 'Create a patch-based COCO dataset from large source images and masks using a YAML config. Tiles images with configurable size and overlap, filters empty patches, and produces COCO annotations for SAM3 training.'
argument-hint: '[--config <path>]'
user-invocable: true
disable-model-invocation: false
---
# Patch Dataset

Create a patch-based COCO dataset from large images/masks.  This is useful when
some source images are much larger than the model's training resolution and you
want to augment the data by tiling them into smaller overlapping patches
(e.g. 256×256 patches with 50% overlap).

## When To Use

- You have large source images (e.g. thick-wire or thin-wire microscopy images)
  that are much bigger than the training input size.
- You want to create additional training data by extracting fixed-size patches
  with overlap.
- You want the patches to keep their COCO polygon annotations so they can be
  merged with other category datasets later.

This skill covers two related scripts:

1. **`create_patched_dataset.py`** — create a new COCO dataset from raw
   `images/` + `masks/` directories.
2. **`patch_coco_categories.py`** — patch **selected categories in-place** inside
   an existing merged COCO dataset, leaving the other categories unchanged.

**Not for:** replacing annotations in an already-merged dataset (use
`patch_bondwire_annotations.py` directly for that).

**HPC queue rule:** patching is CPU-only work. Submit to a CPU queue
(`batch`/`static_hi`/etc.). GPU jobs that stay under ~5 % GPU utilization for
several minutes are killed automatically on this cluster. Use GPU queues only
for training or model inference.

## Input Format

Per split (`train` / `test`):

```text
your_raw_data/
├── train/
│   ├── images/
│   │   ├── img_001.png
│   │   └── ...
│   └── masks/
│       ├── img_001.png      # same stem as the image, one category per mask
│       └── ...
└── test/
    ├── images/
    └── masks/
```

- Image/mask pairs are matched by filename stem (e.g. `img_001.png` ↔ `img_001.png`).
- Masks must be one-category-per-file binary masks (0 = background, >0 = foreground).
  If you have a multi-class label map, split it into per-category binary masks first
  (see `inspect-mask-format` skill).

## Output Format

```text
<output_root>/
├── patch_config.yaml          # copy of the config used
├── train/
│   ├── images/                # patched image tiles
│   ├── masks/                 # patched binary masks (optional)
│   ├── annotation.json        # COCO annotations
│   └── annotation_metadata.json   # image/annotation counts per category
└── test/
    ├── images/
    ├── masks/
    ├── annotation.json
    └── annotation_metadata.json
```

`annotation_metadata.json` is written automatically next to every COCO JSON. It
contains total image/annotation/category counts plus, for each category, the
number of unique images containing that category and the number of annotations
of that category.

## Inputs

Only one input is required — a YAML config file.  All paths inside it should be
absolute HPC paths.

Example: `sam3_fine_tune/preprocessing/configs/patch_dataset_example.yaml`

```yaml
patch:
  size: 256
  overlap: 128
  min_mask_ratio: 0.001

category:
  name: thin_wire_HD
  id: 1

splits:
  train:
    images: /home/hpc_storage/data/projects/userB/sam3_data/raw_thin_wire_HD/train/images
    masks: /home/hpc_storage/data/projects/userB/sam3_data/raw_thin_wire_HD/train/masks
  test:
    images: /home/hpc_storage/data/projects/userB/sam3_data/raw_thin_wire_HD/test/images
    masks: /home/hpc_storage/data/projects/userB/sam3_data/raw_thin_wire_HD/test/masks

output:
  root: /home/hpc_storage/data/projects/userB/sam3_data/thin_wire_HD_patched_256
```

### Config fields

- `patch.size` — edge length of each square patch (required).
- `patch.overlap` — overlap between adjacent patches in pixels.  Stride is
  `size - overlap`.  Use `0` for non-overlapping patches (required).
- `patch.min_mask_ratio` — minimum fraction of non-zero mask pixels required to
  keep a patch.  Patches below this are skipped (default: `0.001`).
- `category.name` — category name written into the COCO JSON (required).
- `category.id` — category id written into the COCO JSON.  This may be reassigned
  when the dataset is later merged (required).
- `splits.<split>.images` — absolute path to source images for this split (required).
- `splits.<split>.masks` — absolute path to source masks for this split (required).
- `output.root` — absolute path where the patched dataset will be created (required).

## Procedure

### 0. Run `configure-repo` once (recommended)

```bash
# This populates ~/.cache/hpc_repo_root/sam3_fine_tune and verifies SSH.
# Skip if you already did it recently.
```

### 1. Prepare your source images and masks

Make sure each split has matching `images/` and `masks/` directories.  If the mask
format is unknown, run the `inspect-mask-format` skill first on a sample mask.

### 2. Create the YAML config

Copy the example config and edit the paths and patch parameters:

```bash
ssh hpc_muc 'mkdir -p /home/hpc_storage/data/projects/$USER/sam3_data/patched_configs'
scp x:\data\projects\userB\test_sam3\sam3_fine_tune\sam3_fine_tune\preprocessing\configs\patch_dataset_example.yaml \
  hpc_muc:/home/hpc_storage/data/projects/$USER/sam3_data/patched_configs/my_patch_config.yaml
```

Then edit the config on HPC (or locally before copying) with the actual source
paths, category name, and patch parameters.

### 3. Submit the patching job to the HPC CPU queue

Patching is CPU-only. Use the `batch` queue with modest memory (the script
processes one image at a time). Do **not** use a GPU queue.

```bash
ssh hpc_muc '
REPO_ROOT=$(bash /home/hpc_storage/data/projects/$USER/test_sam3/sam3_fine_tune/.github/scripts/discover_repo_root.sh sam3_fine_tune)
CONFIG=/home/hpc_storage/data/projects/$USER/sam3_data/patched_configs/my_patch_config.yaml
LOGDIR=$REPO_ROOT/logs/patch_dataset
mkdir -p "$LOGDIR"
cd "$REPO_ROOT"
source .venv/bin/activate
bsub -q batch -R "type=X64LIN && osrel>=70 rusage[mem=8G]" -n 1 -W 240 \
  -o "$LOGDIR/patch.%J.out" -e "$LOGDIR/patch.%J.err" \
  "python $REPO_ROOT/sam3_fine_tune/preprocessing/create_patched_dataset.py --config $CONFIG"
'
```

### 4. Verify the output

```bash
ssh hpc_muc '
ROOT=/home/hpc_storage/data/projects/$USER/sam3_data/thin_wire_HD_patched_256
for split in train test; do
  echo "=== $split ==="
  python - <<"PY"
import json
from pathlib import Path
root = Path("$ROOT")
for split in ["train", "test"]:
    ann = json.load(open(root / split / "annotation.json"))
    meta = json.load(open(root / split / "annotation_metadata.json"))
    print(f"{split}: {len(ann['images'])} images, {len(ann['annotations'])} annotations")
    print(f"  metadata total_images={meta['total_images']}, total_annotations={meta['total_annotations']}")
    for cat in meta["categories"]:
        print(f"    {cat['name']}: {cat['image_count']} images, {cat['annotation_count']} annotations")
PY
done
'
```

---

## In-Place Category Patching with `patch_coco_categories.py`

Use this variant when you already have a merged COCO dataset (e.g.
`9cat_separated_wires_v3`) and you only want to re-tile **some** categories,
leaving the rest untouched. Images that contain any `target_categories` are
replaced by overlapping patches; images without target categories are copied
as-is; annotations from other categories that happen to intersect a patch are
clipped and preserved. It writes `annotation_merged.json` plus
`annotation_merged_metadata.json` per split, with image and annotation counts
per category.

### Config (`patch_coco_categories_example.yaml`)

```yaml
patch:
  size: 256
  overlap: 128
  min_target_area: 1.0

target_categories:
  - thick_wire_HD
  - thin_wire_HD
  - thick_wire_panda
  - thin_wire_panda

splits:
  - train
  - test

input:
  root: /home/hpc_storage/data/projects/userB/sam3_data/9cat_separated_wires_v3

output:
  root: /home/hpc_storage/data/projects/userB/sam3_data/9cat_separated_wires_v3_patched_wires_256
```

### Submit to CPU queue

```bash
ssh hpc_muc '
REPO_ROOT=$(bash /home/hpc_storage/data/projects/$USER/test_sam3/sam3_fine_tune/.github/scripts/discover_repo_root.sh sam3_fine_tune)
CONFIG=$REPO_ROOT/sam3_fine_tune/preprocessing/configs/patch_coco_categories_example.yaml
LOGDIR=$REPO_ROOT/logs/patch_coco
mkdir -p "$LOGDIR"
cd "$REPO_ROOT"
source .venv/bin/activate
bsub -q batch -R "type=X64LIN && osrel>=70 rusage[mem=8G]" -n 1 -W 240 \
  -o "$LOGDIR/patch_coco.%J.out" -e "$LOGDIR/patch_coco.%J.err" \
  "python $REPO_ROOT/sam3_fine_tune/preprocessing/patch_coco_categories.py --config $CONFIG"
'
```

## What the script does

1. For every image/mask pair in each split, it computes a sliding-window grid
   using `stride = patch_size - overlap`.
2. The last patch in each row/column is aligned to the far edge, so no padding
   is needed unless the image is smaller than the patch size.
3. Each patch is saved as `<stem>_py{row}_px{col}_y{y}_x{x}.png`.
4. Patches with foreground mask area below `min_mask_ratio` are skipped.
5. COCO annotations are generated from the patched masks using the same logic as
   `coco_format.py`, producing `annotation.json` per split.

## Common Pitfalls

- **Multi-class label maps:** `create_patched_dataset.py` expects one category per
  mask file.  If your mask encodes multiple categories with different pixel
  values, split it first (e.g. with `split_multiclass_masks.py` or a custom
  script), then run this skill separately per category.
- **Absolute paths:** All paths in the YAML config must be absolute HPC paths.
  Relative paths are resolved from the job's working directory and usually break.
- **Category id collisions:** The `category.id` is local to this dataset.  When you
  later merge it with other categories using `merge-category-dataset`, the merge
  step will reassign ids, so it is fine to use `id: 1` here.
- **Host memory:** For very large source images or many categories, increase
  `rusage[mem=...]` above 32 GB.

## Next Steps

After patching, the dataset is ready for the standard pipeline:

1. `coco-format-conversion` — already done; the output is COCO format.
2. `merge-category-dataset` — if you need to combine this category with others.
3. `finetune-merged-dataset` — to update the training config and launch training.
