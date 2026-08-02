
---
name: coco-format-conversion
description: 'Convert a set of masks and images into COCO format using sam3_fine_tune/preprocessing/coco_format.py on the Acme HPC CPU queue. Validates or creates the expected folder structure before submitting.'
argument-hint: '[--mask-dir <path>] [--img-dir <path>] [--output-dir <path>] [--category-name <name>] [--category-id <id>]'
user-invocable: true
disable-model-invocation: false
---
# COCO Format Conversion

Convert a dataset of masks and images into COCO annotation format using
`sam3_fine_tune/preprocessing/coco_format.py`, submitted to the Acme HPC
CPU queue.

## When To Use

- You have a directory of binary masks and corresponding images.
- You need a COCO-format JSON annotation file for training or evaluation.
- You want the conversion to run on HPC CPU.

## Expected folder structure

`coco_format.py` itself only needs two directories passed as `--mask_dir` and
`--img_dir`. The README convention is to organize each split like this:

```text
your_dataset/
├── train/
│   ├── images/
│   │   ├── 0001.png
│   │   └── ...
│   └── masks/
│       ├── 0001.png
│       └── ...
└── test/
    ├── images/
    │   ├── 0101.png
    │   └── ...
    └── masks/
        ├── 0101.png
        └── ...
```

The COCO JSON will be written to `<output_dir>/<file_name>.json`. A metadata
summary `<output_dir>/<file_name>_metadata.json` is also created automatically,
reporting total image/annotation/category counts and per-category image and
annotation counts.

**Important distinction:**

- **`images/` + `masks/`** are the **input** to `coco_format.py`.
- **`images/` + `annotation.json`** are the **output** used by the training pipeline.
- The `train/` and `test/` split is **not required by `coco_format.py`**, but it **is required for the full training pipeline**. Therefore, the recommended workflow is to run the conversion separately for each split so that the final dataset looks like:
  ```text
  your_dataset/
  ├── train/
  │   ├── images/
  │   └── annotation.json
  └── test/
      ├── images/
      └── annotation.json
  ```

## Inputs

- `mask_dir`: directory containing binary mask images (required).
- `img_dir`: directory containing source images (required).
- `output_dir`: directory where the COCO JSON will be written (required).
- `file_name`: base name of the output JSON file (default: `annotation`).
- `category_name`: name of the object category (required).
- `category_id`: integer category ID (required).

## Procedure

### 0. Inspect the mask format if not already known

If the mask format/category encoding for this dataset isn't already
documented (e.g. new data from a source you haven't converted before), run
the `inspect-mask-format` skill against ONE sample mask file first:

```bash
ssh hpc_muc 'bash -s -- <one_mask_path>' < .github/scripts/inspect_mask.sh
```

This distinguishes a plain binary mask (proceed as normal below) from a
multi-class label map or RGB color-coded mask (which needs a per-category
splitting step BEFORE this skill's conversion — see `inspect-mask-format`'s
"Critical constraint" section on why raw label values like `1`/`2`/`3` get
silently zeroed out by `coco_format.py`'s internal threshold). Skip this
step if you already know the masks are one-category-per-file binary masks.

### 1. Resolve repo context

Run the `configure-repo` skill first to resolve `REPO_ROOT` in one
round-trip (it also verifies SSH connectivity as part of reaching HPC), in
place of the standalone connectivity check below.

Alternatively, verify SSH HPC connectivity directly:

```bash
ssh hpc_muc echo connected
```

If this prompts for a password, stop and set up passwordless SSH first.

### 2. Verify or create the folder structure

Ask the user:

> "Do you already have separate `masks/` and `images/` folders ready on HPC? Do you want to convert `train`, `test`, or both splits?"

- If **yes**, ask for the absolute HPC paths for each split and proceed to step 3.
- If **no**, offer to create the structure. Ask for:
  - Base data directory on HPC (e.g., `/home/hpc_storage/data/projects/$USER/sam3_data/<category>`).
  - Which splits to prepare (`train`, `test`, or both).
  - Whether to also create the output directory.

Create the structure with:

```bash
ssh hpc_muc 'mkdir -p /home/hpc_storage/data/projects/$USER/sam3_data/<category>/train/images'
ssh hpc_muc 'mkdir -p /home/hpc_storage/data/projects/$USER/sam3_data/<category>/train/masks'
ssh hpc_muc 'mkdir -p /home/hpc_storage/data/projects/$USER/sam3_data/<category>/test/images'
ssh hpc_muc 'mkdir -p /home/hpc_storage/data/projects/$USER/sam3_data/<category>/test/masks'
ssh hpc_muc 'mkdir -p /home/hpc_storage/data/projects/$USER/sam3_fine_tune/<output-folder>'
```

Then instruct the user to upload/copy their images and masks into those folders
before running the conversion.

### 3. Build and submit the conversion command

**Recommended (fastest): use the one-shot helper.**

Given a base directory containing `images/` and `masks/`, run:

```bash
ssh hpc_muc 'bash -s -- <base_dir> <category_name> <category_id> [file_name]' \
  < .github/scripts/run_coco_conversion.sh
```

Example:

```bash
ssh hpc_muc 'bash -s -- /home/hpc_storage/data/projects/userB/data/cegled_econo/47834/curated8bit/train matrix_chip 1 annotations' \
  < .github/scripts/run_coco_conversion.sh
```

The helper will:

1. Discover the repo root on HPC (cached per-user at `~/.cache/hpc_repo_root/sam3_fine_tune` after the first run, so repeat calls skip the `find` scan; set `FORCE_REFRESH=1` to bypass the cache).
2. Activate the project `.venv` on the login node.
3. Submit the conversion to the HPC CPU queue via `bsub`.
4. Validate the output JSON.

**Manual fallback:** if the helper is not suitable, activate the project venv on
HPC first, then submit:

```bash
ssh hpc_muc '
  REPO_ROOT=$(bash /home/hpc_storage/data/projects/$USER/test_sam3/sam3_fine_tune/.github/scripts/discover_repo_root.sh sam3_fine_tune)
  cd "$REPO_ROOT"
  source .venv/bin/activate
  bsub -Is -R "type=X64LIN && osrel>=70 && ui=pytest rusage[mem=50G]" -n 30 \
    "python $REPO_ROOT/sam3_fine_tune/preprocessing/coco_format.py \
      --mask_dir <mask_dir> \
      --img_dir <img_dir> \
      --file_name <file_name> \
      --output_dir <output_dir> \
      --category_name <category_name> \
      --category_id <category_id>"
'
```

The venv is already activated in the current shell before `bsub -Is` is called,
so the interactive job inherits it — no need to re-source `.venv/bin/activate`
inside the bsub command.

**Important:** call `coco_format.py` with its **absolute path**, not
`cd <preprocessing_dir> && python coco_format.py`. When the `cd && python ...`
form is passed as the bsub job command, the `cd` can silently fail to apply on
the dispatched compute node, causing `python: can't open file '.../coco_format.py'` (it looks for the script in `$REPO_ROOT` instead of
`preprocessing/`). Using the absolute script path avoids this entirely and was
confirmed working in testing.

For a `train` + `test` setup, submit twice:

- `<mask_dir>` = `<base>/train/masks`, `<img_dir>` = `<base>/train/images`, `<output_dir>` = `<base>/train`
- `<mask_dir>` = `<base>/test/masks`, `<img_dir>` = `<base>/test/images`, `<output_dir>` = `<base>/test`

This produces `<base>/train/annotation.json` and `<base>/test/annotation.json`,
which is the structure expected by the training pipeline.

If you only have a single dataset (no train/test split), set `<output_dir>` to
the same `<base>` directory that contains `images/` so the final layout is:

```text
<base>/
├── images/
├── masks/
└── annotation.json
```

### 4. Validate the output

After each job finishes, check that the output JSON exists:

```bash
ssh hpc_muc 'ls -lh <output_dir>/<file_name>.json'
```

Optionally, print the number of images and annotations, plus the per-category
summary from the metadata file:

```bash
ssh hpc_muc 'python - <<"PY"
import json
from pathlib import Path
ann = json.load(open("<output_dir>/<file_name>.json"))
meta = json.load(open("<output_dir>/<file_name>_metadata.json"))
print(f"{len(ann['images'])} images, {len(ann['annotations'])} annotations")
for cat in meta["categories"]:
    print(f"  {cat['name']}: {cat['image_count']} images, {cat['annotation_count']} annotations")
PY'
```

Confirm the final structure matches the training pipeline expectation:

```text
your_dataset/
├── train/
│   ├── images/
│   └── annotation.json
└── test/
    ├── images/
    └── annotation.json
```

## Decision Logic

- If the user provides Windows-style paths, convert them to HPC mount paths
  (`/home/hpc_storage/data/projects/...`).
- If `mask_dir` or `img_dir` does not exist on HPC, stop and ask whether to
  create the structure or correct the path.
- If the README describes a different folder convention, prefer the README
  convention and explain the difference to the user.

## Output Format

1. Verified HPC paths (or created directories).
2. Full bsub command submitted.
3. Job status / first log output.
4. Path to the generated COCO JSON and a quick validation summary.

## One-shot helper

For a directory that already contains `images/` and `masks/`, the fastest path
is to use the helper script from your local machine:

```bash
ssh hpc_muc 'bash -s -- <base_dir> <category_name> <category_id> [file_name]' \
  < .github/scripts/run_coco_conversion.sh
```

Example:

```bash
ssh hpc_muc \
  'bash -s -- /home/hpc_storage/data/projects/userB/data/cegled_econo/47834/curated8bit/train matrix_chip 1 annotations' \
  < .github/scripts/run_coco_conversion.sh
```

This discovers the repo, activates the venv, submits the CPU job, waits for it
to finish, and validates the output JSON in one step.

## Example prompts

- `Convert my matrix_chip masks and images to COCO format on HPC CPU.`
- `Prepare folder structure for COCO conversion and submit it.`
- `Run coco_format.py for category matrix_chip with category_id 1.`
