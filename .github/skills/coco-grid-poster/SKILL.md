---
name: coco-grid-poster
description: 'Generate a large grid poster of COCO category images with mask overlays, for visual inspection of labeling quality. Supports a single category (all images) or multiple categories (one labeled row per category).'
argument-hint: '[--coco-json <path>] [--categories <name1,name2,...>] [--output <path>]'
user-invocable: true
disable-model-invocation: false
---
# COCO Grid Poster

Generate a single JPEG poster that shows many COCO images for one or more
categories with their mask overlays. Useful for quickly spotting missing or
wrong labels (e.g. "thin wire panda" instances that are unlabeled because only
`thick wire` is annotated in the same image).

## When To Use

- You need to visually verify that every image containing a category is
  actually labeled.
- You want to compare several categories side-by-side in one poster.
- The existing `save_coco_overlay_poster.py` only samples a few random images;
  this skill renders **all** images for a single category or a configurable
  number per category when multiple categories are requested.

## Script

The helper is checked in at
`scripts/save_coco_grid_poster.py`.

## Behavior

- **Single category** (`--categories thin wire panda`):
  - Renders **every** image that contains the category.
  - Default layout: square-ish grid (`n_cols ~= sqrt(n_images)`), 256 px panels.
  - Output example: `thin_wire_panda_all_1000_overlay_poster.jpg`.

- **Multiple categories** (`--categories "thin wire panda,thick wire panda,thin wire HD"`):
  - Renders **one labeled row per category** in a single poster.
  - Default: 16 image columns per row.
  - Use `--max-per-category N` to cap images per row (default 0 = all, which can
    make the poster very wide).
  - Each category row is labeled with the category name on the left and uses a
    distinct mask color.

## Inputs

- `coco_json`: path to COCO annotation JSON (uses sibling `images/` dir by
  default).
- `categories`: comma-separated category names.
- `output`: output JPEG path.
- Optional: `--panel-size`, `--n-cols`, `--max-per-category`, `--alpha`,
  `--dpi`, `--jpeg-quality`.

## Procedure

1. **Verify the category names** exist in the COCO file:

   ```bash
   ssh hpc_muc "python3 - << 'PYEOF'
   import json
   from pathlib import Path
   coco = json.loads(Path('/home/hpc_storage/data/projects/userB/sam3_data/9cat_diverse_wires_v1/train/annotation_merged.json').read_text())
   print([c['name'] for c in coco['categories']])
   PYEOF"
   ```

2. **Submit the poster job to the HPC `batch` queue** as a non-interactive job
   (rendering 1000 images at 256 px is fast but produces a large figure, so run
   it on the cluster):

   ```bash
   ssh hpc_muc '
   REPO=/home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune
   DATASET=/home/hpc_storage/data/projects/userB/sam3_data/9cat_diverse_wires_v1
   LOGDIR=$DATASET/poster_logs
   mkdir -p "$LOGDIR"
   cat > "$LOGDIR/run_grid_poster.sh" << "EOF"
   #!/usr/bin/env bash
   source /ifxhome/$USER/.bashrc
   source /home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune/.venv/bin/activate
   python /home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune/scripts/save_coco_grid_poster.py \
     --coco-json /home/hpc_storage/data/projects/userB/sam3_data/9cat_diverse_wires_v1/train/annotation_merged.json \
     --categories "thin wire panda" \
     --output /home/hpc_storage/data/projects/userB/sam3_data/9cat_diverse_wires_v1/thin_wire_panda_all_overlay_poster.jpg \
     --panel-size 256 --alpha 0.5 --dpi 100 --jpeg-quality 90
   EOF
   chmod +x "$LOGDIR/run_grid_poster.sh"
   bsub -q batch -P ma_ai -n 1 -R "rusage[mem=8000]" \
     -o "$LOGDIR/grid_poster.%J.out" -e "$LOGDIR/grid_poster.%J.err" \
     "$LOGDIR/run_grid_poster.sh"
   '
   ```

   For a **multi-category comparison poster**:

   ```bash
   ssh hpc_muc '
   DATASET=/home/hpc_storage/data/projects/userB/sam3_data/9cat_diverse_wires_v1
   LOGDIR=$DATASET/poster_logs
   cat > "$LOGDIR/run_multi_grid_poster.sh" << "EOF"
   #!/usr/bin/env bash
   source /ifxhome/$USER/.bashrc
   source /home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune/.venv/bin/activate
   python /home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune/scripts/save_coco_grid_poster.py \
     --coco-json /home/hpc_storage/data/projects/userB/sam3_data/9cat_diverse_wires_v1/train/annotation_merged.json \
     --categories "thin wire panda,thick wire panda,thin wire HD,thick wire HD" \
     --output /home/hpc_storage/data/projects/userB/sam3_data/9cat_diverse_wires_v1/wire_categories_comparison_poster.jpg \
     --panel-size 256 --n-cols 16 --max-per-category 32 --alpha 0.5 --dpi 100 --jpeg-quality 90
   EOF
   chmod +x "$LOGDIR/run_multi_grid_poster.sh"
   bsub -q batch -P ma_ai -n 1 -R "rusage[mem=8000]" \
     -o "$LOGDIR/grid_poster.%J.out" -e "$LOGDIR/grid_poster.%J.err" \
     "$LOGDIR/run_multi_grid_poster.sh"
   '
   ```

3. **Wait for completion** and check the output:

   ```bash
   ssh hpc_muc "bjobs -w <job_id>"
   ```

4. **Verify the JPEG** was created:

   ```bash
   ssh hpc_muc "ls -lh /home/hpc_storage/data/projects/userB/sam3_data/9cat_diverse_wires_v1/*.jpg"
   ```

## Decision Logic

- Use **single-category mode** when the goal is a full census of one category
  (e.g. "show me all 1000 thin wire panda images so I can spot missing masks").
- Use **multi-category mode** when the goal is a compact comparison across
  categories.
- Keep `--panel-size 256` as a good balance between readability and file size.
- For very large single-category posters (1000+ images), the resulting JPEG can
  be ~10–20 MB and 8000+ px wide; use `--panel-size 128` and `--dpi 72` if you
  need a smaller file for quick review.

## Output Format

1. Job ID of the submitted batch job.
2. Path to the generated JPEG poster.
3. Poster dimensions (rows × columns) and file size.
