# Thin Wire Panda Error Analysis

Branch: `feat/thin-wire-panda-error-analysis`

## Context

Thin wire panda (category 9) is the lowest-performing class in the current
best checkpoint (`9cat_diverse_wires_v1_2gpu`, AP ≈ 0.90 overall). This folder
tracks the investigation into why that class lags behind the others.

## How the analysis was generated

```bash
python scripts/thin_wire_panda_error_analysis.py \
  --gt /home/hpc_storage/data/projects/userB/sam3_data/9cat_diverse_wires_v1/test/annotation_merged.json \
  --pred /home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune/experiments/9cat_diverse_wires_v1_2gpu/dumps/pins/coco_predictions_segm.json \
  --images /home/hpc_storage/data/projects/userB/sam3_data/9cat_diverse_wires_v1/test/images \
  --out /home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune/experiments/9cat_diverse_wires_v1_2gpu/dumps/thin_wire_panda_error_analysis \
  --category 9 \
  --max-cases 64
```

## Key findings (first pass)

- **Very small class**: only 257 GT annotations across 200 test images. For
  comparison, thick wire panda has 4,638 annotations and thin wire HD has 441.
- **Severe over-segmentation** is the dominant failure mode. Many images have
  1 GT instance but 20–100 predicted instances (capped by the model's top-k
  output). Example worst cases:
  - image 3583: GT=1, predictions=100, TP=0, FP=100, FN=1
  - image 3642: GT=1, predictions=79, TP=1, FP=78, FN=0
  - image 3708: GT=1, predictions=78, TP=1, FP=77, FN=0
- **False negatives are rare** at the image level: only 1 image with thin wire
  panda GT had zero predictions.
- **Prediction score calibration is weak**: many low-confidence predictions
  survive because the model outputs a large number of masks for this class.
  The top prediction on over-segmented images is often reasonable (scores
  0.3–0.8), but the tail is long.
- **Category confusion is minimal**: on thin-wire-panda GT images, almost all
  predictions are class 9; only 6 predictions were class 8 (thick wire panda).

## Possible reasons for low AP

1. **Class imbalance / insufficient data**: 257 training examples for thin
   wire panda vs. thousands for other wire categories.
2. **Small/thin structures**: thin wire panda instances are small and have
   lower area (mean ≈ 1,300 px, median ≈ 1,100 px) than thick wire panda
   (mean ≈ 6,000 px), making them harder to segment cleanly.
3. **Model over-segmentation**: SAM3 is producing many overlapping candidate
   masks for thin wire regions; the correct mask may be present but drowned
   out by false positives at evaluation.
4. **Prompt/assignment ambiguity**: thin wire panda may be visually close to
   thin wire HD (similar aspect ratios) but from a different source, leading
   to inconsistent learning signals.

## Next steps to try

- Increase thin wire panda representation in the next dataset version.
- Apply post-processing NMS or score threshold tuning specific to this class.
- Inspect the top false-positive masks to see if they correspond to real thin
  wires that were simply not annotated.
- Compare with thin wire HD: both are thin wires and may benefit from merged
  training or shared heads.

## Label-quality cleanup

Based on visual review, two buckets were created for data correction:

1. **`no_ground_truth_available`** — 2 images where the model predicted thin
   wire panda masks but no GT existed. Predictions were converted into 4 new
   GT annotations.
2. **`uncorrect_label`** — 26 images with large prediction masks that did not
   overlap the existing GT (<10% IoU), indicating likely wrong or incomplete
   labels. These images were removed from the cleaned annotation file.

Cleaned annotation file:
`9cat_diverse_wires_v1/test/annotation_merged_cleaned_thin_wire_panda.json`

Cleaning record:
`9cat_diverse_wires_v1/test/thin_wire_panda_cleaning_record.json`

Effect: original 4,179 test images / 13,249 annotations → cleaned 4,153
images / 13,226 annotations; thin wire panda annotations: 257 → 234.

## Training-set cleanup

The same label-quality pipeline was run on the training split. Findings:

- **201 training images** had category-9 predictions but no thin wire panda GT.
- **0 training images** were flagged as `uncorrect_label`.
- Using a conservative filter (predictions converted to GT only when an image
  had ≤5 predictions and the best prediction scored ≥0.5):
  - 162 images were converted, adding 320 new thin wire panda annotations.
  - The remaining 39 images were kept unchanged; their existing annotations
    (other categories) were **not** removed. Only the unreliable thin wire
    panda predictions were ignored.
- Cleaned training annotation file:
  `9cat_diverse_wires_v1/train/annotation_merged_cleaned_thin_wire_panda.json`
- Training cleanup script: `scripts/thin_wire_panda_clean_train_annotations.py`
- Converted-image review poster:
  `sam3_data/9cat_diverse_wires_v1/error_analysis_thin_wire_panda_train/converted_training_poster.jpg`

Effect: original 10,228 training images / 42,000 annotations → cleaned
10,228 images / 42,320 annotations; thin wire panda annotations: 1,302 → 1,622.

### Data-source note

Thin wire panda in `9cat_diverse_wires_v1` comes from diversity (DINOv3)
selection over a pool of ~48k patched train images from
`9cat_separated_wires_v3_patched_wires_256_v8`. Only 1,000 train images were
selected for this category, which is why the class is still small. The source
pool has many more thin wire panda patches; a future dataset version could
increase the selected count or bias the selection toward images with larger
continuous thin-wire coverage.

## Expanded dataset (`9cat_diverse_wires_v2`)

To address the thin wire panda data shortage, a new dataset version was built
with more thin wire panda images selected for larger, continuous masks:

- **Builder changes**: `sam3_fine_tune/preprocessing/build_diverse_wire_dataset.py`
  now supports per-diverse-category mask filters:
  - `min_total_area`: minimum combined thin-wire-panda mask area per image
  - `min_max_component_area`: minimum area of the largest connected component
  - `min_coverage`: optional minimum mask coverage ratio
- **Config**: `sam3_fine_tune/preprocessing/configs/9cat_diverse_wires_v2_config.yaml`
  - `thin_wire_panda` target raised from 1,200 to **2,000** images
    (train 1,800 / test 200)
  - Filter applied: `min_total_area: 1000`, `min_max_component_area: 1000`
  - Other categories unchanged (`thick_wire_HD`, `thin_wire_HD` at 1,000 each)
- **Result**:
  - Train: 11,028 images / 43,150 annotations (vs. v1 10,228 / 42,000)
  - Test: 4,179 images / 13,220 annotations
  - Thin wire panda train annotations: 2,412 (vs. v1 1,302)
  - Thin wire panda train images: 1,800 (vs. v1 1,000)
- **Mask-quality comparison (train thin wire panda)**:
  - v1: mean total mask area 1,876 px, mean largest component 1,653 px
  - v2: mean total mask area 2,410 px, mean largest component 2,124 px
  - v2 has zero images with no thin wire panda mask; the mask-area filter
    excluded patches with zero/small/fragmented thin wire panda labels.

Generated review posters (GT overlay):

| Poster | Path |
|---|---|
| v2 all categories (one row per category, 16 images each) | `9cat_diverse_wires_v2/overlay_poster_all_categories.jpg` |
| v2 thin wire panda all 1,800 train images | `9cat_diverse_wires_v2/thin_wire_panda_all_2000_overlay_poster.jpg` |
| v2 wire categories comparison | `9cat_diverse_wires_v2/wire_categories_comparison_poster.jpg` |
| v1 thin wire panda all 1,000 train images (for comparison) | `9cat_diverse_wires_v1/thin_wire_panda_all_1000_overlay_poster.jpg` |

The v2 dataset is ready at:
`/home/hpc_storage/data/projects/userB/sam3_data/9cat_diverse_wires_v2`

## Generated artifacts (not committed because of size)

- Poster of the 64 worst cases:
  `experiments/9cat_diverse_wires_v1_2gpu/dumps/thin_wire_panda_error_analysis/thin_wire_panda_bad_cases_poster.jpg`
- Individual side-by-side overlays:
  `experiments/9cat_diverse_wires_v1_2gpu/dumps/thin_wire_panda_error_analysis/visualizations/`
- Raw copies of the bad-case source images:
  `experiments/9cat_diverse_wires_v1_2gpu/dumps/thin_wire_panda_error_analysis/bad_case_raw_images/`
- Label-quality review posters (image / GT / prediction / overlay):
  `sam3_data/9cat_diverse_wires_v1/error_analysis_thin_wire_panda/label_quality_with_overlay/`
- Training label-quality review posters:
  `sam3_data/9cat_diverse_wires_v1/error_analysis_thin_wire_panda_train/label_quality_with_overlay/`
