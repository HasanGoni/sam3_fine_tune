#!/bin/bash
# Post-training error analysis and overlay generation for the v2 9-category
# diverse-wires experiment.
#
# Runs:
#   1. Test-set thin-wire-panda error analysis + bad-case poster
#   2. Test-set thin-wire-panda overlay poster
#   3. Training-set label overlay poster
#   4. Training-set error analysis, if val-on-train predictions already exist.
#
# Usage (from the repo root on HPC):
#   bash scripts/run_post_training_analysis.sh

set -e

cd /home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune
source /home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune/.venv/bin/activate

echo "=== 1/4 Test-set error analysis + bad-case poster ==="
bash scripts/run_test_error_analysis.sh

echo "=== 2/4 Test-set thin wire panda overlay poster ==="
DATA_ROOT=/home/hpc_storage/data/projects/userB/sam3_data/9cat_diverse_wires_v2
python scripts/save_coco_overlay_poster.py \
    --coco-json "${DATA_ROOT}/test/annotation_merged.json" \
    --image-dir "${DATA_ROOT}/test/images" \
    --output "${DATA_ROOT}/test/error_analysis/thin_wire_panda_overlay_poster.jpg" \
    --category "thin wire panda" \
    --n-samples 16 \
    --seed 42 \
    --alpha 0.45 \
    --dpi 200

echo "=== 3/4 Training-set label overlay poster ==="
bash scripts/run_train_overlay_poster.sh

VAL_PRED=/home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune/experiments/9cat_diverse_wires_v2_2gpu_val_on_train/dumps/train_pins/coco_predictions_segm.json
if [ -f "${VAL_PRED}" ]; then
    echo "=== 4/4 Training-set error analysis ==="
    bash scripts/run_train_error_analysis.sh
else
    echo "=== 4/4 Skipped: training-set predictions not found ==="
    echo "To generate them, submit: bash scripts/run_val_on_train.sh"
    echo "Then re-run this script or run: bash scripts/run_train_error_analysis.sh"
fi

echo "Post-training analysis complete. Artifacts saved under:"
echo "  ${DATA_ROOT}/test/error_analysis"
echo "  ${DATA_ROOT}/train/error_analysis/label_review"
