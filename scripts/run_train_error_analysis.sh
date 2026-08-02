#!/bin/bash
# Thin wire panda error analysis on the v2 training set predictions.

set -e

cd /home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune
source /home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune/.venv/bin/activate

DATA_ROOT=/home/hpc_storage/data/projects/userB/sam3_data/9cat_diverse_wires_v2
VAL_EXPERIMENT=/home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune/experiments/9cat_diverse_wires_v2_2gpu_val_on_train
TRAIN_EXPERIMENT=/home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune/experiments/9cat_diverse_wires_v2_2gpu
OUT_DIR=${DATA_ROOT}/train/error_analysis

mkdir -p "${OUT_DIR}"

python scripts/thin_wire_panda_error_analysis.py \
    --gt "${DATA_ROOT}/train/annotation_merged.json" \
    --pred "${VAL_EXPERIMENT}/dumps/train_pins/coco_predictions_segm.json" \
    --images "${DATA_ROOT}/train/images" \
    --out "${OUT_DIR}" \
    --category 9 \
    --max-cases 64 \
    --iou-thresh 0.5 \
    --score-thresh 0.05
