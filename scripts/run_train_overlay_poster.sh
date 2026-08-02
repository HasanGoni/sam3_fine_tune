#!/bin/bash
# Overlay poster for thin wire panda annotations in the v2 training set.

set -e

cd /home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune
source /home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune/.venv/bin/activate

DATA_ROOT=/home/hpc_storage/data/projects/userB/sam3_data/9cat_diverse_wires_v2
EXPERIMENT=/home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune/experiments/9cat_diverse_wires_v2_2gpu
OUT_DIR=${DATA_ROOT}/train/error_analysis/label_review

mkdir -p "${OUT_DIR}"

python scripts/save_coco_overlay_poster.py \
    --coco-json "${DATA_ROOT}/train/annotation_merged.json" \
    --image-dir "${DATA_ROOT}/train/images" \
    --output "${OUT_DIR}/thin_wire_panda_train_overlay_poster.jpg" \
    --category "thin wire panda" \
    --n-samples 16 \
    --seed 42 \
    --alpha 0.45 \
    --dpi 200
