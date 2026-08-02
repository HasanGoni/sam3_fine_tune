#!/bin/bash
# Thin wire panda error analysis on the v2 test set.

set -e

cd /home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune
source /home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune/.venv/bin/activate

DATA_ROOT=/home/hpc_storage/data/projects/userB/sam3_data/9cat_diverse_wires_v2
EXPERIMENT=/home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune/experiments/9cat_diverse_wires_v2_2gpu
OUT_DIR=${DATA_ROOT}/test/error_analysis

mkdir -p "${OUT_DIR}"

python scripts/thin_wire_panda_error_analysis.py \
    --gt "${DATA_ROOT}/test/annotation_merged.json" \
    --pred "${EXPERIMENT}/dumps/pins/coco_predictions_segm.json" \
    --images "${DATA_ROOT}/test/images" \
    --out "${OUT_DIR}" \
    --category 9 \
    --max-cases 64 \
    --iou-thresh 0.5 \
    --score-thresh 0.05

echo "=== Test-set thin wire panda overlay poster ==="
python scripts/save_coco_overlay_poster.py \
    --coco-json "${DATA_ROOT}/test/annotation_merged.json" \
    --image-dir "${DATA_ROOT}/test/images" \
    --output "${OUT_DIR}/thin_wire_panda_overlay_poster.jpg" \
    --category "thin wire panda" \
    --n-samples 16 \
    --seed 42 \
    --alpha 0.45 \
    --dpi 200
