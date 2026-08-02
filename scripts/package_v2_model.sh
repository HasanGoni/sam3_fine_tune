#!/bin/bash
# Package the v2 2-GPU trained model into the Models registry with metadata.

set -e

EXPERIMENT=/home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune/experiments/9cat_diverse_wires_v2_2gpu
BEST_CKPT=${EXPERIMENT}/checkpoints/val_pin_detection_coco_eval_segm_AP.pt
CONFIG=${EXPERIMENT}/config.yaml
RESOLVED_CONFIG=${EXPERIMENT}/config_resolved.yaml

DATE=$(date +%Y-%m-%d)
REGISTRY=/home/hpc_storage/Models/sam3-finetuned
DEST=${REGISTRY}/${DATE}_9cat_diverse_wires_v2_2gpu

if [ ! -f "${BEST_CKPT}" ]; then
    echo "Best checkpoint not found: ${BEST_CKPT}" >&2
    exit 1
fi

mkdir -p "${DEST}"

cp -v "${BEST_CKPT}" "${DEST}/model.pt"
[ -f "${CONFIG}" ] && cp -v "${CONFIG}" "${DEST}/config.yaml"
[ -f "${RESOLVED_CONFIG}" ] && cp -v "${RESOLVED_CONFIG}" "${DEST}/config_resolved.yaml"

# Pull the best validation AP from best_stats.json.
BEST_STATS=${EXPERIMENT}/logs/best_stats.json
python3 - <<PYEOF > "${DEST}/metadata.json"
import json
from datetime import datetime

stats_path = "${BEST_STATS}"
val_ap = None
thin_wire_panda_ap = None
best_epoch = None
if stats_path:
    try:
        best_ap = -1.0
        with open(stats_path) as f:
            for line in f:
                d = json.loads(line)
                ap = d.get("val_pin/detection/coco_eval_segm_AP")
                if ap is not None and ap > best_ap:
                    best_ap = ap
                    val_ap = ap
                    best_epoch = d.get("Trainer/epoch")
                    thin_wire_panda_ap = d.get("val_pin/detection/coco_eval_segm_AP_thin wire panda")
    except Exception as e:
        print(f"warning: could not read stats: {e}", file=__import__("sys").stderr)

metadata = {
    "model_name": "sam3-finetuned-9cat-diverse-wires-v2-2gpu",
    "source_experiment": "${EXPERIMENT}",
    "source_checkpoint": "${BEST_CKPT}",
    "packaged_at": datetime.utcnow().isoformat() + "Z",
    "date": "${DATE}",
    "dataset": "/home/hpc_storage/data/projects/userB/sam3_data/9cat_diverse_wires_v2",
    "config": "sam3/train/configs/full_train_9cat_diverse_wires_v2_2gpu.yaml",
    "metrics": {
        "best_val_coco_eval_segm_AP": val_ap,
        "best_epoch": best_epoch,
        "thin_wire_panda_AP": thin_wire_panda_ap
    },
    "hardware": "2x GPU",
    "queue": "gpu (2-GPU v2 training)",
    "notes": "Best checkpoint selected by val_pin/detection/coco_eval_segm_AP."
}
print(json.dumps(metadata, indent=2))
PYEOF

echo "Packaged model to ${DEST}"
ls -la "${DEST}"
