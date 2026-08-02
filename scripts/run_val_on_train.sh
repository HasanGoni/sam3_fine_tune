#!/bin/bash
# Run validation on the v2 training set to generate COCO predictions.

set -e

. /opt/ifxenvmods/3.2.13/init/bash
cd /home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune/sam3/train
source /home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune/.venv/bin/activate
module load cuda/12.8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py -c configs/full_train_9cat_diverse_wires_v2_2gpu_val_on_train.yaml --num-gpus 1 --num-nodes 1
