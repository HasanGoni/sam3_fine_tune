#! /bin/bash
mask_dir='/home/hpc_storage/data/projects/userA/sam3_data/wuxi_pin/source_test/mask'
img_dir='/home/hpc_storage/data/projects/userA/sam3_data/wuxi_pin/source_test/image'
file_name='annotation'
output_dir='/home/hpc_storage/data/projects/userA/sam3_fine_tune/test'
category_name='wuxi_pin'
category_id=6

pc="python coco_format.py \
    --mask_dir $mask_dir \
    --img_dir $img_dir \
    --file_name $file_name \
    --output_dir $output_dir \
    --category_name $category_name \
    --category_id $category_id"

bsub -Is -R "type=X64LIN && osrel>=70 && ui=pytest rusage[mem=50G]" -n 30 "$pc"