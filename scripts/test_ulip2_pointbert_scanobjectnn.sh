#!/bin/bash

if [ -z "$1" ]; then
echo "Please provide a *.pt file as input"
exit 1
fi

model_file=$1
output_dir=./outputs/pointBERT

CUDA_VISIBLE_DEVICES=0 python main.py \
--no-distributed \
--model ULIP2_PointBERT \
--npoints 8192 \
--workers 10 \
--linear-projection \
--output-dir $output_dir \
--evaluate_3d_ulip2 \
--validate_dataset_name scanobjectnn \
--validate_dataset_prompt modelnet40_64 \
--test_repeat 1 \
--test_ckpt_addr $model_file 2>&1 | tee $output_dir/log_scanobjectnn.txt