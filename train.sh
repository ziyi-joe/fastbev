#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LOCAL_MODE=True
export BATCH_SIZE=2
# export NCCL_P2P_DISABLE=1

torchrun --nproc_per_node=1 --master_port=10292 tools/train.py \
    ./configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v128x128x4_c192_d2_f4_train.py \
    --work-dir=./work_dirs/exp/ \
    --launcher="pytorch" \
    --gpus 1
