#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    tools/train.py \
    ./configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4_train.py \
    --work-dir=./work_dirs/my/exp/ \
    --launcher="pytorch" \
    --gpus 4
