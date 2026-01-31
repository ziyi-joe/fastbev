#!/bin/bash

BACKBONE_ONNX_PATH=$1
HEAD_ONNX_PATH=$2

CUDA_VISIBLE_DEVICES=0 python tools/auto_test.py \
    ./configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py \
    ./ckpts/epoch15_batch4_lr0002.pth \
    --out work_dirs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4/onnx_results/results.pkl \
    --format-only \
    --eval-options jsonfile_prefix=work_dirs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4/onnx_results \
    --backbone_onnx_path $BACKBONE_ONNX_PATH \
    --head_onnx_path $HEAD_ONNX_PATH 
    

CUDA_VISIBLE_DEVICES=0 python tools/eval.py \
    ./configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py \
    --out work_dirs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4/onnx_results/results.pkl \
    --eval bbox
