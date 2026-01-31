#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    ./configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py \
    ./ckpts/epoch15_batch4_lr0002.pth \
    --out work_dirs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4/onnx_results/results.pkl \
    --format-only \
    --eval-options jsonfile_prefix=work_dirs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4/onnx_results

CUDA_VISIBLE_DEVICES=0 python tools/eval.py \
    ./configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py \
    --out work_dirs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4/onnx_results/results.pkl \
    --eval bbox
