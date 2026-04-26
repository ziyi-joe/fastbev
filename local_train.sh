export LOCAL_MODE=True

torchrun --nproc_per_node=1 tools/train.py \
    ./configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v128x128x4_c192_d2_f4_train.py \
    --work-dir=./work_dirs/exps/ \
    --launcher="pytorch" \
    --gpus 1