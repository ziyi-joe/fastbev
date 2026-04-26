export LOCAL_MODE=False

python tools/single_eval.py \
    configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v128x128x4_c192_d2_f4_train.py \
    work_dirs/0418/epoch_8.pth