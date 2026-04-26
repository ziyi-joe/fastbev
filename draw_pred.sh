export LOCAL_MODE=False
export BATCH_SIZE=1
CUDA_VISIBLE_DEVICES=0 python tools/draw_pred.py \
    ./configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v128x128x4_c192_d2_f4_train.py \
    work_dirs/0418/epoch_8.pth \
    work_dirs/0418/8_vis