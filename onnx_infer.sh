export LOCAL_MODE=True
CUDA_VISIBLE_DEVICES=0 python tools/onnx_infer_w_vel.py \
    ./configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v128x128x4_c192_d2_f4_train.py \
    /root/ziyi/product_e2e_demo-main-fastbev/fastbev/train/fastbev/work_dirs/onnx_viz