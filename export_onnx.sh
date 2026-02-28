export LOCAL_MODE=True

python ./tools/export_onnx.py \
--config ./configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v128x128x4_c192_d2_f4_train.py \
--checkpoint ./work_dirs/0228_2/epoch_1.pth