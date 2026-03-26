# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastBEV is a Bird's-Eye View (BEV) 3D object detection system for autonomous driving using multi-view camera images. Based on MMDetection3D framework and targets nuScenes dataset.

**Architecture Pipeline:**
1. **2D Backbone**: ResNet-based feature extraction from multi-view cameras
2. **2D Neck**: FPN multi-scale feature fusion
3. **View Transformation**: Projects 2D features to 3D BEV space
4. **3D Neck**: M2BevNeck processes BEV features
5. **Detection Head**: FreeAnchor3DHead for 3D bounding box prediction

## Common Commands

### Environment Setup
```bash
conda create -n fastbev python=3.8
conda activate fastbev
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.0 mmdet==2.14.0 mmsegmentation==0.14.1 ipdb timm
python setup.py develop
```

### Training
```bash
# Using shell script (8 GPUs)
bash tools/run_train.sh

# Single GPU
torchrun --nproc_per_node=1 --master_port=10292 tools/train.py \
    ./configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v128x128x4_c192_d2_f4_train.py \
    --work-dir=./work_dirs/exp/ \
    --launcher="pytorch" \
    --gpus 1
```

### Testing
```bash
# Test PyTorch model
bash tools/run_test.sh

# Or directly
CUDA_VISIBLE_DEVICES=0 python tools/test.py <config_path> <checkpoint_path> --out results.pkl --format-only
CUDA_VISIBLE_DEVICES=0 python tools/eval.py <config_path> --out results.pkl --eval bbox
```

### Visualization
```bash
# Draw predictions (uses LOCAL_MODE=True for mini dataset)
bash draw_pred.sh
```

### ONNX Export
```bash
python ./tools/export_onnx.py --config <config_path> --checkpoint <checkpoint_path>
```

## Environment Variables

- `LOCAL_MODE=True` - Use mini dataset for local testing
- `BATCH_SIZE=<n>` - Override training batch size (default: 6)
- `CUDA_VISIBLE_DEVICES=<ids>` - GPU selection

## Test Modes

Config `test_cfg.test_mode` controls inference backend:
- `test_pth` - PyTorch model
- `test_onnx` - ONNX model (requires onnx_path in config)
- `test_custom` - Custom deployment backend

## Key Files

| File | Purpose |
|------|---------|
| `mmdet3d/models/detectors/fastbev.py` | Main detector implementation |
| `mmdet3d/datasets/pipelines/multi_view.py` | MultiViewPipeline for camera inputs |
| `mmdet3d/datasets/nuscenes_monocular_dataset_map_2.py` | Dataset implementation |
| `configs/fastbev/exp/paper/*.py` | Model variant configs (M0-M5) |

## Model Variants

Config naming: `fastbev_m{variant}_r{resnet}_s{H}x{W}_v{BEV}_c{channels}_d{depth}_f{frames}`

- M0: ResNet18, 256x704 input, 128x200x4 BEV
- M1: ResNet18, 320x880 input
- M2: ResNet34 backbone
- M4/M5: ResNet50 with larger BEV grid

## Code Style

- yapf with PEP8 base style
- isort with 79 char line length
- Known first-party: mmdet, mmseg, mmdet3d
