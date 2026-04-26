# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import mmcv
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from os import path as osp

import copy
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

mmdet3d_root = os.environ.get('MMDET3D')
if mmdet3d_root is not None and osp.exists(mmdet3d_root):
    import sys
    sys.path.insert(0, mmdet3d_root)
    print(f"using mmdet3d: {mmdet3d_root}")

from mmdet3d.apis import single_gpu_test, multi_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import  set_random_seed
from mmdet.datasets import replace_ImageToTensor
from IPython import embed
import ipdb

import onnx, onnxruntime
from onnxsim import simplify

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config', type=str, required=True, help='test config file path')
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint file')
    parser.add_argument('--out', type=str, default='name', help='output result file in pickle format')

    
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    model.eval()

    with torch.no_grad():
        for data in data_loader:
            export_onnx(model, data, args.out, export_2d=True, export_3d=True)
            break

def export_onnx(model, data, name, export_2d=False, export_3d=False):
    model = copy.deepcopy(model).cuda()
    model.eval()
    # Prepare dummy inputs (modify according to your model's input format)
    # img_metas = val_dataset['img_metas']
    # img = val_dataset['img']

    # 导出2D模型    # backbone + neck + neck_fuse
    if export_2d:
        img = [torch.randn((1, 3, 256, 704)).cuda()]
        onnx_path_2d = f'/data_nas/ziyi/onnx/{name}_2d_model.onnx'
        # 导出ONNX模型
        torch.onnx.export(
            model,
            (img, None),
            onnx_path_2d,
            verbose=True,
            opset_version=13,
            input_names=['img'],
            output_names=['feat_2d']
        )
        onnx_model_2d = onnx.load(onnx_path_2d)
        onnx.checker.check_model(onnx_model_2d)
        simplified_model_2d, check_ok = simplify(onnx_model_2d)
        onnx.save_model(simplified_model_2d, f'/data_nas/ziyi/onnx/{name}_2d_model.onnx')

    # 导出3D模型
    if export_3d:
        dummy_data = torch.randn((1, 256, 200, 128)).cuda()
        bev_feat = [dummy_data, dummy_data, dummy_data, dummy_data]
        # img = [dummy_data]
        onnx_path_3d = f'/data_nas/ziyi/onnx/{name}_3d_model.onnx'
        # 导出ONNX模型
        torch.onnx.export(
            model,
            (bev_feat, None),
            onnx_path_3d,
            verbose=True,
            opset_version=13,
            input_names=['bev_feat0', 'bev_feat1', 'bev_feat2', 'bev_feat3'],
            output_names=['cls_score', 'bbox_pred', 'dir_cls_preds', 'pred_bev_map', 'instance_map']
        )
        onnx_model_3d = onnx.load(onnx_path_3d)

        # 检查ONNX模型
        onnx.checker.check_model(onnx_model_3d)
        simplified_model_3d, check_ok = simplify(onnx_model_3d)
        onnx.save_model(simplified_model_3d, f'/data_nas/ziyi/onnx/{name}_3d_model.onnx')

if __name__ == '__main__':
    main()
