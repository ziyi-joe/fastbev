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
    parser.add_argument('--out', type=str, default='./output/result.pkl', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', type=bool, default=False, help='show results')
    parser.add_argument(
        '--show-dir', type=str, default='./show_dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument(
        '--vis',
        action='store_true',
        help='whether vis')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='whether debug')
    parser.add_argument('--debug_num', type=int, default=50)
    
    parser.add_argument('--extrinsic-noise', '-n', type=float, default=0)
    
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

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

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    if args.vis:
        nms_thr = 0.0001
        try:
            cfg.model.test_cfg.nms_thr = nms_thr
        except:
            print('### imvoxelnet except in train.py ###')
            cfg.test_cfg.nms_thr = nms_thr

    if args.extrinsic_noise > 0:
        for i in range(3):
            print('### test camera extrinsic robustness ###')
        cfg.model.extrinsic_noise = args.extrinsic_noise

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    model.eval()

    with torch.no_grad():
        for data in data_loader:
            export_onnx(model, data, export_2d=True, export_3d=True)
            break

def export_onnx(model, data, export_2d=False, export_3d=False):
    model = copy.deepcopy(model).cuda()
    model.eval()
    # Prepare dummy inputs (modify according to your model's input format)
    # img_metas = val_dataset['img_metas']
    # img = val_dataset['img']

    # 导出2D模型    # backbone + neck + neck_fuse
    if export_2d:
        img = [torch.randn((1, 3, 256, 704)).cuda()]
        onnx_path_2d = f'./work_dirs/onnx/export_2d_model.onnx'
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
        onnx.save_model(simplified_model_2d, f'./work_dirs/onnx/simplified_export_2d_model.onnx')

    # 导出3D模型
    if export_3d:
        dummy_data = torch.randn((1, 256, 128, 128)).cuda()
        bev_feat = [dummy_data, dummy_data, dummy_data, dummy_data]
        # img = [dummy_data]
        onnx_path_3d = f'./work_dirs/onnx/export_3d_model.onnx'
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
        onnx.save_model(simplified_model_3d, f'./work_dirs/onnx/simplified_export_3d_model.onnx')

if __name__ == '__main__':
    main()
