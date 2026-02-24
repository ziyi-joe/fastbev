# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import mmdet
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from nuscenes.utils.geometry_utils import transform_matrix
if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
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
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--no-aavt',
        action='store_true',
        help='Do not align after view transformer.')
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

import torch
import numpy as np

def calculate_distance_ap(all_preds, all_gts, dist_threshold=2.0):
    """
    基于中心点距离匹配的 mAP 计算
    dist_threshold: 距离阈值（米），NuScenes 常用 0.5m, 1.0m, 2.0m, 4.0m
    """
    all_labels_list = [g['labels'] for g in all_gts if len(g['labels']) > 0]
    if not all_labels_list:
        return 0
    unique_labels = np.unique(np.concatenate(all_labels_list))
    aps = []

    for label in unique_labels:
        scores, tps, fps = [], [], []
        total_gts = sum([sum(g['labels'] == label) for g in all_gts])
        
        if total_gts == 0: continue

        for p, g in zip(all_preds, all_gts):
            p_mask = p['labels'] == label
            g_mask = g['labels'] == label
            
            if not p_mask.any(): continue
            
            # 获取预测框和 GT 框的中心点 (x, y)
            # p['boxes'].gravity_center 是 [N, 3], 我们取前两维 [N, 2]
            p_centers = p['boxes'].gravity_center[p_mask][:, :2]
            p_score = p['scores'][p_mask]
            
            if not g_mask.any():
                fps.extend([1] * len(p_centers))
                tps.extend([0] * len(p_centers))
                scores.extend(p_score.tolist())
                continue

            g_centers = g['boxes'].gravity_center[g_mask][:, :2]

            # 1. 计算距离矩阵 [num_p, num_g]
            # 使用 broadcasting 计算欧氏距离
            # dists = sqrt((x1-x2)^2 + (y1-y2)^2)
            dists = torch.cdist(p_centers, g_centers, p=2) 
            
            # 2. 匹配逻辑
            sort_idx = np.argsort(-p_score)
            matched_gts = set()
            
            for idx in sort_idx:
                # 寻找最近的 GT
                min_dist, argmin_dist = dists[idx].min(dim=0)
                
                # 如果距离小于阈值且该 GT 未被匹配
                if min_dist < dist_threshold and argmin_dist.item() not in matched_gts:
                    tps.append(1)
                    fps.append(0)
                    matched_gts.add(argmin_dist.item())
                else:
                    tps.append(0)
                    fps.append(1)
                scores.append(p_score[idx])

        # 3. 计算 AP (Precision-Recall 曲线下面积)
        scores = np.array(scores)
        tps = np.array(tps)
        fps = np.array(fps)
        
        sort_indices = np.argsort(-scores)
        tps = np.cumsum(tps[sort_indices])
        fps = np.cumsum(fps[sort_indices])
        
        recalls = tps / total_gts
        precisions = tps / (tps + fps)
        
        # 积分法计算 AP
        ap = np.trapz(precisions, recalls) if len(recalls) > 0 else 0
        aps.append(ap)
        print(f"Class {label} AP@{dist_threshold}m: {ap:.4f}")

    return np.mean(aps) if aps else 0

def main():
    args = parse_args()

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')


    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # TODO： hard code

    # build the model and load checkpoint
    if not args.no_aavt:
        if '4D' in cfg.model.type:
            cfg.model.align_after_view_transfromation=True
    if 'num_proposals_test' in cfg and cfg.model.type=='DAL':
        cfg.model.pts_bbox_head.num_proposals=cfg.num_proposals_test
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
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    model = MMDataParallel(model, device_ids=cfg.gpu_ids)

    import numpy as np
    from tqdm import tqdm
    print("ready to infer")
    model.eval()
    all_preds = []
    all_gts = []

    for i, data in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            
            # 收集预测
            all_preds.append({
                'boxes': result[0]['boxes_3d'], # LiDARInstance3DBoxes
                'scores': result[0]['scores_3d'].cpu().numpy(),
                'labels': result[0]['labels_3d'].cpu().numpy()
            })
            
            # 收集真值
            all_gts.append({
                'boxes': data['gt_bboxes_3d'].data[0][0], # LiDARInstance3DBoxes
                'labels': data['gt_labels_3d'].data[0][0].cpu().numpy()
            })

    thresholds = [0.5, 1.0, 2.0, 4.0]
    mAPs = []

    for d_thresh in thresholds:
        print(f"\n--- Calculating AP for Distance Threshold: {d_thresh}m ---")
        current_map = calculate_distance_ap(all_preds, all_gts, dist_threshold=d_thresh) # 评测函数
        mAPs.append(current_map)

    final_nusc_map = np.mean(mAPs)
    print(f"\n" + "="*30)
    print(f"Final NuScenes Style mAP: {final_nusc_map:.4f}")
    print("="*30)




if __name__ == '__main__':
    main()
