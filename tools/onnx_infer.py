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
from draw_det_utils import det_post_process
import ipdb
from tqdm import tqdm
import cv2
import onnxruntime as ort
import math
from mmdet3d.core import bbox3d2result
import copy
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('save_path', help='fig save path')
    parser.add_argument('--out', help='output result file in pickle format')
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
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
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


def _compute_projection(img_meta, stride):
    projection = []
    intrinsic = torch.tensor(img_meta["lidar2img"]["intrinsic"][:3, :3])
    intrinsic[:2] /= stride
    extrinsics = map(torch.tensor, img_meta["lidar2img"]["extrinsic"])
    for extrinsic in extrinsics:
        projection.append(intrinsic @ extrinsic[:3])
    return torch.stack(projection)


def get_points(n_voxels, voxel_size, origin):
    points = torch.stack(
        torch.meshgrid(
            [
                torch.arange(n_voxels[0]),
                torch.arange(n_voxels[1]),
                torch.arange(n_voxels[2]),
            ],
            indexing='ij'
        )
    )
    new_origin = origin - n_voxels / 2.0 * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points


def backproject_inplace(features, points, projection):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [64, 200, 200, 12]
    '''
    # input: features
    # no change: points
    # change: projection
    # projection = torch.tensor(np.load("/workspace/fastbev/debug_gather/projection.npy")).to(features.device)  # test fix index
    n_images, n_channels, height, width = features.shape  # 6, 64, 64, 176
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]  # 3, 200, 200, 4
    # [3, 200, 200, 4] -> [1, 3, 160000] -> [6, 3, 160000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # [6, 3, 160000] -> [6, 4, 160000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 160000] -> [6, 3, 160000]
    points_2d_3 = torch.bmm(projection, points)  # lidar2img
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 160000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 160000]
    z = points_2d_3[:, 2]  # [6, 160000]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 160000]

    # method2：特征填充，只填充有效特征，重复特征直接覆盖
    volume = torch.zeros(
        (n_channels, points.shape[-1]), device=features.device
    ).type_as(features)  # 64, 160000
    for i in range(n_images):
        volume[:, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]

    volume = volume.view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume



def project_2d_to_3d(mlvl_feat, img_metas, stride):
    n_voxels = [128, 128, 4]
    voxel_size = [0.8, 0.8, 1.0]
    mlvl_volumes = []
    
    # [bs*seq*nv, c, h, w] -> [bs, seq*nv, c, h, w]
    mlvl_feat = mlvl_feat.reshape([1, -1] + list(mlvl_feat.shape[1:]))  # 1,24,64,64,176
    # [bs, seq*nv, c, h, w] -> list([bs, nv, c, h, w])
    mlvl_feat_split = torch.split(mlvl_feat, 2, dim=1)  # bs, 2, 64, 64, 176 * 4(seq)

    volume_list = []
    for seq_id in range(len(mlvl_feat_split)):
        volumes = []
        img_meta = copy.deepcopy(img_metas)
        
        feat_i = mlvl_feat_split[seq_id][0]  # [nv, c, h, w]
        img_meta["lidar2img"]["extrinsic"] = img_meta["lidar2img"]["extrinsic"][seq_id*2:(seq_id+1)*2]
        if isinstance(img_meta["img_shape"], list):
            img_meta["img_shape"] = img_meta["img_shape"][seq_id*2:(seq_id+1)*2]
            img_meta["img_shape"] = img_meta["img_shape"][0]
        height = math.ceil(img_meta["img_shape"][0] / stride)
        width = math.ceil(img_meta["img_shape"][1] / stride)

        # 通过内外参，计算3D空间中某个点在图像上的哪个像素
        projection = _compute_projection(
            img_meta, stride).to(feat_i.device)
        # 生成3D采样点
        points = get_points(  # [3, vx, vy, vz]  -> 3, 200, 200, 4 (数值 * 坐标)
            n_voxels=torch.tensor(n_voxels),
            voxel_size=torch.tensor(voxel_size),
            origin=torch.tensor(img_meta["lidar2img"]["origin"]),
        ).to(feat_i.device)

        volume = backproject_inplace(
            feat_i[:, :, :height, :width], points, projection)  # [c, vx, vy, vz]
        
        volume = volume.permute(3, 0, 1, 2).reshape(1, 256, 128, 128)  # 64, 200, 200, 4 -> 4, 64, 200, 200, -> 1, 256, 200, 200
        volume_list.append(volume)
    return torch.cat(volume_list, dim=0)

def main():
    args = parse_args()


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

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    onnx_2d = ort.InferenceSession("/root/ziyi/product_e2e_demo-main-fastbev/fastbev/train/fastbev/work_dirs/onnx/simplified_export_2d_model.onnx")
    onnx_3d = ort.InferenceSession("/root/ziyi/product_e2e_demo-main-fastbev/fastbev/train/fastbev/work_dirs/onnx/simplified_export_3d_model.onnx")
    
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model.eval()
    for i, data in tqdm(enumerate(data_loader)):
        img = data['img'].data[0][0]
        img_metas = data['img_metas'].data[0][0]
        # 一阶段
        feat_2d = []
        for j in range(8):
            input_dict = {
                "img": img[j:j+1].detach().cpu().numpy()
            }
            feat_2d.append(torch.from_numpy(onnx_2d.run(None, input_dict)[0]))
        feat_2d = torch.cat(feat_2d, dim=0) # [8, 64, 64, 176]
        # 2d -> 3d
        stride = math.ceil(img.shape[-1] / feat_2d.shape[-1])
        bev_feat = project_2d_to_3d(feat_2d, img_metas, stride)
        # 二阶段
        input_dict = {
            'bev_feat0': bev_feat[0:1].detach().cpu().numpy(),
            'bev_feat1': bev_feat[1:2].detach().cpu().numpy(),
            'bev_feat2': bev_feat[2:3].detach().cpu().numpy(),
            'bev_feat3': bev_feat[3:4].detach().cpu().numpy(),
        }
        output = onnx_3d.run(['dir_cls_preds', 'bbox_pred', 'cls_score'], input_dict)
        dir_cls_preds = torch.from_numpy(output[0]).cuda()
        bbox_pred = torch.from_numpy(output[1]).cuda()
        cls_score = torch.from_numpy(output[2]).cuda()
        # 解算box
        bbox_list = model.bbox_head.get_bboxes([cls_score], [bbox_pred], [dir_cls_preds], [img_metas])
        bbox_results = [
            bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in bbox_list
        ]
        # 画图
        front_img = cv2.imread(data['img_metas'].data[0][0]['img_info'][0]['filename'])
        vis_info = data['vis_info']
        vis_info['cam_info']['token'] = i
        vis_info['cam_info']['ego2global_translation'] = [tmp.item() for tmp in vis_info['cam_info']['ego2global_translation']]
        vis_info['cam_info']['ego2global_rotation'] = [tmp.item() for tmp in vis_info['cam_info']['ego2global_rotation']]
        
        # gt_len = len(data['gt_bboxes_3d'].data[0][0].tensor)
        # result[0]['boxes_3d'] = data['gt_bboxes_3d'].data[0][0]
        # result[0]['scores_3d'] = torch.ones(gt_len)
        
        show_imgs = det_post_process(bbox_results[0], front_img, vis_info, None)
        cv2.imwrite(f"{save_path}/{i}.jpg", show_imgs[0])
    


if __name__ == '__main__':
    main()
