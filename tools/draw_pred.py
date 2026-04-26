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
import matplotlib.pyplot as plt

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
import numpy as np
from scipy.ndimage import label
from sklearn.cluster import DBSCAN
from onnx_infer import get_ego_transforms
from collections import deque
from pyquaternion import Quaternion
from sklearn.decomposition import PCA

def fit_line_segment_pca(points):
    # 1. PCA 拟合直线方向
    pca = PCA(n_components=1)
    pca.fit(points)
    
    # 2. 得到中心点和方向向量
    center = pca.mean_
    direction = pca.components_[0]
    
    # 3. 将所有点投影到该方向上
    # 投影公式：p' = center + ( (p - center) · direction ) * direction
    projections = []
    for p in points:
        dist = np.dot(p - center, direction)
        projections.append(dist)
    
    # 4. 找到投影的最远两端
    min_dist, max_dist = min(projections), max(projections)
    endpoint1 = center + min_dist * direction
    endpoint2 = center + max_dist * direction
    
    return np.array([endpoint1, endpoint2])

def lane_postprocess(bev_map, instance_map, voxel_size=(0.5, 0.4), pc_range=(0, -25.6)):
    """
    车道线后处理：从 bev_map 和 instance_map 提取向量化车道线

    Args:
        bev_map: [1, 4, 200, 128] 语义分割图
            C维度: 0=空白, 1=divider(车道线), 2=crossing(人行横道), 3=curb(路沿)
        instance_map: [1, 16, 200, 128] 实例嵌入图
            同一instance的像素在channel维度距离较近
        voxel_size: (dx, dy) 每个voxel的物理尺寸(米)
        pc_range: (x_min, y_min) 点云范围起点

    Returns:
        list of dict: 每个dict包含:
            - 'category': 类别名称 ('divider', 'crossing', 'curb')
            - 'class_id': 类别ID (1, 2, 3)
            - 'points': Nx2 array, 自车坐标系下的(x, y)坐标
    """
    if isinstance(bev_map, torch.Tensor):
        bev_map = bev_map[0].cpu().numpy()  # [4, 200, 128]
    if isinstance(instance_map, torch.Tensor):
        instance_map = instance_map[0].cpu().numpy()  # [16, 200, 128]

    # 获取语义分割结果 (argmax)
    semantic_map = np.argmax(bev_map, axis=0)  # [200, 128]

    # 类别映射
    category_map = {0: 'divider', 1: 'crossing'}

    # 存储所有车道线向量
    lane_vectors = []

    # 对每个类别分别处理
    cls_thr = [0.5, 0.45]
    for class_id, category_name in category_map.items():
        # 获取当前类别的mask
        # class_mask = (semantic_map == class_id)  # [200, 128]
        class_mask = (bev_map[class_id] > cls_thr[class_id])

        if not np.any(class_mask):
            continue

        # 获取该类别像素的instance embedding
        ys, xs = np.where(class_mask)
        if len(xs) < 3:  # 像素太少，跳过
            continue

        # 提取这些像素的embedding [N, 16]
        embeddings = instance_map[:, ys, xs].T  # [N, 16]

        # 使用DBSCAN聚类实例
        # 同一instance的像素embedding距离较近
        clustering = DBSCAN(eps=0.6, min_samples=5, metric='euclidean')
        labels = clustering.fit_predict(embeddings)

        # 对每个实例进行处理
        unique_labels = set(labels) - {-1}  # 排除噪声点

        for inst_label in unique_labels:
            # 获取该实例的像素坐标
            inst_mask = (labels == inst_label)
            inst_ys = ys[inst_mask]
            inst_xs = xs[inst_mask]

            if len(inst_xs) < 3:  # 像素太少
                continue

            # 将像素坐标转换为自车坐标系
            # BEV图: 行对应x轴(前向), 列对应y轴(横向)
            # 行索引0对应x_min(近处), 行索引199对应x_max(远处)
            # 列索引0对应y_min(左侧), 列索引127对应y_max(右侧)

            # 方法1: 使用拟合+排序来得到有序点集
            points = pixel_to_ego_coords(inst_xs, inst_ys, voxel_size, pc_range)
            if len(points) < 5:
                continue
            if class_id == 0:
                ordered_points = fit_line_segment_pca(points)
            else:
                # 对点进行排序，使其成为有序向量
                ordered_points = order_points_to_vector(points)

            lane_vectors.append({
                'category': category_name,
                'class_id': class_id,
                'points': ordered_points
            })

    return lane_vectors


def pixel_to_ego_coords(xs, ys, voxel_size=(0.5, 0.4), pc_range=(0, -25.6)):
    """
    将像素坐标转换为自车坐标系

    Args:
        xs: 像素的列坐标 (y方向)
        ys: 像素的行坐标 (x方向)
        voxel_size: (dx, dy) 物理尺寸
        pc_range: (x_min, y_min) 点云范围

    Returns:
        points: Nx2 array, (x, y) 坐标
    """
    dx, dy = voxel_size
    x_min, y_min = pc_range

    # BEV图的行对应x轴(前向)，列对应y轴(横向)
    # x = x_min + row * dx
    # y = y_min + col * dy
    x_coords = x_min + ys * dx
    y_coords = y_min + xs * dy

    points = np.stack([x_coords, y_coords], axis=1)
    return points


def order_points_to_vector(points):
    """
    将无序点集排序为有序向量

    Args:
        points: Nx2 array, 无序的(x, y)坐标

    Returns:
        ordered_points: Mx2 array, 有序的(x, y)坐标
    """
    if len(points) < 3:
        return points

    # 使用基于距离的方法排序
    # 方法: 找到距离最远的两个点作为端点，然后沿着主轴方向排序

    # 1. 计算主方向 (PCA)
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    main_dir = eigenvectors[:, np.argmax(eigenvalues)]

    # 2. 沿主方向投影并排序
    projections = centered @ main_dir
    sorted_indices = np.argsort(projections)
    ordered_points = points[sorted_indices]

    # 3. (可选) 对点进行稀疏化，减少点数
    # 使用 Douglas-Peucker 算法或简单采样
    ordered_points = simplify_points(ordered_points, min_distance=0.3)

    return ordered_points


def simplify_points(points, min_distance=0.3):
    """
    简化点集，保持最小距离

    Args:
        points: Nx2 array
        min_distance: 点之间的最小距离(米)

    Returns:
        simplified_points: Mx2 array
    """
    if len(points) <= 2:
        return points

    simplified = [points[0]]
    for i in range(1, len(points)):
        dist = np.linalg.norm(points[i] - simplified[-1])
        if dist >= min_distance:
            simplified.append(points[i])

    # 确保最后一个点被包含
    if np.linalg.norm(points[-1] - simplified[-1]) > 0.1:
        simplified.append(points[-1])

    return np.array(simplified)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
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
    model = MMDataParallel(model, device_ids=[0])
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    
    model.eval()
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i, data in tqdm(enumerate(data_loader)):
        if i > 195:
            break
        # breakpoint()
        intrinsic0 = data['intrinsic'][0].cpu().numpy()[0].astype(np.float32)
        intrinsic1 = data['intrinsic'][1].cpu().numpy()[0].astype(np.float32)

        viewpad0 = np.eye(4, dtype=np.float32)
        viewpad1 = np.eye(4, dtype=np.float32)
        viewpad0[:intrinsic0.shape[0], :intrinsic0.shape[1]] = intrinsic0
        viewpad1[:intrinsic1.shape[0], :intrinsic1.shape[1]] = intrinsic1
        data['ego2cam'] = [ego2cam.cpu().numpy()[0].astype(np.float32) for ego2cam in data['ego2cam']]
        
        with torch.no_grad():
            result, bev_map, instance_map = model(return_loss=False, rescale=True, **data)
            bev_map = bev_map[0].sigmoid().cpu().numpy()
            instance_map = instance_map[0].cpu().numpy()
        front_img = cv2.imread(data['img_metas'].data[0][0]['img_info'][0]['filename'])
        vis_info = data['vis_info']
        final_output_list = []
        cam_output = {}
        for bbox in result[0]['boxes_3d'].tensor:
            final_output_list.append(dict(
                Obstacle_Rel_Vel_X=bbox[7],
                Obstacle_Rel_Vel_Y=bbox[8],
                Obstacle_TTC = 99.0
            ))
        keys = ['Obstacle_Rel_Vel_X', 'Obstacle_Rel_Vel_Y', 'Obstacle_TTC']
        for k in keys:
            cam_output[k] = [item[k] for item in final_output_list]

        # 可视化gt
        # gt_len = len(data['gt_bboxes_3d'].data[0][0].tensor)
        # result[0]['boxes_3d'] = data['gt_bboxes_3d'].data[0][0]
        # result[0]['scores_3d'] = torch.ones(gt_len)
        # final_output_list = []
        # cam_output = {}
        # for bbox in data['gt_bboxes_3d'].data[0][0].tensor:
        #     final_output_list.append(dict(
        #         Obstacle_Rel_Vel_X=bbox[7],
        #         Obstacle_Rel_Vel_Y=bbox[8],
        #         Obstacle_TTC = 99.0
        #     ))
        # keys = ['Obstacle_Rel_Vel_X', 'Obstacle_Rel_Vel_Y', 'Obstacle_TTC']
        # for k in keys:
        #     cam_output[k] = [item[k] for item in final_output_list]
        # bev_map = data['bev_map'][0].cpu().numpy()
        
        # 车道线后处理
        lane_vectors = lane_postprocess(bev_map, instance_map)
        
        ego2cam = data['ego2cam'][0]
        intrinsic0 = vis_info['cam_intrinsic'].cpu().numpy()[0].astype(np.float32)
        bev_clr = {
            "divider": [0, 255, 0],
            "crossing": [255, 0, 0],
            "curb": [0, 0, 255]
        }
        # 投影到前视图上可视化
        for vector in lane_vectors:
            if vector['category'] == 'curb':
                continue
            points = vector['points']
            points_ego_homogeneous = \
                np.concatenate([points, 0.0 * np.ones((points.shape[0], 1), dtype=points.dtype),
                                np.ones((points.shape[0], 1), dtype=points.dtype)
                                ], axis=1)
            points_camera_3d = (points_ego_homogeneous @ ego2cam.T)[:, :3]
            # 过滤掉在相机后方的点（z <= 0）
            depths = points_camera_3d[:, 2]
            if np.any(depths <= 0):
                continue
            points_camera = points_camera_3d / depths[:, None]
            points_img = (points_camera @ intrinsic0.T)[:, :2]
            h, w = front_img.shape[:2]
            if vector['category'] == 'divider':
                p0 = points_img[0].astype(np.int64)
                p1 = points_img[1].astype(np.int64)
                ret, p0c, p1c = cv2.clipLine((0, 0, w, h), tuple(p0), tuple(p1))
                if ret:
                    cv2.line(front_img, p0c, p1c, color=bev_clr[vector['category']], thickness=5)
            else:
                for point in points_img:
                    pt = point.astype(np.int64)
                    if 0 <= pt[0] < w and 0 <= pt[1] < h:
                        cv2.circle(front_img, tuple(pt), radius=5, color=bev_clr[vector['category']], thickness=-1)

        if len(result[0]['boxes_3d'].tensor) > 0:
            show_imgs = det_post_process(result[0], front_img, ego2cam, intrinsic0, vis_info, 0.0, cam_output)[0]
        else:
            show_imgs = front_img
        # cv2.imwrite(f"{save_path}/{i}.jpg", show_imgs)
        # continue
        
        
        # 可视化bev
        fig, ax = plt.subplots(1, 2, figsize=(24, 12))
        bev_vis = np.ones((200, 128, 3)) * 255
        lane_clr = [255, 0, 0]
        road_clr = [0, 255, 0]
        crossing_clr = [0, 255, 255]
        # bev_map = data['bev_map'][0].cpu().numpy()
        # bev_vis[bev_map[2]>0.4] = tuple([int(0.6*x) for x in lane_clr]) # 红色
        bev_vis[bev_map[0]>0.45] = tuple([int(0.6*x) for x in road_clr]) # 绿色
        bev_vis[bev_map[1]>0.5] = tuple([int(0.6*x) for x in crossing_clr]) # 青色
        bev_vis = np.flip(bev_vis, 0)
        bev_vis = np.flip(bev_vis, 1)
        ax[0].imshow(bev_vis, extent=[-64, 64, 0, 200])
        show_imgs = cv2.cvtColor(show_imgs, cv2.COLOR_BGR2RGB)
        ax[1].imshow(show_imgs)

        # 绘制ego
        mot_length=3
        mot_width=2
        ego_polygon = np.array(
            [[mot_length/2, -mot_length/2, -mot_length/2, mot_length/2, mot_length/2],
             [mot_width/2, mot_width/2, -mot_width/2, -mot_width/2, mot_width/2]
            ]
        ).T
        ego_polygon_vis = np.flip(ego_polygon, axis=1)
        ego_polygon_vis[..., 0] *= -1
        ax[0].plot(ego_polygon_vis[:,0]/0.5, ego_polygon_vis[:,1]/0.4, color='red')
        # 绘制det
        for mot_id in range(len(result[0]['boxes_3d'])):
            mot_polygon = result[0]['boxes_3d'].corners[mot_id, :, :2][::2].cpu().numpy()
            mot_polygon = np.array([mot_polygon[0], mot_polygon[1], mot_polygon[3],
                                    mot_polygon[2], mot_polygon[0]])
            mot_polygon[..., 0] /= 0.5
            mot_polygon[..., 1] /= 0.4
            mot_polygon_vis = np.flip(mot_polygon, axis=1)
            mot_polygon_vis[..., 0] *= -1
            ax[0].plot(mot_polygon_vis[:,0], mot_polygon_vis[:,1], color="blue", linestyle='-')
        ax[0].grid(True, alpha=0.3)
        ax[0].set_xlim(-64,64)
        ax[0].set_ylim(-10,200)
        ax[0].axis('equal')
        ax[1].grid(False)

        fig.savefig(f"{save_path}/{i}.jpg")
        plt.close(fig)
        


if __name__ == '__main__':
    main()