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
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

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


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('output_path', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

# def lidar2img(points_lidar, a, b, camrera_info):
#     points_lidar_homogeneous = \
#         np.concatenate([points_lidar,
#                         np.ones((points_lidar.shape[0], 1),
#                                 dtype=points_lidar.dtype)], axis=1)
#     camera2lidar = np.eye(4, dtype=np.float32)
#     camera2lidar[:3, :3] = camrera_info['sensor2lidar_rotation']
#     camera2lidar[:3, 3] = camrera_info['sensor2lidar_translation']
#     lidar2camera = np.linalg.inv(camera2lidar)
#     points_camera_homogeneous = points_lidar_homogeneous @ lidar2camera.T
#     points_camera = points_camera_homogeneous[:, :3]
#     valid = np.ones((points_camera.shape[0]), dtype=bool)
#     valid = np.logical_and(points_camera[:, -1] > 0.5, valid)
#     points_camera = points_camera / points_camera[:, 2:3]
#     camera2img = camrera_info['cam_intrinsic'][0].numpy()
#     points_img = points_camera @ camera2img.T
#     points_img = points_img[:, :2]
#     return points_img, valid


def ego2img(points_ego, info):
    points_ego_homogeneous = \
        np.concatenate([points_ego,
                        np.ones((points_ego.shape[0], 1),
                                dtype=points_ego.dtype)], axis=1)
    camera2lidar = np.eye(4, dtype=np.float32)
    camera2lidar[:3, :3] = info['sensor2lidar_rotation'].numpy()[0]
    camera2lidar[:3, 3] = info['sensor2lidar_translation'].numpy()[0]
    lidar2camera = np.linalg.inv(camera2lidar)
    lidar2ego = np.eye(4, dtype=np.float32)
    lidar2ego[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = info['lidar2ego_translation']
    ego2lidar = np.linalg.inv(lidar2ego)
    ego2camera = lidar2camera @ ego2lidar
    points_camera_homogeneous = (ego2camera @ points_ego_homogeneous.T).T
    points_camera = points_camera_homogeneous[:, :3]
    valid = np.ones((points_camera.shape[0]), dtype=bool)
    valid = np.logical_and(points_camera[:, -1] > 0.5, valid)
    points_camera = points_camera / points_camera[:, 2:3]
    camera2img = info['cam_intrinsic'][0].numpy()
    points_img = points_camera @ camera2img.T
    points_img = points_img[:, :2]
    return points_img, valid


def lidar2img(points_lidar, lidar2camera, intrinsic, info):
    points_lidar_homogeneous = \
        np.concatenate([points_lidar,
                        np.ones((points_lidar.shape[0], 1),
                                dtype=points_lidar.dtype)], axis=1)
    points_camera_homogeneous = points_lidar_homogeneous @ lidar2camera.T
    points_camera = points_camera_homogeneous[:, :3]
    valid = np.ones((points_camera.shape[0]), dtype=bool)
    valid = np.logical_and(points_camera[:, -1] > 0.5, valid)
    points_camera = points_camera / points_camera[:, 2:3]
    points_img = points_camera @ intrinsic.T
    points_img = points_img[:, :2]
    return points_img, valid


def lidar2camera_coords(points_lidar, lidar2camera):
    """将 LiDAR 坐标转换为相机坐标系下的 3D 坐标（不做投影）"""
    points_lidar_homogeneous = np.concatenate(
        [points_lidar, np.ones((points_lidar.shape[0], 1), dtype=points_lidar.dtype)], axis=1)
    points_camera_homogeneous = points_lidar_homogeneous @ lidar2camera.T
    return points_camera_homogeneous[:, :3]


def project_camera_to_img(points_camera, intrinsic):
    """将相机坐标系下的 3D 点投影到图像平面"""
    # 只对有效点做透视除法
    points_img = points_camera.copy()
    valid = points_img[:, 2] > 0.5
    # 避免除零
    z_safe = np.where(points_img[:, 2:3] > 0.5, points_img[:, 2:3], 1.0)
    points_normalized = points_img[:, :3] / z_safe
    points_img_2d = points_normalized @ intrinsic.T
    return points_img_2d[:, :2], valid


def clip_line_at_camera_plane(p1_cam, p2_cam, z_threshold=0.5):
    """
    在相机平面 z=z_threshold 处裁剪线段

    Args:
        p1_cam, p2_cam: 相机坐标系下的两个 3D 端点
        z_threshold: 相机前方阈值

    Returns:
        p1_clip, p2_clip: 裁剪后的两个端点
        should_draw: 是否应该绘制这条线
    """
    z1, z2 = p1_cam[2], p2_cam[2]

    # 两点都在前方，不需要裁剪
    if z1 >= z_threshold and z2 >= z_threshold:
        return p1_cam.copy(), p2_cam.copy(), True

    # 两点都在后方，不绘制
    if z1 < z_threshold and z2 < z_threshold:
        return None, None, False

    # 一个在前一个在后，计算与 z=z_threshold 平面的交点
    t = (z_threshold - z1) / (z2 - z1)

    # 交点坐标
    p_clip = p1_cam + t * (p2_cam - p1_cam)

    if z1 < z_threshold:
        # p1 在后，用交点替换 p1
        return p_clip, p2_cam.copy(), True
    else:
        # p2 在后，用交点替换 p2
        return p1_cam.copy(), p_clip, True


def clip_line_to_img_bounds(p1, p2, width, height):
    """
    将 2D 线段裁剪到图像边界内 (Cohen-Sutherland 简化版)

    Args:
        p1, p2: 两个 2D 端点 (x, y)
        width, height: 图像尺寸

    Returns:
        p1_clip, p2_clip: 裁剪后的端点
        should_draw: 是否应该绘制
    """
    x1, y1 = p1
    x2, y2 = p2

    # 参数化裁剪
    t_min, t_max = 0.0, 1.0
    dx, dy = x2 - x1, y2 - y1

    # 检查四条边界: x=0, x=width, y=0, y=height
    for p, q in [(x1, -dx), (width - x1, dx), (y1, -dy), (height - y1, dy)]:
        if abs(q) < 1e-6:
            # 线段平行于边界
            if p < 0:
                return None, None, False  # 线段完全在边界外
        else:
            t = p / q
            if q < 0:
                t_min = max(t_min, t)
            else:
                t_max = min(t_max, t)
            if t_min > t_max:
                return None, None, False  # 线段完全在边界外

    new_p1 = (int(x1 + t_min * dx), int(y1 + t_min * dy))
    new_p2 = (int(x1 + t_max * dx), int(y1 + t_max * dy))
    return new_p1, new_p2, True


def check_point_in_img(points, height, width):
    valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
    valid = np.logical_and(
        valid, np.logical_and(points[:, 0] < width, points[:, 1] < height))
    return valid


def get_lidar2global(infos):
    lidar2ego = np.eye(4, dtype=np.float32)
    lidar2ego[:3, :3] = Quaternion(infos['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = infos['lidar2ego_translation']
    ego2global = np.eye(4, dtype=np.float32)
    ego2global[:3, :3] = Quaternion(
        infos['ego2global_rotation']).rotation_matrix
    ego2global[:3, 3] = infos['ego2global_translation']
    return ego2global @ lidar2ego

def get_lidar_to_ego_pose(rec, nusc):
    lidar_sample_data = nusc.get('sample_data', rec['data']['LIDAR_TOP'])
    sd_cs = nusc.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"]) # 传感器相对于自车坐标系的安装位置和朝向
    lidar_to_ego = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False)
    return lidar_to_ego


def write_text(img, text, pos):
    color = (255, 139, 0)
    img = cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return img


color_map = {
    0: (255, 255, 0),   # 亮黄 (Yellow) - car
    1: (0, 255, 255),   # 青色 (Cyan)   - 
    2: (255, 0, 255),   # 品红 (Magenta)- 
    3: (0, 255, 0),     # 纯绿 (Green)  - 
    4: (0, 0, 255),     # 纯蓝 (Blue)   - 
    5: (255, 0, 0),     # 纯红 (Red)    - 
    6: (255, 165, 0),   # 橙色 (Orange) - motocycle
    7: (128, 0, 128),   # 深紫 (Purple) - bicycle
    8: (0, 128, 128),   # 深青 (Teal)   - ped
    9: (128, 128, 128)  # 灰色 (Gray)   - traffic cone
}


def det_post_process(pts_bbox, img, lidar2camera, intrinsic, info, ego_cur_vel=0.0, cam_output=None):
    from mmdet3d.core import bbox3d2result
    from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes as LB
    import numpy as np
    draw_boxes_indexes_img_view = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5),
                                   (5, 6), (6, 7), (7, 4), (0, 4), (1, 5),
                                   (2, 6), (3, 7)]
    color_map = {0: (255, 255, 0), 1: (0, 255, 255)}

    corners_ego = pts_bbox['boxes_3d'].corners.numpy()
    corners_ego = corners_ego.reshape(-1, 3)
    pred_flag = np.ones((corners_ego.shape[0] // 8, ), dtype=np.bool)

    imgs = []
    views = ['CAM_FRONT']
    img_height, img_width = img.shape[0], img.shape[1]

    for view in views:
        # 1. 获取相机坐标系下的 3D 角点
        corners_camera = lidar2camera_coords(corners_ego, lidar2camera)
        corners_camera = corners_camera.reshape(-1, 8, 3)

        # 2. 投影到图像平面（用于文字显示位置计算）
        corners_img_flat, _ = project_camera_to_img(corners_ego, intrinsic)
        corners_img_for_text = corners_img_flat.reshape(-1, 8, 2).astype(np.int)

        cam_idx = 0
        for aid in range(corners_camera.shape[0]):
            is_dangerous = False
            if cam_output is not None:
                vx = cam_output['Obstacle_Rel_Vel_X'][cam_idx]
                vy = cam_output['Obstacle_Rel_Vel_Y'][cam_idx]
                vel = np.sqrt(vx**2 + vy**2)
                pos = ((corners_img_for_text[aid][0] + corners_img_for_text[aid][1]) / 2.0).astype(np.int64)
                ttc = cam_output['Obstacle_TTC'][cam_idx]
                # 只是为了可视化临时取3.0阈值，后续根据实际需求调整
                if ttc < 3.0:
                    is_dangerous = True
                img = write_text(img, f"{vel:.1f}m/s", pos + np.array([-20, -80]))
                cam_idx += 1

            # color = color_map[pts_bbox['labels_3d'][aid].item()]
            # 3. 绘制每条边，使用正确的裁剪逻辑
            for index in draw_boxes_indexes_img_view:
                color = color_map[int(pred_flag[aid])] if not is_dangerous else (0, 0, 255)

                p1_cam = corners_camera[aid, index[0]]
                p2_cam = corners_camera[aid, index[1]]

                # 在相机平面处裁剪
                p1_clip, p2_clip, should_draw = clip_line_at_camera_plane(p1_cam, p2_cam)

                if should_draw:
                    # 投影到图像
                    pts_cam = np.array([p1_clip, p2_clip])
                    pts_img, _ = project_camera_to_img(pts_cam, intrinsic)

                    p1_img = pts_img[0].astype(int)
                    p2_img = pts_img[1].astype(int)

                    # 裁剪到图像边界
                    p1_img, p2_img, should_draw = clip_line_to_img_bounds(
                        p1_img, p2_img, img_width, img_height)

                    if should_draw:
                        cv2.line(img, tuple(p1_img), tuple(p2_img), color=color, thickness=4)

        imgs.append(img)
    return imgs
 