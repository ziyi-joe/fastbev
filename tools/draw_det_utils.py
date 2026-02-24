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

def lidar2img(points_lidar, camrera_info):
    points_lidar_homogeneous = \
        np.concatenate([points_lidar,
                        np.ones((points_lidar.shape[0], 1),
                                dtype=points_lidar.dtype)], axis=1)
    camera2lidar = np.eye(4, dtype=np.float32)
    camera2lidar[:3, :3] = camrera_info['sensor2lidar_rotation']
    camera2lidar[:3, 3] = camrera_info['sensor2lidar_translation']
    lidar2camera = np.linalg.inv(camera2lidar)
    points_camera_homogeneous = points_lidar_homogeneous @ lidar2camera.T
    points_camera = points_camera_homogeneous[:, :3]
    valid = np.ones((points_camera.shape[0]), dtype=bool)
    valid = np.logical_and(points_camera[:, -1] > 0.5, valid)
    points_camera = points_camera / points_camera[:, 2:3]
    camera2img = camrera_info['cam_intrinsic'][0].numpy()
    points_img = points_camera @ camera2img.T
    points_img = points_img[:, :2]
    return points_img, valid


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


def lidar2img(points_lidar, info):
    points_lidar_homogeneous = \
        np.concatenate([points_lidar,
                        np.ones((points_lidar.shape[0], 1),
                                dtype=points_lidar.dtype)], axis=1)
    camera2lidar = np.eye(4, dtype=np.float32)
    camera2lidar[:3, :3] = info['sensor2lidar_rotation'].numpy()[0]
    camera2lidar[:3, 3] = info['sensor2lidar_translation'].numpy()[0]
    lidar2camera = np.linalg.inv(camera2lidar)
    lidar2ego = np.eye(4, dtype=np.float32)
    lidar2ego[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = info['lidar2ego_translation']
    ego2lidar = np.linalg.inv(lidar2ego)
    ego2camera = lidar2camera @ ego2lidar
    points_camera_homogeneous = (lidar2camera @ points_lidar_homogeneous.T).T
    points_camera = points_camera_homogeneous[:, :3]
    valid = np.ones((points_camera.shape[0]), dtype=bool)
    valid = np.logical_and(points_camera[:, -1] > 0.5, valid)
    points_camera = points_camera / points_camera[:, 2:3]
    camera2img = info['cam_intrinsic'][0].numpy()
    points_img = points_camera @ camera2img.T
    points_img = points_img[:, :2]
    return points_img, valid


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

def det_post_process(pts_bbox, img, infos, ego_cur_vel=0.0, cam_output=None):
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
    for view in views:
        # corners_img, valid = lidar2img(corners_lidar, infos)
        corners_img, valid = lidar2img(corners_ego, infos)
        valid = np.logical_and(
            valid,
            check_point_in_img(corners_img, img.shape[0], img.shape[1]))
        valid = valid.reshape(-1, 8)
        # print(valid)
        corners_img = corners_img.reshape(-1, 8, 2).astype(np.int)
        cam_idx = 0
        for aid in range(valid.shape[0]):
            if pts_bbox['scores_3d'][aid] < 0.4:
                continue
            is_dangerous = False
            if cam_output is not None:
                vx = cam_output['Obstacle_Rel_Vel_X'][cam_idx] + ego_cur_vel
                vy = cam_output['Obstacle_Rel_Vel_Y'][cam_idx]
                vel = np.sqrt(vx**2 + vy**2)
                pos = ((corners_img[aid][0] + corners_img[aid][1]) / 2.0).astype(np.int64)
                ttc = cam_output['Obstacle_TTC'][cam_idx]
                # 只是为了可视化临时取3.0阈值，后续根据实际需求调整
                if ttc < 3.0:
                    is_dangerous = True
                img = write_text(img, f"{vel:.1f}m/s", pos + np.array([-20, -80]))
                cam_idx += 1
                
            for index in draw_boxes_indexes_img_view:
                color = color_map[int(pred_flag[aid])] if not is_dangerous else (0, 0, 255)
                if valid[aid, index[0]] and valid[aid, index[1]]:
                    cv2.line(
                        img,
                        corners_img[aid, index[0]],
                        corners_img[aid, index[1]],
                        color=color,
                        thickness=2)

        imgs.append(img)
    return imgs
 