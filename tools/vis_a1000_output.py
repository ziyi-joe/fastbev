"""
只支持可视化nuscenes mini所选取的24张图片
"""
import cv2
import os
import numpy as np
import torch
import onnxruntime
import math
from PIL import Image
from tools.utils import get_bboxes
from mmdet3d.core import bbox3d2result

from mmcv import Config
from mmdet3d.datasets import build_dataloader, build_dataset

def create_data():
    cfg = Config.fromfile("configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py")
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=8,
        dist=False,
        shuffle=False)
    dataset = data_loader.dataset

    return dataset

def get_img_data_from_origin_path(img_path):
    img_data = []
    for filename in sorted(os.listdir(img_path)):
        img_file = os.path.join(img_path, filename)
        if os.path.isfile(img_file):
            # img = Image.open(img_file)
            img = cv2.imread(img_file)
            # img_data.append(img)
            img_data.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # cv2 -> BGR to RGB; PIL -> RGB to BGR
    return img_data

def get_img_data_from_path(img_path):
    img_data = []
    for filename in sorted(os.listdir(img_path)):
        img_file = os.path.join(img_path, filename)
        if os.path.isfile(img_file):
            img = Image.open(img_file)
            img = img.resize((704, 396))  # Resize
            img = img.crop((0, 70, 704, 326))  # Crop from (70, 0) to (326, 704)
            img_data.append(np.asarray(img))
    return img_data

def normalize_img(img):
    img = img.copy().astype(np.float32)
    img[0] = (img[0] - 123.675) * 0.01708984375
    img[1] = (img[1] - 116.28) * 0.017578125
    img[2] = (img[2] - 103.53) * 0.017333984375
    
    return img

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
    # change: 
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

def main():
    ############################################ 前处理 ############################################
    img_path = "./bst_deploy/img_input"
    imgs = get_img_data_from_origin_path(img_path)
    # imgs = get_img_data_from_path(img_path)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    imgs = [torch.from_numpy(normalize_img(img.transpose(2, 0, 1))).unsqueeze(0).cuda() for img in imgs]
    # imgs = [torch.from_numpy(normalize_img(img.transpose(2, 0, 1), mean, std)).unsqueeze(0).cuda() for img in imgs]
    imgs = torch.cat(imgs, dim=0)
    ############################################ 前处理 ############################################

    ############################################ backbone resnet + fpn ############################################
    onnx_model_path = 'your/2d_quant.onnx'  # TODO
    providers = ['CUDAExecutionProvider']
    sess = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
    # model = bstnnx.load(onnx_model_path)
    # input_name = sess.get_inputs()[0].name
    all_outputs = []
    for i in range(imgs.size(0)):
        input_data_single = imgs[i].unsqueeze(0).cpu().float()  # 1,3,256,704
        # onnx_output = bstnnx.backend.CPUBackend.run_model(model,
        #                                                   input_data_single.numpy(),
        #                                                   custom_op_lib=bstnnx.backend.custom_op.get_custom_op_lib_path())
        onnx_input = {'prep_input_input.1': input_data_single.numpy()}  # prep_input_input.1
        onnx_output = sess.run(None, onnx_input)
        output_data_single = torch.from_numpy(onnx_output[0]).to(imgs[0].device)
        all_outputs.append(output_data_single)

    mlvl_feats = [torch.cat(all_outputs, dim=0)]
    mlvl_feat_split = torch.split(mlvl_feats[0], 6, dim=0)
    ############################################ backbone resnet + fpn ############################################

    ############################################ 2d -> 3d ############################################
    # v3 bev ms
    mlvl_volumes = []
    for lvl, mlvl_feat in enumerate(mlvl_feats):  # 24,64,64,176
        stride_i = math.ceil(imgs.shape[-1] / mlvl_feat.shape[-1])  # P4 880 / 32 = 27.5
        # [bs*seq*nv, c, h, w] -> [bs, seq*nv, c, h, w]
        mlvl_feat = mlvl_feat.reshape([1, -1] + list(mlvl_feat.shape[1:]))  # 1,24,64,64,176
        # [bs, seq*nv, c, h, w] -> list([bs, nv, c, h, w])
        mlvl_feat_split = torch.split(mlvl_feat, 6, dim=1)  # 1, 6, 64, 64, 176 * 4(seq)
        
        volume_list = []
        
        projection = torch.tensor(np.load("./bst_deploy/index/projection.npy")).cuda()  # fix index
        points = torch.tensor(np.load("./bst_deploy/index/points.npy")).cuda()

        height, width = 64, 176
        for seq_id in range(len(mlvl_feat_split)):
            volumes = []
            for batch_id in range(1):
                feat_i = mlvl_feat_split[seq_id][batch_id]  # [nv, c, h, w]

                volume = backproject_inplace(feat_i[:, :, :height, :width], points, projection)  # [c, vx, vy, vz]

                volumes.append(volume.permute(3, 0, 1, 2).reshape(1, 256, 200, 200))  # 64, 200, 200, 4 (点云特征)
            volume_list.append(volumes)  # list([bs, c, vx, vy, vz])  64, 200, 200, 4 * 4 (single point feature * seq)

        # mlvl_volumes.append(torch.cat(volume_list, dim=1))  # list([bs, seq*c, vx, vy, vz])
    
    # mlvl_volumes = torch.cat(mlvl_volumes, dim=1)  # [bs, lvl*seq*c, vx, vy, vz]
    ############################################ 2d -> 3d ############################################

    ############################################ 3d neck ############################################
    # x = mlvl_volumes
    # N, C*T, X, Y, Z -> N, X, Y, Z, C -> N, X, Y, Z*C*T -> N, Z*C*T, X, Y
    # N, C, X, Y, Z = mlvl_volumes.shape
    # x = mlvl_volumes.permute(0, 2, 3, 4, 1).reshape(N, X, Y, Z*C).permute(0, 3, 1, 2)
    onnx_model_path = 'your/3d_quant.onnx'  # TODO
    # model = bstnnx.load(onnx_model_path)
    # input_data_single = x.cpu().float().numpy()  # 1,3,256,704
    input_0 = volume_list[0][0].cpu().float().numpy()
    input_1 = volume_list[1][0].cpu().float().numpy()
    input_2 = volume_list[2][0].cpu().float().numpy()
    input_3 = volume_list[3][0].cpu().float().numpy()

    providers = ['CUDAExecutionProvider']
    sess = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
    # input_name = sess.get_inputs()[0].name
    onnx_input = {'0': input_0, '1': input_1, '2': input_2, '3': input_3}
    # onnx_input = {'input.1': input_data_single}
    onnx_outputs = sess.run(None, onnx_input)
    # onnx_outputs = bstnnx.backend.CPUBackend.run_model(model,
    #                                                   input_data_single,
    #                                                   custom_op_lib=bstnnx.backend.custom_op.get_custom_op_lib_path())
    out1 = torch.from_numpy(onnx_outputs[0]).to(imgs[0].device)
    out2 = torch.from_numpy(onnx_outputs[1]).to(imgs[0].device)
    out3 = torch.from_numpy(onnx_outputs[2]).to(imgs[0].device)
    x = ([out1], [out2], [out3])

    # vis A1000B0 output
    # a1000_last = np.fromfile("./bst_deploy/A1000B0_output/A1000B0_3d_output.bin", dtype=np.int8)
    # out1 = (a1000_last[:80*100*100]*0.25).reshape(1, 100, 100, 80).transpose(0, 3, 1, 2)
    # out2 = (a1000_last[80*100*100:(80*100*100*2)]*0.03125).reshape(1, 100, 100, 80)[:, :, :, :72].transpose(0, 3, 1, 2)
    # out3 = (a1000_last[80*100*100*2:]*0.125).reshape(1, 100, 100, 16).transpose(0, 3, 1, 2)
    # out1 = torch.from_numpy(out1).to(imgs[0].device).float()
    # out2 = torch.from_numpy(out2).to(imgs[0].device).float()
    # out3 = torch.from_numpy(out3).to(imgs[0].device).float()
    bbox_list = get_bboxes(*x, input_metas=None, valid=None)
    # bbox_list_2 = self.bbox_head.get_bboxes(*x, img_metas, valid=None)
    bbox_results = [
        bbox3d2result(det_bboxes, det_scores, det_labels)
        for det_bboxes, det_scores, det_labels in bbox_list
    ]
    # return bbox_results
    ############################################ 3d neck ############################################

    ############################################ show result ############################################
    dataset = create_data()
    if True:
        dataset.show(bbox_results, f'./show_dir_debug/test_vis/no_fix')
    ############################################ show result ############################################

if __name__ == '__main__':
    main()
