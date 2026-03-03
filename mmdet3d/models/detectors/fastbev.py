# -*- coding: utf-8 -*-
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmseg.models import build_head as build_seg_head
from mmdet.models.detectors import BaseDetector
from mmdet3d.core import bbox3d2result
from mmseg.ops import resize
from mmcv.runner import get_dist_info, auto_fp16
from ..losses import DiscriminativeLoss

import copy
import onnxruntime

from tools.utils import get_bboxes
import numpy as np
import onnxruntime as ort
import shutil
# import bstnnx

dump_id = None

class SegHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        classes
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes
        embed_dim = 16

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, len(classes) + 1, 1)
        )
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU(True),
            nn.Conv2d(in_channels*2, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, embed_dim, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        if isinstance(x, (list, tuple)):
            x = x[0]
        out = self.classifier(x)
        x_emb = self.embed(x)
        return out, x_emb

@DETECTORS.register_module()
class FastBEV(BaseDetector):
    def __init__(
        self,
        backbone,
        neck,
        neck_fuse,
        neck_3d,
        bbox_head,
        seg_head,
        n_voxels,
        voxel_size,
        bbox_head_2d=None,
        train_cfg=None,
        test_cfg=None,
        train_cfg_2d=None,
        test_cfg_2d=None,
        pretrained=None,
        init_cfg=None,
        extrinsic_noise=0,
        seq_detach=False,
        multi_scale_id=None,
        multi_scale_3d_scaler=None,
        with_cp=False,
        backproject='inplace',
        style='v4',
    ):
        super().__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.neck_3d = build_neck(neck_3d)
        if isinstance(neck_fuse['in_channels'], list):
            for i, (in_channels, out_channels) in enumerate(zip(neck_fuse['in_channels'], neck_fuse['out_channels'])):
                self.add_module(
                    f'neck_fuse_{i}', 
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        else:
            self.neck_fuse = nn.Conv2d(neck_fuse["in_channels"], neck_fuse["out_channels"], 3, 1, 1)
        
        # style
        # v1: fastbev wo/ ms
        # v2: fastbev + img ms
        # v3: fastbev + bev ms
        # v4: fastbev + img/bev ms
        self.style = style
        assert self.style in ['v1', 'v2', 'v3', 'v4'], self.style
        self.multi_scale_id = multi_scale_id
        self.multi_scale_3d_scaler = multi_scale_3d_scaler

        if bbox_head is not None:
            bbox_head.update(train_cfg=train_cfg)
            bbox_head.update(test_cfg=test_cfg)
            self.bbox_head = build_head(bbox_head)
            self.bbox_head.voxel_size = voxel_size
        else:
            self.bbox_head = None

        if seg_head is not None:
            self.seg_head = build_seg_head(seg_head)
        else:
            self.seg_head = None

        if bbox_head_2d is not None:
            bbox_head_2d.update(train_cfg=train_cfg_2d)
            bbox_head_2d.update(test_cfg=test_cfg_2d)
            self.bbox_head_2d = build_head(bbox_head_2d)
        else:
            self.bbox_head_2d = None
            
        classes = ['lane_divider', 'road_divider', 'ped_crossing']
        self.seghead = SegHead(in_channels=256*4, classes=classes)
        
        self.lane_emb_loss = DiscriminativeLoss(embed_dim=16, delta_v=0.5, delta_d=3.0)
        self.lane_seg_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([2.13]))

        self.n_voxels = n_voxels
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # test time extrinsic noise
        self.extrinsic_noise = extrinsic_noise
        if self.extrinsic_noise > 0:
            for i in range(5):
                print("### extrnsic noise: {} ###".format(self.extrinsic_noise))

        # detach adj feature
        self.seq_detach = seq_detach
        self.backproject = backproject
        # checkpoint
        self.with_cp = with_cp

        # init onnx session one time
        if self.test_cfg.get('test_mode', None) in ['test_onnx', 'test_custom']:
            self.backbone_session = self._init_onnx_session(self.test_cfg['backbone_onnx'])
            self.head_session = self._init_onnx_session(self.test_cfg['head_onnx'])

    def _init_onnx_session(self, onnx_path):
        ort.set_default_logger_severity(3)
        bst_so_path = "./bstnnx_cpp2py_export.cpython-37m-x86_64-linux-gnu.so"
        try:
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        except:
            so = ort.SessionOptions()
            so.register_custom_ops_library(bst_so_path)
            session = ort.InferenceSession(onnx_path, so, providers=['CPUExecutionProvider'])
        return session

    def onnx_infer(self, session, input_data):
        
        input_names = [input.name for input in session.get_inputs()]
        input_dict = {}
        for i, input_name in enumerate(input_names):
            input_dict[input_name] = input_data[i]

        output = session.run(None, input_dict)

        return output

    @staticmethod
    def _compute_projection(img_meta, stride, noise=0):
        projection = []
        intrinsic = torch.tensor(img_meta["lidar2img"]["intrinsic"][:3, :3])
        intrinsic[:2] /= stride
        extrinsics = map(torch.tensor, img_meta["lidar2img"]["extrinsic"])
        for extrinsic in extrinsics:
            if noise > 0:
                projection.append(intrinsic @ extrinsic[:3] + noise)
            else:
                projection.append(intrinsic @ extrinsic[:3])
        return torch.stack(projection)

    def extract_feat(self, img, img_metas, mode):
        batch_size = img.shape[0]
        img = img.reshape(
            [-1] + list(img.shape)[2:]
        )  # [1, 6, 3, 928, 1600] -> [6, 3, 928, 1600]
        ###################### 前处理 ######################

        ###################### backbone resnet + fpn ######################
        if mode in ['test', 'train', 'test_pth']:
            # 把所有图像统一处理
            x = self.backbone(
                img
            )  # [N, 64, 64, 176], [N, 128, 32, 88], [N, 256, 16, 44], [N, 512, 8, 22]
            # use for vovnet
            if isinstance(x, dict):
                tmp = []
                for k in x.keys():
                    tmp.append(x[k])
                x = tmp

            # fuse features
            def _inner_forward(x):
                out = self.neck(x)
                return out

            if self.with_cp and x.requires_grad:
                mlvl_feats = cp.checkpoint(_inner_forward, x)
            else:
                # 出来的H，W没变，C全部变成64
                mlvl_feats = _inner_forward(x)
            mlvl_feats = list(mlvl_feats)

            features_2d = None
            if self.bbox_head_2d: # None
                features_2d = mlvl_feats

            if self.multi_scale_id is not None:
                mlvl_feats_ = []
                for msid in self.multi_scale_id:
                    # fpn output fusion
                    if getattr(self, f'neck_fuse_{msid}', None) is not None:
                        fuse_feats = [mlvl_feats[msid]]
                        # 全部resize成[N, 16, 64, 176]
                        for i in range(msid + 1, len(mlvl_feats)):
                            resized_feat = resize(
                                mlvl_feats[i], 
                                size=mlvl_feats[msid].size()[2:], 
                                mode="bilinear",
                                align_corners=False)
                            fuse_feats.append(resized_feat)
                    
                        if len(fuse_feats) > 1:
                            fuse_feats = torch.cat(fuse_feats, dim=1)
                        else:
                            fuse_feats = fuse_feats[0]
                        # [N, 64*4, 64, 176] -> [N, 64, 64, 176]
                        fuse_feats = getattr(self, f'neck_fuse_{msid}')(fuse_feats)
                        mlvl_feats_.append(fuse_feats)
                    else:
                        mlvl_feats_.append(mlvl_feats[msid])
            mlvl_feats = mlvl_feats_  # N, 64, 64, 176 -> 每一张图的2d特征
        elif mode == 'test_onnx':
            all_outputs = []
            for i in range(img.size(0)):
                input_data_single = img[i].unsqueeze(0).cpu().float().numpy()  # 1,3,256,704
                onnx_output = self.onnx_infer(self.backbone_session, [input_data_single])
                output_data_single = torch.from_numpy(onnx_output[0]).to(img[0].device)
                all_outputs.append(output_data_single)
            mlvl_feats = [torch.cat(all_outputs, dim=0)]
        else:
            raise ValueError(f"Unsupported test mode: {mode}")
        ###################### backbone resnet + fpn ######################

        # 2d feat -> bev feat
        mlvl_volumes = []
        for lvl, mlvl_feat in enumerate(mlvl_feats):
            stride_i = math.ceil(img.shape[-1] / mlvl_feat.shape[-1])  # P4 880 / 32 = 27.5
            # [bs*seq*nv, c, h, w] -> [bs, seq*nv, c, h, w]
            mlvl_feat = mlvl_feat.reshape([batch_size, -1] + list(mlvl_feat.shape[1:]))  # [bz,seq*2,64,64,176]
            # 按帧分开，[bs, seq*nv, c, h, w] -> list([bs, nv, c, h, w])
            mlvl_feat_split = torch.split(mlvl_feat, 2, dim=1)

            volume_list = []
            for seq_id in range(len(mlvl_feat_split)):
                volumes = []
                for batch_id, seq_img_meta in enumerate(img_metas):
                    feat_i = mlvl_feat_split[seq_id][batch_id]  # [nv, c, h, w]
                    img_meta = copy.deepcopy(seq_img_meta)
                    img_meta["lidar2img"]["extrinsic"] = img_meta["lidar2img"]["extrinsic"][seq_id*2:(seq_id+1)*2]
                    if isinstance(img_meta["img_shape"], list):
                        img_meta["img_shape"] = img_meta["img_shape"][seq_id*2:(seq_id+1)*2]
                        img_meta["img_shape"] = img_meta["img_shape"][0]
                    height = math.ceil(img_meta["img_shape"][0] / stride_i)
                    width = math.ceil(img_meta["img_shape"][1] / stride_i)

                    # 通过内外参，计算3D空间中某个点在图像上的哪个像素
                    projection = self._compute_projection(
                        img_meta, stride_i, noise=self.extrinsic_noise).to(feat_i.device)
                    if self.style in ['v1', 'v2']:
                        # wo/ bev ms
                        n_voxels, voxel_size = self.n_voxels[0], self.voxel_size[0]
                    else:
                        # v3/v4 bev ms
                        n_voxels, voxel_size = self.n_voxels[lvl], self.voxel_size[lvl]
                    # 生成3D采样点
                    points = get_points(  # [3, vx, vy, vz]  -> 3, 200, 200, 4 (数值 * 坐标)
                        n_voxels=torch.tensor(n_voxels),
                        voxel_size=torch.tensor(voxel_size),
                        origin=torch.tensor(img_meta["lidar2img"]["origin"]),
                    ).to(feat_i.device)

                    if self.backproject == 'inplace':
                        volume = backproject_inplace(
                            feat_i[:, :, :height, :width], points, projection)  # [c, vx, vy, vz]
                    else:
                        volume, valid = backproject_vanilla(
                            feat_i[:, :, :height, :width], points, projection)
                        volume = volume.sum(dim=0)
                        valid = valid.sum(dim=0)
                        volume = volume / valid
                        valid = valid > 0
                        volume[:, ~valid[0]] = 0.0

                    # volumes.append(volume) # [C, X, Y, Z]
                    # ########## change 1 ##############
                    # 避免后续在neck3d中reshape
                    # [C,X,Y,Z] -> [Z,C,X,Y] -> [1, Z*C, X, Y]
                    volumes.append(volume.permute(3, 0, 1, 2).reshape(1, 256, 128, 200))
                    # ########## change 1 ##############
                volume_list.append(torch.stack(volumes))  # list([bs, 1, Z*C, H, W])
    
            mlvl_volumes.append(torch.cat(volume_list, dim=1))  # list([bs, seq, Z*C, vx, vy])
        
        global dump_id
        if dump_id is not None:
            dump_path = f"/root/ziyi/product_e2e_demo-main-fastbev/fastbev/train/fastbev/work_dirs/0209_dump/{dump_id}"
            if not os.path.exists(dump_path):
                os.makedirs(dump_path)
            np.save(f"{dump_path}/bev_feat.npy",
                    mlvl_volumes[0][:,0].detach().cpu().numpy())
            dump_id += 1
            if dump_id > 1000:
                assert False
            
        if self.style in ['v1', 'v2']:
            mlvl_volumes = torch.cat(mlvl_volumes, dim=1)  # [bs, lvl*seq*c, vx, vy, vz]
        else:
            # bev ms: multi-scale bev map (different x/y/z)
            for i in range(len(mlvl_volumes)):
                mlvl_volume = mlvl_volumes[i]
                bs, c, x, y, z = mlvl_volume.shape
                # collapse h, [bs, seq*c, vx, vy, vz] -> [bs, seq*c*vz, vx, vy]
                mlvl_volume = mlvl_volume.permute(0, 2, 3, 4, 1).reshape(bs, x, y, z*c).permute(0, 3, 1, 2)
                
                # different x/y, [bs, seq*c*vz, vx, vy] -> [bs, seq*c*vz, vx', vy']
                if self.multi_scale_3d_scaler == 'pool' and i != (len(mlvl_volumes) - 1):
                    # pooling to bottom level
                    mlvl_volume = F.adaptive_avg_pool2d(mlvl_volume, mlvl_volumes[-1].size()[2:4])
                elif self.multi_scale_3d_scaler == 'upsample' and i != 0:  
                    # upsampling to top level 
                    mlvl_volume = resize(
                        mlvl_volume,
                        mlvl_volumes[0].size()[2:4],
                        mode='bilinear',
                        align_corners=False)
                else:
                    # same x/y
                    pass

                # [bs, seq*c*vz, vx', vy'] -> [bs, seq*c*vz, vx, vy, 1]
                mlvl_volume = mlvl_volume.unsqueeze(-1)
                mlvl_volumes[i] = mlvl_volume
            mlvl_volumes = torch.cat(mlvl_volumes, dim=1)  # [bs, z1*c1+z2*c2+..., vx, vy, 1]

        x = mlvl_volumes
        B, seq, C, X, Y = x.shape
        # [bs, seq, Z*C, X, Y] -> [bs, seq*Z*C, X, Y]
        x = x.reshape(B, seq*C, X, Y)
        
        pred_bev_map, instance_map = self.seghead(x)
        def _inner_forward(x):
            # v1/v2: [bs, lvl*seq*c, vx, vy, vz] -> [bs, c', vx, vy]
            # v3/v4: [bs, z1*c1+z2*c2+..., vx, vy, 1] -> [bs, c', vx, vy]
            out = self.neck_3d(x)
            return out
            
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x, None, features_2d, pred_bev_map, instance_map

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            if img[0].shape == (1, 3, 256, 704):
                return self.onnx_export_2d(img[0], img_metas)  # backbone + neck + neck_fuse
            elif img[0].shape == (1, 256, 128, 200):
            # elif img.shape == (1, 1024, 200, 200):
                return self.onnx_export_3d(img, img_metas)  # neck_3d: 2d -> 3d
            else:
                raise NotImplementedError
        # if kwargs["export_onnx_flag"] == True:
        #     if kwargs["export_2d"]:
        #         return self.onnx_export_2d(img, img_metas)  # backbone + neck + neck_fuse
        #     elif kwargs["export_3d"]:
        #         return self.onnx_export_3d(img, img_metas)  # neck_3d: 2d -> 3d
        #     else:
        #         raise NotImplementedError

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def forward_train(
        self, img, img_metas, gt_bboxes_3d, gt_labels_3d, gt_bev_seg=None, **kwargs
    ):
        feature_bev, valids, features_2d, pred_bev_map, instance_map = self.extract_feat(img, img_metas, "train")
        """
        feature_bev: [(1, 256, 100, 100)]
        valids: (1, 1, 200, 200, 12)
        features_2d: [[6, 64, 232, 400], [6, 64, 116, 200], [6, 64, 58, 100], [6, 64, 29, 50]]
        """
        assert self.bbox_head is not None or self.seg_head is not None

        losses = dict()
        if self.bbox_head is not None:
            x = self.bbox_head(feature_bev)
            loss_det = self.bbox_head.loss(*x, gt_bboxes_3d, gt_labels_3d, img_metas)
            losses.update(loss_det)
            breakpoint()
            # 计算lane loss
            lane_seg_loss = self.lane_seg_loss(pred_bev_map, kwargs['bev_map'])
            lane_emb_loss = self.lane_emb_loss(pred_embed, kwargs['instance_map'])
            losses.update({"lane_seg_loss": lane_seg_loss})
            losses.update({"lane_emb_loss": lane_emb_loss})

        if self.seg_head is not None:
            assert len(gt_bev_seg) == 1
            x_bev = self.seg_head(feature_bev)
            gt_bev = gt_bev_seg[0][None, ...].long()
            loss_seg = self.seg_head.losses(x_bev, gt_bev)
            losses.update(loss_seg)

        if self.bbox_head_2d is not None:
            gt_bboxes = kwargs["gt_bboxes"][0]
            gt_labels = kwargs["gt_labels"][0]
            assert len(kwargs["gt_bboxes"]) == 1 and len(kwargs["gt_labels"]) == 1
            # hack a img_metas_2d
            img_metas_2d = []
            img_info = img_metas[0]["img_info"]
            for idx, info in enumerate(img_info):
                tmp_dict = dict(
                    filename=info["filename"],
                    ori_filename=info["filename"].split("/")[-1],
                    ori_shape=img_metas[0]["ori_shape"],
                    img_shape=img_metas[0]["img_shape"],
                    pad_shape=img_metas[0]["pad_shape"],
                    scale_factor=img_metas[0]["scale_factor"],
                    flip=False,
                    flip_direction=None,
                )
                img_metas_2d.append(tmp_dict)

            rank, world_size = get_dist_info()
            loss_2d = self.bbox_head_2d.forward_train(
                features_2d, img_metas_2d, gt_bboxes, gt_labels
            )
            losses.update(loss_2d)

        return losses

    def forward_test(self, img, img_metas, **kwargs):
        if not self.test_cfg.get('use_tta', False):
            test_mode = self.test_cfg.get('test_mode', False)
            if test_mode in ['test_pth', 'test_onnx', 'test_custom']:
                return self.simple_test(img, img_metas, mode=test_mode)
            else:
                raise ValueError(f"Unsupported test mode: {self.test_cfg.get('test_mode', False)}")
        return self.aug_test(img, img_metas)

    def onnx_export_2d(self, img, img_metas):
        """
        input: 6, 3, 544, 960
        output: 6, 64, 136, 240
        """
        x = self.backbone(img)
        c1, c2, c3, c4 = self.neck(x)
        c2 = resize(
            c2, size=c1.size()[2:], mode="bilinear", align_corners=False
        )  # [6, 64, 232, 400]
        c3 = resize(
            c3, size=c1.size()[2:], mode="bilinear", align_corners=False
        )  # [6, 64, 232, 400]
        c4 = resize(
            c4, size=c1.size()[2:], mode="bilinear", align_corners=False
        )  # [6, 64, 232, 400]
        x = torch.cat([c1, c2, c3, c4], dim=1)
        x = self.neck_fuse_0(x)

        if bool(os.getenv("DEPLOY", False)):
            x = x.permute(0, 2, 3, 1)
            return x

        return x

    def onnx_export_3d(self, x, _):
        # x: [6, 128, 100, 3, 256]
        # if bool(os.getenv("DEPLOY_DEBUG", False)):
        #     x = x.sum(dim=0, keepdim=True)
        #     return [x]
        x_0, x_1, x_2, x_3 = x  # 1, 256, 128, 128
        # x = x[0]
        x = torch.cat((x_0, x_1, x_2, x_3), dim=1)  # 1, 1024, 128, 128
        
        # 车道线检测
        pred_bev_map, instance_map = self.seghead(x)

        if self.style == "v1":
            # x = x.sum(dim=0, keepdim=True)  # [1, 128, 100, 3, 256]
            x = self.neck_3d(x)  # [[1, 256, 100, 50], ]
        elif self.style == "v2":
            x = self.neck_3d(x)  # [6, 256, 100, 50]
            x = [x[0].sum(dim=0, keepdim=True)]  # [1, 256, 100, 50]
        elif self.style == "v3":
            x = self.neck_3d(x)  # [1, 256, 100, 50]
        else:
            raise NotImplementedError

        if self.bbox_head is not None:
            cls_score, bbox_pred, dir_cls_preds = self.bbox_head(x)
            # cls_score = [item.sigmoid() for item in cls_score]
            
        if dir_cls_preds is None:
            x = [cls_score, bbox_pred]
        else:
            x = [cls_score, bbox_pred, dir_cls_preds, pred_bev_map, instance_map]
        # if os.getenv("DEPLOY", False):
        #     if dir_cls_preds is None:
        #         x = [cls_score, bbox_pred]
        #     else:
        #         x = [cls_score, bbox_pred, dir_cls_preds]
        #     return x

        return x

    def simple_test(self, img, img_metas, mode):
        bbox_results = []
        feature_bev, _, features_2d = self.extract_feat(img, img_metas, mode)

        if self.bbox_head is not None:
            if self.test_cfg.get('test_mode', False) == 'test_pth':
                x = self.bbox_head(feature_bev)
            elif self.test_cfg.get('test_mode', False) == 'test_onnx':
                x = feature_bev
            
            bbox_list = self.bbox_head.get_bboxes(*x, img_metas, valid=None)
            bbox_results = [
                bbox3d2result(det_bboxes, det_scores, det_labels)
                for det_bboxes, det_scores, det_labels in bbox_list
            ]

        else:
            bbox_results = [dict()]

        # BEV semantic seg
        if self.seg_head is not None:
            x_bev = self.seg_head(feature_bev)
            bbox_results[0]['bev_seg'] = x_bev

        return bbox_results

    def aug_test(self, imgs, img_metas):
        img_shape_copy = copy.deepcopy(img_metas[0]['img_shape'])
        extrinsic_copy = copy.deepcopy(img_metas[0]['lidar2img']['extrinsic'])

        x_list = []
        img_metas_list = []
        for tta_id in range(2):

            img_metas[0]['img_shape'] = img_shape_copy[24*tta_id:24*(tta_id+1)]
            img_metas[0]['lidar2img']['extrinsic'] = extrinsic_copy[24*tta_id:24*(tta_id+1)]
            img_metas_list.append(img_metas)

            feature_bev, _, _ = self.extract_feat(imgs[:, 24*tta_id:24*(tta_id+1)], img_metas, "test")
            x = self.bbox_head(feature_bev)
            x_list.append(x)

        bbox_list = self.bbox_head.get_tta_bboxes(x_list, img_metas_list, valid=None)
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in [bbox_list]
        ]
        return bbox_results

    def show_results(self, *args, **kwargs):
        pass


@torch.no_grad()
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


def backproject_vanilla(features, points, projection):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [6, 64, 200, 200, 12]
        valid: [6, 1, 200, 200, 12]
    '''
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # [6, 3, 480000] -> [6, 4, 480000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]
    points_2d_3 = torch.bmm(projection, points)  # lidar2img
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    z = points_2d_3[:, 2]  # [6, 480000]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]
    volume = torch.zeros(
        (n_images, n_channels, points.shape[-1]), device=features.device
    ).type_as(features)  # [6, 64, 480000]
    for i in range(n_images):
        volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
    # [6, 64, 480000] -> [6, 64, 200, 200, 12]
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    # [6, 480000] -> [6, 1, 200, 200, 12]
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume, valid


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
