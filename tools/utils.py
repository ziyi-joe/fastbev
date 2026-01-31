import numpy as np
import torch
from mmdet3d.ops.iou3d import iou3d_cuda

def get_bboxes(cls_scores,
                bbox_preds,
                dir_cls_preds,
                input_metas=None,
                valid=None,
                cfg=None,
                rescale=False):
    """Get bboxes of anchor head.

    Args:
        cls_scores (list[torch.Tensor]): Multi-level class scores.
        bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
        dir_cls_preds (list[torch.Tensor]): Multi-level direction
            class predictions.
        input_metas (list[dict]): Contain pcd and img's meta info.
        cfg (None | :obj:`ConfigDict`): Training or testing config.
        rescale (list[torch.Tensor]): Whether th rescale bbox.

    Returns:
        list[tuple]: Prediction resultes of batches.
    """
    assert len(cls_scores) == len(bbox_preds)
    assert len(cls_scores) == len(dir_cls_preds)
    num_levels = len(cls_scores)
    featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
    device = cls_scores[0].device
    mlvl_anchors = [torch.from_numpy(np.load('/workspace/fastbev/debug_gather/other/mlvl_anchors.npy')).to(device)]

    result_list = []
    cls_score_list = [cls_scores[0][0].detach()]
    bbox_pred_list = [bbox_preds[0][0].detach()]
    dir_cls_pred_list = [dir_cls_preds[0][0].detach()]
    proposals = get_bboxes_single(cls_score_list, bbox_pred_list,
                                        dir_cls_pred_list, mlvl_anchors,
                                        None, cfg, rescale)
    result_list.append(proposals)
    # for img_id in range(len(input_metas)):
    #     cls_score_list = [
    #         cls_scores[i][img_id].detach() for i in range(num_levels)
    #     ]
    #     bbox_pred_list = [
    #         bbox_preds[i][img_id].detach() for i in range(num_levels)
    #     ]
    #     dir_cls_pred_list = [
    #         dir_cls_preds[i][img_id].detach() for i in range(num_levels)
    #     ]

    #     input_meta = input_metas[img_id]
    #     proposals = get_bboxes_single(cls_score_list, bbox_pred_list,
    #                                         dir_cls_pred_list, mlvl_anchors,
    #                                         input_meta, cfg, rescale)
    #     result_list.append(proposals)
    return result_list

def get_bboxes_single(  cls_scores,
                        bbox_preds,
                        dir_cls_preds,
                        mlvl_anchors,
                        input_meta,
                        cfg=None,
                        rescale=False):
    """Get bboxes of single branch.

    Args:
        cls_scores (torch.Tensor): Class score in single batch.
        bbox_preds (torch.Tensor): Bbox prediction in single batch.
        dir_cls_preds (torch.Tensor): Predictions of direction class
            in single batch.
        mlvl_anchors (List[torch.Tensor]): Multi-level anchors
            in single batch.
        input_meta (list[dict]): Contain pcd and img's meta info.
        cfg (None | :obj:`ConfigDict`): Training or testing config.
        rescale (list[torch.Tensor]): whether th rescale bbox.

    Returns:
        tuple: Contain predictions of single batch.

            - bboxes (:obj:`BaseInstance3DBoxes`): Predicted 3d bboxes.
            - scores (torch.Tensor): Class score of each bbox.
            - labels (torch.Tensor): Label of each bbox.
    """
    # cfg = self.test_cfg if cfg is None else cfg
    assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
    mlvl_bboxes = []
    mlvl_scores = []
    mlvl_dir_scores = []
    for cls_score, bbox_pred, dir_cls_pred, anchors in zip(
            cls_scores, bbox_preds, dir_cls_preds, mlvl_anchors):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        assert cls_score.size()[-2:] == dir_cls_pred.size()[-2:]
        dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)  # 16,100,100,-> 100,100,16 -> 80000,2
        dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]  # 80000

        cls_score = cls_score.permute(1, 2, 0).reshape(-1, 10)  # 80,100,100, -> 100,100,80, -> 80000,10 -> sigmoid

        scores = cls_score.sigmoid()

        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 9)  # 72,100,100, -> 100,100,72 -> 80000,9

        nms_pre = 1000
        if nms_pre > 0 and scores.shape[0] > nms_pre:
            max_scores, _ = scores.max(dim=1)# 80000,10 -> 80000
            _, topk_inds = max_scores.topk(nms_pre)  # 取前1000个值的索引
            anchors = anchors[topk_inds, :]  # 80000,9 -> 1000,9
            bbox_pred = bbox_pred[topk_inds, :]  # 80000,9 -> 1000,9
            scores = scores[topk_inds, :]  # 80000,10 -> 1000,10
            dir_cls_score = dir_cls_score[topk_inds]  # 80000 -> 1000

        bboxes = decode(anchors, bbox_pred)
        mlvl_bboxes.append(bboxes)
        mlvl_scores.append(scores)
        mlvl_dir_scores.append(dir_cls_score)

    mlvl_bboxes = torch.cat(mlvl_bboxes)
    mlvl_scores = torch.cat(mlvl_scores)
    mlvl_dir_scores = torch.cat(mlvl_dir_scores)

    # Add a dummy background class to the front when using sigmoid
    padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
    mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

    score_thr = 0.05
    # mlvl_bboxes_for_nms = input_meta['box_type_3d'](mlvl_bboxes, box_dim=9).bev
    mlvl_bboxes_for_nms = mlvl_bboxes[:, [0, 1, 3, 4, 6]]
    results = box3d_multiclass_scale_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                            mlvl_scores, score_thr, 500,
                                            cfg, mlvl_dir_scores)

    bboxes, scores, labels, dir_scores = results
    if bboxes.shape[0] > 0:
        dir_rot = limit_period(bboxes[..., 6] - 0.7854, 0, np.pi)
        bboxes[..., 6] = (dir_rot + 0.7854 + np.pi * dir_scores.to(bboxes.dtype))
    # bboxes = input_meta['box_type_3d'](bboxes, box_dim=9)
    corners = corners_get(bboxes)
    # bboxes = corners(bboxes)
    # bboxes = bboxes
    return corners, scores, labels

def box3d_multiclass_scale_nms(mlvl_bboxes,
                               mlvl_bboxes_for_nms,
                               mlvl_scores,
                               score_thr,
                               max_num,
                               cfg,
                               mlvl_dir_scores=None,
                               mlvl_attr_scores=None,
                               mlvl_bboxes2d=None):
    """Multi-class nms for 3D boxes.

    Args:
        mlvl_bboxes (torch.Tensor): Multi-level boxes with shape (N, M).
            M is the dimensions of boxes.
        mlvl_bboxes_for_nms (torch.Tensor): Multi-level boxes with shape
            (N, 5) ([x1, y1, x2, y2, ry]). N is the number of boxes.
        mlvl_scores (torch.Tensor): Multi-level boxes with shape
            (N, C + 1). N is the number of boxes. C is the number of classes.
        score_thr (float): Score thredhold to filter boxes with low
            confidence.
        max_num (int): Maximum number of boxes will be kept.
        cfg (dict): Configuration dict of NMS.
        mlvl_dir_scores (torch.Tensor, optional): Multi-level scores
            of direction classifier. Defaults to None.
        mlvl_attr_scores (torch.Tensor, optional): Multi-level scores
            of attribute classifier. Defaults to None.
        mlvl_bboxes2d (torch.Tensor, optional): Multi-level 2D bounding
            boxes. Defaults to None.

    Returns:
        tuple[torch.Tensor]: Return results after nms, including 3D \
            bounding boxes, scores, labels, direction scores, attribute \
            scores (optional) and 2D bounding boxes (optional).
    """
    # do multi class nms
    # the fg class id range: [0, num_classes-1]
    num_classes = mlvl_scores.shape[1] - 1
    bboxes = []
    scores = []
    labels = []
    dir_scores = []
    attr_scores = []
    bboxes2d = []
    nms_type_list = ['rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'circle']
    nms_thr_list = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.2]
    nms_radius_thr_list = [4, 12, 10, 10, 12, 0.85, 0.85, 0.175, 0.175, 1]
    nms_rescale_factor = [1.0, 0.7, 0.55, 0.4, 0.7, 1.0, 1.0, 4.5, 9.0, 1.0]
    for i in range(0, num_classes):
        # get bboxes and scores of this class
        cls_inds = mlvl_scores[:, i] > score_thr
        if not cls_inds.any():
            continue

        _scores = mlvl_scores[cls_inds, i]
        _bboxes_for_nms = mlvl_bboxes_for_nms[cls_inds, :]
        _mlvl_bboxes = mlvl_bboxes[cls_inds, :]

        nms_func = {'rotate': nms_gpu, 'circle': circle_nms}[nms_type_list[i]]

        nms_thre = nms_thr_list[i]
        nms_radius_thre = nms_radius_thr_list[i]
        nms_target_thre = {'rotate': nms_thre, 'circle': nms_radius_thre}[nms_type_list[i]]

        nms_rescale = nms_rescale_factor[i]
        _bboxes_for_nms[:, 2:4] *= nms_rescale

        if nms_type_list[i] == 'rotate':
            _bboxes_for_nms = xywhr2xyxyr(_bboxes_for_nms)
            selected = nms_func(_bboxes_for_nms, _scores, nms_target_thre)
        else:
            _centers = _bboxes_for_nms[:, [0, 1]]
            _bboxes_for_nms = torch.cat([_centers, _scores.view(-1, 1)], dim=1)
            selected = nms_func(_bboxes_for_nms.detach().cpu().numpy(), nms_target_thre)
            selected = torch.tensor(selected, dtype=torch.long, device=_bboxes_for_nms.device)

        bboxes.append(_mlvl_bboxes[selected])
        scores.append(_scores[selected])
        cls_label = mlvl_bboxes.new_full((len(selected), ), i, dtype=torch.long)
        labels.append(cls_label)

        if mlvl_dir_scores is not None:
            _mlvl_dir_scores = mlvl_dir_scores[cls_inds]
            dir_scores.append(_mlvl_dir_scores[selected])
        if mlvl_attr_scores is not None:
            _mlvl_attr_scores = mlvl_attr_scores[cls_inds]
            attr_scores.append(_mlvl_attr_scores[selected])
        if mlvl_bboxes2d is not None:
            _mlvl_bboxes2d = mlvl_bboxes2d[cls_inds]
            bboxes2d.append(_mlvl_bboxes2d[selected])

    bboxes = torch.cat(bboxes, dim=0)
    scores = torch.cat(scores, dim=0)
    labels = torch.cat(labels, dim=0)
    if mlvl_dir_scores is not None:
        dir_scores = torch.cat(dir_scores, dim=0)
    if mlvl_attr_scores is not None:
        attr_scores = torch.cat(attr_scores, dim=0)
    if mlvl_bboxes2d is not None:
        bboxes2d = torch.cat(bboxes2d, dim=0)
    if bboxes.shape[0] > max_num:
        _, inds = scores.sort(descending=True)
        inds = inds[:max_num]
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        scores = scores[inds]
        if mlvl_dir_scores is not None:
            dir_scores = dir_scores[inds]
        if mlvl_attr_scores is not None:
            attr_scores = attr_scores[inds]
        if mlvl_bboxes2d is not None:
            bboxes2d = bboxes2d[inds]

    results = (bboxes, scores, labels)

    if mlvl_dir_scores is not None:
        results = results + (dir_scores, )
    if mlvl_attr_scores is not None:
        results = results + (attr_scores, )
    if mlvl_bboxes2d is not None:
        results = results + (bboxes2d, )

    return results

def limit_period(val, offset=0.5, period=np.pi):
    """Limit the value into a period for periodic function.

    Args:
        val (torch.Tensor): The value to be converted.
        offset (float, optional): Offset to set the value range. \
            Defaults to 0.5.
        period ([type], optional): Period of the value. Defaults to np.pi.

    Returns:
        torch.Tensor: Value in the range of \
            [-offset * period, (1-offset) * period]
    """
    return val - torch.floor(val / period + offset) * period

def decode(anchors, deltas):
    """Apply transformation `deltas` (dx, dy, dz, dw, dh, dl, dr, dv*) to
    `boxes`.

    Args:
        anchors (torch.Tensor): Parameters of anchors with shape (N, 7).
        deltas (torch.Tensor): Encoded boxes with shape
            (N, 7+n) [x, y, z, w, l, h, r, velo*].

    Returns:
        torch.Tensor: Decoded boxes.
    """
    cas, cts = [], []
    box_ndim = anchors.shape[-1]
    if box_ndim > 7:
        xa, ya, za, wa, la, ha, ra, *cas = torch.split(anchors, 1, dim=-1)  # x,y,z, w,l,h, r旋转角
        xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(deltas, 1, dim=-1)
    else:
        xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt = torch.split(deltas, 1, dim=-1)

    za = za + ha / 2
    diagonal = torch.sqrt(la**2 + wa**2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * ha + za

    lg = torch.exp(lt) * la
    wg = torch.exp(wt) * wa
    hg = torch.exp(ht) * ha
    rg = rt + ra
    zg = zg - hg / 2
    cgs = [t + a for t, a in zip(cts, cas)]
    return torch.cat([xg, yg, zg, wg, lg, hg, rg, *cgs], dim=-1)

def nms_gpu(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """Nms function with gpu implementation.

    Args:
        boxes (torch.Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of boxes with the shape of [N].
        thresh (int): Threshold.
        pre_maxsize (int): Max size of boxes before nms. Default: None.
        post_maxsize (int): Max size of boxes after nms. Default: None.

    Returns:
        torch.Tensor: Indexes after nms.
    """
    order = scores.sort(0, descending=True)[1]

    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes = boxes[order].contiguous()

    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = iou3d_cuda.nms_gpu(boxes, keep, thresh, boxes.device.index)
    keep = order[keep[:num_out].cuda(boxes.device)].contiguous()
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep

def circle_nms(dets, thresh, post_max_size=83):
    """Circular NMS.

    An object is only counted as positive if no other center
    with a higher confidence exists within a radius r using a
    bird-eye view distance metric.

    Args:
        dets (torch.Tensor): Detection results with the shape of [N, 3].
        thresh (float): Value of threshold.
        post_max_size (int): Max number of prediction to be kept. Defaults
            to 83

    Returns:
        torch.Tensor: Indexes of the detections to be kept.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[
                i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate center distance between i and j box
            dist = (x1[i] - x1[j])**2 + (y1[i] - y1[j])**2

            # ovr = inter / areas[j]
            if dist <= thresh:
                suppressed[j] = 1
    return keep[:post_max_size]

def xywhr2xyxyr(boxes_xywhr):
    """Convert a rotated boxes in XYWHR format to XYXYR format.

    Args:
        boxes_xywhr (torch.Tensor): Rotated boxes in XYWHR format.

    Returns:
        torch.Tensor: Converted boxes in XYXYR format.
    """
    boxes = torch.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[:, 2] / 2
    half_h = boxes_xywhr[:, 3] / 2

    boxes[:, 0] = boxes_xywhr[:, 0] - half_w
    boxes[:, 1] = boxes_xywhr[:, 1] - half_h
    boxes[:, 2] = boxes_xywhr[:, 0] + half_w
    boxes[:, 3] = boxes_xywhr[:, 1] + half_h
    boxes[:, 4] = boxes_xywhr[:, 4]
    return boxes

def box_type_3d(tensor, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0)):
    if isinstance(tensor, torch.Tensor):
        device = tensor.device
    else:
        device = torch.device('cpu')
    tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)

    assert tensor.dim() == 2 and tensor.size(-1) == box_dim, tensor.size()

    # box_dim = box_dim
    # with_yaw = with_yaw
    tensor = tensor.clone()
    return tensor

def corners_get(tensor):
    """torch.Tensor: Coordinates of corners of all the boxes
    in shape (N, 8, 3).

    Convert the boxes to corners in clockwise order, in form of
    ``(x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)``

    .. code-block:: none

                                        up z
                        front x           ^
                                /            |
                            /             |
                (x1, y0, z1) + -----------  + (x1, y1, z1)
                            /|            / |
                            / |           /  |
            (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                        |  /      .   |  /
                        | / oriign    | /
        left y<-------- + ----------- + (x0, y1, z0)
            (x0, y0, z0)
    """
    # TODO: rotation_3d_in_axis function do not support
    #  empty tensor currently.
    assert len(tensor) != 0
    dims = tensor[:, 3:6]
    corners_norm = torch.from_numpy(
        np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)).to(
            device=dims.device, dtype=dims.dtype)

    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    # use relative origin [0.5, 0.5, 0]
    corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0])
    corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

    # rotate around z axis
    corners = rotation_3d_in_axis(corners, tensor[:, 6], axis=2)
    corners += tensor[:, :3].view(-1, 1, 3)
    return corners

def rotation_3d_in_axis(points, angles, axis=0):
    """Rotate points by angles according to axis.

    Args:
        points (torch.Tensor): Points of shape (N, M, 3).
        angles (torch.Tensor): Vector of angles in shape (N,)
        axis (int, optional): The axis to be rotated. Defaults to 0.

    Raises:
        ValueError: when the axis is not in range [0, 1, 2], it will \
            raise value error.

    Returns:
        torch.Tensor: Rotated points in shape (N, M, 3)
    """
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = torch.stack([
            torch.stack([rot_cos, zeros, -rot_sin]),
            torch.stack([zeros, ones, zeros]),
            torch.stack([rot_sin, zeros, rot_cos])
        ])
    elif axis == 2 or axis == -1:
        rot_mat_T = torch.stack([
            torch.stack([rot_cos, -rot_sin, zeros]),
            torch.stack([rot_sin, rot_cos, zeros]),
            torch.stack([zeros, zeros, ones])
        ])
    elif axis == 0:
        rot_mat_T = torch.stack([
            torch.stack([zeros, rot_cos, -rot_sin]),
            torch.stack([zeros, rot_sin, rot_cos]),
            torch.stack([ones, zeros, zeros])
        ])
    else:
        raise ValueError(f'axis should in range [0, 1, 2], got {axis}')

    return torch.einsum('aij,jka->aik', (points, rot_mat_T))