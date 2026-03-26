import torch
from torch import nn
from torch.nn import functional as F

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75, reduction='mean', loss_weight=1.0):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        # 针对严重漏检，将 alpha 调高，偏向正样本（车道线/边界）
        self.alpha = torch.tensor([[0.8, 0.8]]) 
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        """
        Args:
            pred: logits with shape (B, N, H, W)
            target: float targets with shape (B, N, H, W)
        """
        # 1. 计算 sigmoid 概率
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)

        # 2. 计算 pt (模型对真值的预测概率)
        # 当 target=1 时，pt = pred_sigmoid
        # 当 target=0 时，pt = 1 - pred_sigmoid
        pt = (pred_sigmoid * target) + ((1 - pred_sigmoid) * (1 - target))
        
        # 3. 计算 Focal Weight
        # 这里的 alpha 处理：给正样本 alpha 权重，给负样本 (1-alpha) 权重
        self.alpha = self.alpha.to(pred.device)
        alpha_weight = (self.alpha[:, :, None, None] * target + (1 - self.alpha)[:, :, None, None] * (1 - target))
        
        # 调节因子：(1-pt)^gamma。当预测越准时（pt接近1），权重越小
        focal_weight = alpha_weight * (1 - pt).pow(self.gamma)

        # 4. 计算 BCE Loss 并加权
        # 使用 binary_cross_entropy_with_logits 保证数值稳定性
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        loss = bce_loss * focal_weight

        # 5. Reduction
        if self.reduction == 'mean':
            # 注意：在稀疏任务中，有时 sum(loss) / num_positive_samples 效果更好
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss * self.loss_weight

class DiscriminativeLoss(nn.Module):
    def __init__(self, embed_dim, delta_v, delta_d):
        super(DiscriminativeLoss, self).__init__()
        self.embed_dim = embed_dim
        self.delta_v = delta_v
        self.delta_d = delta_d

    def forward(self, embedding, seg_gt):
        if embedding is None:
            return 0, 0, 0
        bs = embedding.shape[0]

        var_loss = embedding.sum() * 0.0
        dist_loss = embedding.sum() * 0.0
        reg_loss = embedding.sum() * 0.0

        for b in range(bs):
            embedding_b = embedding[b]  # (embed_dim, H, W)
            seg_gt_b = seg_gt[b][0]

            labels = torch.unique(seg_gt_b)
            labels = labels[labels != 0]
            num_lanes = len(labels)
            if num_lanes == 0:
                continue

            centroid_mean = []
            for lane_idx in labels:
                seg_mask_i = (seg_gt_b == lane_idx)
                if not seg_mask_i.any():
                    continue
                embedding_i = embedding_b[:, seg_mask_i]

                mean_i = torch.mean(embedding_i, dim=1)
                centroid_mean.append(mean_i)

                # ---------- var_loss -------------
                var_loss = var_loss + torch.mean(F.relu(torch.norm(embedding_i-mean_i.reshape(self.embed_dim, 1), dim=0) - self.delta_v) ** 2) / num_lanes
            centroid_mean = torch.stack(centroid_mean)  # (n_lane, embed_dim)

            if num_lanes > 1:
                centroid_mean1 = centroid_mean.reshape(-1, 1, self.embed_dim)
                centroid_mean2 = centroid_mean.reshape(1, -1, self.embed_dim)
                dist = torch.norm(centroid_mean1-centroid_mean2, dim=2)  # shape (num_lanes, num_lanes)
                dist = dist + torch.eye(num_lanes, dtype=dist.dtype, device=dist.device) * self.delta_d  # diagonal elements are 0, now mask above delta_d

                # divided by two for double calculated loss above, for implementation convenience
                dist_loss = dist_loss + torch.sum(F.relu(-dist + self.delta_d)**2) / (num_lanes * (num_lanes-1)) / 2

            # reg_loss is not used in original paper
            # reg_loss = reg_loss + torch.mean(torch.norm(centroid_mean, dim=1))

        var_loss = var_loss / bs
        dist_loss = dist_loss / bs
        reg_loss = reg_loss / bs
        return var_loss + dist_loss + reg_loss