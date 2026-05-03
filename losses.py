# losses.py
# -*- coding: utf-8 -*-
"""
多标签分类用损失函数合集：
- AsymmetricLossMultiLabel（ASL，支持按类别加权）
- BCEWithLogitsIgnore（支持 -1 忽略标签 + 按类别加权）
- GraphSmoothingLoss（基于标签图的平滑正则）
- PairwiseAUCLoss（批内按类别的排序/AUC 近似损失）
- CombinedMultiLabelLoss（上面几项的可加权组合）

注意：
- logits 形状 (B, K)，targets 取值 {1, 0, -1}，其中 -1 表示忽略。
- 可选 mask 形状 (B, K)，1 表示参与损失，0 表示忽略。
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "AsymmetricLossMultiLabel",
    "BCEWithLogitsIgnore",
    "GraphSmoothingLoss",
    "PairwiseAUCLoss",
    "CombinedMultiLabelLoss",
]


# ============== 1) ASL: Asymmetric Loss（多标签稳定实现） ==============
class AsymmetricLossMultiLabel(nn.Module):
    """
    适用于多标签不平衡，支持 -1 忽略标签、按类别加权
    参考：Ridnik et al., "Asymmetric Loss For Multi-Label Classification"
    """
    def __init__(self,
                 gamma_pos: float = 0.0,
                 gamma_neg: float = 4.0,
                 clip: float = 0.05,
                 eps: float = 0.0,
                 reduction: str = 'mean',
                 label_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip) if clip is not None else None
        self.eps = float(eps)
        self.reduction = reduction

        # 按类别权重：shape = (1, K)，用于对每一列 loss 缩放
        if label_weights is not None:
            lw = label_weights.view(1, -1).float()
            self.register_buffer("label_weights", lw)
        else:
            self.label_weights = None

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        logits:  (B, K) 未经 sigmoid
        targets: (B, K) ∈ {1,0,-1}，-1 表示忽略
        mask:    (B, K) 1/0，若为 None，则自动使用 (targets != -1)
        """
        if mask is None:
            mask = (targets != -1).float()
            targets = targets.clamp(min=0)  # -1 -> 0

        # Sigmoid 概率
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        # 负样本概率下界，避免极端值导致梯度爆炸
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # 基础 BCE（对数形式），加 clamp 以避免 log(0)
        loss_pos = targets * torch.log(xs_pos.clamp(min=1e-8))
        loss_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=1e-8))
        loss = loss_pos + loss_neg  # 这里是负值

        # Asymmetric Focusing
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt = targets * xs_pos + (1.0 - targets) * xs_neg  # p_t
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            loss = loss * torch.pow(1.0 - pt, one_sided_gamma)

        # Label smoothing（仅作轻微常数平滑）
        if self.eps > 0:
            num_classes = targets.size(1)
            const_term = -torch.log(torch.tensor(0.5, device=logits.device))
            loss = (1.0 - self.eps) * loss + self.eps * const_term / num_classes

        loss = -loss * mask  # 取负 & 忽略

        # 按类别权重缩放（重点照顾难类）
        if self.label_weights is not None:
            # self.label_weights: (1, K)，广播到 (B, K)
            loss = loss * self.label_weights

        if self.reduction == 'mean':
            denom = mask.sum().clamp(min=1.0)
            return loss.sum() / denom
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ============== 2) BCEWithLogits（带忽略标签 + 按类别加权） ==============
class BCEWithLogitsIgnore(nn.Module):
    """
    与 PyTorch BCEWithLogits 相同，但支持：
    - targets = -1 表示忽略
    - label_weights 对每个类别单独加权
    """
    def __init__(self,
                 reduction: str = 'mean',
                 pos_weight: Optional[torch.Tensor] = None,
                 label_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.reduction = reduction
        self.register_buffer('pos_weight_buf', pos_weight if pos_weight is not None else None)

        if label_weights is not None:
            lw = label_weights.view(1, -1).float()
            self.register_buffer("label_weights", lw)
        else:
            self.label_weights = None

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = (targets != -1).float()
            targets = targets.clamp(min=0)

        loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(),
            pos_weight=self.pos_weight_buf,
            reduction='none'
        )
        loss = loss * mask

        # 按类别权重缩放
        if self.label_weights is not None:
            loss = loss * self.label_weights

        if self.reduction == 'mean':
            denom = mask.sum().clamp(min=1.0)
            return loss.sum() / denom
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ============== 3) 图平滑正则（基于标签图邻接矩阵 A） ==============
class GraphSmoothingLoss(nn.Module):
    """
    让相邻标签预测概率更一致（对多标签相关性建模的正则项）
    简化目标：sum_{i<j} A_ij * |p_i - p_j|

    其中 A 为 (K,K) 的相似度/邻接矩阵（对角置零，可选择行归一化）。
    """
    def __init__(self, A: torch.Tensor, normalized: bool = True, reduction: str = 'mean'):
        super().__init__()
        assert isinstance(A, torch.Tensor) and A.ndim == 2 and A.size(0) == A.size(1), "A 必须是 (K,K) 方阵"
        A = A.clone().float()
        A.fill_diagonal_(0.0)
        if normalized:
            deg = A.sum(dim=1, keepdim=True).clamp(min=1e-6)
            A = A / deg  # 行归一化
        self.register_buffer('A', A)
        # 只取上三角避免重复计算
        triu = torch.triu(torch.ones_like(A), diagonal=1)
        self.register_buffer('triu_mask', triu)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # logits -> 概率
        P = torch.sigmoid(logits)  # (B, K)
        if mask is not None:
            valid = (mask != 0).float()
            # 对无效维度填充中性概率 0.5，降低其对 pairwise 的影响
            P = P * valid + 0.5 * (1.0 - valid)

        # pairwise |p_i - p_j| * A_ij
        diff = torch.abs(P.unsqueeze(2) - P.unsqueeze(1))  # (B, K, K)
        loss_mat = diff * self.A  # (B, K, K)
        loss_mat = loss_mat * self.triu_mask  # 仅上三角
        loss = loss_mat.sum(dim=(1, 2))  # (B,)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ============== 4) 批内 Pairwise AUC 近似损失 ==============
class PairwiseAUCLoss(nn.Module):
    """
    批内按类别的排序损失（AUC 友好）：
    对类别 c，采样若干正/负预测分数 s_pos, s_neg，最小化：
        softplus( -(s_pos - s_neg - margin) )
    使正样本 logit 系统性高于负样本，有助于提升阈值无关指标（ROC-AUC / PR-AUC）。
    """
    def __init__(self,
                 max_pos: int = 32,
                 max_neg: int = 64,
                 margin: float = 0.0,
                 reduction: str = 'mean',
                 cls_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.max_pos = max_pos
        self.max_neg = max_neg
        self.margin = float(margin)
        self.reduction = reduction

        if cls_weights is not None:
            cw = cls_weights.view(-1).float()
            self.register_buffer("cls_weights", cw)
        else:
            self.cls_weights = None

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        logits:  (B, K) 未过 Sigmoid
        targets: (B, K) ∈ {1,0,-1}，-1 表示忽略
        mask:    (B, K) 1/0（可选）
        """
        B, C = logits.shape
        if mask is None:
            valid = (targets != -1)
        else:
            valid = (mask != 0)
        t = targets.clamp(min=0)  # -1 -> 0

        loss_sum = logits.new_tensor(0.0)
        count = 0

        for c in range(C):
            v = valid[:, c]
            if v.sum() < 2:
                continue
            tc = t[v, c]            # (n,)
            lc = logits[v, c]       # (n,)

            pos = lc[tc == 1]
            neg = lc[tc == 0]
            if pos.numel() == 0 or neg.numel() == 0:
                continue

            # 子采样控制计算量，增强稳定性（长尾场景尤为重要）
            if self.max_pos is not None and pos.numel() > self.max_pos:
                idx = torch.randperm(pos.numel(), device=pos.device)[:self.max_pos]
                pos = pos[idx]
            if self.max_neg is not None and neg.numel() > self.max_neg:
                idx = torch.randperm(neg.numel(), device=neg.device)[:self.max_neg]
                neg = neg[idx]

            # 计算配对差值：s_pos - s_neg - margin
            diff = pos.view(-1, 1) - neg.view(1, -1) - self.margin
            loss_c = F.softplus(-diff).mean()  # log(1+exp(-diff))

            # 类别级权重：难类的 AUC loss 更重一点
            if self.cls_weights is not None and c < self.cls_weights.numel():
                loss_c = loss_c * self.cls_weights[c]

            loss_sum = loss_sum + loss_c
            count += 1

        if count == 0:
            return logits.new_zeros(())
        return loss_sum / count if self.reduction == 'mean' else loss_sum


# ============== 5) 混合 Loss 封装 ==============
class CombinedMultiLabelLoss(nn.Module):
    """
    总损失 = w_asl * ASL + w_bce * BCE + w_gsl * GraphSmooth + w_auc * PairwiseAUC
    - 支持任意 {1,0,-1} / {1,0} / {1,255} 之类的编码
    - 在这里统一把标签/掩码处理掉，再喂给各个子 loss
    """
    def __init__(self,
                 w_asl: float = 1.0,
                 w_bce: float = 0.0,
                 w_gsl: float = 0.0,
                 w_auc: float = 0.1,
                 # ASL 参数
                 gamma_pos: float = 0.0,
                 gamma_neg: float = 4.0,
                 clip: float = 0.05,
                 eps: float = 0.0,
                 # BCE / 类别权重
                 pos_weight: Optional[torch.Tensor] = None,
                 label_weights: Optional[torch.Tensor] = None,
                 # 图平滑
                 A: Optional[torch.Tensor] = None,
                 A_normalized: bool = True,
                 # AUC 排序
                 auc_max_pos: int = 32,
                 auc_max_neg: int = 64,
                 auc_margin: float = 0.0):
        super().__init__()
        self.w_asl = float(w_asl)
        self.w_bce = float(w_bce)
        self.w_gsl = float(w_gsl)
        self.w_auc = float(w_auc)

        if label_weights is not None and not isinstance(label_weights, torch.Tensor):
            label_weights = torch.tensor(label_weights, dtype=torch.float32)

        self.asl = AsymmetricLossMultiLabel(
            gamma_pos=gamma_pos,
            gamma_neg=gamma_neg,
            clip=clip,
            eps=eps,
            reduction='mean',
            label_weights=label_weights
        ) if self.w_asl > 0 else None

        self.bce = BCEWithLogitsIgnore(
            reduction='mean',
            pos_weight=pos_weight,
            label_weights=label_weights
        ) if self.w_bce > 0 else None

        self.gsl = GraphSmoothingLoss(
            A=A, normalized=A_normalized, reduction='mean'
        ) if (self.w_gsl > 0 and A is not None) else None

        # AUC 分支也使用同一套 label_weights 对难类加权
        self.auc = PairwiseAUCLoss(
            max_pos=auc_max_pos,
            max_neg=auc_max_neg,
            margin=auc_margin,
            reduction='mean',
            cls_weights=label_weights
        ) if self.w_auc > 0 else None

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        在这里统一：
        - 任意 dtype 的 targets -> float
        - -1 视作忽略
        - >0 视作正类(1)，其它视作负类(0)
        """
        # 1) 转 float
        if not torch.is_floating_point(targets):
            t = targets.float()
        else:
            t = targets

        # 2) 构造统一的 mask：-1 为忽略，其它参与 loss
        if mask is None:
            mask_internal = (t != -1).float()
        else:
            mask_internal = mask.float()

        # 3) 所有 >0 的标签都当成正类 1，其余 0（兼容 {1,0,-1} / {1,0} / {255,0} 等）
        t = (t > 0).float()

        loss = logits.new_tensor(0.0)

        if self.asl is not None and self.w_asl != 0:
            loss = loss + self.w_asl * self.asl(logits, t, mask=mask_internal)

        if self.bce is not None and self.w_bce != 0:
            loss = loss + self.w_bce * self.bce(logits, t, mask=mask_internal)

        if self.gsl is not None and self.w_gsl != 0:
            loss = loss + self.w_gsl * self.gsl(logits, mask=mask_internal)

        if self.auc is not None and self.w_auc != 0:
            loss = loss + self.w_auc * self.auc(logits, t, mask=mask_internal)

        return loss
