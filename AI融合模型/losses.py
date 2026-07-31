"""多目标损失函数

L_total = λ_a·L_acc + λ_x·L_state + λ_rtn·L_RTN
        + λ_e·L_endpoint + λ_s·L_smooth + λ_b·L_bound + λ_g·L_graph

所有损失加权质量权重 w_q。
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HuberLoss(nn.Module):
    """Huber 损失 (delta=1.0 即 smooth L1)。"""

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(pred, target, delta=self.delta, reduction="none")


def acceleration_loss(
    delta_a_pred: torch.Tensor,
    delta_a_target: torch.Tensor,
    quality_weight: Optional[torch.Tensor] = None,
    axis_weights: tuple[float, float, float] = (1.0, 2.0, 1.0),
) -> torch.Tensor:
    """RTN 残差加速度监督损失 (Huber)。

    L_a = sum w_q * Huber(pred - target) / sum w_q

    Args:
        delta_a_pred:   (B, N, H, 3)
        delta_a_target: (B, N, H, 3)
        quality_weight: (B, N, H) quality weights [0, 1]
        axis_weights: (w_R, w_T, w_N) per-axis weights

    Returns:
        scalar loss
    """
    axis_w = torch.tensor(axis_weights, device=delta_a_pred.device).view(1, 1, 1, 3)
    huber = F.smooth_l1_loss(delta_a_pred, delta_a_target, reduction="none")  # (B, N, H, 3)
    weighted = huber * axis_w

    if quality_weight is not None:
        weighted = weighted * quality_weight.unsqueeze(-1)

    return weighted.sum() / max(weighted.numel(), 1)


def state_loss(
    state_pred: torch.Tensor,
    state_ref: torch.Tensor,
    pos_weight: float = 1.0,
    vel_weight: float = 0.1,
) -> torch.Tensor:
    """再积分后轨道状态损失。

    L_x = (w_r * |r_pred - r_ref|^2 + w_v * |v_pred - v_ref|^2) / (N*H)
    """
    pos_err = ((state_pred[..., :3] - state_ref[..., :3]) ** 2).mean()
    vel_err = ((state_pred[..., 3:6] - state_ref[..., 3:6]) ** 2).mean()
    return pos_weight * pos_err + vel_weight * vel_err


def rtn_loss(
    state_pred: torch.Tensor,
    state_ref: torch.Tensor,
    axis_weights: tuple[float, float, float] = (1.0, 2.0, 1.0),
) -> torch.Tensor:
    """RTN 分量误差损失 (需要调用方提前将状态投影到RTN)。

    L_rtn = mean(w_R*ΔR^2 + w_T*ΔT^2 + w_N*ΔN^2)
    """
    # 简化: 此处假设输入已是 RTN 坐标
    w = torch.tensor(axis_weights, device=state_pred.device).view(1, 1, 1, 3)
    diff_sq = (state_pred - state_ref) ** 2
    return (diff_sq * w).mean()


def endpoint_loss(
    state_pred: torch.Tensor,
    state_ref: torch.Tensor,
) -> torch.Tensor:
    """末端位置误差: |r_pred(H) - r_ref(H)|^2 / N"""
    end_pred = state_pred[:, :, -1, :3]
    end_ref = state_ref[:, :, -1, :3]
    return ((end_pred - end_ref) ** 2).sum(dim=-1).mean()


def smoothness_loss(
    delta_a_pred: torch.Tensor,
) -> torch.Tensor:
    """加速度时间平滑损失。

    L_smooth = mean(|Δa_{k+1} - Δa_k|^2) / (N*(H-1))
    """
    diff = delta_a_pred[:, :, 1:] - delta_a_pred[:, :, :-1]  # (B, N, H-1, 3)
    return (diff ** 2).mean()


def bound_loss(
    delta_a_pred: torch.Tensor,
    acc_bounds: tuple[float, float, float] = (2e-5, 2e-5, 2e-5),
) -> torch.Tensor:
    """边界损失: 惩罚超出物理边界的加速度。

    L_bound = sum_c ReLU(|a_c| - bound_c)^2
    """
    bounds = torch.tensor(acc_bounds, device=delta_a_pred.device).view(1, 1, 1, 3)
    excess = torch.abs(delta_a_pred) - bounds
    return (F.relu(excess) ** 2).mean()


def graph_consistency_loss(
    delta_a_pred: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """图一致性弱正则: 相邻卫星预测的残差加速度应相似。

    仅对同轨道面边施加。
    """
    if edge_index.size(1) == 0:
        return torch.tensor(0.0, device=delta_a_pred.device)

    src = edge_index[0]
    dst = edge_index[1]
    diff = delta_a_pred[:, src] - delta_a_pred[:, dst]  # (B, E, H, 3)
    return (diff ** 2).mean() * 0.001  # 弱正则


class CompositeLoss(nn.Module):
    """组合多目标损失。

    预训练: λ_a=1.0, λ_smooth=0.01, λ_bound=0.001
    联合微调: λ_a=0.2, λ_x=1.0, λ_rtn=1.0, λ_end=0.5, λ_smooth=0.01
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        acc_bounds: tuple[float, float, float] = (2e-5, 2e-5, 2e-5),
    ):
        super().__init__()
        self.weights = weights or {
            "lambda_acc": 1.0,
            "lambda_state": 0.0,
            "lambda_rtn": 0.0,
            "lambda_endpoint": 0.0,
            "lambda_smooth": 0.01,
            "lambda_bound": 0.001,
            "lambda_graph": 0.0,
        }
        self.acc_bounds = acc_bounds

    def forward(
        self,
        delta_a_pred: torch.Tensor,
        delta_a_target: torch.Tensor,
        state_pred: Optional[torch.Tensor] = None,
        state_ref: Optional[torch.Tensor] = None,
        quality_weight: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict of {loss_name: scalar_tensor, "total": ...}
        """
        losses = {}

        # Acceleration supervision
        losses["L_acc"] = acceleration_loss(
            delta_a_pred, delta_a_target, quality_weight,
        )
        losses["total"] = self.weights["lambda_acc"] * losses["L_acc"]

        # State loss (if available)
        if state_pred is not None and state_ref is not None:
            losses["L_state"] = state_loss(state_pred, state_ref)
            losses["total"] = losses["total"] + self.weights["lambda_state"] * losses["L_state"]

            losses["L_endpoint"] = endpoint_loss(state_pred, state_ref)
            losses["total"] = losses["total"] + self.weights["lambda_endpoint"] * losses["L_endpoint"]

        # Smoothness
        losses["L_smooth"] = smoothness_loss(delta_a_pred)
        losses["total"] = losses["total"] + self.weights["lambda_smooth"] * losses["L_smooth"]

        # Bound
        losses["L_bound"] = bound_loss(delta_a_pred, self.acc_bounds)
        losses["total"] = losses["total"] + self.weights["lambda_bound"] * losses["L_bound"]

        # Graph consistency
        if edge_index is not None and self.weights.get("lambda_graph", 0) > 0:
            losses["L_graph"] = graph_consistency_loss(delta_a_pred, edge_index)
            losses["total"] = losses["total"] + self.weights["lambda_graph"] * losses["L_graph"]

        return losses
