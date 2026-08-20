"""物理-AI 融合模型 — 完整的 Informer + GNN + Decoder 架构

Architecture:
  Input  →  Informer (单星时间编码)
         →  GNN (星间空间融合)
         →  Decoder (未来多步 RTN 残差加速度预测)

Output: Δa_RTN ∈ R^{H×3}  for each satellite

其中 H = prediction_length (72 for 12h, 144 for 24h @ 600s step)
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from informer.model import InformerEncoder
from gnn.model import GNNEncoder


class FutureDecoder(nn.Module):
    """未来 RTN 残差加速度多步解码器。

    输入:
      - h_space: (B, N, d_model) 空间融合特征
      - physics_future: (B, N, H, 6) 物理预测状态
      - future_time_features: (B, N, H, F_t) 未来时间编码 (可选)

    输出:
      - delta_a_rtn: (B, N, H, 3) RTN残差加速度 [ΔaR, ΔaT, ΔaN]

    架构:
      Per-step MLP: concat(h_space, physics_state(t), time(t)) → MLP → Δa
      使用 Tanh 限制输出幅值。
    """

    def __init__(
        self,
        d_model: int = 128,
        num_time_features: int = 6,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        acc_bounds: tuple[float, float, float] = (5e-5, 2e-4, 5e-5),
    ):
        super().__init__()
        # Input: h_space(128) + physics(6) [+ time if present]
        self._base_dim = d_model + 6  # = 134
        self._pad_time_dim = max(num_time_features, 0)
        input_dim = self._base_dim + self._pad_time_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
            nn.Tanh(),
        )

        # Output scaling — must cover 99.5% of label values from piecewise fitting
        # Default: [5e-5, 1e-4, 5e-5] for [ΔaR, ΔaT, ΔaN]
        # ΔaT is always largest (along-track error accumulation)
        self.register_buffer(
            "acc_scale",
            torch.tensor(acc_bounds, dtype=torch.float).view(1, 1, 1, 3),
        )

    def forward(
        self,
        h_space: torch.Tensor,
        physics_future: torch.Tensor,
        future_time_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            h_space: (B, N, d_model)
            physics_future: (B, N, H, 6)
            future_time_features: (B, N, H, F_t) optional

        Returns:
            delta_a_rtn: (B, N, H, 3)
        """
        B, N, H, _ = physics_future.shape

        # Expand h_space to all time steps
        h_expanded = h_space.unsqueeze(2).expand(-1, -1, H, -1)  # (B, N, H, d_model)

        # Concatenate per time step; pad with zeros if no time features
        if future_time_features is not None:
            combined = torch.cat([h_expanded, physics_future, future_time_features], dim=-1)
        else:
            # Pad to match MLP expected input dim
            padding = torch.zeros(B, N, H, self._pad_time_dim, device=physics_future.device)
            combined = torch.cat([h_expanded, physics_future, padding], dim=-1)

        # Reshape for MLP: (B*N*H, input_dim)
        flat = combined.view(B * N * H, -1)
        out = self.mlp(flat)  # (B*N*H, 3)
        out = out.view(B, N, H, 3)

        return out * self.acc_scale


class PhysicsAIFusionModel(nn.Module):
    """完整的物理-AI融合模型。

    Architecture:
      Starlink history states (N satellites × L steps)
              │
      ┌───────┴────────┐
      │  Informer       │  ← 单星时间编码
      │  d_model=128    │
      └───────┬────────┘
              │ h_time: (B, N, 128)
      ┌───────┴────────┐
      │  GNN (GATv2)    │  ← 星间空间融合
      │  2 layers       │
      └───────┬────────┘
              │ h_space: (B, N, 128)
      ┌───────┴────────┐
      │  Future Decoder │  ← 未来RTN残差加速度
      │  MLP per step   │
      └───────┬────────┘
              │
      Δa_RTN ∈ R^{B×N×H×3}
    """

    def __init__(self, cfg: dict):
        super().__init__()
        inf = cfg.get("informer", {})
        g = cfg.get("gnn", {})

        d_model = inf.get("d_model", 128)

        self.informer = InformerEncoder(
            d_model=d_model,
            n_heads=inf.get("n_heads", 8),
            n_encoder_layers=inf.get("n_encoder_layers", 3),
            ffn_dim=inf.get("ffn_dim", 256),
            dropout=inf.get("dropout", 0.1),
            activation=inf.get("activation", "gelu"),
            num_value_features=14,
            num_time_features=6,
            max_seq_len=inf.get("history_length", 288) + 10,
        )

        self.gnn = GNNEncoder(
            d_model=d_model,
            n_layers=g.get("n_layers", 2),
            n_heads=g.get("n_heads", 4),
            edge_dim=13,  # match graph_builder output
            dropout=g.get("dropout", 0.1),
        )

        self.decoder = FutureDecoder(
            d_model=d_model,
            hidden_dim=d_model,
            dropout=inf.get("dropout", 0.1),
            acc_bounds=(5e-5, 2e-4, 5e-5),  # [R,T,N] — T dominant for along-track error accumulation
        )

    def forward(
        self,
        history_features: torch.Tensor,           # (B, N, L, F)
        edge_index: torch.Tensor,                  # (2, E)
        edge_attr: torch.Tensor,                   # (E, F_e)
        physics_future: torch.Tensor,              # (B, N, H, 6)
        history_mask: Optional[torch.Tensor] = None,     # (B, N, L)
        time_features: Optional[torch.Tensor] = None,    # (B, N, L, F_t)
        future_time_features: Optional[torch.Tensor] = None,  # (B, N, H, F_t)
    ) -> torch.Tensor:
        """
        Returns:
            delta_a_rtn: (B, N, H, 3)
        """
        B, N, L, F = history_features.shape

        # 1. Informer: per-satellite time encoding
        # Reshape: (B, N, L, F) → (B*N, L, F)
        hist_flat = history_features.view(B * N, L, F)
        mask_flat = history_mask.view(B * N, L) if history_mask is not None else None
        time_flat = time_features.view(B * N, L, -1) if time_features is not None else None

        h_time = self.informer(hist_flat, time_flat, None, mask_flat)
        # h_time: (B*N, d_model) → (B, N, d_model)
        h_time = h_time.view(B, N, -1)

        # 2. GNN: spatial fusion
        # Handle batched edge_index: if 3D (B, 2, E), take first batch
        if edge_index.dim() == 3:
            edge_index = edge_index[0]  # (2, E)
        if edge_attr.dim() == 3:
            edge_attr = edge_attr[0]    # (E, F_e)
        h_space = self.gnn(h_time, edge_index, edge_attr)
        # h_space: (B, N, d_model)

        # 3. Decoder: multi-step RTN residual acceleration
        delta_a_rtn = self.decoder(h_space, physics_future, future_time_features)
        # (B, N, H, 3)

        return delta_a_rtn


def build_fusion_model(cfg: dict) -> PhysicsAIFusionModel:
    """从配置构建完整的融合模型。"""
    return PhysicsAIFusionModel(cfg)


def count_parameters(model: nn.Module) -> int:
    """统计模型参数总数。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
