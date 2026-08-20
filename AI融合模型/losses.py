"""Multi-objective PINN loss for orbit prediction with unknown thrust.

Loss components (all in normalized coordinates unless noted):
  1. L_position:   MSE(predicted_r, observed_r) — core data term
  2. L_endpoint:   MSE(predicted_r[-1], observed_r[-1]) — terminal accuracy
  3. L_smooth:     mean(|d(Δa)/dt|²) — thrust temporal smoothness
  4. L_energy:     |dE/dt_actual - v·a_thrust|² — physics consistency
  5. L_magnitude:  mean(|Δa|²) — L2 penalty on thrust magnitude
  6. L_boundary:   soft penalty for exceeding thrust bounds
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class PINNLoss(nn.Module):
    """Composite physics-informed loss for orbit prediction."""

    def __init__(self,
                 w_position: float = 1.0,
                 w_endpoint: float = 2.0,
                 w_smooth: float = 0.1,
                 w_energy: float = 0.05,
                 w_magnitude: float = 0.01,
                 w_boundary: float = 0.5,
                 mu: float = 3.986004418e14,
                 thrust_smooth_sigma: float = 300.0,  # smoothing timescale [s]
                 thrust_max: tuple = (5e-5, 2e-4, 5e-5),  # max plausible thrust [m/s²]
                 use_huber: bool = True,
                 huber_delta: float = 0.001):  # ~6m in normalized coords
        super().__init__()
        self.w_pos = w_position
        self.w_end = w_endpoint
        self.w_sm = w_smooth
        self.w_en = w_energy
        self.w_mag = w_magnitude
        self.w_bnd = w_boundary
        self.mu = mu
        self.smooth_sigma = thrust_smooth_sigma
        self.register_buffer("thrust_max", torch.tensor(thrust_max))
        self.use_huber = use_huber
        self.huber_delta = huber_delta

    def _pos_loss(self, pred: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor | None = None) -> torch.Tensor:
        """Position error loss.

        pred, target: (B, H, 3)
        mask: (B, H) optional, 1=valid
        """
        diff = pred - target
        if self.use_huber:
            loss = nn.functional.huber_loss(diff, torch.zeros_like(diff),
                                            delta=self.huber_delta, reduction='none')
        else:
            loss = diff ** 2

        loss = loss.mean(dim=-1)  # average over x,y,z

        if mask is not None:
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()

        return loss

    def endpoint_loss(self, pred: torch.Tensor, target: torch.Tensor,
                      n_steps: torch.Tensor | None = None) -> torch.Tensor:
        """Loss at final prediction step (most important for forecasting)."""
        B, H, _ = pred.shape
        if n_steps is not None:
            # Use actual final step index per sample
            indices = (n_steps - 1).clamp(min=0).long()
            pred_end = pred[torch.arange(B, device=pred.device), indices, :]
            target_end = target[torch.arange(B, device=pred.device), indices, :]
        else:
            pred_end = pred[:, -1, :]
            target_end = target[:, -1, :]

        diff = pred_end - target_end
        if self.use_huber:
            return nn.functional.huber_loss(diff, torch.zeros_like(diff),
                                            delta=self.huber_delta)
        return (diff ** 2).mean()

    def smoothness_loss(self, thrust: torch.Tensor, dt: float) -> torch.Tensor:
        """Penalize rapid changes in thrust.

        thrust: (B, T, 3) predicted Δa_RTN over time
        dt: time step between successive thrust predictions [s]
        """
        if thrust.shape[1] < 2:
            return torch.tensor(0.0, device=thrust.device)

        # d(Δa)/dt via finite differences
        da_dt = (thrust[:, 1:, :] - thrust[:, :-1, :]) / dt

        # Gaussian-weighted: allow gradual changes, penalize abrupt ones
        # Weight = 1 - exp(-dt² / (2*sigma²)) → emphasizes large jumps
        weight = 1.0 - math.exp(-0.5 * (dt / self.smooth_sigma) ** 2)
        return weight * (da_dt ** 2).mean()

    def energy_loss(self, pred_pos: torch.Tensor, pred_vel: torch.Tensor,
                    thrust_eci: torch.Tensor, dt: float,
                    mask: torch.Tensor | None = None) -> torch.Tensor:
        """Physics consistency: dE/dt should equal work done by thrust.

        For a non-conservative system:
          dE/dt = d/dt(½mv² - μm/|r|) = m * v · a_nonconservative

        This loss penalizes violations of the work-energy theorem.
        """
        B, H, _ = pred_pos.shape

        # Specific orbital energy (per kg): e = ½v² - μ/|r|
        v2 = (pred_vel ** 2).sum(dim=-1)  # (B, H)
        r_norm = torch.norm(pred_pos, dim=-1).clamp(min=1e-3)  # (B, H)
        e_specific = 0.5 * v2 - self.mu / r_norm  # (B, H)

        # dE/dt via finite differences
        de_dt = (e_specific[:, 1:] - e_specific[:, :-1]) / dt  # (B, H-1)

        # Work done by thrust: v · a_thrust (per kg)
        v_mid = 0.5 * (pred_vel[:, 1:, :] + pred_vel[:, :-1, :])  # midpoint velocity
        work_rate = (v_mid * thrust_eci[:, 1:, :]).sum(dim=-1)  # (B, H-1)

        residual = de_dt - work_rate

        if mask is not None:
            mask_mid = mask[:, 1:] * mask[:, :-1]
            loss = ((residual ** 2) * mask_mid).sum() / mask_mid.sum().clamp(min=1)
        else:
            loss = (residual ** 2).mean()

        return loss

    def magnitude_loss(self, thrust: torch.Tensor) -> torch.Tensor:
        """L2 penalty on thrust magnitude — prefer minimal thrust solutions."""
        return (thrust ** 2).mean()

    def boundary_loss(self, thrust: torch.Tensor) -> torch.Tensor:
        """Soft penalty for exceeding physically plausible thrust bounds.

        Uses softplus: penalty = log(1 + exp(|Δa| - max)) — zero when in bounds.
        """
        max_val = self.thrust_max.unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
        excess = torch.abs(thrust) - max_val  # positive only when |Δa| > max
        penalty = nn.functional.softplus(excess, beta=10.0)  # sharp but differentiable
        return penalty.mean()

    def forward(self,
                pred_pos: torch.Tensor,
                target_pos: torch.Tensor,
                thrust_rtn: torch.Tensor,
                thrust_eci: torch.Tensor | None = None,
                pred_vel: torch.Tensor | None = None,
                pos_mask: torch.Tensor | None = None,
                n_steps: torch.Tensor | None = None,
                dt: float = 60.0,
                return_components: bool = False,
                ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Compute composite PINN loss.

        Args:
            pred_pos: (B, H, 3) predicted positions [m]
            target_pos: (B, H, 3) observed positions [m]
            thrust_rtn: (B, T, 3) predicted Δa in RTN [m/s²]
            thrust_eci: (B, T, 3) optional, thrust in ECI for energy loss
            pred_vel: (B, H, 3) optional, predicted velocities for energy loss
            pos_mask: (B, H) optional valid-position mask
            n_steps: (B,) optional actual sequence lengths
            dt: time step [s]
            return_components: if True, return (total_loss, dict_of_components)

        Returns:
            total_loss, or (total_loss, components_dict)
        """
        components = {}

        # 1. Position loss (primary)
        l_pos = self._pos_loss(pred_pos, target_pos, pos_mask)
        components["position"] = l_pos.item() if isinstance(l_pos, torch.Tensor) else l_pos

        # 2. Endpoint loss
        l_end = self.endpoint_loss(pred_pos, target_pos, n_steps)
        components["endpoint"] = l_end.item()

        # 3. Thrust smoothness
        l_sm = self.smoothness_loss(thrust_rtn, dt)
        components["smoothness"] = l_sm.item() if isinstance(l_sm, torch.Tensor) else l_sm

        # 4. Energy conservation (only if velocities provided)
        l_en = torch.tensor(0.0, device=pred_pos.device)
        if pred_vel is not None and thrust_eci is not None:
            l_en = self.energy_loss(pred_pos, pred_vel, thrust_eci, dt, pos_mask)
        components["energy"] = l_en.item() if isinstance(l_en, torch.Tensor) else l_en

        # 5. Thrust magnitude
        l_mag = self.magnitude_loss(thrust_rtn)
        components["magnitude"] = l_mag.item()

        # 6. Boundary
        l_bnd = self.boundary_loss(thrust_rtn)
        components["boundary"] = l_bnd.item()

        total = (self.w_pos * l_pos + self.w_end * l_end +
                 self.w_sm * l_sm + self.w_en * l_en +
                 self.w_mag * l_mag + self.w_bnd * l_bnd)

        if return_components:
            return total, components
        return total


# ===== Simple Test =====

if __name__ == "__main__":
    B, H, T = 4, 720, 720
    pred_pos = torch.randn(B, H, 3) * 1000
    target_pos = torch.randn(B, H, 3) * 1000
    thrust_rtn = torch.randn(B, T, 3) * 1e-5
    thrust_eci = torch.randn(B, T, 3) * 1e-5
    pred_vel = torch.randn(B, H, 3) * 100

    loss_fn = PINNLoss()
    total, comps = loss_fn(
        pred_pos, target_pos, thrust_rtn, thrust_eci, pred_vel,
        dt=60.0, return_components=True,
    )
    print(f"Total loss: {total.item():.4f}")
    for k, v in comps.items():
        print(f"  {k}: {v:.4f}")
