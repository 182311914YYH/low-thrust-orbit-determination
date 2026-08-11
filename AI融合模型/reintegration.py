"""轨道再积分模块

将网络预测的 RTN 残差加速度加入物理传播器，进行轨道再积分。

融合动力学方程:
    dx/dt = [v; a_phy + C_RTN^ECI * Δa_RTN]

其中:
    a_phy  = 物理传播器加速度
    Δa_RTN = Informer-GNN 预测的 RTN 残差加速度
    C_RTN^ECI = RTN → ECI 方向余弦矩阵 (根据当前状态实时计算)
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from 物理力模型.forces import (
    central_gravity,
    j2_perturbation,
    drag_acceleration_exponential,
    rtn_thrust_to_eci,
)


def compute_total_acceleration_fused(
    t: float,
    state: np.ndarray,
    cfg: dict[str, Any],
    delta_a_rtn: np.ndarray,
    a_t_estimated: float = 0.0,
    model_name: str = "truth",
) -> np.ndarray:
    """融合总加速度: 物理加速度 + AI预测残差加速度。

    a_total = a_physics + C_RTN^ECI · Δa_RTN

    Args:
        t: 当前时刻 [s]
        state: (6,) ECI 状态 [x, y, z, vx, vy, vz]
        cfg: 配置
        delta_a_rtn: (3,) RTN 残差加速度 [ΔaR, ΔaT, ΔaN]
        a_t_estimated: 估计的沿迹加速度 (用于物理模型)
        model_name: 模型名

    Returns:
        (6,) 状态导数
    """
    r = state[:3]
    v = state[3:6]
    r_norm = np.linalg.norm(r)
    model = cfg["model"][model_name]
    const = cfg["const"]
    mu = const["mu_earth"]

    if r_norm <= const["radius_earth"]:
        raise RuntimeError(f"传播状态进入地球内部 (r = {r_norm:.1f} m)。")

    # 物理加速度
    a_phy = np.zeros(3)

    # 中心引力
    a_phy += central_gravity(r, mu)

    # J2
    if model.get("use_J2", False):
        a_phy += j2_perturbation(r, mu, const["J2"], const["radius_earth"])

    # 大气阻力
    if model.get("use_drag", False):
        a_phy += drag_acceleration_exponential(r, v, cfg)

    # 沿迹估计推力 (物理模型中的经验加速度)
    if model.get("use_thrust", False) and abs(a_t_estimated) > 1e-16:
        speed = np.linalg.norm(v)
        if speed > 1e-12:
            a_phy += a_t_estimated * v / speed

    # AI 残差加速度 (RTN → ECI)
    if delta_a_rtn is not None and np.any(np.abs(delta_a_rtn) > 1e-16):
        a_fused = a_phy + rtn_thrust_to_eci(r, v, delta_a_rtn)
    else:
        a_fused = a_phy

    return np.concatenate([v, a_fused])


def reintegrate_orbit(
    cfg: dict[str, Any],
    t0: float,
    state0: np.ndarray,
    t_eval: np.ndarray,
    delta_a_rtn_sequence: np.ndarray,
    a_t_estimated: float = 0.0,
    model_name: str = "truth",
) -> np.ndarray:
    """融合再积分: 使用 AI 残差加速度修正物理传播器。

    在每个输出步长之间，残差加速度保持为零阶保持 (ZOH)。

    Args:
        cfg: 配置
        t0: 初始时刻 [s]
        state0: (6,) 初始 ECI 状态
        t_eval: (M,) 输出时刻 [s]
        delta_a_rtn_sequence: (M, 3) 或 (M-1, 3) RTN 残差加速度序列
        a_t_estimated: 估计的沿迹加速度
        model_name: 模型名

    Returns:
        (M, 6) 融合轨道状态
    """
    t_eval = np.asarray(t_eval, dtype=float).ravel()
    M = len(t_eval)
    state0 = np.asarray(state0, dtype=float).ravel()

    if len(delta_a_rtn_sequence) == M - 1:
        # 残差加速度在区间上定义 (常值 ZOH)
        delta_a_rtn_sequence = np.vstack([
            delta_a_rtn_sequence,
            delta_a_rtn_sequence[-1],
        ])

    if len(delta_a_rtn_sequence) != M:
        raise ValueError(
            f"delta_a_rtn_sequence 长度 ({len(delta_a_rtn_sequence)}) "
            f"必须为 M ({M}) 或 M-1 ({M-1})"
        )

    sim = cfg["sim"]
    rtol = sim["rel_tol"]
    atol_pos = max(sim["abs_tol"], 1e-3)
    atol_vel = max(sim["abs_tol"], 1e-6)

    state_fused = np.zeros((M, 6))
    state_fused[0] = state0

    for k in range(M - 1):
        t_start = t_eval[k]
        t_end = t_eval[k + 1]
        dt = t_end - t_start

        if dt <= 0:
            state_fused[k + 1] = state_fused[k]
            continue

        # ZOH: 在此子区间内使用常值 Δa_RTN
        delta_a = delta_a_rtn_sequence[k]

        def rhs(t, y):
            return compute_total_acceleration_fused(
                t, y, cfg, delta_a, a_t_estimated, model_name,
            )

        # 短区间内使用 RK4 或 DOP853
        result = solve_ivp(
            rhs,
            (t_start, t_end),
            state_fused[k],
            method="RK45",
            rtol=rtol,
            atol=[atol_pos] * 3 + [atol_vel] * 3,
            t_eval=[t_end] if abs(t_end - t_start) < 3600 else None,
        )

        if not result.success:
            raise RuntimeError(f"再积分失败 (k={k}): {result.message}")

        state_fused[k + 1] = result.y[:, -1]

    return state_fused


def physics_ai_fusion_propagate(
    cfg: dict[str, Any],
    t0: float,
    state0: np.ndarray,
    t_eval: np.ndarray,
    delta_a_rtn_sequence: np.ndarray,
    a_t_estimated: float = 0.0,
    model_name: str = "truth",
) -> np.ndarray:
    """物理 + AI 融合轨道传播 (便捷函数)。

    等价于 reintegrate_orbit()，提供统一调用接口。
    """
    return reintegrate_orbit(
        cfg, t0, state0, t_eval, delta_a_rtn_sequence,
        a_t_estimated, model_name,
    )
