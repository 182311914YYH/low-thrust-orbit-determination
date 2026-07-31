"""轨道数值传播器 (v2 — 全高精度力模型)

使用 SciPy DOP853/RK45 进行高精度数值积分。
支持时变力模型 (SRP, 三体) 通过 JD 参数传递。
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from .forces import compute_total_acceleration_upgraded, compute_total_acceleration

# J2000 epoch as JD
JD_J2000 = 2451545.0
SECONDS_PER_DAY = 86400.0


def _jd_from_seconds(t_seconds: float, epoch_jd: float) -> float:
    """从参考历元秒数计算 JD。"""
    return epoch_jd + t_seconds / SECONDS_PER_DAY


def _get_epoch_jd(cfg: dict[str, Any]) -> float:
    """从配置获取 epoch 的 JD。"""
    epoch_str = cfg.get("time", {}).get("epoch_utc", "2026-07-01T00:00:00Z")
    epoch_dt = datetime.fromisoformat(epoch_str.replace("Z", "+00:00"))
    return (epoch_dt - datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)).total_seconds() / SECONDS_PER_DAY + JD_J2000


class OrbitalPropagator:
    """轨道传播器 (支持旧接口 + 新高精度接口)。

    使用例:
      # 旧接口 (仅沿迹推力)
      state = prop.propagate(t0, state0, t_eval, a_t=5e-5)

      # 新接口 (RTN 3轴 + drag_scale)
      state = prop.propagate_upgraded(
          t0, state0, t_eval,
          a_rtn=np.array([1e-7, 5e-5, -2e-7]),
          drag_scale=0.95,
      )
    """

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.sim = cfg["sim"]
        self._rtol = cfg["sim"]["rel_tol"]
        self._atol_pos = max(cfg["sim"]["abs_tol"], 1e-3)
        self._atol_vel = max(cfg["sim"]["abs_tol"], 1e-6)
        self._epoch_jd = _get_epoch_jd(cfg)

    # ---- 旧接口 (向后兼容) ----

    def propagate(
        self, t0: float, state0: np.ndarray, t_eval: np.ndarray,
        a_t: float = 0.0, model_name: str = "truth",
    ) -> np.ndarray:
        """旧接口: 仅沿迹推力。"""
        a_rtn = np.array([0.0, a_t, 0.0]) if abs(a_t) > 1e-16 else None
        return self.propagate_upgraded(t0, state0, t_eval, a_rtn, 1.0, model_name)

    # ---- 新高精度接口 ----

    def propagate_upgraded(
        self, t0: float, state0: np.ndarray, t_eval: np.ndarray,
        a_rtn: np.ndarray | None = None, drag_scale: float = 1.0,
        model_name: str = "truth",
    ) -> np.ndarray:
        """高精度轨道传播。

        Args:
            t0: 初始时刻 [s from epoch]
            state0: (6,) ECI 初始状态
            t_eval: (M,) 输出时刻 [s]
            a_rtn: (3,) RTN 三轴经验加速度 [aR, aT, aN] or None
            drag_scale: 阻力缩放因子 (1.0=名义, 用于 OD)
            model_name: "truth" | "estimate"

        Returns:
            (M, 6) 状态序列
        """
        state0 = np.asarray(state0, dtype=float).ravel()
        if state0.size != 6:
            raise ValueError(f"state0 must be 6-dim, got {state0.size}")
        t_eval = np.asarray(t_eval, dtype=float).ravel()
        if len(t_eval) == 0:
            return np.zeros((0, 6))
        if np.any(t_eval < t0 - 1e-10):
            raise ValueError("All t_eval must be >= t0 (no backward propagation)")

        # Sort & dedup
        sort_idx = np.argsort(t_eval)
        t_sorted = t_eval[sort_idx]
        t_unique, unique_inv = np.unique(t_sorted, return_inverse=True)

        if t_unique[0] > t0 + 1e-12:
            t_integrate = np.concatenate([[t0], t_unique])
            remove_first = True
        else:
            t_integrate = t_unique
            remove_first = False

        t_span = (t_integrate[0], t_integrate[-1])
        epoch_jd = self._epoch_jd

        def rhs(t_s, y):
            t_jd = _jd_from_seconds(float(t_s), epoch_jd)
            return compute_total_acceleration_upgraded(
                float(t_s), float(t_jd), y, self.cfg,
                a_rtn, drag_scale, model_name,
            )

        method = self.sim.get("integrator", "DOP853")
        solver_method = method if method in ("RK45", "LSODA", "DOP853") else "DOP853"

        result = solve_ivp(
            rhs, t_span, state0, method=solver_method,
            t_eval=t_integrate if len(t_integrate) > 2 else None,
            rtol=self._rtol,
            atol=[self._atol_pos]*3 + [self._atol_vel]*3,
            dense_output=(len(t_integrate) <= 2),
        )
        if not result.success:
            raise RuntimeError(f"Integration failed: {result.message}")

        if result.y.shape[1] == len(t_integrate):
            state_int = result.y.T
        else:
            state_int = np.array([result.sol(ti) for ti in t_integrate])

        if remove_first:
            state_unique = state_int[1:]
        else:
            state_unique = state_int

        state_sorted = state_unique[unique_inv]
        state_result = np.zeros_like(state_sorted)
        state_result[sort_idx] = state_sorted
        return state_result


    # ---- Fixed-step RK4 (fast mode for physics cache) ----

    def propagate_fixed_rk4(
        self, t0: float, state0: np.ndarray, t_eval: np.ndarray,
        a_rtn: np.ndarray | None = None, drag_scale: float = 1.0,
        model_name: str = "truth",
    ) -> np.ndarray:
        """Fast fixed-step RK4 propagation — exactly 4 RHS evals per step.

        For physics cache computation where DOP853's adaptive overhead
        is unnecessary. ~100x faster for degree-20 gravity.

        Args:
            t0, state0, t_eval, a_rtn, drag_scale, model_name: same as propagate_upgraded

        Returns:
            (M, 6) state trajectory
        """
        state0 = np.asarray(state0, dtype=float).ravel()
        t_eval = np.asarray(t_eval, dtype=float).ravel()
        if len(t_eval) == 0:
            return np.zeros((0, 6))

        M = len(t_eval)
        epoch_jd = self._epoch_jd
        result = np.zeros((M, 6))
        state = state0.copy()

        # Track current time
        t_current = t0
        out_idx = 0

        for i, t_target in enumerate(t_eval):
            while t_current < t_target - 1e-12:
                dt = min(t_target - t_current, 60.0)
                if dt <= 0:
                    break

                # RK4 — must pass exact state for each sub-step
                def rhs(s, st):
                    t_jd = _jd_from_seconds(t_current + s, epoch_jd)
                    return compute_total_acceleration_upgraded(
                        t_current + s, t_jd, st, self.cfg,
                        a_rtn, drag_scale, model_name)

                k1 = rhs(0.0, state)
                k2 = rhs(0.5 * dt, state + 0.5 * dt * k1)
                k3 = rhs(0.5 * dt, state + 0.5 * dt * k2)
                k4 = rhs(dt, state + dt * k3)

                state = state + dt / 6.0 * (k1 + 2*k2 + 2*k3 + k4)
                t_current += dt

            result[i] = state.copy()

        return result


# ---- 便捷函数 ----

def propagate_orbit(
    cfg: dict[str, Any], t0: float, state0: np.ndarray, a_t: float,
    t_eval: np.ndarray, model_name: str = "truth",
) -> np.ndarray:
    """便捷: 旧接口传播。"""
    prop = OrbitalPropagator(cfg)
    return prop.propagate(t0, state0, t_eval, a_t, model_name)


def propagate_orbit_upgraded(
    cfg: dict[str, Any], t0: float, state0: np.ndarray,
    t_eval: np.ndarray,
    a_rtn: np.ndarray | None = None, drag_scale: float = 1.0,
    model_name: str = "truth",
) -> np.ndarray:
    """便捷: 高精度传播。"""
    prop = OrbitalPropagator(cfg)
    return prop.propagate_upgraded(t0, state0, t_eval, a_rtn, drag_scale, model_name)


def propagate_orbit_batch(
    cfg: dict[str, Any], t0_list: np.ndarray, state0_array: np.ndarray,
    a_t_array: np.ndarray, t_eval: np.ndarray, model_name: str = "truth",
) -> np.ndarray:
    """批量传播 (旧接口)。"""
    B = len(t0_list)
    M = len(t_eval)
    results = np.zeros((B, M, 6))
    prop = OrbitalPropagator(cfg)
    for b in range(B):
        results[b] = prop.propagate(t0_list[b], state0_array[b], t_eval, a_t_array[b], model_name)
    return results
