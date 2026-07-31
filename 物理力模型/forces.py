"""轨道动力学力模型 (v2 — 高精度版)

支持的力模型:
  - 中心引力 (二体)
  - J2 摄动
  - 高阶球谐引力 (up to degree 20)
  - 指数大气阻力 (+ NRLMSISE-00 接口)
  - 太阳光压 (圆柱/圆锥阴影)
  - 日月三体引力 (astropy / 简化解析)
  - RTN 三轴经验加速度 / 连续小推力

约定:
  - r, v: (3,) ndarray, ECI 坐标系
  - 所有加速度单位为 m/s^2
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np


# ============================
# 中心引力
# ============================

def central_gravity(r: np.ndarray, mu: float) -> np.ndarray:
    """a = -mu * r / |r|^3"""
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-6:
        raise ValueError("位置范数过小。")
    return -mu * r / r_norm**3


# ============================
# J2 摄动
# ============================

def j2_perturbation(r: np.ndarray, mu: float, J2: float, Re: float) -> np.ndarray:
    """J2 带谐项摄动加速度 (解析公式)。"""
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-6:
        return np.zeros(3)
    x, y, z = r
    r2 = r_norm**2
    z2 = z**2
    factor = 1.5 * J2 * mu * Re**2 / r_norm**5
    return factor * np.array([
        x * (5.0 * z2 / r2 - 1.0),
        y * (5.0 * z2 / r2 - 1.0),
        z * (5.0 * z2 / r2 - 3.0),
    ])


# ============================
# 球谐引力 (高阶)
# ============================

from 物理力模型.gravity import (
    load_gravity_coefficients,
    spherical_harmonic_gravity,
    ecef_gravity_to_eci,
)
from 坐标时间系统.frames import eci_to_ecef
from 坐标时间系统.time_systems import gmst_from_jd_batch


def high_order_gravity_eci(
    r_eci: np.ndarray,
    t_jd: float,
    cfg: dict[str, Any],
) -> np.ndarray:
    """ECI 球谐引力加速度。

    自动使用配置中的 max_degree。degree=2 时回退到 J2 解析公式。
    """
    hp = cfg.get("high_precision", {})
    max_degree = hp.get("gravity_degree", 2)

    const = cfg["const"]
    mu = const["mu_earth"]
    Re = const["radius_earth"]

    if max_degree <= 2:
        # 使用 J2 解析公式 (更快)
        a_central = central_gravity(r_eci, mu)
        a_j2 = j2_perturbation(r_eci, mu, const["J2"], Re)
        return a_central + a_j2

    # 球谐引力 (通过 ECEF 计算)
    gmst = gmst_from_jd_batch(np.array([t_jd]))[0]
    r_ecef = eci_to_ecef(r_eci, gmst)

    C, S = load_gravity_coefficients(max_degree)
    a_ecef = spherical_harmonic_gravity(r_ecef, mu, Re, max_degree, C, S)

    # ECEF → ECI (位置和加速度的旋转相同)
    from 坐标时间系统.frames import ecef_to_eci
    return ecef_to_eci(a_ecef, gmst)


# ============================
# 大气阻力 (指数模型)
# ============================

def drag_acceleration_exponential(
    r: np.ndarray, v: np.ndarray, cfg: dict[str, Any],
    drag_scale: float = 1.0,
) -> np.ndarray:
    """指数大气阻力。

    Args:
        drag_scale: 阻力系数缩放因子 (用于参数估计, default=1.0)
    """
    const = cfg["const"]
    atm = cfg["atmosphere"]
    sc = cfg["spacecraft"]
    Re = const["radius_earth"]
    omega_earth = const["omega_earth"]

    altitude = np.linalg.norm(r) - Re
    rho = atm["reference_density"] * math.exp(
        -(altitude - atm["reference_altitude"]) / atm["scale_height"]
    )

    omega_vec = np.array([0.0, 0.0, omega_earth])
    v_atm = np.cross(omega_vec, r)
    v_rel = v - v_atm
    v_rel_norm = np.linalg.norm(v_rel)
    if v_rel_norm < 1e-12:
        return np.zeros(3)

    ballistic_coeff = sc["Cd"] * sc["area_drag"] / sc["mass"]
    return -0.5 * rho * ballistic_coeff * drag_scale * v_rel_norm * v_rel


# ============================
# 太阳光压
# ============================

def srp_acceleration_eci(
    r_sat: np.ndarray,
    t_jd: float,
    cfg: dict[str, Any],
) -> np.ndarray:
    """ECI 太阳光压加速度 (薄封装)。"""
    from 物理力模型.srp import srp_acceleration
    return srp_acceleration(r_sat, None, cfg, t_jd)


# ============================
# 日月三体引力
# ============================

def third_body_acceleration_eci(
    r_sat: np.ndarray,
    t_jd: float,
    cfg: dict[str, Any],
) -> np.ndarray:
    """ECI 日月三体引力总加速度 (薄封装)。"""
    from 物理力模型.third_body import third_body_sun_moon
    return third_body_sun_moon(r_sat, t_jd, cfg)


# ============================
# RTN → ECI 转换
# ============================

def rtn_thrust_to_eci(r: np.ndarray, v: np.ndarray, a_rtn: np.ndarray) -> np.ndarray:
    """RTN 加速度 → ECI 加速度。"""
    R_hat = r / np.linalg.norm(r)
    speed = np.linalg.norm(v)
    if speed < 1e-12:
        raise ValueError("速度范数过小。")
    T_hat = v / speed
    N_hat = np.cross(R_hat, T_hat)
    N_norm = np.linalg.norm(N_hat)
    if N_norm < 1e-12:
        raise ValueError("法向计算异常。")
    N_hat = N_hat / N_norm
    C = np.column_stack([R_hat, T_hat, N_hat])
    return C @ a_rtn


def along_track_thrust_to_eci(r: np.ndarray, v: np.ndarray, a_t: float) -> np.ndarray:
    """沿迹推力 → ECI (兼容旧接口)。"""
    speed = np.linalg.norm(v)
    if speed < 1e-12:
        return np.zeros(3)
    return a_t * v / speed


# ============================
# 总加速度 (完整版)
# ============================

def compute_total_acceleration_upgraded(
    t_seconds: float,
    t_jd: float,
    state: np.ndarray,
    cfg: dict[str, Any],
    a_rtn: np.ndarray | None = None,
    drag_scale: float = 1.0,
    model_name: str = "truth",
) -> np.ndarray:
    """完整高精度总加速度。

    力模型顺序:
      1. 中心引力 (必定包含)
      2. 球谐引力 (J2 或高阶, 可配置)
      3. 大气阻力 (可开关; drag_scale 可估计)
      4. 太阳光压 (可开关)
      5. 日月三体引力 (可开关)
      6. RTN 经验加速度 (a_rtn)

    Args:
        t_seconds: 从参考历元起算的秒数
        t_jd: 当前 Julian Date
        state: (6,) ECI [x, y, z, vx, vy, vz]
        cfg: 配置字典
        a_rtn: (3,) RTN 残差/经验加速度 [aR, aT, aN], None=零
        drag_scale: 阻力缩放因子 (1.0=名义)
        model_name: "truth" | "estimate"

    Returns:
        (6,) 状态导数 [vx, vy, vz, ax, ay, az]
    """
    r = state[:3]
    v = state[3:6]
    r_norm = np.linalg.norm(r)
    model = cfg["model"][model_name]
    const = cfg["const"]
    mu = const["mu_earth"]

    if r_norm <= const["radius_earth"]:
        raise RuntimeError(f"状态进入地球内部 (r={r_norm:.1f} m)。")

    # 1. 引力场 (中心 + 高阶球谐)
    max_deg = model.get("gravity_order", cfg.get("high_precision", {}).get("gravity_degree", 2))

    if max_deg <= 2:
        a_total = central_gravity(r, mu)
        if model.get("use_J2", False):
            a_total += j2_perturbation(r, mu, const["J2"], const["radius_earth"])
    else:
        # High-order EGM96 spherical harmonic gravity (fast recurrence)
        from 物理力模型.gravity_fast import (
            spherical_harmonic_gravity_fast, load_gravity_coefficients_fast,
        )
        # Central gravity
        a_total = central_gravity(r, mu)
        # Add perturbation from spherical harmonics
        gmst = gmst_from_jd_batch(np.array([t_jd]))[0]
        r_ecef = eci_to_ecef(r, gmst)
        C, S = load_gravity_coefficients_fast(max_deg)
        a_ecef = spherical_harmonic_gravity_fast(
            r_ecef, mu, const["radius_earth"], max_deg, C, S)
        from 坐标时间系统.frames import ecef_to_eci
        a_total += ecef_to_eci(a_ecef, gmst)

    # 2. 大气阻力 (NRLMSISE-00 or exponential fallback)
    if model.get("use_drag", False):
        use_nrl = model.get("use_nrlmsise", False)
        if use_nrl:
            from 物理力模型.nrlmsise00 import drag_acceleration_nrlmsise
            # JD to Unix: unix = (jd - 2440587.5) * 86400
            t_unix = (t_jd - 2440587.5) * 86400.0
            a_total += drag_acceleration_nrlmsise(r, v, t_unix, cfg, drag_scale)
        else:
            a_total += drag_acceleration_exponential(r, v, cfg, drag_scale)

    # 3. 太阳光压
    if model.get("use_srp", False):
        a_total += srp_acceleration_eci(r, t_jd, cfg)

    # 4. 日月三体
    if model.get("use_third_body", False):
        a_total += third_body_acceleration_eci(r, t_jd, cfg)

    # 5. RTN 经验加速度
    if a_rtn is not None and np.any(np.abs(a_rtn) > 1e-16):
        a_total += rtn_thrust_to_eci(r, v, a_rtn)

    return np.concatenate([v, a_total])


# ============================
# 兼容旧接口
# ============================

def compute_total_acceleration(
    t: float,
    state: np.ndarray,
    cfg: dict[str, Any],
    a_t: float = 0.0,
    model_name: str = "truth",
) -> np.ndarray:
    """旧接口: 仅支持沿迹加速度标量。向后兼容。"""
    a_rtn = np.array([0.0, a_t, 0.0]) if abs(a_t) > 1e-16 else None
    # Estimate JD from epoch
    import datetime
    epoch_dt = datetime.fromisoformat(
        cfg.get("time", {}).get("epoch_utc", "2026-07-01T00:00:00Z").replace("Z", "+00:00")
    )
    epoch_jd = (epoch_dt - datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)).total_seconds() / 86400.0 + 2451545.0
    t_jd = epoch_jd + t / 86400.0
    return compute_total_acceleration_upgraded(t, float(t_jd), state, cfg, a_rtn, 1.0, model_name)
