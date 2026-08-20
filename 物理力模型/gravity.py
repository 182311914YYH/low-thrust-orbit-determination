"""高阶球谐引力场模型

使用归一化伴随 Legendre 函数 (scipy.special.lpmv) 计算球谐引力加速度。
支持可配置的地球引力场阶次 (degree/order)。

参考: Montenbruck & Gill, "Satellite Orbits", Section 3.2
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.special import lpmv  # 改用新版 lpmv，旧版 lpmn 已移除


# ---- EGM96 归一化系数 (仅前几阶，用于验证) ----
# C_nm, S_nm — fully normalized, dimensionless
# 完整系数应从外部文件加载

EGM96_LOW_ORDER: dict[tuple[int, int], tuple[float, float]] = {
    # (n, m): (C_nm, S_nm)
    # Degree 2 (zonal + sectorial + tesseral)
    (2, 0): (-0.484165371736e-3, 0.0),           # J2 = -sqrt(5)*C20
    (2, 1): (-0.186987635955e-9, 0.119528012031e-8),
    (2, 2): (0.243914352398e-5, -0.140016683654e-5),
    # Degree 3
    (3, 0): (0.957254173792e-6, 0.0),             # J3
    (3, 1): (0.202998882184e-5, 0.248513158716e-6),
    (3, 2): (0.904627768605e-6, -0.619025944215e-6),
    (3, 3): (0.721072657057e-6, 0.141435626958e-5),
    # Degree 4
    (4, 0): (0.539873863789e-6, 0.0),             # J4
    (4, 1): (-0.536321616971e-6, -0.473440265853e-6),
    (4, 2): (0.350694105785e-6, 0.662480026275e-6),
    (4, 3): (0.990771803829e-6, -0.200928368177e-6),
    (4, 4): (-0.188560801735e-6, 0.308853169333e-6),
}

# 补充到 degree 8 的 zonal 项 (非零 C_n0)
EGM96_ZONAL_SUPPLEMENT: dict[int, float] = {
    5: 0.685323475630e-7,   # C50
    6: -0.149957994714e-6,  # C60
    7: 0.905133793367e-7,   # C70
    8: 0.494750154184e-7,   # C80
}


def load_gravity_coefficients(
    max_degree: int = 8,
    coefficients: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """构造 C_nm, S_nm 系数矩阵 (n=0..max_degree, m=0..n)。

    Args:
        max_degree: 最大阶次
        coefficients: 补充系数, 格式 {(n,m): (C,S)}

    Returns:
        C, S: (max_degree+1, max_degree+1) ndarray (上三角有效)
    """
    N = max_degree + 1
    C = np.zeros((N, N))
    S = np.zeros((N, N))

    # 从内置 EGM96 低阶系数提取
    for (n, m), (cnm, snm) in EGM96_LOW_ORDER.items():
        if n <= max_degree:
            C[n, m] = cnm
            S[n, m] = snm

    # 补充 zonal 项
    for n, c_val in EGM96_ZONAL_SUPPLEMENT.items():
        if n <= max_degree:
            C[n, 0] = c_val

    # 外部提供的补充系数
    if coefficients is not None:
        for (n, m), (cnm, snm) in coefficients.items():
            if n <= max_degree and m <= n:
                C[n, m] = cnm
                S[n, m] = snm

    return C, S


def spherical_harmonic_gravity(
    r_ecef: np.ndarray,
    mu: float,
    R_earth: float,
    max_degree: int,
    C: np.ndarray,
    S: np.ndarray,
) -> np.ndarray:
    """球谐引力加速度 (ECEF 坐标系)。

    直接调用快速实现，已适配新版 scipy.special.lpmv。
    """
    return _spherical_harmonic_gradient_fast(r_ecef, mu, R_earth, max_degree, C, S)


def _spherical_harmonic_gradient_fast(
    r_ecef: np.ndarray,
    mu: float,
    R_earth: float,
    max_degree: int,
    C: np.ndarray,
    S: np.ndarray,
) -> np.ndarray:
    """球谐引力梯度 (快速实现)

    使用 lpmv 逐阶计算归一化 Legendre 函数。
    对于 max_degree <= 20，性能可接受。
    """
    x, y, z = r_ecef
    r = math.sqrt(x**2 + y**2 + z**2)
    if r < R_earth * 0.99:
        return np.zeros(3)

    sin_phi = z / r
    cos_phi = math.sqrt(x**2 + y**2) / r

    # longitude
    if cos_phi > 1e-15:
        cos_lambda = x / (r * cos_phi)
        sin_lambda = y / (r * cos_phi)
    else:
        cos_lambda = 1.0
        sin_lambda = 0.0

    # Precompute powers
    R_over_r = R_earth / r

    # dU/dr, dU/d(phi), dU/d(lambda)  partial derivatives
    dU_dr = 0.0
    dU_dphi = 0.0
    dU_dlambda = 0.0

    for n in range(2, max_degree + 1):
        r_pow = R_over_r ** n
        for m in range(n + 1):
            c_nm = C[n, m]
            s_nm = S[n, m]

            if abs(c_nm) < 1e-25 and abs(s_nm) < 1e-25:
                continue

            # Normalized associated Legendre Pbar_nm(sin_phi)
            # lpmv returns Ferrers function (unnormalized)
            # Conversion: Pbar_nm = P_nm * sqrt((2n+1)*(n-m)!/(n+m)!) * sqrt((2-delta_{0m}))
            delta = 1.0 if m == 0 else 2.0
            norm = math.sqrt((2 * n + 1) * math.factorial(n - m)
                             / math.factorial(n + m) * delta)

            P_bar = lpmv(m, n, sin_phi) * norm  # normalized

            # cos/sin terms
            cos_ml = math.cos(m * math.atan2(sin_lambda, cos_lambda))
            sin_ml = math.sin(m * math.atan2(sin_lambda, cos_lambda))

            cos_part = c_nm * cos_ml + s_nm * sin_ml
            sin_part = -c_nm * sin_ml + s_nm * cos_ml

            # dU/dr
            dU_dr += -(n + 1) * r_pow * P_bar * cos_part

            # dP/d(phi)
            # Use recurrence: dPbar_nm/dphi
            if sin_phi < 1.0 - 1e-15:
                dP = _dPbar_dphi(m, n, sin_phi, P_bar)
                dU_dphi += r_pow * dP * cos_part

            # dU/d(lambda)
            if m > 0:
                dU_dlambda += r_pow * P_bar * m * sin_part

    # Convert partials to Cartesian accelerations
    factor = mu / r

    # Spherical → Cartesian gradient mapping
    dU_dr *= factor
    dU_dphi *= factor / r
    dU_dlambda *= factor / (r * cos_phi + 1e-30)

    # Build the transformation matrix
    ax_r = dU_dr * cos_phi * cos_lambda
    ay_r = dU_dr * cos_phi * sin_lambda
    az_r = dU_dr * sin_phi

    ax_phi = dU_dphi * (-sin_phi * cos_lambda)
    ay_phi = dU_dphi * (-sin_phi * sin_lambda)
    az_phi = dU_dphi * cos_phi

    ax_lambda = dU_dlambda * (-sin_lambda)
    ay_lambda = dU_dlambda * cos_lambda
    az_lambda = 0.0

    return np.array([
        ax_r + ax_phi + ax_lambda,
        ay_r + ay_phi + ay_lambda,
        az_r + az_phi + az_lambda,
    ])


def _dPbar_dphi(m: int, n: int, sin_phi: float, Pbar_nm: float) -> float:
    """计算归一化伴随 Legendre 函数对 phi 的导数。

    使用递推关系:
    dPbar_nm/dphi = n*sin_phi/cos_phi * Pbar_nm
                    - sqrt((2n+1)/(2n-1)*(n+m)*(n-m)) / cos_phi * Pbar_{n-1,m}
    """
    cos_phi = math.sqrt(max(0.0, 1.0 - sin_phi**2))
    if cos_phi < 1e-15:
        # 极点: 用极限
        return 0.0

    if n == 0:
        return 0.0

    if n == m:
        # dP_mm/dphi = m*tan(phi)*P_mm = m*sin_phi/cos_phi*Pbar
        return m * sin_phi / cos_phi * Pbar_nm

    # Pbar_{n-1,m}
    delta = 1.0 if m == 0 else 2.0
    norm_prev = math.sqrt((2 * (n - 1) + 1) * math.factorial(n - 1 - m)
                          / math.factorial(n - 1 + m) * delta)
    P_prev_bar = lpmv(m, n - 1, sin_phi) * norm_prev

    coef = math.sqrt((2.0 * n + 1) / (2.0 * n - 1) * (n + m) * (n - m))

    return (n * sin_phi / cos_phi * Pbar_nm - coef / cos_phi * P_prev_bar)


def compute_gravity_ecef(
    r_ecef: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray:
    """统一接口: 计算 ECEF 球谐引力加速度。

    自动使用配置中指定的重力阶次。
    """
    hp = cfg.get("high_precision", {})
    max_degree = hp.get("gravity_degree", 2)

    C, S = load_gravity_coefficients(max_degree)

    return spherical_harmonic_gravity(
        r_ecef,
        cfg["const"]["mu_earth"],
        cfg["const"]["radius_earth"],
        max_degree,
        C, S,
    )


def ecef_gravity_to_eci(
    r_eci: np.ndarray,
    gmst_rad: float,
    mu: float,
    R_earth: float,
    max_degree: int,
    C: np.ndarray,
    S: np.ndarray,
) -> np.ndarray:
    """通过 ECEF 球谐引力计算 → 转回 ECI。

    Args:
        r_eci: (3,) ECI 位置 [m]
        gmst_rad: GMST [rad]
        mu, R_earth: 引力常数和地球半径
        max_degree: 最大阶次
        C, S: 系数矩阵

    Returns:
        (3,) ECI 加速度 [m/s^2]
    """
    from 坐标时间系统.frames import eci_to_ecef, ecef_to_eci

    r_ecef = eci_to_ecef(r_eci, gmst_rad)
    a_ecef = spherical_harmonic_gravity(r_ecef, mu, R_earth, max_degree, C, S)
    return ecef_to_eci(a_ecef, gmst_rad)
