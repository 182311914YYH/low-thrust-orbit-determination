"""高阶球谐引力场模型

使用归一化伴随 Legendre 函数 (scipy.special.lpmn) 计算球谐引力加速度。
支持可配置的地球引力场阶次 (degree/order)。

参考: Montenbruck & Gill, "Satellite Orbits", Section 3.2
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.special import lpmn  # normalized associated Legendre


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

    U(r,phi,lambda) = mu/r * [1 + sum_{n=2}^N (R/r)^n sum_{m=0}^n
        Pbar_{nm}(sin(phi)) * (C_{nm}*cos(m*lambda) + S_{nm}*sin(m*lambda))]

    a = grad(U) 使用 Cunningham 递推 (Montenbruck 3.2.4)。

    Args:
        r_ecef: (3,) ECEF 位置 [m]
        mu: 引力常数 [m^3/s^2]
        R_earth: 地球赤道半径 [m]
        max_degree: 最大阶次 N
        C: (N+1, N+1) C_nm 矩阵
        S: (N+1, N+1) S_nm 矩阵

    Returns:
        (3,) ECEF 加速度 [m/s^2]
    """
    x, y, z = r_ecef
    r = math.sqrt(x**2 + y**2 + z**2)

    if r < 1e-6:
        return np.zeros(3)

    # 球坐标
    # phi = geocentric latitude, lambda = longitude
    lambda_ = math.atan2(y, x)
    sin_phi = z / r
    cos_phi = math.sqrt(x**2 + y**2) / r

    # Cunningham 递推计算 V_nm, W_nm (非归一化)
    # 使用归一化 Legendre 函数
    N = max_degree + 1

    # 计算归一化 Legendre 函数 Pbar_nm(sin_phi)
    # lpmn 返回 (P_nm, dP_nm): (N+1, N+1) 数组 (非归一化! scipy lpmn 返回的是非归一化)
    # 需要手动归一化
    P_unnorm, dP_unnorm = lpmn(max_degree, max_degree, sin_phi)
    # P_unnorm[n,m] = 非归一化; dP_unnorm[n,m] = dP/d(phi)

    # 递归: 计算归一化 Pbar 和 dPbar/d(phi)
    # V_nm 和 W_nm (Cunningham formulation)
    # 使用 Montenbruck 的递推关系

    # 预计算 cos(m*lambda), sin(m*lambda)
    cos_ml = np.ones(N)
    sin_ml = np.zeros(N)
    for m in range(1, N):
        cos_ml[m] = cos_ml[m - 1] * math.cos(lambda_) - sin_ml[m - 1] * math.sin(lambda_)
        sin_ml[m] = sin_ml[m - 1] * math.cos(lambda_) + cos_ml[m - 1] * math.sin(lambda_)

    # 比例因子
    R_over_r = R_earth / r
    factor = mu / r

    # 加速度递推 (Montenbruck Algorithm 3.2)
    a_x, a_y, a_z = 0.0, 0.0, 0.0

    for n in range(2, N):
        R_pow = R_over_r ** n
        for m in range(0, n + 1):
            # 归一化系数
            # lpmn returns unnormalized legendre functions
            # Normalize: Pbar_nm = P_nm * sqrt((2n+1)*(n-m)!/(n+m)!) * (2-delta_{0m})
            # But scipy lpmn returns standard (not Schmidt) normalization

            # For scipy.special.lpmn:
            # P_n^m returned, these are Ferrers functions (unnormalized)
            # We need to normalize for geopotential use

            # Normalization factor
            delta_0m = 2.0 if m == 0 else 1.0
            norm_fac = math.sqrt(
                (2 * n + 1) * math.factorial(n - m) / math.factorial(n + m)
            ) * delta_0m

            Pnm = P_unnorm[n, m] * norm_fac
            dPnm = dP_unnorm[n, m] * norm_fac

            C_nm = C[n, m]
            S_nm = S[n, m]

            cos_term = C_nm * cos_ml[m] + S_nm * sin_ml[m]
            sin_term = S_nm * cos_ml[m] - C_nm * sin_ml[m]

            # Radial component
            radial = -(n + 1) * R_pow * Pnm * cos_term

            # 累积三角函数部分 (会在下方组装为直角坐标加速度)
            a_x += R_pow * (radial * x / r
                            + factor * R_pow * (dPnm * cos_phi * x / r * cos_term
                            - m * Pnm * sin_term * (-y) / (x**2 + y**2 + 1e-30)))

    # Simplified: use direct gradient formula instead
    # Re-implement with the standard recurrence for the gravity potential gradient

    # Clear and use direct method
    a_x, a_y, a_z = 0.0, 0.0, 0.0

    # Recompute using the recurrence for V_nm and W_nm
    # V_nm = (R/r)^{n+1} * Pbar_nm(sin_phi) * cos(m*lambda)
    # W_nm = (R/r)^{n+1} * Pbar_nm(sin_phi) * sin(m*lambda)

    V = np.zeros((N, N))
    W = np.zeros((N, N))

    for n in range(2, N):
        R_pow_n = R_over_r ** (n + 1)
        for m in range(n + 1):
            delta_0m = 2.0 if m == 0 else 1.0
            norm_fac = math.sqrt(
                (2 * n + 1) * math.factorial(n - m) / math.factorial(n + m)
            ) * delta_0m
            Pbar = P_unnorm[n, m] * norm_fac

            V[n, m] = R_pow_n * Pbar * cos_ml[m]
            W[n, m] = R_pow_n * Pbar * sin_ml[m]

    # Accelerations (Montenbruck Eq. 3.33, simplified for ECEF)
    for n in range(2, N):
        for m in range(n + 1):
            C_nm = C[n, m]
            S_nm = S[n, m]

            if abs(C_nm) < 1e-30 and abs(S_nm) < 1e-30:
                continue

            if m == 0:
                # Zonal term: only radial and z components affected
                # dPbar/d(phi) needed
                delta_0m = 2.0
                norm_fac = math.sqrt((2*n+1) * math.factorial(n-m) / math.factorial(n+m)) * delta_0m
                dPbar = dP_unnorm[n, m] * norm_fac

                a_x += factor * C_nm * V[n + 1, 1] if n + 1 < N and 1 <= n + 1 else 0
                # Use recurrence for zonal: simpler in cartesian from
                # a_x = factor * sum C_n0 * [V_{n+1,1}]
                # Practical: compute using explicit zonal formula
                pass
            else:
                pass

    # ---- 简化方案: 使用直接坐标梯度 (球谐引力 pot 的梯度) ----
    # 对于低阶场, 使用 Explicit zonal formula + low-order tesseral

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

    使用脚本预计算的递推关系。
    对于 max_degree <= 20，性能可接受。
    """
    from scipy.special import lpmv  # associated Legendre values (not normalized)

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
            cm = c_nm * math.cos(m * math.acos(cos_lambda)) if m > 0 else c_nm
            sm = s_nm * math.sin(m * math.acos(cos_lambda)) if m > 0 else 0.0
            # Better:
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
    # da = grad(U) in spherical → Cartesian
    factor = mu / r

    # Spherical → Cartesian gradient mapping
    # a = [dx, dy, dz]^T = R * [dU/dr, 1/r*dU/dphi, 1/(r*cos_phi)*dU/dlambda]^T
    dU_dr *= factor
    dU_dphi *= factor / r
    dU_dlambda *= factor / (r * cos_phi + 1e-30)

    # Build the transformation matrix
    # d/dr → Cartesian
    ax_r = dU_dr * cos_phi * cos_lambda
    ay_r = dU_dr * cos_phi * sin_lambda
    az_r = dU_dr * sin_phi

    # d/d(phi) → Cartesian
    ax_phi = dU_dphi * (-sin_phi * cos_lambda)
    ay_phi = dU_dphi * (-sin_phi * sin_lambda)
    az_phi = dU_dphi * cos_phi

    # d/d(lambda) → Cartesian
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

    from scipy.special import lpmv

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
