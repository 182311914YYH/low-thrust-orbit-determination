"""坐标系转换模块

支持:
  ECI/J2000 (GCRF) ↔ ECEF/ITRF (简化旋转)
  ECEF → ENU (站心东北天)
  ECI state → RTN basis
  Kepler ↔ Cartesian

约定:
  - 所有角度为 rad
  - 位置/速度为 m, m/s
  - 矩阵约定: r_ecef = R_eci2ecef @ r_eci
"""
from __future__ import annotations

import math

import numpy as np


# ---- ECI ↔ ECEF (简化: 仅考虑地球自转, 忽略极移/章动/岁差) ----


def eci_to_ecef(r_eci: np.ndarray, gmst_rad: float) -> np.ndarray:
    """ECI → ECEF 位置 (Z轴旋转 -GMST)。

    Args:
        r_eci: (3,) ECI 位置 [m]
        gmst_rad: Greenwich Mean Sidereal Time [rad]

    Returns:
        (3,) ECEF 位置 [m]
    """
    c = math.cos(gmst_rad)
    s = math.sin(gmst_rad)
    R = np.array([
        [ c, s, 0],
        [-s, c, 0],
        [ 0, 0, 1],
    ])
    return R @ r_eci


def ecef_to_eci(r_ecef: np.ndarray, gmst_rad: float) -> np.ndarray:
    """ECEF → ECI 位置 (Z轴旋转 +GMST)。"""
    c = math.cos(gmst_rad)
    s = math.sin(gmst_rad)
    R = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ])
    return R @ r_ecef


# ---- ECEF → ENU (站心东北天) ----


def ecef_to_enu(
    r_ecef: np.ndarray,
    station_ecef: np.ndarray,
    lat: float,
    lon: float,
) -> np.ndarray:
    """ECEF 位置 → 站心 ENU 位置。

    Args:
        r_ecef: (3,) or (N,3) 目标 ECEF 位置
        station_ecef: (3,) 测站 ECEF 坐标
        lat: 测站大地纬度 [rad]
        lon: 测站大地经度 [rad]

    Returns:
        (3,) or (N,3) ENU [East, North, Up] [m]
    """
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)

    R_ecef2enu = np.array([
        [-sin_lon,               cos_lon,              0],
        [-sin_lat * cos_lon,    -sin_lat * sin_lon,    cos_lat],
        [ cos_lat * cos_lon,     cos_lat * sin_lon,    sin_lat],
    ])

    rho_ecef = np.atleast_2d(r_ecef) - station_ecef.reshape(1, 3)
    rho_enu = rho_ecef @ R_ecef2enu.T
    if rho_enu.shape[0] == 1:
        return rho_enu[0]
    return rho_enu


def enu_to_ecef(
    rho_enu: np.ndarray,
    station_ecef: np.ndarray,
    lat: float,
    lon: float,
) -> np.ndarray:
    """站心 ENU → ECEF (R_ecef2enu 的转置即为其逆)。"""
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)

    R_enu2ecef = np.array([
        [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
        [ cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
        [ 0,        cos_lat,           sin_lat         ],
    ])

    rho_ecef = np.atleast_2d(rho_enu) @ R_enu2ecef.T
    result = rho_ecef + station_ecef.reshape(1, 3)
    if result.shape[0] == 1:
        return result[0]
    return result


# ---- 大地坐标 → ECEF ----


def geodetic_to_ecef(
    lat: float, lon: float, h: float,
    a: float = 6378137.0,
    f: float = 1.0 / 298.257223563,
) -> np.ndarray:
    """大地经纬高 → ECEF (WGS-84)。

    Args:
        lat: 大地纬度 [rad]
        lon: 大地经度 [rad]
        h: 大地高 [m]
        a: 椭球长半轴 [m]
        f: 椭球扁率

    Returns:
        (3,) ECEF [m]
    """
    e2 = f * (2 - f)
    sin_lat = math.sin(lat)
    N = a / math.sqrt(1 - e2 * sin_lat**2)
    cos_lat = math.cos(lat)
    cos_lon = math.cos(lon)
    sin_lon = math.sin(lon)

    return np.array([
        (N + h) * cos_lat * cos_lon,
        (N + h) * cos_lat * sin_lon,
        (N * (1 - e2) + h) * sin_lat,
    ])


# ---- RTN 基底 ----


def eci_to_rtn_basis(r: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """根据 ECI 位置和速度计算 RTN 正交基底。

    R (Radial):      r / |r|
    N (Normal):      (r × v) / |r × v|
    T (Track):       N × R

    Returns:
        (R_hat, T_hat, N_hat): 三个 (3,) 单位向量
    """
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-6:
        raise ValueError("位置范数过小，无法确定 R 方向。")
    R_hat = r / r_norm

    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    if h_norm < 1e-12:
        raise ValueError("角动量过小，无法确定 N 方向。")
    N_hat = h / h_norm

    T_hat = np.cross(N_hat, R_hat)
    T_norm = np.linalg.norm(T_hat)
    if T_norm < 1e-12:
        raise ValueError("T 方向计算异常。")
    T_hat = T_hat / T_norm

    return R_hat, T_hat, N_hat


def rtn_to_eci_matrix(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """计算 RTN → ECI 方向余弦矩阵 C_RTN^ECI。

    v_eci = C_RTN^ECI @ v_rtn
    """
    R_hat, T_hat, N_hat = eci_to_rtn_basis(r, v)
    return np.column_stack([R_hat, T_hat, N_hat])  # (3, 3)


def eci_error_to_rtn(
    r_ref: np.ndarray,
    v_ref: np.ndarray,
    error_eci: np.ndarray,
) -> np.ndarray:
    """将 ECI 误差投影到参考轨道的 RTN 坐标系。

    Args:
        r_ref: (N, 3) 参考位置
        v_ref: (N, 3) 参考速度
        error_eci: (N, 3) ECI 位置误差

    Returns:
        (N, 3) RTN 误差 [R, T, N]
    """
    N = r_ref.shape[0]
    error_rtn = np.zeros((N, 3))
    for i in range(N):
        C = rtn_to_eci_matrix(r_ref[i], v_ref[i])
        error_rtn[i] = C.T @ error_eci[i]
    return error_rtn


# ---- Kepler ↔ Cartesian ----


def kep2cart(
    a: float, e: float, i: float, raan: float, argp: float, nu: float, mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """经典轨道根数 → ECI 位置速度。

    Args:
        a: 半长轴 [m]
        e: 偏心率
        i: 倾角 [rad]
        raan: 升交点赤经 [rad]
        argp: 近地点幅角 [rad]
        nu: 真近点角 [rad]
        mu: 引力常数 [m^3/s^2]

    Returns:
        (r_eci, v_eci): (3,), (3,) [m, m/s]
    """
    p = a * (1 - e**2)
    r_pqw = (p / (1 + e * math.cos(nu))) * np.array([
        math.cos(nu), math.sin(nu), 0.0
    ])
    v_pqw = math.sqrt(mu / p) * np.array([
        -math.sin(nu), e + math.cos(nu), 0.0
    ])

    c_raan, s_raan = math.cos(raan), math.sin(raan)
    c_inc, s_inc = math.cos(i), math.sin(i)
    c_argp, s_argp = math.cos(argp), math.sin(argp)

    R3_W = np.array([
        [c_raan, -s_raan, 0],
        [s_raan,  c_raan, 0],
        [0,       0,      1],
    ])
    R1_i = np.array([
        [1, 0,      0],
        [0, c_inc, -s_inc],
        [0, s_inc,  c_inc],
    ])
    R3_w = np.array([
        [c_argp, -s_argp, 0],
        [s_argp,  c_argp, 0],
        [0,       0,      1],
    ])

    Q = R3_W @ R1_i @ R3_w
    return Q @ r_pqw, Q @ v_pqw


def cart2kep(r: np.ndarray, v: np.ndarray, mu: float) -> dict[str, float]:
    """ECI 位置速度 → 经典轨道根数。

    Returns:
        dict with keys: a, e, i, raan, arg_perigee, true_anomaly (rad)
    """
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)

    # 半长轴
    specific_energy = 0.5 * v_norm**2 - mu / r_norm
    if abs(specific_energy) < 1e-14:
        a = float("inf")
    else:
        a = -mu / (2 * specific_energy)

    # 偏心率矢量
    e_vec = ((v_norm**2 - mu / r_norm) * r - np.dot(r, v) * v) / mu
    e = float(np.linalg.norm(e_vec))

    # 倾角
    i = math.acos(max(-1.0, min(1.0, h[2] / h_norm)))

    # 升交点赤经
    n_vec = np.cross([0, 0, 1], h)
    n_norm = np.linalg.norm(n_vec)
    if n_norm > 1e-12:
        raan = math.acos(max(-1.0, min(1.0, n_vec[0] / n_norm)))
        if n_vec[1] < 0:
            raan = 2 * math.pi - raan
    else:
        raan = 0.0

    # 近地点幅角
    if e > 1e-12 and n_norm > 1e-12:
        argp = math.acos(
            max(-1.0, min(1.0, np.dot(n_vec, e_vec) / (n_norm * e)))
        )
        if e_vec[2] < 0:
            argp = 2 * math.pi - argp
    else:
        argp = 0.0

    # 真近点角
    if e > 1e-12:
        nu = math.acos(
            max(-1.0, min(1.0, np.dot(e_vec, r) / (e * r_norm)))
        )
        if np.dot(r, v) < 0:
            nu = 2 * math.pi - nu
    else:
        # 圆轨道: 用纬度幅角
        if n_norm > 1e-12:
            nu = math.acos(
                max(-1.0, min(1.0, np.dot(n_vec, r) / (n_norm * r_norm)))
            )
            if r[2] < 0:
                nu = 2 * math.pi - nu
        else:
            nu = math.acos(max(-1.0, min(1.0, r[0] / r_norm)))
            if r[1] < 0:
                nu = 2 * math.pi - nu

    return {
        "a": a,
        "e": e,
        "i": i,
        "raan": raan,
        "arg_perigee": argp,
        "true_anomaly": nu,
    }
