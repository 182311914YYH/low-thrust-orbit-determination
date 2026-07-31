"""太阳光压 (Solar Radiation Pressure) 模型

a_srp = nu * P0 * Cr * (A/m) * (AU/|r-r_sun|)^2 * (r - r_sun)/|r - r_sun|

其中:
  nu = 阴影函数 (0=本影, 1=全光照)
  P0 = 太阳辐射压力常数 ~4.56e-6 N/m^2 at 1 AU
  Cr = 反射系数
  A/m = 面质比 [m^2/kg]

阴影模型: 圆柱形 (cylindrical) 或 圆锥形 (conical)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

# 太阳辐射压力常数 at 1 AU [N/m^2 = Pa]
P0_SUN = 4.56e-6

# 天文单位 [m]
AU = 149597870700.0

# 太阳半径 [m]
R_SUN = 695700000.0


def sun_position_eci(t_jd: float) -> np.ndarray:
    """计算太阳在 ECI 中的近似位置。

    使用简化解析公式 (精度 ~0.01 deg)。

    Args:
        t_jd: Julian Date

    Returns:
        (3,) ECI 位置 [m]
    """
    # Julian centuries from J2000
    T = (t_jd - 2451545.0) / 36525.0

    # Mean longitude [deg]
    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T**2
    # Mean anomaly [deg]
    M = 357.52911 + 35999.05029 * T - 0.0001537 * T**2

    L0_rad = math.radians(L0 % 360.0)
    M_rad = math.radians(M % 360.0)

    # Ecliptic longitude [rad] (simplified)
    lambda_sun = L0_rad + math.radians(1.914602) * math.sin(M_rad) \
                 + math.radians(0.019993) * math.sin(2 * M_rad)

    # Obliquity [deg]
    epsilon = math.radians(23.439291 - 0.0130042 * T)

    # Distance [AU]
    R = 1.000140612 - 0.016708617 * math.cos(M_rad) - 0.000139589 * math.cos(2 * M_rad)

    r_au = R * AU

    # Ecliptic → Equatorial (ECI)
    cos_lambda = math.cos(lambda_sun)
    sin_lambda = math.sin(lambda_sun)
    cos_eps = math.cos(epsilon)
    sin_eps = math.sin(epsilon)

    return np.array([
        r_au * cos_lambda,
        r_au * sin_lambda * cos_eps,
        r_au * sin_lambda * sin_eps,
    ])


def sun_position_eci_astropy(t_jd: float) -> np.ndarray:
    """Use astropy to compute sun ECI position (high precision)."""
    try:
        from astropy.coordinates import get_sun, GCRS
        from astropy.time import Time
        t = Time(t_jd, format="jd", scale="tt")
        sun_gcrs = get_sun(t)
        pos = sun_gcrs.cartesian.xyz.to("m").value
        return np.array(pos)
    except ImportError:
        return sun_position_eci(t_jd)


# Module-level cache for sun position (reused across RHS calls)
_SUN_CACHE: dict = {}


def _get_sun_position(t_jd: float) -> np.ndarray:
    """Get cached sun position (nearest hour)."""
    jd_key = int(t_jd * 24)
    if jd_key in _SUN_CACHE:
        return _SUN_CACHE[jd_key]
    pos = sun_position_eci_astropy(t_jd)
    _SUN_CACHE[jd_key] = pos
    return pos


def srp_acceleration(
    r_sat: np.ndarray,
    r_sun: np.ndarray | None,
    cfg: dict[str, Any],
    t_jd: float | None = None,
) -> np.ndarray:
    """Solar radiation pressure acceleration [m/s2].

    Args:
        r_sat: (3,) ECI satellite position [m]
        r_sun: (3,) ECI sun position [m] (None to auto-compute)
        cfg: config
        t_jd: JD (needed if r_sun=None)

    Returns:
        (3,) ECI SRP acceleration [m/s2]
    """
    # Note: the caller (forces.py) already checks model.use_srp.
    # We don't double-check here — just compute if called.
    sc = cfg["spacecraft"]

    if r_sun is None and t_jd is not None:
        r_sun = _get_sun_position(t_jd)

    if r_sun is None:
        return np.zeros(3)

    dr = r_sat - r_sun
    dist = np.linalg.norm(dr)
    if dist < 1e-10:
        return np.zeros(3)

    # Shadow function
    srp_cfg = cfg.get("high_precision", {}).get("srp", {})
    shadow_model = srp_cfg.get("shadow_model", "cylindrical")
    nu = _shadow_function(r_sat, r_sun, shadow_model, cfg)

    if nu < 1e-10:
        return np.zeros(3)

    # SRP 加速度
    Cr = sc.get("Cr", 1.3)
    A_srp = sc.get("area_srp", sc.get("area_drag", 4.0))
    mass = sc["mass"]
    area_to_mass = A_srp / mass

    a_magnitude = nu * P0_SUN * Cr * area_to_mass * (AU / dist)**2

    return a_magnitude * dr / dist


def _shadow_function(
    r_sat: np.ndarray,
    r_sun: np.ndarray,
    model: str,
    cfg: dict[str, Any],
) -> float:
    """计算阴影函数 nu ∈ [0, 1]。

    圆柱形模型: 地球视为圆柱形遮光体
    圆锥形模型: 考虑本影/半影
    """
    const = cfg["const"]
    R_earth = const["radius_earth"]

    # 太阳方向
    dr_sun = r_sun - r_sat
    dist_sun = np.linalg.norm(dr_sun)
    dir_sun = dr_sun / dist_sun if dist_sun > 1e-10 else np.zeros(3)

    # 地球方向
    r_sat_norm = np.linalg.norm(r_sat)

    # 卫星在地球阴影中的投影
    dot = np.dot(-r_sat, dir_sun)
    if dot < 0:
        return 1.0  # 太阳在地球后方 (从卫星看)

    if model == "cylindrical":
        # 圆柱形阴影
        sat_to_earth_center = r_sat_norm
        tangent_height = R_earth / math.tan(math.asin(R_earth / sat_to_earth_center)) \
            if sat_to_earth_center > R_earth else 0.0

        # 卫星到地影轴的距离
        d_perp_sq = r_sat_norm**2 - dot**2
        d_perp = math.sqrt(max(0.0, d_perp_sq))

        if d_perp < R_earth and dot < sat_to_earth_center:
            return 0.0  # 本影
        elif d_perp < R_earth + 100e3:
            # 半影 (简化: 线性过渡)
            penumbra_width = 100e3
            frac = (d_perp - R_earth) / penumbra_width
            return max(0.0, min(1.0, frac))

    elif model == "conical":
        # 圆锥形阴影 (Montenbruck & Gill 3.4.2)
        # 太阳视半径 ~0.266 deg
        a_sun = math.asin(R_SUN / dist_sun)
        a_earth = math.asin(R_earth / r_sat_norm)

        c1 = math.sin(a_earth)
        c2 = math.sin(a_sun)

        if dot < 0 and abs(dot) > R_earth:
            return 1.0

        # 简化处理
        d_perp_sq = r_sat_norm**2 - dot**2
        d_perp = math.sqrt(max(0.0, d_perp_sq))

        if d_perp < R_earth and dot > 0:
            return 0.0  # 本影

    return 1.0
