"""日月三体引力摄动

a_b = mu_b * [(r_b - r)/(|r_b - r|^3) - r_b/(|r_b|^3)]

其中 b = Sun, Moon.
r_b = 第三天体在 ECI 中的位置矢量

使用简化解析公式或 astropy/jplephem 获取天体位置。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def moon_position_eci(t_jd: float) -> np.ndarray:
    """月球 ECI 近似位置 (简化解析, 精度 ~0.1 deg)。

    参考: Meeus, "Astronomical Algorithms", Chapter 47.
    """
    T = (t_jd - 2451545.0) / 36525.0

    # Moon mean longitude [deg]
    Lp = 218.3164477 + 481267.88123421 * T - 0.0015786 * T**2
    # Moon mean elongation
    D = 297.8501921 + 445267.1114034 * T - 0.0018819 * T**2
    # Sun mean anomaly
    M = 357.5291092 + 35999.0502909 * T - 0.0001536 * T**2
    # Moon mean anomaly
    Mp = 134.9633964 + 477198.8675055 * T + 0.0087414 * T**2
    # Moon argument of latitude
    F = 93.2720950 + 483202.0175233 * T - 0.0036539 * T**2

    Lp_rad = math.radians(Lp % 360.0)
    D_rad = math.radians(D % 360.0)
    M_rad = math.radians(M % 360.0)
    Mp_rad = math.radians(Mp % 360.0)
    F_rad = math.radians(F % 360.0)

    # Ecliptic longitude corrections (arcseconds → deg)
    dlambda = (
        22640 * math.sin(Mp_rad)
        - 4586 * math.sin(Mp_rad - 2 * D_rad)
        + 2370 * math.sin(2 * D_rad)
        + 769 * math.sin(2 * Mp_rad)
        - 668 * math.sin(M_rad)
        - 412 * math.sin(2 * F_rad)
        - 212 * math.sin(2 * Mp_rad - 2 * D_rad)
        - 206 * math.sin(Mp_rad + M_rad - 2 * D_rad)
        + 192 * math.sin(Mp_rad + 2 * D_rad)
        - 165 * math.sin(M_rad - 2 * D_rad)
        + 148 * math.sin(Mp_rad - M_rad)
        - 125 * math.sin(D_rad)
        - 110 * math.sin(Mp_rad + M_rad)
    ) / 3600.0

    lambda_moon = Lp_rad + math.radians(dlambda)

    # Ecliptic latitude
    dB = (
        18461 * math.sin(F_rad)
        + 1010 * math.sin(Mp_rad + F_rad)
        + 1000 * math.sin(Mp_rad - F_rad)
        - 624 * math.sin(F_rad - 2 * D_rad)
        - 199 * math.sin(Mp_rad - F_rad - 2 * D_rad)
        - 167 * math.sin(Mp_rad + F_rad - 2 * D_rad)
    ) / 3600.0
    beta_moon = math.radians(dB)

    # Horizontal parallax [arcsec → deg]
    pi_moon = math.radians((3422.7 + 186.6 * math.cos(Mp_rad)) / 3600.0)

    # Distance [km]
    dist_moon_km = 6378.137 / math.sin(pi_moon)
    # Convert to m
    dist_moon = dist_moon_km * 1000.0 + 384400000.0  # rough adjustment

    # Better: use mean distance 384400 km + corrections
    dist_moon = 385000560.0  # mean [m]

    # Obliquity [deg]
    epsilon = math.radians(23.439291 - 0.0130042 * T)

    # Ecliptic → Equatorial rotation
    sin_lambda = math.sin(lambda_moon)
    cos_lambda = math.cos(lambda_moon)
    sin_beta = math.sin(beta_moon)
    cos_beta = math.cos(beta_moon)
    cos_eps = math.cos(epsilon)
    sin_eps = math.sin(epsilon)

    return np.array([
        dist_moon * (cos_beta * cos_lambda),
        dist_moon * (cos_beta * sin_lambda * cos_eps - sin_beta * sin_eps),
        dist_moon * (cos_beta * sin_lambda * sin_eps + sin_beta * cos_eps),
    ])


def third_body_acceleration(
    r_sat: np.ndarray,
    mu_body: float,
    r_body: np.ndarray,
) -> np.ndarray:
    """第三天体引力摄动加速度。

    a_b = mu_b * [(r_b - r)/(|r_b - r|^3) - r_b/(|r_b|^3)]

    Args:
        r_sat: (3,) 卫星 ECI 位置 [m]
        mu_body: 天体引力常数 [m^3/s^2]
        r_body: (3,) 天体 ECI 位置 [m]

    Returns:
        (3,) ECI 摄动加速度 [m/s^2]
    """
    dr = r_body - r_sat
    dist_sat = np.linalg.norm(dr)
    dist_earth = np.linalg.norm(r_body)

    if dist_sat < 1e-10 or dist_earth < 1e-10:
        return np.zeros(3)

    return mu_body * (dr / dist_sat**3 - r_body / dist_earth**3)


# Module-level cache for astropy imports and positions
_ASTROPY_AVAILABLE = None
_SUN_MOON_CACHE: dict = {}  # cache by jd_int (nearest hour)


def _get_astropy():
    global _ASTROPY_AVAILABLE
    if _ASTROPY_AVAILABLE is None:
        try:
            from astropy.coordinates import get_sun, get_body, GCRS  # noqa: F401
            from astropy.time import Time  # noqa: F401
            _ASTROPY_AVAILABLE = True
        except ImportError:
            _ASTROPY_AVAILABLE = False
    return _ASTROPY_AVAILABLE


def _get_sun_moon_positions(t_jd: float) -> tuple[np.ndarray, np.ndarray]:
    """Get sun and moon ECI positions, cached by nearest hour."""
    jd_key = int(t_jd * 24)  # nearest hour
    if jd_key in _SUN_MOON_CACHE:
        return _SUN_MOON_CACHE[jd_key]

    from astropy.coordinates import get_sun, get_body, GCRS
    from astropy.time import Time

    t = Time(t_jd, format="jd", scale="tt")
    sun_gcrs = get_sun(t)
    r_sun = np.array(sun_gcrs.cartesian.xyz.to("m").value)
    moon_gcrs = get_body("moon", t)
    r_moon = np.array(moon_gcrs.cartesian.xyz.to("m").value)

    _SUN_MOON_CACHE[jd_key] = (r_sun, r_moon)
    return r_sun, r_moon


def third_body_sun_moon(
    r_sat: np.ndarray,
    t_jd: float,
    cfg: dict[str, Any],
) -> np.ndarray:
    """Sun+Moon third-body perturbation acceleration (cached).

    Args:
        r_sat: (3,) ECI satellite position [m]
        t_jd: Julian Date
        cfg: config

    Returns:
        (3,) ECI acceleration [m/s^2]
    """
    hp = cfg.get("high_precision", {})
    tb = hp.get("third_body", {})
    const = cfg["const"]
    a_total = np.zeros(3)

    if not tb.get("sun", True) and not tb.get("moon", True):
        return a_total

    if _get_astropy():
        r_sun, r_moon = _get_sun_moon_positions(t_jd)
        if tb.get("sun", True):
            a_total += third_body_acceleration(r_sat, const.get("mu_sun", 1.32712440018e20), r_sun)
        if tb.get("moon", True):
            a_total += third_body_acceleration(r_sat, const["mu_moon"], r_moon)
    else:
        from 物理力模型.srp import sun_position_eci
        if tb.get("sun", True):
            r_sun = sun_position_eci(t_jd)
            a_total += third_body_acceleration(r_sat, const.get("mu_sun", 1.32712440018e20), r_sun)
        if tb.get("moon", True):
            r_moon = moon_position_eci(t_jd)
            a_total += third_body_acceleration(r_sat, const["mu_moon"], r_moon)

    return a_total
