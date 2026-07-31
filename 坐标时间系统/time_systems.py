"""时间系统转换模块

支持:
  UTC datetime → Julian Date
  Julian Date → UTC datetime
  UTC → GMST (简化模型)
  TAI → TT

参考: Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np


# ---- Julian Date ----


def utc_to_jd(dt: datetime) -> float:
    """UTC datetime → Julian Date (JD)。

    使用标准天文算法。输入必须带时区信息。
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour + dt.minute / 60 + dt.second / 3600 + dt.microsecond / 3.6e9

    if month <= 2:
        year -= 1
        month += 12

    A = int(year / 100)
    B = 2 - A + int(A / 4)

    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + hour / 24 + B - 1524.5
    return jd


def jd_to_utc(jd: float) -> datetime:
    """Julian Date → UTC datetime。"""
    jd += 0.5
    Z = int(jd)
    F = jd - Z

    if Z < 2299161:
        A = Z
    else:
        alpha = int((Z - 1867216.25) / 36524.25)
        A = Z + 1 + alpha - int(alpha / 4)

    B = A + 1524
    C = int((B - 122.1) / 365.25)
    D = int(365.25 * C)
    E = int((B - D) / 30.6001)

    day = B - D - int(30.6001 * E) + F
    month = E - 1 if E < 14 else E - 13
    year = C - 4716 if month > 2 else C - 4715

    day_int = int(day)
    day_frac = day - day_int
    hour = int(day_frac * 24)
    minute = int((day_frac * 24 - hour) * 60)
    second = (day_frac * 24 - hour) * 3600 - minute * 60
    microsecond = int(round((second - int(second)) * 1e6))

    return datetime(year, month, day_int, hour, minute, int(second), microsecond, tzinfo=timezone.utc)


# ---- GMST ----


def utc_to_gmst(dt: datetime) -> float:
    """UTC → GMST (Greenwich Mean Sidereal Time) [rad]。

    使用简化公式 (Vallado Eq. 3-47)，精度 ~1 arcsec。
    """
    jd = utc_to_jd(dt)
    # Julian centuries from J2000.0
    T_ut1 = (jd - 2451545.0) / 36525.0

    # GMST in seconds at 0h UT1 (Vallado Eq. 3-47)
    gmst_sec = (
        24110.54841
        + 8640184.812866 * T_ut1
        + 0.093104 * T_ut1**2
        - 6.2e-6 * T_ut1**3
    )

    # Add fraction of day
    day_frac = (jd - 0.5) % 1.0
    gmst_sec += day_frac * 86400.0 * 1.002737909350795

    # Normalize to [0, 86400)
    gmst_sec = gmst_sec % 86400.0

    # Convert to rad
    return gmst_sec * (2 * math.pi / 86400.0)


def _gmst_rad_from_jd(jd: float) -> float:
    """从 JD 直接计算 GMST [rad] (方便批量计算)。"""
    T_ut1 = (jd - 2451545.0) / 36525.0
    gmst_sec = (
        24110.54841
        + 8640184.812866 * T_ut1
        + 0.093104 * T_ut1**2
        - 6.2e-6 * T_ut1**3
    )
    day_frac = (jd - 0.5) % 1.0
    gmst_sec += day_frac * 86400.0 * 1.002737909350795
    gmst_sec = gmst_sec % 86400.0
    return gmst_sec * (2 * math.pi / 86400.0)


def gmst_from_jd_batch(jd_array: np.ndarray) -> np.ndarray:
    """批量计算 GMST [rad] from JD array。"""
    T_ut1 = (jd_array - 2451545.0) / 36525.0
    gmst_sec = (
        24110.54841
        + 8640184.812866 * T_ut1
        + 0.093104 * T_ut1**2
        - 6.2e-6 * T_ut1**3
    )
    day_frac = (jd_array - 0.5) % 1.0
    gmst_sec += day_frac * 86400.0 * 1.002737909350795
    gmst_sec = gmst_sec % 86400.0
    return gmst_sec * (2 * math.pi / 86400.0)


# ---- TAI → TT ----


def tai_to_tt(tai_jd: float) -> float:
    """TAI (International Atomic Time) → TT (Terrestrial Time)。

    TT = TAI + 32.184 s
    """
    return tai_jd + 32.184 / 86400.0


# ---- 常用常量和工具 ----


SECONDS_PER_DAY = 86400.0
JD_J2000 = 2451545.0  # J2000.0 epoch


def datetime_to_seconds_since_j2000(dt: datetime) -> float:
    """UTC datetime → 从 J2000.0 起算的秒数。"""
    jd = utc_to_jd(dt)
    return (jd - JD_J2000) * SECONDS_PER_DAY


def seconds_since_j2000_to_datetime(seconds: float) -> datetime:
    """从 J2000.0 起算的秒数 → UTC datetime。"""
    jd = JD_J2000 + seconds / SECONDS_PER_DAY
    return jd_to_utc(jd)
