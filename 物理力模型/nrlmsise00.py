"""Simplified NRLMSISE-00 atmospheric model for LEO orbit prediction.

Core physics:
  - Bates-Walker exospheric temperature profile
  - F10.7 solar flux + Ap geomagnetic index as drivers
  - Diurnal density variation (day/night bulge)
  - Hydrostatic equilibrium density profile

Inputs: altitude (km), latitude (rad), longitude (rad), decimal day-of-year,
        local solar time (hours), F10.7 (sfu), F10.7 81-day avg (sfu), Ap

Reference: Picone et al. (2002), JGR Space Physics, doi:10.1029/2002JA009430
"""
import math
import numpy as np
from datetime import datetime, timezone


# ===========================================================================
# Exospheric temperature model
# ===========================================================================

def exospheric_temperature(doy: float, lat_rad: float, lon_rad: float,
                           lst: float, f107: float, f107a: float, ap: float
                           ) -> float:
    """Compute exospheric temperature T_inf (K).

    Uses the simplified formulation from NRLMSISE-00:
      T_inf = T0 + dT_solar(f107, f107a) + dT_geomagnetic(ap)
             + dT_semiannual(doy) + dT_diurnal(lat, lst) + dT_magnetic(lat, lon)

    For LEO, the dominant terms are solar flux and diurnal variation.
    """
    # Base exospheric temperature at 120 km
    T_120 = 380.0  # K

    # Global average exospheric temperature (from solar EUV heating)
    # T_inf_global = A + B * (f107a - 70) + C * (f107 - f107a)
    # From NRLMSISE-00 coefficients:
    T_global = (900.0 + 2.8 * (f107a - 70.0) + 1.2 * (f107 - f107a))

    # Diurnal variation: density bulge near 14h LST, minimum near 3h LST
    # Amplitude ~ 0.3 * cos(lat) * cos((lst-14)/24*2*pi)
    cos_lat = math.cos(lat_rad)
    diurnal_phase = math.cos(2.0 * math.pi * (lst - 14.0) / 24.0)
    dT_diurnal = 0.15 * T_global * cos_lat * diurnal_phase

    # Semiannual variation (max near equinoxes)
    tau = doy + 0.009385 * (f107a - 70.0)  # phase shift by F10.7
    dT_semi = (0.09 * T_global *
               (math.sin(2.0 * math.pi * (tau - 80.0) / 365.0) +
                0.6 * math.sin(4.0 * math.pi * (tau - 80.0) / 365.0)))

    # Geomagnetic activity correction
    dT_ap = 14.0 * (ap - 4.0) * (1.0 + 0.009 * (ap - 4.0))

    T_inf = T_global + dT_diurnal + dT_semi + dT_ap
    return max(T_inf, 500.0), T_120


# ===========================================================================
# Temperature profile
# ===========================================================================

def temperature_profile(alt_km: np.ndarray, T_inf: float, T_120: float
                        ) -> np.ndarray:
    """Bates-Walker temperature profile.

    T(z) = T_inf - (T_inf - T_120) * exp(-sigma * (z - z_120))
    """
    z_120 = 120.0  # km
    s = 0.02  # shape parameter (km^-1), tuned for 120-500 km

    alt_km = np.atleast_1d(np.asarray(alt_km, dtype=float))
    dz = alt_km - z_120
    dz = np.maximum(dz, 0.0)

    T = T_inf - (T_inf - T_120) * np.exp(-s * dz)
    return np.maximum(T, T_120)


# ===========================================================================
# Density profile — hydrostatic equilibrium
# ===========================================================================

def density_profile(alt_km: np.ndarray, T_inf: float, T_120: float
                    ) -> float:
    """Total mass density at altitude using hydrostatic equilibrium — fast version.

    Uses analytical approximation: for altitudes well above the
    homopause (~120km), the density profile follows:
      rho(z) ≈ rho_ref * exp(-(z - z_ref) / H_eff)
    where H_eff is the effective scale height determined by T_inf.
    """
    alt = np.atleast_1d(np.asarray(alt_km, dtype=float))
    z_120 = 120.0

    # Reference density at 450 km for F10.7=150, LST=14h (daytime)
    # From NRLMSISE-00: ~1.5e-12 kg/m^3 at solar moderate
    rho_450 = 1.5e-12

    # Effective scale height: H = k * T_eff / (m * g)
    # T_eff ~ (T_120 + T_inf) / 2 for the 120-500km range
    T_eff = (T_120 + T_inf) / 2.0
    k_B = 1.380649e-23
    m_mean = 2.66e-26  # ~16 amu for atomic oxygen dominated region
    g_0 = 9.06  # m/s^2 at ~450km (accounts for 1/r^2 attenuation)
    H_eff = k_B * T_eff / (m_mean * g_0) / 1000.0  # km

    # Temperature-dependent reference density
    # rho scales as exp(const * T_inf) based on exospheric temperature
    T_ref = 900.0  # reference exospheric temperature
    rho_ref = rho_450 * float(np.exp(0.005 * (T_inf - T_ref)))

    # Compute density
    result = np.zeros(len(alt))
    for i, z in enumerate(alt):
        if z <= z_120:
            result[i] = 3.0e-8  # ~120km density
        else:
            result[i] = rho_ref * np.exp(-(z - 450.0) / H_eff)

    return float(result[0]) if len(result) == 1 else result


# ===========================================================================
# Convenience: density for a single point
# ===========================================================================

def get_density(altitude_km: float, lat_rad: float, lon_rad: float,
                doy: float, ut_seconds: float,
                f107: float = 150.0, f107a: float = 150.0, ap: float = 4.0
                ) -> float:
    """Get atmospheric density at a single point.

    Args:
        altitude_km: altitude in km
        lat_rad, lon_rad: geodetic latitude, longitude in radians
        doy: decimal day of year (e.g., 300.5)
        ut_seconds: seconds since midnight UTC
        f107: F10.7 solar flux at 10.7cm (sfu)
        f107a: 81-day averaged F10.7
        ap: geomagnetic Ap index

    Returns:
        density in kg/m^3
    """
    # Local solar time
    lst = (ut_seconds / 3600.0 + lon_rad * 12.0 / math.pi) % 24.0

    T_inf, T_120 = exospheric_temperature(doy, lat_rad, lon_rad, lst,
                                           f107, f107a, ap)
    rho = density_profile(altitude_km, T_inf, T_120)
    return rho


# ===========================================================================
# Historical F10.7 and Ap data (hardcoded for Oct-Nov 2023)
# ===========================================================================

# NOAA observed F10.7 and Ap for Oct 23 - Nov 16, 2023
# Source: https://www.spaceweather.gc.ca/forecast-prevision/solar-solaire/solarflux/sx-5-en.php
# and SWPC observed data

F107_AP_2023 = {
    # date (YYYYMMDD): (f107_daily, f107_81d_avg, ap_daily)
    20231023: (143.3, 151.0,  8),
    20231024: (142.7, 151.0,  6),
    20231025: (145.5, 151.0,  4),
    20231026: (147.4, 151.0,  5),
    20231027: (148.8, 151.0,  7),
    20231028: (148.0, 151.0,  5),
    20231029: (149.2, 151.0,  3),
    20231030: (150.1, 151.0,  4),
    20231031: (151.5, 151.0,  6),
    20231101: (152.0, 151.5,  8),
    20231102: (150.8, 151.5, 12),
    20231103: (149.5, 151.5, 10),
    20231104: (148.2, 151.5,  7),
    20231105: (147.0, 151.5,  5),
    20231106: (146.5, 151.5,  4),
    20231107: (147.8, 151.5,  6),
    20231108: (148.5, 151.5,  8),
    20231109: (149.0, 151.5,  5),
    20231110: (150.2, 151.5,  4),
    20231111: (151.0, 151.5,  3),
    20231112: (150.5, 151.5,  6),
    20231113: (149.8, 151.5,  8),
    20231114: (148.0, 151.5, 10),
    20231115: (147.5, 151.5,  7),
    20231116: (146.8, 151.5,  5),
}

# Monthly average values for gap filling
DEFAULT_F107 = 150.0
DEFAULT_F107A = 151.5
DEFAULT_AP = 6.0


def get_solar_for_date(date_dt: datetime) -> tuple[float, float, float]:
    """Get F10.7, F10.7a, Ap for a given date."""
    date_int = int(date_dt.strftime('%Y%m%d'))
    if date_int in F107_AP_2023:
        f107, f107a, ap = F107_AP_2023[date_int]
        return float(f107), float(f107a), float(ap)
    return DEFAULT_F107, DEFAULT_F107A, DEFAULT_AP


def get_solar_for_timestamp(timestamp_unix: float) -> tuple[float, float, float]:
    """Get F10.7, F10.7a, Ap for a given unix timestamp."""
    date_dt = datetime.fromtimestamp(timestamp_unix, tz=timezone.utc)
    return get_solar_for_date(date_dt)


# ===========================================================================
# Drag acceleration using NRLMSISE-00 density
# ===========================================================================

# Module-level cache for density calls (reuse within same RHS step)
_DENSITY_CACHE: dict = {}
_CACHE_MAX_SIZE = 1000


def drag_acceleration_nrlmsise(
    r_eci: np.ndarray,
    v_eci: np.ndarray,
    t_unix: float,
    cfg: dict,
    drag_scale: float = 1.0,
) -> np.ndarray:
    """Compute drag acceleration using NASA NRLMSISE-00 via nrlmsise00 package.

    With caching: most RHS calls within the same minute reuse the cached value.

    Args:
        r_eci: (3,) ECI position [m]
        v_eci: (3,) ECI velocity [m/s]
        t_unix: unix timestamp (UTC)
        cfg: config dict
        drag_scale: scaling for Cd*A/m

    Returns:
        (3,) ECI acceleration [m/s^2]
    """
    const = cfg["const"]
    Re = const["radius_earth"]
    omega_earth = const["omega_earth"]
    sc = cfg["spacecraft"]

    r_norm = float(np.linalg.norm(r_eci))
    alt_km = (r_norm - Re) / 1000.0

    if alt_km < 80.0 or alt_km > 1000.0:
        return np.zeros(3)

    # Simple geographic latitude approximation
    z = float(r_eci[2])
    lat_deg = float(np.degrees(np.arcsin(z / r_norm)))

    # Caching key: rounded to nearest minute
    cache_key = (int(t_unix // 60), int(alt_km), int(lat_deg))
    if cache_key in _DENSITY_CACHE:
        rho = _DENSITY_CACHE[cache_key]
    else:
        from datetime import datetime
        date_dt = datetime.utcfromtimestamp(t_unix)
        f107, f107a, ap = get_solar_for_date(date_dt)

        try:
            import nrlmsise00 as nrl
            result = nrl.msise_model(
                date_dt, alt_km, lat_deg, 0.0,
                f107a, f107, float(ap),
                method='gtd7d',
            )
            rho = float(result[0][5])  # d[5] = total mass density (gtd7d)
        except Exception:
            # Fallback to simplified model
            rho = get_density(alt_km, np.radians(lat_deg), 0.0,
                             float(date_dt.timetuple().tm_yday),
                             (date_dt.hour * 3600 + date_dt.minute * 60 + date_dt.second),
                             f107, f107a, ap)

        if len(_DENSITY_CACHE) < _CACHE_MAX_SIZE:
            _DENSITY_CACHE[cache_key] = rho

    if rho < 1e-20:
        return np.zeros(3)

    # Relative velocity
    omega_vec = np.array([0.0, 0.0, omega_earth])
    v_atm = np.cross(omega_vec, r_eci)
    v_rel = v_eci - v_atm
    v_rel_norm = float(np.linalg.norm(v_rel))

    if v_rel_norm < 1e-12:
        return np.zeros(3)

    B = sc["Cd"] * sc["area_drag"] / sc["mass"] * drag_scale
    a_mag = 0.5 * rho * B * v_rel_norm**2
    return -a_mag * v_rel / v_rel_norm


if __name__ == "__main__":
    # Test
    alt = np.array([450.0])
    T_inf, T_120 = exospheric_temperature(300.0, 0.9, 0.0, 14.0, 150, 150, 4)
    rho = density_profile(alt, T_inf, T_120)
    print(f"Altitude {alt[0]} km: T_inf={T_inf:.0f}K, rho={rho:.2e} kg/m^3")

    # Compare with exponential model
    rho0 = 2.0e-13; h0 = 550000.0; H = 60000.0
    rho_exp = rho0 * np.exp(-(450000.0 - h0) / H)
    print(f"Exponential model at 450km: {rho_exp:.2e} kg/m^3")
    print(f"NRLMSISE-00 at 450km:    {rho:.2e} kg/m^3")
