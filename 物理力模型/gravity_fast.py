"""Fast spherical harmonic gravity using standard recurrence.

Algorithm: Cunningham/Montenbruck recurrence for fully-normalized
associated Legendre functions and their derivatives.

For degree N, computes all Pbar_nm(sin_phi) in O(N²) using recurrence,
rather than calling scipy.special.lpmv N² times (which is ~100x slower).

Reference: Montenbruck & Gill, "Satellite Orbits", Section 3.2.4
"""
import math
import numpy as np
from 物理力模型.egm96_full import get_egm96_coefficients


def compute_pbar_recurrence(sin_phi: float, max_degree: int):
    """Compute all normalized Pbar and dPbar/dphi using standard recurrence.

    Recurrences (Montenbruck 3.25-3.28, adapted for normalized):
      Pbar[0,0] = 1
      Pbar[m,m] = sqrt((2m+1)/(2m)) * cos(phi) * Pbar[m-1,m-1]  (m ≥ 1)
      Pbar[m+1,m] = sqrt(2m+3) * sin(phi) * Pbar[m,m]  (m ≥ 0)
      Pbar[n,m] = a_nm * sin(phi) * Pbar[n-1,m] - b_nm * Pbar[n-2,m]  (n > m+1)
    where:
      a_nm = sqrt(((2n-1)(2n+1))/((n-m)(n+m)))
      b_nm = sqrt(((2n+1)(n+m-1)(n-m-1))/((2n-3)(n-m)(n+m)))

    Args:
        sin_phi: sine of geocentric latitude
        max_degree: maximum degree

    Returns:
        Pbar: (N+1, N+1) matrix of Pbar[n,m] for n≥m
        dPbar: (N+1, N+1) matrix of dPbar/dphi[n,m]
    """
    N = max_degree + 1
    cos_phi = math.sqrt(max(0.0, 1.0 - sin_phi**2))

    Pbar = np.zeros((N, N))
    dPbar = np.zeros((N, N))

    # Seed
    Pbar[0, 0] = 1.0
    dPbar[0, 0] = 0.0

    if max_degree < 1:
        return Pbar, dPbar

    # ---- Diagonal terms: Pbar[m,m] ----
    for m in range(1, N):
        # Pbar[m,m] = W_mm * cos(phi) * Pbar[m-1,m-1]
        # W_mm = sqrt((2m+1)/(2m))
        w_mm = math.sqrt((2.0 * m + 1.0) / (2.0 * m))
        Pbar[m, m] = w_mm * cos_phi * Pbar[m - 1, m - 1]

        # dPbar[m,m]/dphi = m * tan(phi) * Pbar[m,m]
        if cos_phi > 1e-15:
            dPbar[m, m] = m * sin_phi / cos_phi * Pbar[m, m]

    # ---- First off-diagonal: Pbar[m+1,m] ----
    for m in range(N - 1):
        # Pbar[m+1,m] = sqrt(2m+3) * sin(phi) * Pbar[m,m]
        Pbar[m + 1, m] = math.sqrt(2.0 * m + 3.0) * sin_phi * Pbar[m, m]

        # dPbar[m+1,m]/dphi = sqrt(2m+3)*(cos(phi)*Pbar[m,m] + sin(phi)*dPbar[m,m])
        dPbar[m + 1, m] = (math.sqrt(2.0 * m + 3.0) *
                           (cos_phi * Pbar[m, m] + sin_phi * dPbar[m, m]))

    # ---- General recurrence: Pbar[n,m] for n > m+1 ----
    for n in range(2, N):
        for m in range(n - 1):
            # a_nm = sqrt(((2n-1)(2n+1))/((n-m)(n+m)))
            denom = (n - m) * (n + m)
            if denom <= 0:
                continue
            a_nm = math.sqrt((2.0 * n - 1.0) * (2.0 * n + 1.0) / denom)

            # b_nm = sqrt(((2n+1)(n+m-1)(n-m-1))/((2n-3)(n-m)(n+m)))
            num_b = (2.0 * n + 1.0) * (n + m - 1.0) * (n - m - 1.0)
            denom_b = (2.0 * n - 3.0) * denom
            if denom_b > 0 and num_b >= 0:
                b_nm = math.sqrt(num_b / denom_b)
            else:
                b_nm = 0.0

            # Recurrence
            Pbar[n, m] = (a_nm * sin_phi * Pbar[n - 1, m] -
                          b_nm * Pbar[n - 2, m])

            # dPbar/dphi recurrence (derived from recurrence above)
            dPbar[n, m] = (a_nm * sin_phi * dPbar[n - 1, m] +
                           a_nm * cos_phi * Pbar[n - 1, m] -
                           b_nm * dPbar[n - 2, m])

    return Pbar, dPbar


def spherical_harmonic_gravity_fast(
    r_ecef: np.ndarray,
    mu: float,
    R_earth: float,
    max_degree: int,
    C: np.ndarray,
    S: np.ndarray,
) -> np.ndarray:
    """Compute spherical harmonic gravity acceleration (ECEF).

    Uses Cunningham's method with the standard recurrence for Legendre
    functions. This is ~100x faster than per-term lpmv for degree 20.

    Montenbruck & Gill Algorithm 3.2 (Cunningham 1970).

    Args:
        r_ecef: (3,) ECEF position [m]
        mu: gravitational parameter [m³/s²]
        R_earth: Earth equatorial radius [m]
        max_degree: maximum degree (>0)
        C: (N+1,N+1) C_nm coefficients (fully normalized)
        S: (N+1,N+1) S_nm coefficients

    Returns:
        (3,) ECEF acceleration [m/s2]
    """
    x, y, z = r_ecef
    r = math.sqrt(x**2 + y**2 + z**2)

    if r < R_earth * 0.5 or r < 1e-3:
        return np.zeros(3)

    sin_phi = z / r
    cos_phi = math.sqrt(max(0.0, x**2 + y**2)) / r

    # Longitude
    if cos_phi > 1e-15:
        cos_lambda = x / (r * cos_phi)
        sin_lambda = y / (r * cos_phi)
    else:
        # At poles, longitude is undefined — use 0
        cos_lambda = 1.0
        sin_lambda = 0.0

    # Compute all Pbar_nm and dPbar/dphi via recurrence
    Pbar, dPbar = compute_pbar_recurrence(sin_phi, max_degree)

    # Precompute cos(m*lambda), sin(m*lambda) via Chebyshev recurrence
    N = max_degree + 1
    cos_ml = np.ones(N)
    sin_ml = np.zeros(N)
    for m in range(1, N):
        cos_ml[m] = cos_ml[m-1] * cos_lambda - sin_ml[m-1] * sin_lambda
        sin_ml[m] = sin_ml[m-1] * cos_lambda + cos_ml[m-1] * sin_lambda

    # Cunningham V_nm and W_nm
    # V_nm = (R/r)^{n+1} * Pbar_nm * cos(m*lambda)
    # W_nm = (R/r)^{n+1} * Pbar_nm * sin(m*lambda)
    R_over_r = R_earth / r

    # Precompute powers of R/r
    r_pow = np.ones(N + 1)
    for n in range(1, N + 1):
        r_pow[n] = r_pow[n - 1] * R_over_r

    # Accumulate accelerations using Cunningham formulation
    # (Montenbruck Eq. 3.33)
    a_x, a_y, a_z = 0.0, 0.0, 0.0
    factor = mu / r

    for n in range(2, N):
        r_pow_n1 = r_pow[n + 1]  # (R/r)^{n+1}
        for m in range(n + 1):
            c_nm = C[n, m]
            s_nm = S[n, m]

            if abs(c_nm) < 1e-30 and abs(s_nm) < 1e-30:
                continue

            P = Pbar[n, m]
            if abs(P) < 1e-30:
                continue

            cos_term = c_nm * cos_ml[m] + s_nm * sin_ml[m]
            sin_term = s_nm * cos_ml[m] - c_nm * sin_ml[m]

            # V_nm = r_pow_n1 * Pbar * cos(m*lambda)
            V_nm = r_pow_n1 * P * cos_ml[m]
            # W_nm = r_pow_n1 * Pbar * sin(m*lambda)
            W_nm = r_pow_n1 * P * sin_ml[m]

            # Radial contribution (m=0 terms)
            # For zonal: simpler expressions

            # Cunningham accelerations:
            # d²x/dt² from V, W recurrence:
            # a_x = factor * sum_n sum_m [C_nm * V_{n+1,m+1} corrections...]

            # Instead use the direct gradient approach:
            # dU/dr, dU/d(phi), dU/d(lambda) → Cartesian

            # This is simpler and equally accurate:
            if m == 0:
                # Zonal: only radial and z (sin_phi) terms
                # Contribution to radial:
                # a_r += (n+1) * r_pow_n1 * Pbar_n0 * C_n0
                pass  # handled below with unified approach

    # ---- Direct potential gradient (more straightforward) ----
    # Reset and use the direct method
    a_x, a_y, a_z = 0.0, 0.0, 0.0

    # dU/dr component
    dU_dr = 0.0
    dU_dphi = 0.0
    dU_dlambda = 0.0

    for n in range(2, N):
        r_pow_n = r_pow[n]  # (R/r)^n
        for m in range(n + 1):
            c_nm = C[n, m]
            s_nm = S[n, m]

            if abs(c_nm) < 1e-30 and abs(s_nm) < 1e-30:
                continue

            P = Pbar[n, m]
            cos_term = c_nm * cos_ml[m] + s_nm * sin_ml[m]
            sin_term = -c_nm * sin_ml[m] + s_nm * cos_ml[m]

            # dU/dr: -(n+1)/r * (R/r)^n * Pbar * cos_term
            dU_dr += -(n + 1) * r_pow_n * P * cos_term

            # dU/d(phi): (R/r)^n * dPbar/dphi * cos_term
            dU_dphi += r_pow_n * dPbar[n, m] * cos_term

            # dU/d(lambda): (R/r)^n * Pbar * m * (-C*sin(m*lambda) + S*cos(m*lambda))
            # = (R/r)^n * Pbar * m * sin_term
            if m > 0:
                dU_dlambda += r_pow_n * P * m * sin_term

    # Convert partials to Cartesian
    dU_dr *= factor / r
    dU_dphi *= factor / r
    dU_dlambda *= factor / (r * cos_phi + 1e-30)

    # Transformation matrix (spherical → Cartesian gradient)
    # ∂/∂r     → [cos(phi)*cos(lambda), cos(phi)*sin(lambda), sin(phi)]
    # ∂/∂(phi) → [-sin(phi)*cos(lambda), -sin(phi)*sin(lambda), cos(phi)]
    # ∂/∂(lambda) → [-sin(lambda), cos(lambda), 0]

    a_x = (dU_dr * cos_phi * cos_lambda +
           dU_dphi * (-sin_phi * cos_lambda) +
           dU_dlambda * (-sin_lambda))

    a_y = (dU_dr * cos_phi * sin_lambda +
           dU_dphi * (-sin_phi * sin_lambda) +
           dU_dlambda * cos_lambda)

    a_z = (dU_dr * sin_phi +
           dU_dphi * cos_phi)

    return np.array([a_x, a_y, a_z])


def load_gravity_coefficients_fast(max_degree: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Load EGM96 coefficients for use with fast gravity."""
    return get_egm96_coefficients(max_degree)


# ===========================================================================
# Test
# ===========================================================================

if __name__ == "__main__":
    import time

    # Test accuracy against J2 analytical formula
    mu = 3.986004418e14
    Re = 6378137.0

    # Test position: 550km altitude, 45 deg latitude
    alt = 550000.0
    lat = math.radians(45.0)
    lon = math.radians(120.0)

    r_mag = Re + alt
    x = r_mag * math.cos(lat) * math.cos(lon)
    y = r_mag * math.cos(lat) * math.sin(lon)
    z = r_mag * math.sin(lat)
    r_ecef = np.array([x, y, z])

    print(f"Test position: r={r_mag/1000:.1f} km, lat={math.degrees(lat):.1f}°")

    # Compare degree 2 vs J2 analytical
    C, S = load_gravity_coefficients_fast(2)
    a_harmonic = spherical_harmonic_gravity_fast(r_ecef, mu, Re, 2, C, S)

    # J2 analytical
    r_norm = np.linalg.norm(r_ecef)
    J2 = 1.08262668e-3
    factor_j2 = 1.5 * J2 * mu * Re**2 / r_norm**5
    x_, y_, z_ = r_ecef
    r2 = r_norm**2
    z2 = z_**2
    a_j2 = factor_j2 * np.array([
        x_ * (5*z2/r2 - 1),
        y_ * (5*z2/r2 - 1),
        z_ * (5*z2/r2 - 3),
    ])
    central = -mu * r_ecef / r_norm**3
    a_j2_total = central + a_j2

    diff = np.abs(a_harmonic - a_j2_total).max()
    print(f"Degree 2 vs J2 analytical: max diff = {diff:.2e} m/s2")
    if diff < 1e-10:
        print("  ✓ PASS — harmonic matches J2 analytical")

    # Benchmark
    for deg in [2, 8, 20]:
        C, S = load_gravity_coefficients_fast(deg)
        t0 = time.time()
        for _ in range(100):
            spherical_harmonic_gravity_fast(r_ecef, mu, Re, deg, C, S)
        dt = (time.time() - t0) / 100
        print(f"Degree {deg:2d}: {dt*1000:.2f} ms/call")

    # Test contribution of higher degrees
    a_deg2, _ = spherical_harmonic_gravity_fast(r_ecef, mu, Re, 2, C, S), None
    C20, S20 = load_gravity_coefficients_fast(20)
    a_deg20 = spherical_harmonic_gravity_fast(r_ecef, mu, Re, 20, C20, S20)
    diff_2_20 = np.linalg.norm(a_deg20 - a_deg2)
    print(f"\nDifference (deg 20 - deg 2): {diff_2_20:.4e} m/s2")
    print(f"As fraction of central accel: {diff_2_20 / (mu/r_mag**2):.2e}")
