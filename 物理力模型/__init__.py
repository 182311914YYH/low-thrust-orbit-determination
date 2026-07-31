"""05_物理传播器 — 高精度轨道动力学模型与数值积分 (v2)。"""
from .forces import (
    central_gravity, j2_perturbation, drag_acceleration_exponential,
    compute_total_acceleration, compute_total_acceleration_upgraded,
    high_order_gravity_eci, srp_acceleration_eci, third_body_acceleration_eci,
)
from .propagator import (
    propagate_orbit, propagate_orbit_upgraded, propagate_orbit_batch, OrbitalPropagator,
)
from .rtn_frame import compute_rtn_basis, rtn_to_eci_transform, eci_to_rtn_transform
from .gravity import spherical_harmonic_gravity, load_gravity_coefficients
from .srp import srp_acceleration as srp_force
from .third_body import third_body_acceleration as third_body_force

__all__ = [
    "central_gravity", "j2_perturbation", "drag_acceleration_exponential",
    "compute_total_acceleration", "compute_total_acceleration_upgraded",
    "high_order_gravity_eci", "srp_acceleration_eci", "third_body_acceleration_eci",
    "spherical_harmonic_gravity", "load_gravity_coefficients",
    "propagate_orbit", "propagate_orbit_upgraded", "propagate_orbit_batch",
    "OrbitalPropagator", "compute_rtn_basis", "rtn_to_eci_transform",
    "eci_to_rtn_transform", "srp_force", "third_body_force",
]
