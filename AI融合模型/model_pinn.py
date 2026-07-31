"""PINN + LSTM model for non-cooperative spacecraft thrust estimation.

Architecture:
  144-step history → 2-layer LSTM → hidden state h
  h + time encoding → ThrustMLP → Δa_RTN(t)  (continuous neural thrust function)
  Δa_RTN(t) + known physics → Differentiable RK4 → predicted orbit

Key difference from Informer approach:
  - LSTM is naturally suited for sequence-to-function tasks
  - Time-varying thrust (not constant over prediction window)
  - Physics-informed loss constrains the model to physically plausible solutions
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


# ===== Differentiable Physics Engine (pure torch) =====

def _compute_physics_accel(
    r: torch.Tensor,
    v: torch.Tensor,
    mu: float, J2: float, Re: float,
    omega: float,
    rho0: float, h0: float, H_scale: float, Cd_A_m: float,
    a_rtn: torch.Tensor | None,
) -> torch.Tensor:
    """Compute total acceleration: gravity + J2 + drag + RTN thrust.

    All inputs are torch tensors for differentiability.
    r, v: (B, 3)
    a_rtn: (B, 3) or None
    Returns: (B, 3) total acceleration in ECI
    """
    rn = torch.norm(r, dim=-1, keepdim=True).clamp(min=1e-3)

    # 1. Central gravity: a = -mu * r / |r|³
    a = -mu * r / rn**3

    # 2. J2 perturbation
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    r2 = rn.squeeze(-1)**2
    z2 = z**2
    factor = 1.5 * J2 * mu * Re**2 / rn.squeeze(-1)**5
    a_j2 = torch.stack([
        factor * x * (5.0 * z2 / r2 - 1.0),
        factor * y * (5.0 * z2 / r2 - 1.0),
        factor * z * (5.0 * z2 / r2 - 3.0),
    ], dim=-1)
    a = a + a_j2

    # 3. Exponential atmospheric drag
    alt = rn.squeeze(-1) - Re
    rho = rho0 * torch.exp(-(alt - h0) / H_scale)
    omega_tensor = torch.tensor([0.0, 0.0, omega], device=r.device, dtype=r.dtype)
    v_atm = torch.cross(omega_tensor.expand_as(r), r, dim=-1)
    v_rel = v - v_atm
    v_rel_n = torch.norm(v_rel, dim=-1, keepdim=True).clamp(min=1e-12)
    a_drag = -0.5 * rho.unsqueeze(-1) * Cd_A_m * v_rel_n * v_rel
    a = a + a_drag

    # 4. RTN thrust → ECI
    if a_rtn is not None:
        R_hat = r / rn
        speed = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-12)
        T_hat = v / speed
        N_hat = torch.cross(R_hat, T_hat, dim=-1)
        N_hat = N_hat / torch.norm(N_hat, dim=-1, keepdim=True).clamp(min=1e-12)
        # Rotation matrix R: columns = [R_hat, T_hat, N_hat]
        C = torch.stack([R_hat, T_hat, N_hat], dim=-1)  # (B, 3, 3)
        a_rtn_eci = torch.bmm(C, a_rtn.unsqueeze(-1)).squeeze(-1)
        a = a + a_rtn_eci

    return a


def differentiable_rk4_step(
    r: torch.Tensor, v: torch.Tensor,
    dt: float,
    mu: float, J2: float, Re: float, omega: float,
    rho0: float, h0: float, H_scale: float, Cd_A_m: float,
    a_rtn: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single RK4 step. r,v are (B,3). Returns (r_new, v_new)."""
    def acc(rr, vv, aa_rtn=None):
        return _compute_physics_accel(rr, vv, mu, J2, Re, omega, rho0, h0, H_scale, Cd_A_m, aa_rtn)

    a1 = acc(r, v, a_rtn)
    k1v, k1r = a1, v

    a2 = acc(r + 0.5*dt*k1r, v + 0.5*dt*k1v, a_rtn)
    k2v, k2r = a2, v + 0.5*dt*k1v

    a3 = acc(r + 0.5*dt*k2r, v + 0.5*dt*k2v, a_rtn)
    k3v, k3r = a3, v + 0.5*dt*k2v

    a4 = acc(r + dt*k3r, v + dt*k3v, a_rtn)
    k4v, k4r = a4, v + dt*k3v

    r_new = r + dt/6.0 * (k1r + 2*k2r + 2*k3r + k4r)
    v_new = v + dt/6.0 * (k1v + 2*k2v + 2*k3v + k4v)
    return r_new, v_new


def differentiable_propagate_with_thrust_fn(
    r0: torch.Tensor,
    v0: torch.Tensor,
    dt_output: float,
    n_steps: int,
    thrust_fn,  # callable: thrust_fn(t_vec) -> (B, 3) Δa_RTN
    physics_params: dict,
    substeps: int = 10,
    return_full: bool = False,
):
    """Propagate orbit with time-varying thrust from a neural function.

    Args:
        r0, v0: (B, 3) initial position [m], velocity [m/s]
        dt_output: output time step [s]
        n_steps: number of output steps (H)
        thrust_fn: callable that takes (B, T) time tensor [s from t0] → (B, T, 3) a_rtn
        physics_params: dict of {mu, J2, Re, omega, rho0, h0, H_scale, Cd_A_m}
        substeps: number of internal steps per output step (default 10)
        return_full: if True, return (positions, velocities); else just positions

    Returns:
        positions: (B, n_steps, 3) position history [m]
        if return_full: also velocities: (B, n_steps, 3)
    """
    B = r0.shape[0]
    device = r0.device
    dt_sub = dt_output / substeps
    total_substeps = n_steps * substeps

    mu = physics_params["mu"]
    J2 = physics_params["J2"]
    Re = physics_params["Re"]
    omega = physics_params["omega"]
    rho0 = physics_params["rho0"]
    h0 = physics_params["h0"]
    H_scale = physics_params["H_scale"]
    Cd_A_m = physics_params["Cd_A_m"]

    r, v = r0, v0
    positions = torch.zeros(B, n_steps, 3, device=device, dtype=r.dtype)
    velocities = torch.zeros(B, n_steps, 3, device=device, dtype=r.dtype) if return_full else None

    # Pre-compute thrust at all output steps (query thrust_fn once per output step)
    t_output = torch.arange(n_steps, device=device, dtype=r.dtype) * dt_output
    a_rtn_output = thrust_fn(t_output.unsqueeze(0).expand(B, -1))  # (B, n_steps, 3)

    for i in range(total_substeps):
        out_idx = i // substeps
        a_rtn_current = a_rtn_output[:, out_idx, :]  # (B, 3) — piecewise constant

        r, v = differentiable_rk4_step(
            r, v, dt_sub,
            mu, J2, Re, omega, rho0, h0, H_scale, Cd_A_m,
            a_rtn_current,
        )

        if (i + 1) % substeps == 0:
            k = (i + 1) // substeps - 1
            positions[:, k, :] = r
            if return_full:
                velocities[:, k, :] = v

    if return_full:
        return positions, velocities
    return positions


# ===== LSTM Encoder =====

class LSTMEncoder(nn.Module):
    """2-layer LSTM to encode orbital history into a compact hidden state.

    Input: (B, L, F) — L=144 history steps, F=feature dim
    Output: h_final (B, 2*d_hidden) — concatenated final hidden states from both layers
    """

    def __init__(self, input_dim: int = 14, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode sequence.

        Args:
            x: (B, L, F) input features
            mask: (B, L) optional mask, 1=valid, 0=pad

        Returns:
            h: (B, 2 * hidden_dim) concatenated final states from layer 1 & 2
        """
        B, L, _ = x.shape

        # Pack padded sequence if mask provided
        if mask is not None and mask.sum() < B * L:
            lengths = mask.sum(dim=1).cpu().long()
            lengths = lengths.clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(x)

        # h_n: (num_layers, B, hidden_dim)
        # Concatenate final states from layer 0 and layer 1
        h = torch.cat([h_n[0], h_n[1]], dim=-1)  # (B, 2*d)
        return self.dropout(h)


# ===== Neural Thrust Function =====

class NeuralThrustFunction(nn.Module):
    """MLP that maps (hidden_state, time_encoding) → Δa_RTN(t).

    The hidden_state encodes "what thrust pattern is the satellite exhibiting",
    and the time encoding tells "what time in the prediction window we're at".

    Time encoding uses sin/cos of orbital and daily periods to capture
    the periodic patterns in thrust (e.g., thrust during eclipse vs sunlight).
    """

    def __init__(self, hidden_dim: int = 128, num_harmonics: int = 4,
                 thrust_bounds: tuple[float, float, float] = (5e-5, 2e-4, 5e-5)):
        """
        Args:
            hidden_dim: dimension of LSTM encoding
            num_harmonics: number of sin/cos pairs for time encoding (→ 2*n_harmonics dims)
            thrust_bounds: (max_R, max_T, max_N) for soft-clamping thrust output [m/s²]
        """
        super().__init__()
        time_dim = 2 * num_harmonics  # sin/cos for each harmonic
        input_dim = hidden_dim + time_dim

        self.time_dim = time_dim
        self.num_harmonics = num_harmonics

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 3),
        )

        # Soft bounds for each RTN component
        # Use softplus to constrain output magnitude
        self.register_buffer("bounds", torch.tensor(thrust_bounds))

    def time_encoding(self, t: torch.Tensor) -> torch.Tensor:
        """Encode time with sin/cos of multiple harmonics.

        Args:
            t: (B, T) time values in seconds from prediction start

        Returns:
            (B, T, time_dim) time features
        """
        harmonics = []
        # Orbital period ~5700s (95 min), daily ~86400s, monthly ~2.6e6s, yearly ~3.15e7s
        periods = [5700.0, 86400.0, 2.6e6, 3.15e7]
        for T_period in periods[:self.num_harmonics]:
            omega = 2 * math.pi / T_period
            harmonics.append(torch.sin(omega * t))
            harmonics.append(torch.cos(omega * t))
        return torch.stack(harmonics, dim=-1)  # (B, T, time_dim)

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict thrust at time points t.

        Args:
            h: (B, hidden_dim) LSTM hidden state
            t: (B, T) time values [s from prediction start]

        Returns:
            (B, T, 3) Δa_RTN [m/s²]
        """
        B, T = t.shape
        # Expand h to match time dimension
        h_exp = h.unsqueeze(1).expand(-1, T, -1)  # (B, T, hidden_dim)
        t_enc = self.time_encoding(t)              # (B, T, time_dim)
        combined = torch.cat([h_exp, t_enc], dim=-1)  # (B, T, input_dim)

        # MLP per time step
        combined_flat = combined.reshape(B * T, -1)
        out_flat = self.mlp(combined_flat)
        out = out_flat.reshape(B, T, 3)

        # Soft clamping: bounded but smooth, prevents extreme predictions
        # a_out = bounds * tanh(raw / bounds) — scales to [-bounds, bounds]
        # Using a softer version: bounds * (raw / sqrt(raw² + bounds²)) ≈ bounds * tanh(raw/bounds)
        bound = self.bounds.unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
        out = bound * torch.tanh(out / bound)

        return out


# ===== Full PINN-LSTM Model =====

class PINNLSTMModel(nn.Module):
    """Complete PINN+LSTM model for orbit prediction with unknown thrust.

    Pipeline:
      history (B, L, F) → LSTMEncoder → h (B, 2*d)
      h → NeuralThrustFunction → Δa_RTN(t) for any future t
      Δa_RTN(t) + physics → differentiable RK4 → predicted orbit
    """

    def __init__(self,
                 input_dim: int = 14,
                 lstm_hidden: int = 64,
                 lstm_layers: int = 2,
                 lstm_dropout: float = 0.1,
                 num_harmonics: int = 4,
                 thrust_bounds: tuple = (5e-5, 2e-4, 5e-5)):
        super().__init__()

        self.encoder = LSTMEncoder(input_dim, lstm_hidden, lstm_layers, lstm_dropout)
        hidden_dim = 2 * lstm_hidden  # concatenated from 2 layers

        self.thrust_fn = NeuralThrustFunction(hidden_dim, num_harmonics, thrust_bounds)

        # Store for easy access
        self.hidden_dim = hidden_dim

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode history → hidden state."""
        return self.encoder(x, mask)

    def predict_thrust(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict thrust at given times.

        Args:
            h: (B, hidden_dim) from encoder
            t: (B, T) times [s]

        Returns:
            (B, T, 3) Δa_RTN [m/s²]
        """
        return self.thrust_fn(h, t)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None,
                t_query: torch.Tensor | None = None) -> torch.Tensor:
        """Simple forward: history → Δa_RTN at query times.

        Args:
            x: (B, L, F) history features
            mask: (B, L) optional mask
            t_query: (B, T) query times [s]. If None, returns single Δa at t=0

        Returns:
            (B, T, 3) or (B, 3) Δa_RTN predictions
        """
        h = self.encode(x, mask)
        if t_query is None:
            t_query = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        return self.predict_thrust(h, t_query)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===== Linearized Thrust Propagation (FAST — no RK4 autograd) =====

def linearized_thrust_propagation(
    r_phys: torch.Tensor,
    v_phys: torch.Tensor,
    rtn_basis: torch.Tensor,
    a_rtn: torch.Tensor,
    dt: float = 60.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute thrust-corrected trajectory via linearized recurrence.

    Uses the state-correction recurrence:
      δv_{k+1} = δv_k + Δt · R_k · Δa_k
      δr_{k+1} = δr_k + Δt · δv_k + ½Δt² · R_k · Δa_k

    where R_k is the (3,3) RTN→ECI rotation matrix at step k.

    This is O(H) matrix operations — ~1000x faster than differentiable RK4.

    Args:
        r_phys:  (B, H, 3) pre-computed physics-only positions [m]
        v_phys:  (B, H, 3) pre-computed physics-only velocities [m/s]
        rtn_basis: (B, H, 3, 3) pre-computed RTN→ECI rotation matrices
        a_rtn:   (B, H, 3) predicted RTN thrust from model [m/s²]
        dt:      time step [s]

    Returns:
        r_corrected: (B, H, 3) physics + thrust-corrected positions [m]
        v_corrected: (B, H, 3) physics + thrust-corrected velocities [m/s]
    """
    B, H, _ = a_rtn.shape
    device = a_rtn.device
    dtype = a_rtn.dtype

    # Allocate output
    r_corrected = torch.zeros_like(r_phys)
    v_corrected = torch.zeros_like(v_phys)

    # Initial state: t0 has no correction (thrust hasn't acted yet)
    dr = torch.zeros(B, 3, device=device, dtype=dtype)
    dv = torch.zeros(B, 3, device=device, dtype=dtype)

    r_corrected[:, 0, :] = r_phys[:, 0, :]
    v_corrected[:, 0, :] = v_phys[:, 0, :]

    dt2_half = 0.5 * dt * dt

    for k in range(H):
        # RTN thrust at step k → ECI
        R_k = rtn_basis[:, k, :, :]  # (B, 3, 3)
        a_eci_k = torch.bmm(R_k, a_rtn[:, k, :].unsqueeze(-1)).squeeze(-1)  # (B, 3)

        # Update corrections
        dr_new = dr + dt * dv + dt2_half * a_eci_k
        dv_new = dv + dt * a_eci_k

        dr, dv = dr_new, dv_new

        # Apply correction to physics trajectory
        r_corrected[:, k, :] = r_phys[:, k, :] + dr
        v_corrected[:, k, :] = v_phys[:, k, :] + dv

    return r_corrected, v_corrected


def compute_rtn_basis(r: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute RTN→ECI rotation matrices from position and velocity.

    Args:
        r: (B, H, 3) positions [m]
        v: (B, H, 3) velocities [m/s]

    Returns:
        C: (B, H, 3, 3) rotation matrices [R_hat, T_hat, N_hat] as columns
    """
    rn = torch.norm(r, dim=-1, keepdim=True).clamp(min=1e-3)
    R_hat = r / rn
    speed = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-12)
    T_hat = v / speed
    N_hat = torch.cross(R_hat, T_hat, dim=-1)
    N_hat = N_hat / torch.norm(N_hat, dim=-1, keepdim=True).clamp(min=1e-12)
    return torch.stack([R_hat, T_hat, N_hat], dim=-1)


# ===== Test =====

if __name__ == "__main__":
    # Quick smoke test
    B, L, F = 4, 144, 14
    H = 720  # 12h at 60s step

    x = torch.randn(B, L, F)
    mask = torch.ones(B, L)
    t = torch.arange(H, dtype=torch.float32).unsqueeze(0).expand(B, -1) * 60.0  # 60s step

    model = PINNLSTMModel(input_dim=F)
    print(f"PINN+LSTM Model: {count_parameters(model):,} parameters")

    h = model.encode(x, mask)
    print(f"Hidden state: {h.shape}")

    thrust = model.predict_thrust(h, t)
    print(f"Thrust predictions: {thrust.shape} (range: [{thrust.min().item():.2e}, {thrust.max().item():.2e}])")

    # Test propagation
    physics_params = {
        "mu": 3.986004418e14,
        "J2": 1.08262668e-3,
        "Re": 6378137.0,
        "omega": 7.2921150e-5,
        "rho0": 3.6e-12,  # at 550km
        "h0": 550000.0,
        "H_scale": 60000.0,
        "Cd_A_m": 0.002 * 10.0 / 260.0,  # Cd*A/m
    }

    r0 = torch.randn(B, 3) * 1000 + torch.tensor([[6800000.0, 0.0, 0.0]])
    v0 = torch.randn(B, 3) * 10 + torch.tensor([[0.0, 7600.0, 0.0]])

    # Only propagate first few steps for smoke test
    pos = differentiable_propagate_with_thrust_fn(
        r0, v0, dt_output=60.0, n_steps=10,
        thrust_fn=lambda t_q: model.predict_thrust(h, t_q),
        physics_params=physics_params, substeps=10,
    )
    print(f"Propagated positions: {pos.shape}")
    print(f"All tests passed!")
