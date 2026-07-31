"""Train PINN+LSTM on QK-1 OEM precision ephemeris data.

Pipeline:
  1. Load all OEM .dat files → parse → concatenate chronologically
  2. Build sliding-window samples: 144-step history → H-step future
  3. Train PINN+LSTM with differentiable propagation + multi-objective loss
  4. Evaluate: position RMSE, thrust detection, physics baseline comparison

Usage:
  python -m fusion.train_pinn --data-dir <path_to_oem_files> --horizon 12 --epochs 100
"""
from __future__ import annotations

import sys, os, time, math, argparse, json, glob
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from AI融合模型.model_pinn import (
    PINNLSTMModel, differentiable_propagate_with_thrust_fn, count_parameters,
)
from AI融合模型.losses_pinn import PINNLoss
from 系统配置.config_loader import load_config


# ===========================================================================
# QK-1 OEM Data Loader
# ===========================================================================

def parse_qk1_oem_file(filepath: str) -> np.ndarray | None:
    """Parse a single QK-1 CCSDS OEM file.

    Returns: (N, 7) array [t_unix, x_m, y_m, z_m, vx, vy, vz]
             or None if parsing fails.
    """
    data = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line or line[0].isalpha():
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                epoch_str = parts[0].replace('Z', '')
                epoch = datetime.fromisoformat(epoch_str)
                ts = epoch.timestamp()
                x = float(parts[1]) * 1000.0   # km → m
                y = float(parts[2]) * 1000.0
                z = float(parts[3]) * 1000.0
                vx = float(parts[4]) * 1000.0  # km/s → m/s
                vy = float(parts[5]) * 1000.0
                vz = float(parts[6]) * 1000.0
                data.append([ts, x, y, z, vx, vy, vz])
            except (ValueError, IndexError):
                continue

    if len(data) < 200:
        return None

    arr = np.array(data, dtype=np.float64)
    # Sort by time
    arr = arr[arr[:, 0].argsort()]
    return arr


def load_qk1_dataset(data_dir: str) -> np.ndarray:
    """Load all QK-1 OEM files and concatenate into continuous timeline.

    Returns: (N_total, 7) [t_unix, x, y, z, vx, vy, vz] in J2000, meters
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.dat")) + sorted(data_dir.glob("*.txt"))
    if not files:
        # Try recursive
        files = sorted(data_dir.rglob("QK_*_OEM_*.dat"))

    if not files:
        raise FileNotFoundError(f"No QK-1 OEM files found in {data_dir}")

    print(f"[Data] Found {len(files)} OEM files")

    segments = []
    n_failed = 0
    for fp in files:
        arr = parse_qk1_oem_file(str(fp))
        if arr is not None:
            segments.append(arr)
            if len(segments) % 100 == 0:
                print(f"  Loaded {len(segments)}/{len(files)} files...")
        else:
            n_failed += 1

    if not segments:
        raise RuntimeError(f"No valid data from {len(files)} files ({n_failed} failed)")

    full = np.concatenate(segments, axis=0)
    full = full[full[:, 0].argsort()]

    # Deduplicate by time
    _, unique_idx = np.unique(full[:, 0], return_index=True)
    full = full[np.sort(unique_idx)]

    t0 = full[0, 0]
    t_end = full[-1, 0]
    span_days = (t_end - t0) / 86400.0

    print(f"[Data] Total: {len(full):,} records, {span_days:.0f} days")
    print(f"[Data] Range: {datetime.fromtimestamp(t0, tz=timezone.utc).strftime('%Y-%m-%d')}"
          f" → {datetime.fromtimestamp(t_end, tz=timezone.utc).strftime('%Y-%m-%d')}")
    print(f"[Data] Failed files: {n_failed}")

    return full


def find_continuous_segments(data: np.ndarray, max_gap_s: float = 3600.0
                             ) -> list[np.ndarray]:
    """Split data into continuous segments (no gaps > max_gap_s).

    Returns: list of (N_i, 7) arrays
    """
    t = data[:, 0]
    gaps = np.diff(t)
    break_idx = np.where(gaps > max_gap_s)[0] + 1

    if len(break_idx) == 0:
        return [data]

    segments = []
    prev = 0
    for bi in break_idx:
        if bi - prev >= 200:
            segments.append(data[prev:bi])
        prev = bi
    if len(data) - prev >= 200:
        segments.append(data[prev:])

    return segments


# ===========================================================================
# Sample Building
# ===========================================================================

def build_samples(data: np.ndarray, history_len: int = 144,
                  horizon: int = 720, step_s: float = 60.0,
                  stride: int = 72) -> dict:
    """Build training samples from continuous data.

    Each sample:
      history: (L, F) — L steps of features
      target: (H, 6) — H steps of [r, v] in ECI
      init_state: (6,) — ECI state at prediction start

    Args:
        data: (N, 7) [t_unix, x, y, z, vx, vy, vz]
        history_len: L, number of input steps
        horizon: H, number of prediction steps
        step_s: time step between samples [s]
        stride: sliding window stride in steps

    Returns:
        dict with keys: histories, targets, init_states, t0_list
    """
    N = len(data)
    min_len = history_len + horizon

    if N < min_len:
        return None

    histories = []
    targets = []
    init_states = []
    t0_list = []

    positions = data[:, 1:4]  # (N, 3)
    velocities = data[:, 4:7]  # (N, 3)
    times = data[:, 0]

    for start in range(0, N - min_len + 1, stride):
        end_input = start + history_len
        end_target = end_input + horizon

        if end_target > N:
            break

        # History: [pos, vel] at L steps
        hist = np.zeros((history_len, 7), dtype=np.float32)
        hist[:, :3] = positions[start:end_input]
        hist[:, 3:6] = velocities[start:end_input]
        hist[:, 6] = times[start:end_input] - times[start]  # relative time

        # Target: [pos, vel] at H steps
        tgt = np.zeros((horizon, 6), dtype=np.float32)
        tgt[:, :3] = positions[end_input:end_target]
        tgt[:, 3:6] = velocities[end_input:end_target]

        # Initial state for propagation = first target state
        init_s = np.concatenate([
            positions[end_input],
            velocities[end_input],
        ])

        histories.append(hist)
        targets.append(tgt)
        init_states.append(init_s)
        t0_list.append(times[end_input])

    if len(histories) == 0:
        return None

    return {
        "histories": np.stack(histories),
        "targets": np.stack(targets),
        "init_states": np.stack(init_states),
        "t0": np.array(t0_list),
    }


def build_samples_from_segments(segments: list[np.ndarray],
                                history_len: int = 144,
                                horizon_steps: int = 720,
                                step_s: float = 60.0,
                                stride: int = 72) -> dict:
    """Build samples from all continuous segments."""
    all_h, all_t, all_i, all_t0 = [], [], [], []
    n_segments_used = 0

    for seg in segments:
        result = build_samples(seg, history_len, horizon_steps, step_s, stride)
        if result is not None:
            all_h.append(result["histories"])
            all_t.append(result["targets"])
            all_i.append(result["init_states"])
            all_t0.append(result["t0"])
            n_segments_used += 1

    if not all_h:
        raise RuntimeError("No valid samples — need longer continuous data segments")

    return {
        "histories": np.concatenate(all_h),
        "targets": np.concatenate(all_t),
        "init_states": np.concatenate(all_i),
        "t0": np.concatenate(all_t0),
        "n_segments": n_segments_used,
    }


# ===========================================================================
# Feature Normalization
# ===========================================================================

class FeatureNormalizer:
    """Normalize orbital features to ~[0, 1] range."""

    def __init__(self, Re: float = 6378137.0, Vs: float = 7905.0):
        self.Re = Re
        self.Vs = Vs

    def normalize_history(self, hist: np.ndarray) -> np.ndarray:
        """hist: (B, L, 7) [x, y, z, vx, vy, vz, dt]"""
        out = hist.copy()
        out[..., :3] /= self.Re
        out[..., 3:6] /= self.Vs
        out[..., 6] /= 86400.0  # seconds → days
        return out

    def denorm_positions(self, pos: torch.Tensor) -> torch.Tensor:
        return pos * self.Re

    def denorm_velocities(self, vel: torch.Tensor) -> torch.Tensor:
        return vel * self.Vs


# ===========================================================================
# Training
# ===========================================================================

class QK1Dataset(torch.utils.data.Dataset):
    def __init__(self, samples: dict, normalizer: FeatureNormalizer):
        self.histories = normalizer.normalize_history(samples["histories"])
        self.targets = samples["targets"].copy()  # (B, H, 6) raw
        self.init_states = samples["init_states"].copy()  # (B, 6) raw
        self.t0 = samples["t0"].copy()

        # Normalize targets too for loss computation
        self.targets_norm = self.targets.copy()
        self.targets_norm[..., :3] /= normalizer.Re
        self.targets_norm[..., 3:6] /= normalizer.Vs

        self.Re = normalizer.Re
        self.Vs = normalizer.Vs
        self.N = len(self.histories)

    def __len__(self): return self.N

    def __getitem__(self, idx):
        return {
            "history": torch.from_numpy(self.histories[idx]),
            "target": torch.from_numpy(self.targets_norm[idx]),
            "target_raw": torch.from_numpy(self.targets[idx]),
            "init_state": torch.from_numpy(self.init_states[idx]),
            "t0": self.t0[idx],
        }


def collate_fn(batch: list[dict]) -> dict:
    B = len(batch)
    L = batch[0]["history"].shape[0]
    H = max(b["target"].shape[0] for b in batch)

    F = batch[0]["history"].shape[1]
    histories = torch.zeros(B, L, F)
    targets_norm = torch.zeros(B, H, 6)
    targets_raw = torch.zeros(B, H, 6)
    init_states = torch.zeros(B, 6)
    masks = torch.zeros(B, L)
    pos_masks = torch.zeros(B, H)
    n_steps = torch.zeros(B, dtype=torch.long)

    for i, b in enumerate(batch):
        h_actual = b["target"].shape[0]
        histories[i] = b["history"]
        targets_norm[i, :h_actual] = b["target"]
        targets_raw[i, :h_actual] = b["target_raw"]
        init_states[i] = b["init_state"]
        masks[i] = 1.0
        pos_masks[i, :h_actual] = 1.0
        n_steps[i] = h_actual

    return {
        "history": histories, "mask": masks,
        "target_norm": targets_norm, "target_raw": targets_raw,
        "init_state": init_states, "pos_mask": pos_masks,
        "n_steps": n_steps,
    }


def train_pinn(
    data_dir: str,
    horizon_h: float = 12.0,
    history_steps: int = 144,
    step_s: float = 60.0,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 5e-4,
    device_str: str = "cuda",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    save_dir: str = "12_数据与实验结果/models",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    horizon_steps = int(horizon_h * 3600 / step_s)

    print(f"\n{'='*65}")
    print(f"  PINN+LSTM Training — QK-1 Precision Ephemeris")
    print(f"  {history_steps}-step history → {horizon_steps}-step ({horizon_h}h) prediction")
    print(f"{'='*65}\n")

    # ---- Load Data ----
    print("[1/5] Loading QK-1 OEM data...")
    data = load_qk1_dataset(data_dir)

    # Split by time: train → val → test
    N = len(data)
    t_sorted = data[:, 0]
    n_test = int(N * test_ratio)
    n_val = int(N * val_ratio)
    n_train = N - n_val - n_test

    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]

    print(f"  Split: train={n_train:,} ({n_train/1440:.0f}d), "
          f"val={n_val:,} ({n_val/1440:.0f}d), "
          f"test={n_test:,} ({n_test/1440:.0f}d)")

    # Find continuous segments in each split
    train_segs = find_continuous_segments(train_data)
    val_segs = find_continuous_segments(val_data)
    test_segs = find_continuous_segments(test_data)

    print(f"  Continuous segments: train={len(train_segs)}, val={len(val_segs)}, "
          f"test={len(test_segs)}")

    # Build samples
    print("[2/5] Building sliding-window samples...")
    train_samples = build_samples_from_segments(
        train_segs, history_steps, horizon_steps, step_s)
    val_samples = build_samples_from_segments(
        val_segs, history_steps, horizon_steps, step_s,
        stride=history_steps // 4)  # fewer samples for validation
    test_samples = build_samples_from_segments(
        test_segs, history_steps, horizon_steps, step_s,
        stride=history_steps // 4)

    print(f"  Train: {len(train_samples['histories']):,} samples "
          f"(from {train_samples['n_segments']} segments)")
    print(f"  Val:   {len(val_samples['histories']):,} samples")
    print(f"  Test:  {len(test_samples['histories']):,} samples")

    # Normalizer
    norm = FeatureNormalizer()

    # Datasets
    train_ds = QK1Dataset(train_samples, norm)
    val_ds = QK1Dataset(val_samples, norm)
    test_ds = QK1Dataset(test_samples, norm)

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=(device_str == "cuda"))
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=(device_str == "cuda"))
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=(device_str == "cuda"))

    # ---- Model ----
    print("[3/5] Building PINN+LSTM model...")
    model = PINNLSTMModel(
        input_dim=7,  # [x, y, z, vx, vy, vz, dt]
        lstm_hidden=64,
        lstm_layers=2,
        lstm_dropout=0.1,
        num_harmonics=4,
        thrust_bounds=(5e-5, 2e-4, 5e-5),
    ).to(device)

    print(f"  Parameters: {count_parameters(model):,}")

    # Physics params for differentiable propagator
    cfg = load_config("系统配置/default.yaml")
    const = cfg["const"]
    atm = cfg["atmosphere"]
    sc = cfg["spacecraft"]
    physics_params = {
        "mu": const["mu_earth"],
        "J2": const["J2"],
        "Re": const["radius_earth"],
        "omega": const["omega_earth"],
        "rho0": atm["reference_density"],
        "h0": atm["reference_altitude"],
        "H_scale": atm["scale_height"],
        "Cd_A_m": sc["Cd"] * sc["area_drag"] / sc["mass"],
    }

    # ---- Loss ----
    loss_fn = PINNLoss(
        w_position=1.0,
        w_endpoint=3.0,    # endpoint is critical for forecasting
        w_smooth=0.1,      # smooth thrust
        w_energy=0.01,     # physics consistency (low weight initially)
        w_magnitude=0.005,  # light L2 on thrust
        w_boundary=0.5,    # respect physical bounds
        mu=const["mu_earth"],
        thrust_max=(5e-5, 2e-4, 5e-5),
        use_huber=True,
        huber_delta=0.0005,  # ~3m in normalized coords
    ).to(device)

    # ---- Optimizer ----
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2, eta_min=1e-6)

    # ---- AMP for speed ----
    use_amp = (device_str == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ---- Compile model for ~30% speedup (PyTorch >= 2.0) ----
    try:
        if hasattr(torch, 'compile'):
            model = torch.compile(model, mode="reduce-overhead")
            print("  torch.compile enabled")
    except Exception as e:
        print(f"  torch.compile skipped: {e}")

    # ---- Training Loop ----
    print(f"[4/5] Training on {device} (AMP={use_amp})...\n")
    best_val_loss = float("inf")
    patience_counter = 0
    train_history = []

    for epoch in range(epochs):
        t_start = time.time()

        # --- Train ---
        model.train()
        tr_total = 0.0
        tr_components = {"position": 0.0, "endpoint": 0.0, "smoothness": 0.0,
                         "energy": 0.0, "magnitude": 0.0, "boundary": 0.0}
        n_batches = 0

        for batch in train_dl:
            hist = batch["history"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            target_raw = batch["target_raw"].to(device, non_blocking=True)
            init_s = batch["init_state"].to(device, non_blocking=True)
            pos_mask = batch["pos_mask"].to(device, non_blocking=True)
            n_steps = batch["n_steps"]

            B, H_full = target_raw.shape[:2]
            H = n_steps.max().item()

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda") if use_amp else torch.no_grad():
                # Encode history
                h = model.encode(hist, mask)

                # Predict thrust at all H steps
                t_query = torch.arange(H, device=device, dtype=torch.float32) * step_s
                t_query = t_query.unsqueeze(0).expand(B, -1)
                thrust_rtn = model.predict_thrust(h, t_query)

                # Differentiable propagation (substeps=5 for speed)
                r0 = init_s[:, :3]
                v0 = init_s[:, 3:6]

                pos_pred, vel_pred = differentiable_propagate_with_thrust_fn(
                    r0, v0, dt_output=step_s, n_steps=H,
                    thrust_fn=lambda t_q: model.predict_thrust(h, t_q),
                    physics_params=physics_params,
                    substeps=5,
                    return_full=True,
                )

                thrust_eci = _rtn_to_eci_batch(pos_pred, vel_pred, thrust_rtn)

                # Composite loss
                loss, comps = loss_fn(
                    pos_pred, target_raw[..., :3],
                    thrust_rtn, thrust_eci, vel_pred,
                    pos_mask, n_steps,
                    dt=step_s, return_components=True,
                )

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            tr_total += loss.item()
            for k in tr_components:
                tr_components[k] += comps.get(k, 0.0)
            n_batches += 1

        tr_total /= n_batches
        for k in tr_components:
            tr_components[k] /= n_batches

        # --- Validation ---
        model.eval()
        vl_total = 0.0
        vl_pos_rmse = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_dl:
                hist = batch["history"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                target_raw = batch["target_raw"].to(device, non_blocking=True)
                init_s = batch["init_state"].to(device, non_blocking=True)
                pos_mask = batch["pos_mask"].to(device, non_blocking=True)
                n_steps = batch["n_steps"]

                B, _, _ = target_raw.shape
                H = n_steps.max().item()

                with torch.amp.autocast("cuda") if use_amp else torch.no_grad():
                    h = model.encode(hist, mask)
                    t_query = torch.arange(H, device=device, dtype=torch.float32) * step_s
                    t_query = t_query.unsqueeze(0).expand(B, -1)

                    thrust_rtn = model.predict_thrust(h, t_query)
                    r0 = init_s[:, :3]
                    v0 = init_s[:, 3:6]

                    pos_pred, vel_pred = differentiable_propagate_with_thrust_fn(
                        r0, v0, dt_output=step_s, n_steps=H,
                        thrust_fn=lambda t_q: model.predict_thrust(h, t_q),
                        physics_params=physics_params,
                        substeps=5, return_full=True,
                    )

                    thrust_eci = _rtn_to_eci_batch(pos_pred, vel_pred, thrust_rtn)
                    loss, _ = loss_fn(
                        pos_pred, target_raw[..., :3],
                        thrust_rtn, thrust_eci, vel_pred,
                        pos_mask, n_steps, dt=step_s, return_components=True,
                    )
                vl_total += loss.item()

                # Position RMSE
                diff = (pos_pred - target_raw[..., :3]) * pos_mask.unsqueeze(-1)
                rmse = torch.sqrt((diff ** 2).sum(dim=-1).mean())
                vl_pos_rmse += rmse.item()
                n_val_batches += 1

        vl_total /= n_val_batches
        vl_pos_rmse /= n_val_batches
        sched.step()

        # --- Logging ---
        is_best = vl_total < best_val_loss
        if is_best:
            best_val_loss = vl_total
            patience_counter = 0
            # Save best model
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch + 1,
                "val_loss": vl_total,
                "val_pos_rmse": vl_pos_rmse,
            }, save_path / f"pinn_lstm_qk1_{horizon_h}h_best.pt")
        else:
            patience_counter += 1

        dt_epoch = time.time() - t_start
        lr_now = opt.param_groups[0]["lr"]

        if epoch % 5 == 0 or is_best or epoch < 5:
            print(f"E {epoch+1:4d} | tr={tr_total:.4f} vl={vl_total:.4f} | "
                  f"posRMSE={vl_pos_rmse:.0f}m | "
                  f"components: pos={tr_components['position']:.3f} "
                  f"end={tr_components['endpoint']:.3f} "
                  f"sm={tr_components['smoothness']:.3f} "
                  f"mag={tr_components['magnitude']:.3f} | "
                  f"lr={lr_now:.2e} | {dt_epoch:.1f}s"
                  f"{' *' if is_best else ''}")

        if patience_counter >= 25:
            print(f"  Early stop at epoch {epoch + 1}")
            break

        train_history.append({
            "epoch": epoch + 1,
            "train_loss": tr_total,
            "val_loss": vl_total,
            "val_pos_rmse": vl_pos_rmse,
        })

    # ---- Final Evaluation ----
    print(f"\n[5/5] Final evaluation on test set...")
    model.eval()

    # Load best checkpoint
    best_ckpt = torch.load(save_path / f"pinn_lstm_qk1_{horizon_h}h_best.pt",
                           map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model"])

    test_rmse_list = []
    test_endpoint_rmse = []
    thrust_predictions = []

    with torch.no_grad():
        for batch in test_dl:
            hist = batch["history"].to(device)
            mask = batch["mask"].to(device)
            target_raw = batch["target_raw"].to(device)
            init_s = batch["init_state"].to(device)
            pos_mask = batch["pos_mask"].to(device)
            n_steps = batch["n_steps"]

            B, _, _ = target_raw.shape
            H = n_steps.max().item()

            h = model.encode(hist, mask)
            t_query = torch.arange(H, device=device, dtype=torch.float32) * step_s
            t_query = t_query.unsqueeze(0).expand(B, -1)
            thrust_rtn = model.predict_thrust(h, t_query)
            r0 = init_s[:, :3]
            v0 = init_s[:, 3:6]

            pos_pred = differentiable_propagate_with_thrust_fn(
                r0, v0, dt_output=step_s, n_steps=H,
                thrust_fn=lambda t_q: model.predict_thrust(h, t_q),
                physics_params=physics_params,
                substeps=10, return_full=False,
            )

            # Per-sample RMSE
            for j in range(B):
                h_j = n_steps[j].item()
                diff = pos_pred[j, :h_j] - target_raw[j, :h_j, :3]
                rmse = math.sqrt((diff ** 2).sum().item() / h_j)
                test_rmse_list.append(rmse)
                # Endpoint error
                end_err = torch.norm(diff[-1]).item()
                test_endpoint_rmse.append(end_err)

            thrust_predictions.append(thrust_rtn.cpu().numpy())

    test_rmse = np.mean(test_rmse_list)
    test_end_rmse = np.mean(test_endpoint_rmse)
    thrust_all = np.concatenate([t.reshape(-1, 3) for t in thrust_predictions], axis=0)

    print(f"\n{'='*65}")
    print(f"  RESULTS — {horizon_h}h Prediction")
    print(f"{'='*65}")
    print(f"  Test position RMSE (avg):   {test_rmse:.0f} m")
    print(f"  Test endpoint error:        {test_end_rmse:.0f} m")
    print(f"  Thrust magnitude (mean):    [{thrust_all[:, 0].mean():.2e}, "
          f"{thrust_all[:, 1].mean():.2e}, {thrust_all[:, 2].mean():.2e}] m/s²")
    print(f"  Thrust magnitude (max):     [{np.abs(thrust_all[:, 0]).max():.2e}, "
          f"{np.abs(thrust_all[:, 1]).max():.2e}, {np.abs(thrust_all[:, 2]).max():.2e}] m/s²")

    # Physics baseline: pure 2-body+J2+drag propagation (no thrust)
    # This gives the "without AI" benchmark
    physics_rmse_list = []
    for batch in test_dl:
        init_s = batch["init_state"]
        target_raw = batch["target_raw"]
        pos_mask = batch["pos_mask"]
        n_steps = batch["n_steps"]
        B = init_s.shape[0]
        H = n_steps.max().item()

        for j in range(B):
            h_j = n_steps[j].item()
            r0_p = init_s[j, :3].to(device)
            v0_p = init_s[j, 3:6].to(device)
            t_q = torch.arange(h_j, device=device, dtype=torch.float32) * step_s
            t_q = t_q.unsqueeze(0).expand(1, -1)

            pos_phys = differentiable_propagate_with_thrust_fn(
                r0_p.unsqueeze(0), v0_p.unsqueeze(0),
                dt_output=step_s, n_steps=h_j,
                thrust_fn=lambda t: torch.zeros(1, t.shape[1], 3, device=device),
                physics_params=physics_params,
                substeps=10, return_full=False,
            )

            diff_phys = pos_phys[0] - target_raw[j, :h_j, :3].to(device)
            rmse_phys = math.sqrt((diff_phys ** 2).sum().item() / h_j)
            physics_rmse_list.append(rmse_phys)

    physics_rmse = np.mean(physics_rmse_list)
    improvement = (physics_rmse - test_rmse) / physics_rmse * 100

    print(f"\n  Physics-only baseline RMSE:  {physics_rmse:.0f} m")
    print(f"  PINN+LSTM improvement:       {improvement:+.1f}%")
    print(f"\n  Model saved: {save_path / f'pinn_lstm_qk1_{horizon_h}h_best.pt'}")
    print(f"{'='*65}\n")

    return model, {
        "test_rmse": test_rmse,
        "test_endpoint_rmse": test_end_rmse,
        "physics_rmse": physics_rmse,
        "improvement_pct": improvement,
        "thrust_stats": thrust_all,
    }


def _rtn_to_eci_batch(r: torch.Tensor, v: torch.Tensor,
                      a_rtn: torch.Tensor) -> torch.Tensor:
    """Batch RTN→ECI conversion for thrust vectors.

    r, v: (B, H, 3)
    a_rtn: (B, H, 3)
    Returns: a_eci (B, H, 3)
    """
    rn = torch.norm(r, dim=-1, keepdim=True).clamp(min=1e-3)
    R_hat = r / rn
    speed = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-12)
    T_hat = v / speed
    N_hat = torch.cross(R_hat, T_hat, dim=-1)
    N_hat = N_hat / torch.norm(N_hat, dim=-1, keepdim=True).clamp(min=1e-12)

    a_eci = (R_hat * a_rtn[..., 0:1] + T_hat * a_rtn[..., 1:2] +
             N_hat * a_rtn[..., 2:3])
    return a_eci


# ===========================================================================
# CLI
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description="PINN+LSTM Training on QK-1 OEM Data")
    ap.add_argument("--data-dir", required=True,
                    help="Directory containing QK-1 OEM .dat files")
    ap.add_argument("--horizon", type=float, default=12.0,
                    help="Prediction horizon in hours (12 or 24)")
    ap.add_argument("--history-steps", type=int, default=144,
                    help="Number of history steps")
    ap.add_argument("--step-s", type=float, default=60.0,
                    help="Time step between samples in seconds")
    ap.add_argument("--epochs", type=int, default=100,
                    help="Number of training epochs")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-dir", default="12_数据与实验结果/models")
    args = ap.parse_args()

    train_pinn(
        data_dir=args.data_dir,
        horizon_h=args.horizon,
        history_steps=args.history_steps,
        step_s=args.step_s,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
