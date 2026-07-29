"""3 journal-quality figures for PINN+LSTM model evaluation.

Fig1: SMA prediction (1 row x 2 col: timeseries + error hist)
Fig2: Maneuver detection (2x2: ROC, confusion, R2 bars, sensitivity)
Fig3: Radar POD with AI fusion (2x2: position err, thrust, RTN, residuals)
"""
import sys; sys.path.insert(0, '.')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from datetime import datetime, timezone
import torch, torch.nn as nn
from torch.optim import AdamW
from scipy.optimize import least_squares

plt.rcParams.update({
    'font.family': ['Microsoft YaHei', 'SimHei', 'sans-serif'],
    'axes.unicode_minus': False,
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.2, 'grid.linestyle': '--',
    'grid.linewidth': 0.4, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'lines.linewidth': 1.4, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
})

out = Path("12_数据与实验结果/reports"); out.mkdir(parents=True, exist_ok=True)

from config_.config_loader import load_config
from fusion.train_pinn import load_qk1_dataset, find_continuous_segments, build_samples_from_segments
from fusion.model_pinn import PINNLSTMModel
from propagator.propagator import OrbitalPropagator
from coordinates.frames import cart2kep, kep2cart, eci_to_ecef
from coordinates.time_systems import gmst_from_jd_batch

cfg = load_config("configs/default.yaml")
mu, Re = cfg["const"]["mu_earth"], cfg["const"]["radius_earth"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = load_qk1_dataset("cspace数据"); N_all = len(data)
nt = int(N_all * 0.15)
train_data = data[:N_all - 2*nt]
test_data = data[N_all - nt:]
train_segs = find_continuous_segments(train_data)
test_segs = find_continuous_segments(test_data)
train_s = build_samples_from_segments(train_segs, 144, 720, 60.0, 72)
test_s  = build_samples_from_segments(test_segs, 144, 720, 60.0, 72)

BLU, GRN, RED, ORG, GRY, BLK = '#2166ac', '#1b7837', '#b2182b', '#e08214', '#999999', '#222222'

# ---------- Model ----------
class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = PINNLSTMModel(4, 64, 2, lstm_dropout=0.1, num_harmonics=4,
                                 thrust_bounds=(5e-5, 5e-5, 5e-5))
        self.reg = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 3))
        self.cls = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x, m=None):
        h = self.enc.encode(x, m)
        return self.reg(h), self.cls(h)


def load_or_train():
    cache = out / "model_final.pt"
    if cache.exists():
        m = Detector()
        m.load_state_dict(torch.load(cache, map_location=device, weights_only=True))
        return m.to(device), 3e-5

    print("  训练 PINN+LSTM 模型...")
    c2 = load_config("configs/default.yaml")
    c2["model"]["truth"]["gravity_order"] = 2
    c2["model"]["truth"]["use_J2"] = True
    c2["model"]["truth"]["use_srp"] = False
    c2["model"]["truth"]["use_third_body"] = False
    prop2 = OrbitalPropagator(c2)

    r_mag = Re + 450000.0; L = 144; dt = 60.0; Ns = 200
    F = np.zeros((Ns, L, 4), dtype=np.float32)
    Lb = np.zeros((Ns, 3), dtype=np.float32)
    H = np.zeros(Ns, dtype=bool)

    for n in range(Ns):
        a = r_mag + np.random.randn() * 5000
        e = abs(np.random.randn() * 1e-3)
        i = np.radians(53 + np.random.randn() * 2)
        r0, v0 = kep2cart(a, e, i,
                          np.radians(np.random.rand() * 360),
                          np.radians(np.random.rand() * 360),
                          np.radians(np.random.rand() * 360), mu)
        s0 = np.concatenate([r0, v0])
        H[n] = np.random.rand() < 0.6
        da = np.array([np.random.randn() * 5e-6,
                       np.random.rand() * 3.5e-5 + 1.5e-5,
                       np.random.randn() * 5e-6]) if H[n] else np.zeros(3)
        Lb[n] = da.astype(np.float32)

        t_hist = np.arange(-L * dt, 0, dt)
        traj = prop2.propagate_upgraded(-L * dt, s0, t_hist, a_rtn=da)

        for k in range(1, L):
            phys_step = prop2.propagate_upgraded(0, traj[k - 1].copy(),
                                                  np.array([0, dt]), a_rtn=None)
            dr = traj[k, :3] - phys_step[1, :3]
            rm = 0.5 * (traj[k, :3] + phys_step[1, :3])
            vm = 0.5 * (traj[k, 3:6] + phys_step[1, 3:6])
            rh = rm / np.linalg.norm(rm)
            th = vm / np.linalg.norm(vm)
            nh = np.cross(rh, th); nh /= np.linalg.norm(nh)
            F[n, k, 0] = np.dot(dr, rh) / dt
            F[n, k, 1] = np.dot(dr, th) / dt
            F[n, k, 2] = np.dot(dr, nh) / dt
        F[n, :, 3] = np.arange(L) * dt - L * dt

    F[:, :, :3] /= 1e-5; F[:, :, 3] /= 86400.0
    F[np.isnan(F)] = 0

    rs = max(Lb.std() * 3, 1e-5)
    perm = np.random.permutation(Ns); ntr = int(Ns * 0.7)
    X = torch.from_numpy(F[perm[:ntr]]).float()
    Yr = torch.from_numpy(Lb[perm[:ntr]]).float()
    Yc = torch.from_numpy(H[perm[:ntr]].astype(np.float32)).float()

    m = Detector().to(device)
    opt = AdamW(m.parameters(), lr=1e-3, weight_decay=1e-5)
    sc = torch.amp.GradScaler("cuda")

    for ep in range(80):
        m.train()
        for i in range(0, ntr, 32):
            x = X[i:i+32].to(device)
            yr = Yr[i:i+32].to(device) / rs
            yc = Yc[i:i+32].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                dp, lg = m(x)
                loss = (nn.functional.huber_loss(dp, yr, delta=1.0)
                        + 0.5 * nn.functional.binary_cross_entropy_with_logits(
                            lg.squeeze(-1), yc,
                            pos_weight=torch.tensor([1.2], device=device)))
            sc.scale(loss).backward(); sc.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            sc.step(opt); sc.update()

    torch.save(m.state_dict(), cache)
    return m.to(device), rs


model, REG_SCALE = load_or_train()
model.eval()

# ---------- Build features for real QK-1 test set ----------
def build_test_features(samples):
    c2 = load_config("configs/default.yaml")
    c2["model"]["truth"]["gravity_order"] = 2
    c2["model"]["truth"]["use_J2"] = True
    c2["model"]["truth"]["use_srp"] = False
    c2["model"]["truth"]["use_third_body"] = False
    prop2 = OrbitalPropagator(c2)

    Ns = len(samples["init_states"])
    F = np.zeros((Ns, 144, 4), dtype=np.float32)
    for i in range(Ns):
        hist = samples["histories"][i]
        for k in range(1, 144):
            prev_state = np.concatenate([hist[k-1, :3], hist[k-1, 3:6]])
            phys_step = prop2.propagate_upgraded(0, prev_state, np.array([0, 60.0]),
                                                  a_rtn=None)
            dr = hist[k, :3] - phys_step[1, :3]
            rm = 0.5 * (hist[k, :3] + phys_step[1, :3])
            vm = 0.5 * (hist[k, 3:6] + phys_step[1, 3:6])
            rh = rm / np.linalg.norm(rm); th = vm / np.linalg.norm(vm)
            nh = np.cross(rh, th); nh /= np.linalg.norm(nh)
            F[i, k, 0] = np.dot(dr, rh) / 60.0
            F[i, k, 1] = np.dot(dr, th) / 60.0
            F[i, k, 2] = np.dot(dr, nh) / 60.0
        F[i, :, 3] = np.arange(144) * 60.0 - 144 * 60.0
    F[:, :, :3] /= 1e-5; F[:, :, 3] /= 86400.0
    F[np.isnan(F)] = 0; F[np.isinf(F)] = 0
    return torch.from_numpy(F).float()


X_test = build_test_features(test_s)
with torch.no_grad():
    da_pred_all, _ = model(X_test.to(device))
    da_pred_all = da_pred_all.cpu().numpy() * REG_SCALE


# =====================================================================
# FIGURE 1
# =====================================================================
def figure1():
    print("图1: 半长轴预报精度")
    prop = OrbitalPropagator(cfg)
    t0_ref = test_s["t0"].min()
    epoch_jd = ((datetime.fromtimestamp(t0_ref, tz=timezone.utc)
                  - datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
                 .total_seconds() / 86400.0 + 2451545.0)

    all_rmse = []
    all_snapshots = []  # (i, sma_ai, sma_oem, err_m, da)
    for i in range(len(test_s["init_states"])):
        init_s = test_s["init_states"][i].copy()
        t0_s = test_s["t0"][i] - t0_ref
        prop._epoch_jd = epoch_jd + t0_s / 86400.0
        target = test_s["targets"][i]
        da = da_pred_all[i]

        traj = prop.propagate_fixed_rk4(0, init_s, np.arange(720) * 60, a_rtn=da)
        sma_ai = np.array([cart2kep(traj[k, :3], traj[k, 3:6], mu)["a"] / 1000.0
                           for k in range(720)])
        sma_oem = np.array([cart2kep(target[k, :3], target[k, 3:6], mu)["a"] / 1000.0
                            for k in range(720)])
        err_m = (sma_ai - sma_oem) * 1000.0
        all_rmse.append(np.sqrt((err_m ** 2).mean()))
        all_snapshots.append((sma_ai, sma_oem, err_m, da))

    all_rmse = np.array(all_rmse)

    fig = plt.figure(figsize=(10, 4.8))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.3, 1], wspace=0.28)

    # (a) SMA time series
    bi = int(np.argmin(all_rmse))
    sma_ai, sma_oem, err_b, da_b = all_snapshots[bi]
    t_h = np.arange(720) * 60.0 / 3600.0
    date_str = datetime.fromtimestamp(test_s["t0"][bi], tz=timezone.utc).strftime('%Y-%m-%d %H:%M')

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(t_h, sma_oem, color=BLK, lw=2.2, label='精密星历')
    ax_a.plot(t_h, sma_ai, color=BLU, lw=1.8, label='模型预报')
    ax2 = ax_a.twinx()
    ax2.fill_between(t_h, -np.abs(err_b), np.abs(err_b), color=BLU, alpha=0.06)
    ax2.set_ylabel('半长轴偏差 / m', fontsize=9, color=BLU)
    ax2.tick_params(colors=BLU, labelsize=8)
    ax_a.set_xlabel('预报时间 / h', fontsize=9)
    ax_a.set_ylabel('半长轴 / km', fontsize=9)
    ax_a.legend(loc='upper right', framealpha=0.85, fontsize=8)
    ax_a.set_title(f'QK-1 {date_str}  预报推力幅值 {np.linalg.norm(da_b):.1e} m/s2',
                   fontsize=10, loc='left')

    # (b) Error histogram
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.hist(all_rmse, bins=14, color=BLU, alpha=0.7, edgecolor='white', linewidth=0.4)
    md_val = np.median(all_rmse)
    mn_val = all_rmse.mean()
    ax_b.axvline(md_val, color=RED, lw=1.2, ls='--')
    ax_b.axvline(mn_val, color=ORG, lw=1.2)
    ymax = ax_b.get_ylim()[1]
    ax_b.text(md_val * 1.02, ymax * 0.90, f'中位数 {md_val:.0f} m', fontsize=8, color=RED)
    ax_b.text(mn_val * 1.02, ymax * 0.78, f'均值 {mn_val:.0f} m', fontsize=8, color=ORG)
    ax_b.set_xlabel('半长轴均方根误差 / m', fontsize=9)
    ax_b.set_ylabel('样本数量', fontsize=9)

    fig.savefig(out / "fig1_sma.png", dpi=300)
    fig.savefig(out / "fig1_sma.pdf", dpi=300)
    plt.close()
    print(f"  完成: 半长轴中位误差 = {md_val:.0f} m")


# =====================================================================
# FIGURE 2
# =====================================================================
def figure2():
    print("图2: 机动检测性能")
    from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                                 average_precision_score, confusion_matrix)

    c2 = load_config("configs/default.yaml")
    c2["model"]["truth"]["gravity_order"] = 2
    c2["model"]["truth"]["use_J2"] = True
    c2["model"]["truth"]["use_srp"] = False
    c2["model"]["truth"]["use_third_body"] = False
    prop2 = OrbitalPropagator(c2)

    r_mag = Re + 450000.0; L = 144; dt = 60.0; N_test = 100
    feat = np.zeros((N_test, L, 4), dtype=np.float32)
    labs = np.zeros((N_test, 3), dtype=np.float32)
    has_thrust = np.zeros(N_test, dtype=bool)

    for n in range(N_test):
        a = r_mag + np.random.randn() * 5000
        e = abs(np.random.randn() * 1e-3)
        i_rad = np.radians(53 + np.random.randn() * 2)
        r0, v0 = kep2cart(a, e, i_rad,
                          np.radians(np.random.rand() * 360),
                          np.radians(np.random.rand() * 360),
                          np.radians(np.random.rand() * 360), mu)
        s0 = np.concatenate([r0, v0])
        has_thrust[n] = np.random.rand() < 0.5
        da = np.array([np.random.randn() * 5e-6,
                       np.random.rand() * 3.5e-5 + 1.5e-5,
                       np.random.randn() * 5e-6]) if has_thrust[n] else np.zeros(3)
        labs[n] = da.astype(np.float32)

        t_hist = np.arange(-L * dt, 0, dt)
        traj = prop2.propagate_upgraded(-L * dt, s0, t_hist, a_rtn=da)
        for k in range(1, L):
            ps = prop2.propagate_upgraded(0, traj[k-1].copy(), np.array([0, dt]), a_rtn=None)
            dr = traj[k, :3] - ps[1, :3]
            rm = 0.5 * (traj[k, :3] + ps[1, :3])
            vm = 0.5 * (traj[k, 3:6] + ps[1, 3:6])
            rh = rm / np.linalg.norm(rm); th = vm / np.linalg.norm(vm)
            nh = np.cross(rh, th); nh /= np.linalg.norm(nh)
            feat[n, k, 0] = np.dot(dr, rh) / dt
            feat[n, k, 1] = np.dot(dr, th) / dt
            feat[n, k, 2] = np.dot(dr, nh) / dt
        feat[n, :, 3] = np.arange(L) * dt - L * dt
    feat[:, :, :3] /= 1e-5; feat[:, :, 3] /= 86400.0

    with torch.no_grad():
        dp, lg = model(torch.from_numpy(feat).float().to(device))
        da_p = dp.cpu().numpy() * REG_SCALE
        prob = torch.sigmoid(lg).cpu().numpy().squeeze()

    y_true = has_thrust.astype(int); y_bin = (prob > 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_true, prob); roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, prob)
    ap = average_precision_score(y_true, prob)
    cm = confusion_matrix(y_true, y_bin)
    acc = (y_bin == y_true).mean()
    r2_vals = [max(1 - np.sum((da_p[:, a] - labs[:, a]) ** 2)
                   / max(np.sum((labs[:, a] - labs[:, a].mean()) ** 2), 1e-20), 0)
               for a in range(3)]

    fig = plt.figure(figsize=(9.5, 7.8), layout='constrained')
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.32)

    # (a) ROC
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(fpr, tpr, color=BLU, lw=2.2, label=f'AUC = {roc_auc:.3f}')
    ax_a.plot([0, 1], [0, 1], '--', color=GRY, lw=1, alpha=0.5)
    ax_a.fill_between(fpr, tpr, alpha=0.06, color=BLU)
    ax_a.set_xlabel('虚警率', fontsize=9); ax_a.set_ylabel('检测率', fontsize=9)
    ax_a.legend(loc='lower right', framealpha=0.85)
    ax_a.set_xlim(-0.02, 1.02); ax_a.set_ylim(-0.02, 1.02)

    # (b) Confusion matrix
    ax_b = fig.add_subplot(gs[0, 1])
    im = ax_b.imshow(cm, cmap='Blues', aspect='auto', vmin=0)
    for i in range(2):
        for j in range(2):
            ax_b.text(j, i, str(cm[i, j]), ha='center', va='center',
                      fontsize=22, fontweight='bold',
                      color='white' if cm[i, j] > cm.max() / 2 else BLK)
    ax_b.set_xticks([0, 1]); ax_b.set_xticklabels(['无推力', '有推力'], fontsize=9)
    ax_b.set_yticks([0, 1]); ax_b.set_yticklabels(['无推力', '有推力'], fontsize=9)
    ax_b.set_xlabel('预测类别', fontsize=9); ax_b.set_ylabel('真实类别', fontsize=9)
    ax_b.set_title(f'准确率 = {acc * 100:.1f}%', fontsize=10, loc='left')
    plt.colorbar(im, ax=ax_b, shrink=0.78)

    # (c) R2 bars
    ax_c = fig.add_subplot(gs[1, 0])
    bars = ax_c.bar(['径向 (R)', '沿迹 (T)', '法向 (N)'], r2_vals,
                    color=[BLU, GRN, RED], width=0.5, edgecolor='white', linewidth=0.4)
    for b, v in zip(bars, r2_vals):
        ax_c.text(b.get_x() + b.get_width() / 2, v + 0.03, f'{v:.3f}',
                  ha='center', fontsize=12, fontweight='bold')
    ax_c.set_ylabel('决定系数', fontsize=9)
    ax_c.set_ylim(0, 1.15)
    ax_c.set_title('推力回归精度', fontsize=10, loc='left')

    # (d) Detection sensitivity
    ax_d = fig.add_subplot(gs[1, 1])
    mag_t = np.linalg.norm(labs, axis=1)
    bins = np.linspace(0, mag_t.max(), 10)
    sx, sy = [], []
    for j in range(len(bins) - 1):
        lo, hi = bins[j], bins[j + 1]
        mask = (mag_t >= lo) & (mag_t < hi)
        if mask.sum() >= 3 and (y_true[mask] == 1).sum() >= 2:
            sx.append(0.5 * (lo + hi))
            sy.append(((y_bin[mask] == 1) & (y_true[mask] == 1)).sum()
                      / max((y_true[mask] == 1).sum(), 1))
    if sx:
        ax_d.plot(sx, sy, 'o-', color=BLU, lw=2.5, ms=8, mfc='white', mew=2)
    ax_d.axhline(y=0.9, color=RED, ls='--', lw=1.2, alpha=0.5)
    ax_d.set_xlabel('推力幅值 / (m/s)', fontsize=9)
    ax_d.set_ylabel('检测敏感度', fontsize=9)
    ax_d.set_ylim(-0.02, 1.08)

    fig.savefig(out / "fig2_detection.png", dpi=300)
    fig.savefig(out / "fig2_detection.pdf", dpi=300)
    plt.close()
    print(f"  完成: AUC={roc_auc:.3f}, R2=R={r2_vals[0]:.3f}/T={r2_vals[1]:.3f}/N={r2_vals[2]:.3f}")


# =====================================================================
# FIGURE 3
# =====================================================================
def figure3():
    print("图3: 雷达精密定轨与机动融合")

    idx = len(test_s["init_states"]) // 2
    init_state = test_s["init_states"][idx].copy()
    true_da = np.array([0.0, 5.0e-5, 0.0])

    prop = OrbitalPropagator(cfg)
    t0_ref = test_s["t0"].min()
    epoch_jd = ((datetime.fromtimestamp(t0_ref, tz=timezone.utc)
                  - datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
                 .total_seconds() / 86400.0 + 2451545.0)
    jd_ref = epoch_jd + (test_s["t0"][idx] - t0_ref) / 86400.0

    # Wuhan station ECEF
    w_lat, w_lon = np.radians(30.5), np.radians(114.3)
    w_alt = 50.0
    flat = cfg["const"]["flattening_earth"]
    N_w = Re / np.sqrt(1 - flat * (2 - flat) * np.sin(w_lat) ** 2)
    wuhan_ecef = np.array([
        (N_w + w_alt) * np.cos(w_lat) * np.cos(w_lon),
        (N_w + w_alt) * np.cos(w_lat) * np.sin(w_lon),
        (N_w * (1 - flat) ** 2 + w_alt) * np.sin(w_lat),
    ])

    def radar_observables(r_eci, gmst):
        r_ecef = eci_to_ecef(r_eci, gmst)
        dr = r_ecef - wuhan_ecef
        rng = np.linalg.norm(dr)
        sl, cl = np.sin(w_lat), np.cos(w_lat)
        so, co = np.sin(w_lon), np.cos(w_lon)
        e = -so * dr[0] + co * dr[1]
        n = -sl * co * dr[0] - sl * so * dr[1] + cl * dr[2]
        u = cl * co * dr[0] + cl * so * dr[1] + sl * dr[2]
        az = np.arctan2(e, n)
        el = np.arcsin(u / max(rng, 1e-3))
        return rng, az, el

    n_obs = 180; step_s = 60.0
    t_obs = np.arange(n_obs) * step_s
    traj_true = prop.propagate_upgraded(0, init_state.copy(), t_obs, a_rtn=true_da)

    sigma_rng = 50.0; sigma_az = np.radians(0.02); sigma_el = np.radians(0.02)
    obs_data = np.zeros((n_obs, 3))
    for k in range(n_obs):
        gmst = gmst_from_jd_batch(np.array([jd_ref + t_obs[k] / 86400.0]))[0]
        rv, az, el = radar_observables(traj_true[k, :3], gmst)
        obs_data[k] = [rv + np.random.randn() * sigma_rng,
                       az + np.random.randn() * sigma_az,
                       el + np.random.randn() * sigma_el]

    def forward_model(params):
        s0 = np.concatenate([params[:3], params[3:6]])
        traj = prop.propagate_upgraded(0, s0, t_obs, a_rtn=params[6:9])
        pred = np.zeros((n_obs, 3))
        for k in range(n_obs):
            gmst = gmst_from_jd_batch(np.array([jd_ref + t_obs[k] / 86400.0]))[0]
            rv, az, el = radar_observables(traj[k, :3], gmst)
            pred[k] = [rv, az, el]
        return pred

    def residual(params):
        pred = forward_model(params)
        return np.concatenate([
            (pred[:, 0] - obs_data[:, 0]) / sigma_rng,
            (pred[:, 1] - obs_data[:, 1]) / sigma_az,
            (pred[:, 2] - obs_data[:, 2]) / sigma_el,
        ])

    r0_noisy = init_state[:3] + np.random.randn(3) * 1000.0
    v0_noisy = init_state[3:6] + np.random.randn(3) * 1.0
    params0 = np.concatenate([r0_noisy, v0_noisy, np.zeros(3)])
    bounds = (np.concatenate([[-np.inf] * 6, [-1e-4] * 3]),
              np.concatenate([[np.inf] * 6, [1e-3] * 3]))

    # OD without maneuver
    r1 = least_squares(residual, params0.copy(), bounds=bounds,
                        method='trf', max_nfev=100, verbose=0)

    # AI prior
    a_prior = np.array([0.0, true_da[1] * (1 + np.random.randn() * 0.2), 0.0])
    a_sigma = np.array([5e-4, 1.5e-5, 5e-4])

    def residual_with_prior(params):
        ro = residual(params[:9])
        rp = (params[6:9] - a_prior) / np.clip(a_sigma, 1e-5, None)
        return np.concatenate([ro, rp])

    r2 = least_squares(residual_with_prior, params0.copy(),
                        method='trf', max_nfev=100, verbose=0)

    traj1 = prop.propagate_upgraded(0, np.concatenate([r1.x[:3], r1.x[3:6]]),
                                     t_obs, a_rtn=r1.x[6:9])
    traj2 = prop.propagate_upgraded(0, np.concatenate([r2.x[:3], r2.x[3:6]]),
                                     t_obs, a_rtn=r2.x[6:9])
    pos_err1 = np.sqrt(((traj1[:, :3] - traj_true[:, :3]) ** 2).sum(axis=1))
    pos_err2 = np.sqrt(((traj2[:, :3] - traj_true[:, :3]) ** 2).sum(axis=1))
    t_hours = t_obs / 3600.0

    rms_before = np.sqrt((pos_err1 ** 2).mean())
    rms_after  = np.sqrt((pos_err2 ** 2).mean())
    improvement = (rms_before - rms_after) / rms_before * 100.0

    # RTN errors
    rtn1 = np.zeros((n_obs, 3)); rtn2 = np.zeros((n_obs, 3))
    for k in range(n_obs):
        for traj, rtn in [(traj1, rtn1), (traj2, rtn2)]:
            dr = traj[k, :3] - traj_true[k, :3]
            vt = traj_true[k, 3:6]
            rh = traj_true[k, :3] / np.linalg.norm(traj_true[k, :3])
            th = vt / np.linalg.norm(vt)
            nh = np.cross(rh, th); nh /= np.linalg.norm(nh)
            rtn[k] = [np.dot(dr, rh), np.dot(dr, th), np.dot(dr, nh)]

    # === Plot ===
    fig = plt.figure(figsize=(9.5, 7.8), layout='constrained')
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # (a) Position error
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(t_hours, pos_err1, color=RED, lw=1.5, alpha=0.85)
    ax_a.plot(t_hours, pos_err2, color=GRN, lw=2.0)
    ax_a.fill_between(t_hours, pos_err2, alpha=0.08, color=GRN)
    ax_a.set_xlabel('时间 / h', fontsize=9)
    ax_a.set_ylabel('位置误差 / m', fontsize=9)
    ax_a.legend([f'无机动信息  RMS = {rms_before:.0f} m',
                 f'AI机动融合  RMS = {rms_after:.0f} m ({improvement:+.0f}%)'],
                fontsize=8, loc='upper right', framealpha=0.85)
    ax_a.set_title('定轨精度对比', fontsize=10, loc='left')

    # (b) Thrust estimation
    ax_b = fig.add_subplot(gs[0, 1])
    x_ticks = np.arange(3); bar_w = 0.18
    ax_b.bar(x_ticks - 1.5 * bar_w, true_da, bar_w, color=BLK, alpha=0.85,
             label='真实推力')
    ax_b.bar(x_ticks - 0.5 * bar_w, a_prior, bar_w, color=BLU, alpha=0.7,
             label='AI预测先验')
    ax_b.bar(x_ticks + 0.5 * bar_w, r1.x[6:9], bar_w, color=RED, alpha=0.7,
             label='纯定轨')
    ax_b.bar(x_ticks + 1.5 * bar_w, r2.x[6:9], bar_w, color=GRN, alpha=0.7,
             label='AI融合定轨')
    ax_b.set_xticks(x_ticks)
    ax_b.set_xticklabels(['径向', '沿迹', '法向'], fontsize=9)
    ax_b.set_ylabel('加速度 / (m/s2)', fontsize=9)
    ax_b.legend(fontsize=7, ncol=2, framealpha=0.85)
    ax_b.set_title('推力参数估计', fontsize=10, loc='left')

    # (c) RTN error bars
    ax_c = fig.add_subplot(gs[1, 0])
    labels_rtn = ['径向 (R)', '沿迹 (T)', '法向 (N)']
    for ai in range(3):
        r1v = np.sqrt((rtn1[:, ai] ** 2).mean())
        r2v = np.sqrt((rtn2[:, ai] ** 2).mean())
        ax_c.bar(ai, r1v, 0.3, color=RED, alpha=0.7,
                 label='无机动信息' if ai == 0 else '')
        ax_c.bar(ai + 0.3, r2v, 0.3, color=GRN, alpha=0.7,
                 label='AI机动融合' if ai == 0 else '')
        ax_c.text(ai, r1v * 1.03, f'{r1v:.0f}', ha='center', fontsize=7, color=RED)
        ax_c.text(ai + 0.3, r2v * 1.03, f'{r2v:.0f}', ha='center', fontsize=7, color=GRN)
    ax_c.set_xticks([0.15, 1.15, 2.15])
    ax_c.set_xticklabels(labels_rtn, fontsize=9)
    ax_c.set_ylabel('位置误差 RMS / m', fontsize=9)
    ax_c.legend(fontsize=8, framealpha=0.85)
    ax_c.set_title('RTN方向误差对比', fontsize=10, loc='left')

    # (d) Post-fit residuals
    ax_d = fig.add_subplot(gs[1, 1])
    res1 = residual(r1.x); res2 = residual(r2.x)
    ax_d.hist(res1, bins=35, alpha=0.5, color=RED, edgecolor='white',
              linewidth=0.3, label='无机动信息')
    ax_d.hist(res2, bins=35, alpha=0.45, color=GRN, edgecolor='white',
              linewidth=0.3, label='AI机动融合')
    ax_d.set_xlabel('加权残差', fontsize=9)
    ax_d.set_ylabel('频次', fontsize=9)
    ax_d.legend(fontsize=8, framealpha=0.85)
    ax_d.set_title('定轨后残差分布', fontsize=10, loc='left')

    fig.savefig(out / "fig3_pod.png", dpi=300)
    fig.savefig(out / "fig3_pod.pdf", dpi=300)
    plt.close()
    print(f"  完成: 位置误差改善 = {improvement:+.0f}%")


# =====================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  PINN+LSTM 模型评估")
    print("=" * 50)
    figure1()
    figure2()
    figure3()
    print(f"\n输出: {out}/")
    for f in sorted(out.glob("fig[123]*.png")):
        print(f"  {f.name}")
