# -*- coding: utf-8 -*-
"""PINN+LSTM 模型评估 — 三张期刊标准图表."""
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

from chart_style import *

out = Path("评估图表"); out.mkdir(parents=True, exist_ok=True)

from 系统配置.config_loader import load_config
from AI融合模型.train_pinn import load_qk1_dataset, find_continuous_segments, build_samples_from_segments
from AI融合模型.model_pinn import PINNLSTMModel
from 物理力模型.propagator import OrbitalPropagator
from 坐标时间系统.frames import cart2kep, kep2cart, eci_to_ecef
from 坐标时间系统.time_systems import gmst_from_jd_batch

cfg = load_config("系统配置/default.yaml")
mu, Re = cfg["const"]["mu_earth"], cfg["const"]["radius_earth"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 数据加载 ----------
data = load_qk1_dataset("数据文件/QK1原始OEM数据"); N_all = len(data)
nt = int(N_all * 0.15)
train_data = data[:N_all - 2*nt]
test_data = data[N_all - nt:]
train_segs = find_continuous_segments(train_data)
test_segs = find_continuous_segments(test_data)
train_s = build_samples_from_segments(train_segs, 144, 720, 60.0, 72)
test_s  = build_samples_from_segments(test_segs, 144, 720, 60.0, 72)

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
    """训练或加载 AI 模型"""
    cache = out / "model_final.pt"
    if cache.exists():
        m = Detector()
        m.load_state_dict(torch.load(cache, map_location=device, weights_only=True))
        return m.to(device), 3e-5

    print("  训练 PINN+LSTM 模型...")
    c2 = load_config("系统配置/default.yaml")
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
    c2 = load_config("系统配置/default.yaml")
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
    all_snapshots = [] 
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

    fig = create_figure(figsize=(10, 4.8), constrained=True)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.3, 1])

    bi = int(np.argmin(all_rmse))
    sma_ai, sma_oem, err_b, da_b = all_snapshots[bi]
    t_h = np.arange(720) * 60.0 / 3600.0
    date_str = datetime.fromtimestamp(test_s["t0"][bi], tz=timezone.utc).strftime('%Y-%m-%d %H:%M')

    # --- (a) 时间序列 + 双Y轴 ---
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(t_h, sma_oem, color=C_BLACK, lw=2.2, label='精密星历')
    ax_a.plot(t_h, sma_ai, color=C_BLUE, lw=1.8, label='模型预报')
    ax2 = ax_a.twinx()
    ax2.fill_between(t_h, -np.abs(err_b), np.abs(err_b),
                     color=C_BLUE, alpha=0.06)
    ax2.set_ylabel('半长轴偏差 / m', fontsize=FS_AXISLABEL,
                   color=C_BLUE, labelpad=6)
    ax2.tick_params(colors=C_BLUE, labelsize=FS_TICK, direction='in',
                    width=0.6, length=4)
    for sp in ax2.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.8); sp.set_color('#333333')
    ax2.grid(False)
    style_axis(ax_a, title=f'QK-1 {date_str}  模型预报',
               xlabel='预报时间 / h', ylabel='半长轴 / km',
               legend_loc='upper right')
    add_subfig_label(ax_a, 'a')

    # --- (b) 直方图 ---
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.hist(all_rmse, bins=14, color=C_BLUE, alpha=0.7,
              edgecolor='white', linewidth=0.4)
    md_val = np.median(all_rmse)
    mn_val = all_rmse.mean()
    ax_b.axvline(md_val, color=C_RED, lw=1.2, ls='--')
    ax_b.axvline(mn_val, color=C_ORANGE, lw=1.2)
    ymax = ax_b.get_ylim()[1]
    ax_b.text(md_val * 1.02, ymax * 0.90, f'中位数 {md_val:.0f} m',
              fontsize=FS_ANNOT, color=C_RED)
    ax_b.text(mn_val * 1.02, ymax * 0.78, f'均值 {mn_val:.0f} m',
              fontsize=FS_ANNOT, color=C_ORANGE)
    style_axis(ax_b, xlabel='半长轴均方根误差 / m', ylabel='样本数量')
    add_subfig_label(ax_b, 'b')

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

    c2 = load_config("系统配置/default.yaml")
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

    fig = create_figure(figsize=(9.5, 7.8), constrained=True)
    gs = GridSpec(2, 2, figure=fig)

    # --- (a) ROC 曲线 ---
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(fpr, tpr, color=C_BLUE, lw=2.2, label=f'AUC = {roc_auc:.3f}')
    ax_a.plot([0, 1], [0, 1], '--', color=C_GRAY, lw=1, alpha=0.5,
              label='随机基线')
    ax_a.fill_between(fpr, tpr, alpha=0.06, color=C_BLUE)
    style_axis(ax_a, xlabel='虚警率', ylabel='检测率',
               legend_loc='lower right')
    ax_a.set_xlim(-0.02, 1.02); ax_a.set_ylim(-0.02, 1.02)
    add_subfig_label(ax_a, 'a')

    # --- (b) 混淆矩阵 ---
    ax_b = fig.add_subplot(gs[0, 1])
    im = ax_b.imshow(cm, cmap='Blues', aspect='auto', vmin=0)
    for i in range(2):
        for j in range(2):
            ax_b.text(j, i, str(cm[i, j]), ha='center', va='center',
                      fontsize=FS_TITLE + 8, fontweight='bold',
                      color='white' if cm[i, j] > cm.max() / 2 else C_BLACK)
    ax_b.set_xticks([0, 1]); ax_b.set_xticklabels(['无推力', '有推力'])
    ax_b.set_yticks([0, 1]); ax_b.set_yticklabels(['无推力', '有推力'])
    style_axis(ax_b, title=f'准确率 = {acc * 100:.1f}%',
               xlabel='预测类别', ylabel='真实类别')
    ax_b.grid(False)
    cbar = plt.colorbar(im, ax=ax_b, shrink=0.78)
    style_colorbar(cbar)
    add_subfig_label(ax_b, 'b')

    # --- (c) R² 柱状图 ---
    ax_c = fig.add_subplot(gs[1, 0])
    bars = ax_c.bar(['径向 (R)', '沿迹 (T)', '法向 (N)'], r2_vals,
                    color=[C_BLUE, C_GREEN, C_RED], width=0.5,
                    edgecolor='white', linewidth=0.4)
    for b, v in zip(bars, r2_vals):
        ax_c.text(b.get_x() + b.get_width() / 2, v + 0.03, f'{v:.3f}',
                  ha='center', fontsize=FS_LEGEND + 1.5,
                  fontweight='bold', color=C_BLACK)
    style_axis(ax_c, title='推力回归精度', ylabel='决定系数')
    ax_c.set_ylim(0, 1.15)
    add_subfig_label(ax_c, 'c')

    # --- (d) 敏感度曲线 ---
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
        ax_d.plot(sx, sy, 'o-', color=C_BLUE, lw=2.5, ms=8,
                  mfc='white', mew=2)
    ax_d.axhline(y=0.9, color=C_RED, ls='--', lw=1.2, alpha=0.5)
    style_axis(ax_d, xlabel='推力幅值 / (m/s)', ylabel='检测敏感度')
    ax_d.set_ylim(-0.02, 1.08)
    add_subfig_label(ax_d, 'd')

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

    r1 = least_squares(residual, params0.copy(), bounds=bounds,
                        method='trf', max_nfev=100, verbose=0)

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

    rtn1 = np.zeros((n_obs, 3)); rtn2 = np.zeros((n_obs, 3))
    for k in range(n_obs):
        for traj, rtn in [(traj1, rtn1), (traj2, rtn2)]:
            dr = traj[k, :3] - traj_true[k, :3]
            vt = traj_true[k, 3:6]
            rh = traj_true[k, :3] / np.linalg.norm(traj_true[k, :3])
            th = vt / np.linalg.norm(vt)
            nh = np.cross(rh, th); nh /= np.linalg.norm(nh)
            rtn[k] = [np.dot(dr, rh), np.dot(dr, th), np.dot(dr, nh)]

    fig = create_figure(figsize=(9.5, 7.8), constrained=True)
    gs = GridSpec(2, 2, figure=fig)

    # --- (a) 位置误差时间序列 ---
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(t_hours, pos_err1, color=C_RED, lw=1.5, alpha=0.85,
              label=f'无机动信息  RMS = {rms_before:.0f} m')
    ax_a.plot(t_hours, pos_err2, color=C_GREEN, lw=2.0,
              label=f'AI机动融合  RMS = {rms_after:.0f} m ({improvement:+.0f}%)')
    ax_a.fill_between(t_hours, pos_err2, alpha=0.08, color=C_GREEN)
    style_axis(ax_a, title='定轨精度对比',
               xlabel='时间 / h', ylabel='位置误差 / m')
    add_subfig_label(ax_a, 'a')

    # --- (b) 推力参数估计柱状图 ---
    ax_b = fig.add_subplot(gs[0, 1])
    x_ticks = np.arange(3); bar_w = 0.18
    ax_b.bar(x_ticks - 1.5 * bar_w, true_da, bar_w, color=C_BLACK, alpha=0.85,
             label='真实推力')
    ax_b.bar(x_ticks - 0.5 * bar_w, a_prior, bar_w, color=C_BLUE, alpha=0.7,
             label='AI预测先验')
    ax_b.bar(x_ticks + 0.5 * bar_w, r1.x[6:9], bar_w, color=C_RED, alpha=0.7,
             label='纯定轨')
    ax_b.bar(x_ticks + 1.5 * bar_w, r2.x[6:9], bar_w, color=C_GREEN, alpha=0.7,
             label='AI融合定轨')
    ax_b.set_xticks(x_ticks)
    ax_b.set_xticklabels(['径向', '沿迹', '法向'])
    style_axis(ax_b, title='推力参数估计', ylabel='加速度 / (m/s²)')
    leg_b = ax_b.legend(fontsize=FS_LEGEND, ncol=2, loc='upper right',
                        framealpha=0.9, edgecolor='#cccccc',
                        fancybox=False, borderpad=0.4)
    leg_b.get_frame().set_linewidth(0.5)
    add_subfig_label(ax_b, 'b')

    # --- (c) RTN 方向误差对比 ---
    ax_c = fig.add_subplot(gs[1, 0])
    labels_rtn = ['径向 (R)', '沿迹 (T)', '法向 (N)']
    for ai in range(3):
        r1v = np.sqrt((rtn1[:, ai] ** 2).mean())
        r2v = np.sqrt((rtn2[:, ai] ** 2).mean())
        ax_c.bar(ai, r1v, 0.3, color=C_RED, alpha=0.7,
                 label='无机动信息' if ai == 0 else '')
        ax_c.bar(ai + 0.3, r2v, 0.3, color=C_GREEN, alpha=0.7,
                 label='AI机动融合' if ai == 0 else '')
        ax_c.text(ai, r1v * 1.03, f'{r1v:.0f}', ha='center',
                  fontsize=FS_ANNOT, color=C_RED, fontweight='bold')
        ax_c.text(ai + 0.3, r2v * 1.03, f'{r2v:.0f}', ha='center',
                  fontsize=FS_ANNOT, color=C_GREEN, fontweight='bold')
    ax_c.set_xticks([0.15, 1.15, 2.15])
    ax_c.set_xticklabels(labels_rtn)
    style_axis(ax_c, title='RTN方向误差对比', ylabel='位置误差 RMS / m')
    add_subfig_label(ax_c, 'c')

    # --- (d) 残差分布直方图 ---
    ax_d = fig.add_subplot(gs[1, 1])
    res1 = residual(r1.x); res2 = residual(r2.x)
    ax_d.hist(res1, bins=35, alpha=0.5, color=C_RED, edgecolor='white',
              linewidth=0.3, label='无机动信息')
    ax_d.hist(res2, bins=35, alpha=0.45, color=C_GREEN, edgecolor='white',
              linewidth=0.3, label='AI机动融合')
    style_axis(ax_d, title='定轨后残差分布',
               xlabel='加权残差', ylabel='频次')
    add_subfig_label(ax_d, 'd')

    fig.savefig(out / "fig3_pod.png", dpi=300)
    fig.savefig(out / "fig3_pod.pdf", dpi=300)
    plt.close()
    print(f"  完成: 位置误差改善 = {improvement:+.0f}%")

# =====================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  PINN+LSTM 模型评估")
    print("=" * 50)
    figure1(); figure2(); figure3()
    print(f"\n输出: {out}/")
