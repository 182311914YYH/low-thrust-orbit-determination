# -*- coding: utf-8 -*-
"""
演示用仿真流水线（无需 torch/scipy）
=====================================
当 generate_software_figures.py 因缺少 torch 等依赖无法导入时，
GUI 自动回退到此模块，使用模拟数据生成演示图表。
接口与 generate_software_figures.py 完全一致。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pathlib import Path

from chart_style import (
    apply_style, create_figure, style_axis, style_3d_axis,
    style_colorbar, style_table, add_subfig_label,
    C_BLUE, C_GREEN, C_RED, C_ORANGE, C_PURPLE, C_GRAY,
    C_BLACK, C_TEAL, C_GOLD, C_MAGENTA, PALETTE, SEQ_CMAP,
    FS_TITLE, FS_AXISLABEL, FS_TICK,
    FS_LEGEND, FS_ANNOT, FS_SUBLABEL,
)
apply_style()

out = Path("评估图表")
out.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# ============================================================
# 仿真参数（由 GUI 传入，影响模拟数据生成）
# ============================================================
_params = {
    'gravity_order': 20,
    'srp': True,
    'third_body': True,
    'nrlmsise': True,
    'prediction_hours': 12,
    'od_arc_days': 2,
}

def set_params(params):
    """设置仿真参数，影响后续图表数据生成"""
    global _params
    _params.update(params)

def _err_scale():
    """根据参数计算误差缩放系数：低阶引力/缺摄动 → 误差增大"""
    g = _params.get('gravity_order', 20)
    s = {2: 3.5, 10: 1.8, 20: 1.0}.get(g, 1.0)
    if not _params.get('srp', True):
        s *= 1.3
    if not _params.get('third_body', True):
        s *= 1.2
    if not _params.get('nrlmsise', True):
        s *= 1.15
    return s

def _param_desc():
    """生成参数描述字符串，用于图表标题"""
    g = _params['gravity_order']
    pred = _params['prediction_hours']
    arc = _params['od_arc_days']
    flags = []
    if _params['srp']:
        flags.append("SRP")
    if _params['third_body']:
        flags.append("三体")
    if _params['nrlmsise']:
        flags.append("大气")
    flag_str = "+".join(flags) if flags else "无摄动"
    return f"引力{g}阶 | 预报{pred}h | 弧段{arc}天 | {flag_str}"


# ============================================================
# 模拟数据（参数感知）
# ============================================================
_RE = 6371  # km
_ALT = 450  # km

def _mock_orbit_positions(n_pts=500):
    """生成模拟轨道位置数据 — 受摄动参数影响"""
    t = np.linspace(0, 4 * np.pi, n_pts)
    r = _RE + _ALT
    pos = np.column_stack([
        r * np.cos(t) + 50 * np.sin(2 * t),
        r * np.sin(t) + 30 * np.cos(3 * t),
        200 * np.sin(t) + 50 * np.cos(2 * t),
    ])
    # 太阳光压摄动
    if _params.get('srp', True):
        pos[:, 0] += 8 * np.sin(3 * t)
        pos[:, 1] += 6 * np.cos(5 * t)
    # 三体引力摄动
    if _params.get('third_body', True):
        pos[:, 2] += 15 * np.sin(t / 2)
    # 大气阻力 — 轨道高度衰减
    if _params.get('nrlmsise', True):
        decay = np.linspace(0, 2.0, n_pts)
        scale = (r - decay) / r
        pos *= scale[:, np.newaxis]
    return pos

def _mock_error_data(n=720):
    """生成模拟预报误差数据 — 受引力阶数和摄动参数影响"""
    pred_h = _params.get('prediction_hours', 12)
    t_h = np.linspace(0, pred_h, n)
    s = _err_scale()
    err = 30 * np.sin(2 * np.pi * t_h / 6) + 15 * np.random.randn(n) + 10
    err = np.cumsum(err * 0.05) + 50
    err *= s
    return t_h, err

def _mock_thrust_data(n_samples=50):
    """生成模拟推力数据"""
    thrust = np.random.randn(n_samples, 3) * np.array([1e-5, 3e-5, 1e-5])
    thrust[:, 1] += np.random.rand(n_samples) * 2e-5
    return thrust

def _mock_statistics():
    """生成模拟统计数据 — 受参数影响"""
    s = _err_scale()
    pred_h = _params.get('prediction_hours', 12)
    alt_decay = 2.0 if _params.get('nrlmsise', True) else 0.0
    return {
        'alt_min': 448.2 - alt_decay,
        'alt_max': 452.8,
        'n_samples': 20,
        'sma_median': 753.1 * s,
        'sma_mean': 812.4 * s,
        'max_err': 1245.6 * s * (pred_h / 12),
        'final_err': 428.3 * s * (pred_h / 12),
        'thrust_mean': np.array([1.23e-6, 2.87e-5, -3.45e-7]),
        'thrust_max': np.array([4.56e-5, 4.92e-5, 1.78e-5]),
    }


# ============================================================
# 四张图表生成函数（接口与 generate_software_figures.py 一致）
# ============================================================
def figure_3d_orbit(save_path="评估图表/fig_3d_orbit.png", ax=None):
    """轨道三维可视化"""
    positions = _mock_orbit_positions()

    if ax is None:
        fig = create_figure(figsize=(10, 8), constrained=False)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax.clear()

    n_pts = len(positions)
    colors = plt.cm.coolwarm(np.linspace(0.15, 0.85, n_pts))
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c=colors, s=4, alpha=0.85, label='轨道轨迹', zorder=5)

    ax.scatter(*positions[0], color=C_GREEN, s=120, marker='o',
               edgecolor='white', linewidth=1.5, label='起点', zorder=10)
    ax.scatter(*positions[-1], color=C_RED, s=160, marker='*',
               edgecolor='white', linewidth=1.2, label='终点', zorder=10)

    draw_r = _RE * 0.85
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_e = draw_r * np.outer(np.cos(u), np.sin(v))
    y_e = draw_r * np.outer(np.sin(u), np.sin(v))
    z_e = draw_r * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_e, y_e, z_e, color=C_BLUE,
                    alpha=0.25, rstride=2, cstride=2, edgecolor='none')

    max_range = max(positions.max(axis=0) - positions.min(axis=0)) * 0.55
    mid = positions.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    style_3d_axis(ax, title='非合作航天器轨道三维可视化',
                  xlabel='X [km]', ylabel='Y [km]', zlabel='Z [km]')

    leg = ax.legend(loc='upper right', fontsize=FS_LEGEND, framealpha=0.9,
                    edgecolor='#cccccc', fancybox=False, borderpad=0.4)
    if leg:
        leg.get_frame().set_linewidth(0.5)

    add_subfig_label(ax, 'a', x=-0.02, y=1.0)

    if ax is None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    return ax


def figure_error_curve(save_path="评估图表/fig_error_curve.png", ax=None):
    """预报误差分析"""
    t_h, err_m = _mock_error_data()

    if ax is None:
        fig = create_figure(figsize=(10, 5), constrained=False)
        ax = fig.add_subplot(111)

    ax.fill_between(t_h, 0, err_m, color=C_BLUE, alpha=0.18, linewidth=0)
    ax.plot(t_h, err_m, color=C_BLUE, lw=1.8, label='预报误差')
    ax.axhline(y=np.median(err_m), color=C_ORANGE, ls='--', lw=1.6,
               label=f'中位误差: {np.median(err_m):.0f} m')
    ax.axhline(y=0, color=C_BLACK, lw=0.8, alpha=0.4)

    style_axis(ax, title=f'轨道预报误差随时间演化 ({_param_desc()})',
               xlabel='预报时间 [h]', ylabel='半长轴误差 [m]')
    add_subfig_label(ax, 'b')

    if ax is None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    return ax


def figure_thrust_heatmap(save_path="评估图表/fig_thrust_heatmap.png", ax=None):
    """推力强度热图"""
    thrust_all = _mock_thrust_data()
    n_samples = thrust_all.shape[0]
    thrust_abs = np.abs(thrust_all.T)

    if ax is None:
        fig = create_figure(figsize=(12, 4.5), constrained=False)
        ax = fig.add_subplot(111)

    im = ax.imshow(thrust_abs, aspect='auto', cmap=SEQ_CMAP,
                   extent=[0, n_samples, 2.5, -0.5],
                   vmin=0, vmax=np.percentile(thrust_abs, 95))

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['R (径向)', 'T (沿迹)', 'N (法向)'])

    style_axis(ax, title=f'RTN三轴推力强度热图 ({n_samples}个测试样本)',
               xlabel='测试样本序号')
    ax.grid(False)

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    style_colorbar(cbar, label='推力幅值 [m/s²]')
    add_subfig_label(ax, 'c')

    if ax is None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    return ax


def figure_statistics_report(save_dir="评估图表", ax=None):
    """统计评估报告"""
    stats = _mock_statistics()

    table_data = [
        ["指标", "数值", "单位"],
        ["轨道近地点高度", f"{stats['alt_min']:.1f}", "km"],
        ["轨道远地点高度", f"{stats['alt_max']:.1f}", "km"],
        ["样本数量", f"{stats['n_samples']}", "-"],
        ["半长轴中位误差", f"{stats['sma_median']:.1f}", "m"],
        ["半长轴RMS误差", f"{stats['sma_mean']:.1f}", "m"],
        ["最大绝对误差", f"{stats['max_err']:.1f}", "m"],
        ["12h终点误差", f"{stats['final_err']:.1f}", "m"],
        ["R方向推力均值", f"{stats['thrust_mean'][0]:.2e}", "m/s²"],
        ["T方向推力均值", f"{stats['thrust_mean'][1]:.2e}", "m/s²"],
        ["N方向推力均值", f"{stats['thrust_mean'][2]:.2e}", "m/s²"],
        ["R方向推力最大值", f"{stats['thrust_max'][0]:.2e}", "m/s²"],
        ["T方向推力最大值", f"{stats['thrust_max'][1]:.2e}", "m/s²"],
        ["N方向推力最大值", f"{stats['thrust_max'][2]:.2e}", "m/s²"],
    ]

    if ax is None:
        fig = create_figure(figsize=(12, 8), constrained=False)
        ax = fig.add_subplot(111)

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.30, 0.20], edges='open')
    style_table(ax, table, title='非合作航天器精密定轨与智能预报 — 统计评估报告')
    add_subfig_label(ax, 'd', x=0.0, y=1.02)

    if ax is None:
        fig_path = Path(save_dir) / "fig_statistics_table.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    return ax


# ============================================================
# 场景 1：精密定轨与预报 — 新增图表
# ============================================================

def _mock_orbital_elements(n=720):
    """生成模拟轨道根数随时间变化数据 — 受参数影响"""
    pred_h = _params.get('prediction_hours', 12)
    t_h = np.linspace(0, pred_h, n)
    g = _params.get('gravity_order', 20)
    noise_scale = {2: 0.02, 10: 0.008, 20: 0.005}.get(g, 0.005)

    a = 6818.5 + 0.02 * np.sin(2 * np.pi * t_h / 6) + noise_scale * np.random.randn(n)
    e = 0.001 + 0.0003 * np.sin(2 * np.pi * t_h / 4) + noise_scale * 0.02 * np.random.randn(n)
    i_ang = 53.0 + 0.01 * np.sin(2 * np.pi * t_h / 8) + 0.003 * np.random.randn(n)
    # 升交点赤经漂移率 — 三体引力影响
    raan_rate = 0.05
    if _params.get('third_body', True):
        raan_rate += 0.02
    raan = 30.0 + raan_rate * t_h + 0.01 * np.random.randn(n)
    # 大气阻力 — 半长轴衰减
    if _params.get('nrlmsise', True):
        a -= 0.003 * t_h
    # 太阳光压 — 偏心率周期变化
    if _params.get('srp', True):
        e += 0.0002 * np.sin(2 * np.pi * t_h / 12)
    return t_h, a, e, i_ang, raan


def figure_orbital_elements(save_path="评估图表/fig_orbital_elements.png", ax=None, fig=None):
    """轨道根数随时间变化（四子图）"""
    from matplotlib.gridspec import GridSpec
    t_h, a, e, i_ang, raan = _mock_orbital_elements()

    if fig is not None:
        fig.clear()
        gs = GridSpec(2, 2, figure=fig)
    elif ax is not None:
        ax.clear()
        ax.set_axis_off()
        ax.text(0.5, 0.5, '请使用独立窗口查看多子图', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='#cbd5e0')
        return ax
    else:
        fig = create_figure(figsize=(10, 7), constrained=True)
        gs = GridSpec(2, 2, figure=fig)

    # (a) 半长轴
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(t_h, a, color=C_BLUE, lw=1.8)
    style_axis(ax_a, title='半长轴变化', xlabel='时间 / h', ylabel='a / km')
    add_subfig_label(ax_a, 'a')

    # (b) 偏心率
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(t_h, e * 1000, color=C_RED, lw=1.8)
    style_axis(ax_b, title='偏心率变化', xlabel='时间 / h', ylabel='e × 10⁻³')
    add_subfig_label(ax_b, 'b')

    # (c) 轨道倾角
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.plot(t_h, i_ang, color=C_GREEN, lw=1.8)
    style_axis(ax_c, title='轨道倾角变化', xlabel='时间 / h', ylabel='i / deg')
    add_subfig_label(ax_c, 'c')

    # (d) 升交点赤经
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.plot(t_h, raan, color=C_ORANGE, lw=1.8)
    style_axis(ax_d, title='升交点赤经变化', xlabel='时间 / h', ylabel='Ω / deg')
    add_subfig_label(ax_d, 'd')

    if fig is not None and save_path == "评估图表/fig_orbital_elements.png":
        return fig

    if ax is None and fig is None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    return ax


def figure_od_convergence(save_path="评估图表/fig_od_convergence.png", ax=None):
    """定轨收敛曲线 — 弧段越长收敛越快"""
    od_days = _params.get('od_arc_days', 2)
    # 弧段越长，数据越多，收敛越快
    n_iter = max(10, 22 - od_days * 3)
    iterations = np.arange(1, n_iter + 1)
    conv_rate = 0.22 + 0.06 * od_days
    pos_err = 120 * np.exp(-conv_rate * iterations) + 2 + np.random.randn(n_iter) * 3
    vel_err = 15 * np.exp(-conv_rate * 0.8 * iterations) + 0.3 + np.random.randn(n_iter) * 0.4

    if ax is None:
        fig = create_figure(figsize=(10, 5), constrained=False)
        ax = fig.add_subplot(111)

    ax.semilogy(iterations, pos_err, color=C_BLUE, lw=2.0, marker='o',
                markersize=5, label='位置误差')
    ax.semilogy(iterations, vel_err, color=C_RED, lw=2.0, marker='s',
                markersize=5, label='速度误差')
    ax.axhline(y=5, color=C_ORANGE, ls='--', lw=1.2, label='收敛阈值 (5 m)')

    style_axis(ax, title=f'精密定轨迭代收敛曲线 (弧段{od_days}天)',
               xlabel='迭代次数', ylabel='误差 (对数坐标)', legend_loc='upper right')
    add_subfig_label(ax, 'b')

    if ax is None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    return ax


# ============================================================
# 场景 2：机动检测与估计 — 新增图表
# ============================================================

def _mock_roc_data():
    """生成模拟 ROC 曲线数据"""
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(fpr) * 0.95 + 0.05
    tpr[-1] = 1.0
    return fpr, tpr


def figure_roc_curve(save_path="评估图表/fig_roc.png", ax=None):
    """机动检测 ROC 曲线"""
    fpr, tpr = _mock_roc_data()

    if ax is None:
        fig = create_figure(figsize=(8, 6), constrained=False)
        ax = fig.add_subplot(111)

    ax.plot(fpr, tpr, color=C_BLUE, lw=2.2, label=f'AUC = 0.952')
    ax.plot([0, 1], [0, 1], color=C_GRAY, ls='--', lw=1.0, label='随机分类器')
    ax.fill_between(fpr, tpr, 0, color=C_BLUE, alpha=0.08)

    style_axis(ax, title='机动检测 ROC 曲线',
               xlabel='假阳性率 (FPR)', ylabel='真阳性率 (TPR)',
               legend_loc='lower right')
    add_subfig_label(ax, 'a')

    if ax is None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    return ax


def figure_thrust_scatter(save_path="评估图表/fig_thrust_scatter.png", ax=None):
    """推力估计精度散点图"""
    np.random.seed(42)
    true_t = np.random.rand(60) * 4e-5
    pred_t = true_t + np.random.randn(60) * 3e-6
    true_r = np.random.rand(60) * 1e-5
    pred_r = true_r + np.random.randn(60) * 1.5e-6

    if ax is None:
        fig = create_figure(figsize=(8, 6), constrained=False)
        ax = fig.add_subplot(111)

    ax.scatter(true_r, pred_r, color=C_BLUE, s=35, alpha=0.7,
               edgecolor='white', linewidth=0.5, label='R方向')
    ax.scatter(true_t, pred_t, color=C_ORANGE, s=35, alpha=0.7,
               edgecolor='white', linewidth=0.5, label='T方向')

    max_v = 5e-5
    ax.plot([0, max_v], [0, max_v], color=C_RED, ls='--', lw=1.5,
            label='理想估计')

    style_axis(ax, title='推力估计精度 (R/T方向)',
               xlabel='真实推力 [m/s²]', ylabel='估计推力 [m/s²]',
               legend_loc='upper left')
    add_subfig_label(ax, 'c')

    if ax is None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    return ax


def figure_detection_timeline(save_path="评估图表/fig_timeline.png", ax=None):
    """机动检测时间线"""
    np.random.seed(42)
    t_h = np.linspace(0, 24, 480)
    signal = 0.5 * np.random.randn(480) + 0.1
    # 模拟3次机动事件
    events = [(4.2, 5.1), (11.5, 12.8), (18.3, 19.0)]
    for start, end in events:
        mask = (t_h >= start) & (t_h <= end)
        signal[mask] += np.linspace(0, 3, mask.sum()) + 1.5

    threshold = 1.5

    if ax is None:
        fig = create_figure(figsize=(12, 4.5), constrained=False)
        ax = fig.add_subplot(111)

    ax.fill_between(t_h, 0, signal, where=signal > threshold,
                    color=C_RED, alpha=0.2, interpolate=True)
    ax.plot(t_h, signal, color=C_BLUE, lw=1.2, label='检测信号')
    ax.axhline(y=threshold, color=C_ORANGE, ls='--', lw=1.5,
               label=f'检测阈值 ({threshold})')

    # 标注机动事件
    for idx, (start, end) in enumerate(events):
        mid = (start + end) / 2
        ax.axvspan(start, end, alpha=0.08, color=C_RED)
        ax.annotate(f'机动{idx+1}', xy=(mid, 3.5), fontsize=FS_ANNOT,
                    color=C_RED, ha='center', fontweight='bold')

    style_axis(ax, title='机动检测时间线 (24h)',
               xlabel='时间 / h', ylabel='检测信号强度',
               legend_loc='upper right')
    add_subfig_label(ax, 'd')

    if ax is None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    return ax


# ============================================================
# 场景 3：雷达定轨融合 — 新增图表
# ============================================================

def figure_position_error(save_path="评估图表/fig_pos_error.png", ax=None):
    """位置误差对比（雷达 vs 融合 vs 精密）— 受引力阶数影响"""
    np.random.seed(42)
    pred_h = _params.get('prediction_hours', 12)
    s = _err_scale()
    t_h = np.linspace(0, pred_h, 120)
    err_radar = (50 + 20 * np.exp(-t_h / 3) + 5 * np.random.randn(120)) * s
    err_fused = (15 + 8 * np.exp(-t_h / 2) + 3 * np.random.randn(120)) * s
    err_oem = (5 + 2 * np.random.randn(120)) * s

    if ax is None:
        fig = create_figure(figsize=(10, 5), constrained=False)
        ax = fig.add_subplot(111)

    ax.plot(t_h, err_radar, color=C_RED, lw=1.8, label='仅雷达定轨')
    ax.plot(t_h, err_fused, color=C_BLUE, lw=1.8, label='雷达+AI融合')
    ax.plot(t_h, err_oem, color=C_GREEN, lw=1.5, ls='--', label='精密星历')
    ax.fill_between(t_h, err_fused, err_radar, color=C_BLUE, alpha=0.06)

    style_axis(ax, title=f'位置误差对比 — 雷达定轨 vs AI融合 ({_param_desc()})',
               xlabel='预报时间 / h', ylabel='位置误差 / m',
               legend_loc='upper right')
    add_subfig_label(ax, 'a')

    if ax is None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    return ax


def figure_velocity_cdf(save_path="评估图表/fig_vel_cdf.png", ax=None):
    """速度误差累积分布函数 — 受引力阶数影响"""
    np.random.seed(42)
    s = _err_scale()
    vel_err_radar = np.sort(np.abs(np.random.randn(200) * 15 + 20)) * s
    vel_err_fused = np.sort(np.abs(np.random.randn(200) * 5 + 8)) * s
    cdf = np.arange(1, 201) / 200.0

    if ax is None:
        fig = create_figure(figsize=(8, 5), constrained=False)
        ax = fig.add_subplot(111)

    ax.plot(vel_err_radar, cdf, color=C_RED, lw=2.0, label='仅雷达定轨')
    ax.plot(vel_err_fused, cdf, color=C_BLUE, lw=2.0, label='雷达+AI融合')
    ax.axhline(y=0.95, color=C_GRAY, ls=':', lw=1.0, alpha=0.6)
    ax.text(vel_err_radar.max() * 0.6, 0.96, '95%分位线',
            fontsize=FS_ANNOT, color=C_GRAY)

    # 标注P95值
    p95_radar = vel_err_radar[int(0.95 * 200)]
    p95_fused = vel_err_fused[int(0.95 * 200)]
    ax.axvline(x=p95_radar, color=C_RED, ls='--', lw=0.8, alpha=0.5)
    ax.axvline(x=p95_fused, color=C_BLUE, ls='--', lw=0.8, alpha=0.5)

    style_axis(ax, title='速度误差累积分布 (CDF)',
               xlabel='速度误差 / (m/s)', ylabel='累积概率',
               legend_loc='lower right')
    add_subfig_label(ax, 'b')

    if ax is None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    return ax


def figure_radar_residuals(save_path="评估图表/fig_residuals.png", ax=None, fig=None):
    """雷达观测残差图"""
    from matplotlib.gridspec import GridSpec
    np.random.seed(42)
    n_obs = 200
    t_obs = np.linspace(0, 6, n_obs)
    resid_range = np.random.randn(n_obs) * 3 + 0.5
    resid_doppler = np.random.randn(n_obs) * 0.8 + 0.1
    resid_az = np.random.randn(n_obs) * 0.05
    resid_el = np.random.randn(n_obs) * 0.04

    if fig is not None:
        fig.clear()
        gs = GridSpec(2, 2, figure=fig)
    elif ax is not None:
        ax.clear()
        ax.set_axis_off()
        ax.text(0.5, 0.5, '请使用独立窗口查看多子图', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='#cbd5e0')
        return ax
    else:
        fig = create_figure(figsize=(10, 7), constrained=True)
        gs = GridSpec(2, 2, figure=fig)

    # (a) 距离残差
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.scatter(t_obs, resid_range, color=C_BLUE, s=8, alpha=0.6)
    ax_a.axhline(y=0, color=C_BLACK, lw=0.8)
    style_axis(ax_a, title='距离残差', xlabel='时间 / h', ylabel='残差 / m')
    add_subfig_label(ax_a, 'a')

    # (b) 多普勒残差
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.scatter(t_obs, resid_doppler, color=C_RED, s=8, alpha=0.6)
    ax_b.axhline(y=0, color=C_BLACK, lw=0.8)
    style_axis(ax_b, title='多普勒残差', xlabel='时间 / h', ylabel='残差 / (m/s)')
    add_subfig_label(ax_b, 'b')

    # (c) 方位角残差
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.scatter(t_obs, resid_az * 1000, color=C_GREEN, s=8, alpha=0.6)
    ax_c.axhline(y=0, color=C_BLACK, lw=0.8)
    style_axis(ax_c, title='方位角残差', xlabel='时间 / h', ylabel='残差 / mdeg')
    add_subfig_label(ax_c, 'c')

    # (d) 俯仰角残差
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.scatter(t_obs, resid_el * 1000, color=C_ORANGE, s=8, alpha=0.6)
    ax_d.axhline(y=0, color=C_BLACK, lw=0.8)
    style_axis(ax_d, title='俯仰角残差', xlabel='时间 / h', ylabel='残差 / mdeg')
    add_subfig_label(ax_d, 'd')

    if fig is not None and save_path == "评估图表/fig_residuals.png":
        return fig

    if ax is None and fig is None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    return ax


def figure_fusion_table(save_dir="评估图表", ax=None):
    """融合精度对比表"""
    table_data = [
        ["定轨方法", "位置误差 / m", "速度误差 / (m/s)", "12h预报误差 / m"],
        ["仅雷达定轨", "45.3", "12.8", "128.5"],
        ["雷达+AI融合", "12.1", "3.2", "35.2"],
        ["精密星历(OEM)", "3.5", "0.8", "8.1"],
        ["改善率(融合vs雷达)", "73.3%", "75.0%", "72.6%"],
    ]

    if ax is None:
        fig = create_figure(figsize=(12, 5), constrained=False)
        ax = fig.add_subplot(111)

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.22, 0.18, 0.18, 0.18], edges='open')
    style_table(ax, table, title='雷达定轨融合精度对比 — 统计评估报告')
    add_subfig_label(ax, 'd', x=0.0, y=1.02)

    if ax is None:
        fig_path = Path(save_dir) / "fig_fusion_table.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    return ax


# ============================================================
# 主入口
# ============================================================
def run_pipeline(data_folder=None, params=None):
    """执行演示流水线 — 生成全部图表"""
    if params:
        set_params(params)

    np.random.seed(42)
    desc = _param_desc()
    print("\n" + "=" * 55)
    print("  [演示模式] 生成全部展示图表")
    print(f"  参数: {desc}")
    print("=" * 55)

    # 场景1：精密定轨与预报
    figure_3d_orbit()
    figure_error_curve()
    figure_orbital_elements()
    figure_od_convergence()

    # 场景2：机动检测与估计
    figure_roc_curve()
    figure_thrust_heatmap()
    figure_thrust_scatter()
    figure_detection_timeline()

    # 场景3：雷达定轨融合
    figure_position_error()
    figure_velocity_cdf()
    figure_radar_residuals()
    figure_fusion_table()

    # 场景4：综合评估
    figure_statistics_report()

    print("\n" + "=" * 55)
    print("[演示模式] 全部图表已成功生成。")
    print(f"保存目录: {os.path.abspath(out)}")
    print("=" * 55)


if __name__ == "__main__":
    run_pipeline()
