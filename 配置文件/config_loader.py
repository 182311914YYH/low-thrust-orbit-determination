"""统一配置加载器

从 YAML 配置文件读取项目所有公共参数，提供验证和默认值。
所有模块通过此入口获取配置，不得在其他模块中硬编码公共常数。
"""
from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


class ConfigValidator:
    """配置自洽性验证器，等价于 MATLAB default_config.m 中的 localValidateConfig。"""

    @staticmethod
    def validate(cfg: dict[str, Any]) -> None:
        """检查关键配置是否自洽，不满足则抛出 AssertionError。"""
        const = cfg["const"]
        orbit = cfg["orbit"]
        sim = cfg["sim"]
        station = cfg["station"]
        measurement = cfg["measurement"]
        estimation = cfg["estimation"]
        thrust = cfg["thrust"]

        assert const["mu_earth"] > 0, "mu_earth 必须为正。"
        assert const["radius_earth"] > 6.0e6, "地球半径设置异常。"
        assert orbit["a"] > const["radius_earth"], "轨道半长轴必须大于地球半径。"
        assert 0 <= orbit["e"] < 1, "当前项目只支持椭圆轨道。"
        assert sim["integration_step"] > 0, "积分步长必须为正。"
        assert sim["measurement_step"] >= sim["integration_step"], (
            "建议观测步长不小于动力学输出步长。"
        )
        min_el_rad = _deg2rad(station["min_elevation_deg"])
        assert 0 <= min_el_rad < math.pi / 2, "截止高度角应位于[0, pi/2)范围。"

        sigma_vec = _build_sigma_vector(cfg)
        assert all(s > 0 for s in sigma_vec), "观测标准差必须为正。"
        assert estimation["max_iterations"] >= 1, "最大迭代次数必须大于0。"
        assert thrust["truth"]["stop_time"] > thrust["truth"]["start_time"], (
            "推力结束时刻必须晚于开始时刻。"
        )

        print(f"[ConfigValidator] 配置验证通过：{cfg['meta']['version']}")
        print(f"[ConfigValidator] 基准轨道高度：{orbit['altitude'] / 1e3:.1f} km")
        period = _compute_orbital_period(orbit["a"], const["mu_earth"])
        print(f"[ConfigValidator] 轨道周期：{period / 60:.2f} min")
        print(
            f"[ConfigValidator] 真值沿迹加速度："
            f"{thrust['truth']['acceleration_rtn'][1]:.3e} m/s^2"
        )
        print(
            f"[ConfigValidator] 地面站：{station['name']}，"
            f"最低高度角：{station['min_elevation_deg']:.1f} deg"
        )


def _build_sigma_vector(cfg: dict[str, Any]) -> list[float]:
    m = cfg["measurement"]
    return [
        m["sigma_range"],
        _deg2rad(m["sigma_azimuth_deg"]),
        _deg2rad(m["sigma_elevation_deg"]),
    ]


def _compute_orbital_period(a: float, mu: float) -> float:
    mean_motion = math.sqrt(mu / a**3)
    return 2 * math.pi / mean_motion


def _compute_derived_values(cfg: dict[str, Any]) -> dict[str, Any]:
    """根据基础参数计算所有派生值并注入配置。"""
    const = cfg["const"]
    orbit = cfg["orbit"]
    station = cfg["station"]
    estimation = cfg["estimation"]
    thrust = cfg["thrust"]
    measurement = cfg["measurement"]

    # ---- 轨道根数 ----
    oe = {
        "a": orbit["a"],
        "e": orbit["e"],
        "i": _deg2rad(orbit["i_deg"]),
        "raan": _deg2rad(orbit["raan_deg"]),
        "arg_perigee": _deg2rad(orbit["arg_perigee_deg"]),
        "true_anomaly": _deg2rad(orbit["true_anomaly_deg"]),
    }

    # kepler → cartesian
    r0_eci, v0_eci = _kep2cart(oe, const["mu_earth"])
    x0_truth = np.concatenate([r0_eci, v0_eci])  # 6,

    mean_motion = math.sqrt(const["mu_earth"] / orbit["a"] ** 3)
    period = 2 * math.pi / mean_motion

    # ---- 地面站 ECEF ----
    station_lat = _deg2rad(station["latitude_deg"])
    station_lon = _deg2rad(station["longitude_deg"])
    station_ecef = _geodetic2ecef(
        station_lat, station_lon, station["altitude"],
        const["radius_earth"], const["flattening_earth"],
    )

    # ---- 观测噪声 (rad) ----
    sigma_vector = _build_sigma_vector(cfg)

    # ---- 定轨初值 ----
    x0_guess = x0_truth.copy()
    x0_guess[:3] += np.array(estimation["initial_position_error_eci"])
    x0_guess[3:6] += np.array(estimation["initial_velocity_error_eci"])

    # ---- 预报时长的秒表示 ----
    horizons_seconds = [h * 3600 for h in cfg["prediction"]["horizons_hours"]]

    # ---- 分段估计窗口 (s) ----
    seg = cfg["segmentation"]
    fixed_time_windows = seg.get("fixed_time_windows", [43200, 86400, 129600])

    # 注入
    derived = deepcopy(cfg)
    derived["orbit"]["elements"] = oe
    derived["orbit"]["r0_eci"] = r0_eci.tolist()
    derived["orbit"]["v0_eci"] = v0_eci.tolist()
    derived["orbit"]["x0_truth"] = x0_truth.tolist()
    derived["orbit"]["mean_motion"] = mean_motion
    derived["orbit"]["period"] = period
    derived["station"]["latitude"] = station_lat
    derived["station"]["longitude"] = station_lon
    derived["station"]["min_elevation"] = _deg2rad(station["min_elevation_deg"])
    derived["station"]["ecef"] = station_ecef.tolist()
    derived["measurement"]["sigma_vector"] = sigma_vector
    derived["measurement"]["sigma_azimuth"] = _deg2rad(measurement["sigma_azimuth_deg"])
    derived["measurement"]["sigma_elevation"] = _deg2rad(measurement["sigma_elevation_deg"])
    derived["estimation"]["x0_guess"] = x0_guess.tolist()
    derived["prediction"]["horizons"] = horizons_seconds
    derived["prediction"]["horizons_hours"] = cfg["prediction"]["horizons_hours"]
    derived["segmentation"]["fixed_time_windows"] = fixed_time_windows
    derived["thrust"]["estimate"]["initial_guess_at"] = thrust["estimate"]["initial_guess_rtn"][1]
    derived["thrust"]["estimate"]["lower_bound_at"] = thrust["estimate"]["lower_bound_rtn"][1]
    derived["thrust"]["estimate"]["upper_bound_at"] = thrust["estimate"]["upper_bound_rtn"][1]

    return derived


def _kep2cart(
    elements: dict[str, float], mu: float
) -> tuple[np.ndarray, np.ndarray]:
    """经典轨道根数 → ECI 位置速度 (m, m/s)。

    elements: {a, e, i, raan, arg_perigee, true_anomaly} (rad)
    """
    a = elements["a"]
    e = elements["e"]
    inc = elements["i"]
    raan = elements["raan"]
    argp = elements["arg_perigee"]
    nu = elements["true_anomaly"]

    p = a * (1 - e**2)
    r_pqw = (p / (1 + e * math.cos(nu))) * np.array([
        math.cos(nu), math.sin(nu), 0.0
    ])
    v_pqw = math.sqrt(mu / p) * np.array([
        -math.sin(nu), e + math.cos(nu), 0.0
    ])

    # rotation matrices
    c_raan, s_raan = math.cos(raan), math.sin(raan)
    c_inc, s_inc = math.cos(inc), math.sin(inc)
    c_argp, s_argp = math.cos(argp), math.sin(argp)

    R3_W = np.array([
        [c_raan, -s_raan, 0],
        [s_raan,  c_raan, 0],
        [0,       0,      1],
    ])
    R1_i = np.array([
        [1, 0,      0],
        [0, c_inc, -s_inc],
        [0, s_inc,  c_inc],
    ])
    R3_w = np.array([
        [c_argp, -s_argp, 0],
        [s_argp,  c_argp, 0],
        [0,       0,      1],
    ])

    Q = R3_W @ R1_i @ R3_w
    r_eci = Q @ r_pqw
    v_eci = Q @ v_pqw
    return r_eci, v_eci


def _geodetic2ecef(
    lat: float, lon: float, h: float, a: float, f: float
) -> np.ndarray:
    """大地经纬高 → ECEF (m)。"""
    e2 = f * (2 - f)
    sin_lat = math.sin(lat)
    N = a / math.sqrt(1 - e2 * sin_lat**2)
    cos_lat = math.cos(lat)
    cos_lon = math.cos(lon)
    sin_lon = math.sin(lon)
    return np.array([
        (N + h) * cos_lat * cos_lon,
        (N + h) * cos_lat * sin_lon,
        (N * (1 - e2) + h) * sin_lat,
    ])


def load_config(config_path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """加载 YAML 配置文件，计算派生值，验证后返回完整配置字典。"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = _compute_derived_values(cfg)
    ConfigValidator.validate(cfg)

    # 设置 numpy 随机种子
    np.random.seed(cfg["random"]["seed"])

    return cfg
