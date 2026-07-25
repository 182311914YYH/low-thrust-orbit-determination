function y_calc = compute_obs_enu(X_params, t_reference, t_obs, cfg)
%COMPUTE_OBS_ENU  ENU观测模型：用于定轨估计端
%
% 与 04_雷达观测仿真/enu_obs_model 保持一致：
%   ECI → ECEF（地球自转） → ENU站心坐标系 → 距离/方位角/俯仰角
%
% 定轨端的原始 compute_all_observations 使用 radar_observation_model（纯ECI），
% 与数据生成端 enu_obs_model 不一致，导致残差异常。本函数修正此问题。
%
% 输入：
%   X_params    : 7×1 待估参数 [r0(m); v0(m/s); at(m/s^2)]
%   t_reference : 参考历元时刻 (s)
%   t_obs       : N×1 观测时刻
%   cfg         : 全局配置结构体
% 输出：
%   y_calc      : N×3 理论观测值 [rho(m), az(rad), el(rad)]

    r0 = X_params(1:3);
    v0 = X_params(4:6);
    at = X_params(7);

    % 传播轨道
    state_hist = propagate_orbit_j2_thrust( ...
        t_reference, [r0; v0], at, t_obs, cfg);

    % 从配置读取测站参数
    sta_ecef = cfg.station.ecef(:);
    lat0 = cfg.station.latitude;
    lon0 = cfg.station.longitude;
    omega_e = cfg.const.omegaEarth;

    % ENU 转换矩阵（ECEF → ENU）
    sin_lat = sin(lat0);
    cos_lat = cos(lat0);
    sin_lon = sin(lon0);
    cos_lon = cos(lon0);
    R_enu = [-sin_lon,  cos_lon, 0;
             -sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat;
              cos_lat*cos_lon,  cos_lat*sin_lon, sin_lat];

    N = numel(t_obs);
    y_calc = zeros(N, 3);

    for i = 1:N
        r_eci = state_hist(1:3, i);
        ti = t_obs(i);

        % ECI → ECEF 旋转
        theta = -omega_e * ti;
        cos_t = cos(theta);
        sin_t = sin(theta);
        R3 = [cos_t, sin_t, 0;
             -sin_t, cos_t, 0;
              0,     0,     1];
        r_sat_ecef = R3 * r_eci;

        % 站心矢量 ECEF
        rho_ecef = r_sat_ecef - sta_ecef;

        % ECEF → ENU
        rho_enu = R_enu * rho_ecef;
        east  = rho_enu(1);
        north = rho_enu(2);
        up    = rho_enu(3);

        % 观测值
        rho = norm(rho_enu);
        az  = atan2(east, north);
        el  = atan2(up, sqrt(east^2 + north^2) + eps);

        y_calc(i, :) = [rho, az, el];
    end
end
