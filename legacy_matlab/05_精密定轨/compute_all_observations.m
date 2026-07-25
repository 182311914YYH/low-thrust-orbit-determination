function y_calc = compute_all_observations(X_params, t_reference, t_obs, cfg)
% 计算所有观测时刻的雷达理论观测值
% 统一使用 m、rad 单位
% 输入：
%   X_params    : 7×1 待估参数 [r0(m); v0(m/s); at(m/s^2)]
%   t_reference : 参考历元时刻 (s)，初始状态对应时刻
%   t_obs       : N×1 观测时刻
%   cfg         : 全局配置结构体
% 输出：
%   y_calc      : N×3 理论观测值 [rho(m), az(rad), el(rad)]

    r0 = X_params(1:3);
    v0 = X_params(4:6);
    at = X_params(7);

    state_hist = propagate_orbit_j2_thrust( ...
        t_reference, [r0; v0], at, t_obs, cfg);

    % 从配置读取测站位置（m）
    r_station = cfg.station.ecef(:);

    y_calc = radar_observation_model(state_hist, r_station);
end
