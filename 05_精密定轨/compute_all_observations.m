function y_calc = compute_all_observations(X_params, t_obs, r_station, prop_func)
% 计算所有观测时刻的雷达理论观测值
% 输入：
%   X_params  : 7×1 待估参数 [r0; v0; at]
%   t_obs     : N×1 观测时刻
%   r_station : 3×1 测站位置
%   prop_func : 轨道传播函数句柄
% 输出：
%   y_calc    : N×3 理论观测值 [rho, az, el]

    % 拆分待估参数
    r0 = X_params(1:3);
    v0 = X_params(4:6);
    at = X_params(7);
    
    % 轨道传播得到所有观测时刻的状态
    state_hist = prop_func(t_obs(1), [r0; v0], at, t_obs);
    
    % 逐时刻计算观测值
    n_obs = length(t_obs);
    y_calc = zeros(n_obs, 3);
    for i = 1:n_obs
        state_i = state_hist(:, i);
        y_calc(i, :) = radar_observation_model(state_i, r_station);
    end
end