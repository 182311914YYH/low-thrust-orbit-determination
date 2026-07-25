function y = radar_observation_model(state, r_station)
% 雷达观测方程：惯性系状态 -> 距离/方位角/俯仰角
% 统一使用 m、rad 单位
% 输入：
%   state     : 6×1 或 6×M 航天器惯性系状态 [r(m); v(m/s)]
%   r_station : 3×1 测站惯性系位置 (m)
% 输出：
%   y         : N×3 观测值 [距离rho(m), 方位角az(rad), 俯仰角el(rad)]
%             方位角范围 [-pi, pi]（与 atan2 一致）

    if size(state, 1) ~= 6
        error('state 必须为 6×N 矩阵。');
    end

    n = size(state, 2);
    r_station = r_station(:);
    y = zeros(n, 3);

    for i = 1:n
        rho_vec = state(1:3, i) - r_station;

        % 斜距
        rho = norm(rho_vec);

        % 方位角（atan2 输出范围为 [-pi, pi]）
        az = atan2(rho_vec(2), rho_vec(1));

        % 俯仰角
        horizontal = sqrt(rho_vec(1)^2 + rho_vec(2)^2);
        el = atan2(rho_vec(3), horizontal);

        y(i, :) = [rho, az, el];
    end
end
