function H = numerical_design_matrix(X_params, t_reference, t_obs, cfg, y0)
% 数值差分法计算观测设计矩阵 H
% 支持前向差分和中心差分，方位角跨越 ±pi 自动处理
% 统一使用 m、m/s、m/s^2 单位
%
% 输入：
%   X_params    : 7×1 当前参数值 [r0(m); v0(m/s); at(m/s^2)]
%   t_reference : 参考历元时刻 (s)
%   t_obs       : N×1 观测时刻
%   cfg         : 全局配置结构体
%   y0          : (可选) 当前参数对应的基准观测值 N×3
% 输出：
%   H           : (3N)×7 设计矩阵，行按历元交错排列
%                 [rho1,az1,el1, rho2,az2,el2, ...]'

    steps = [
        repmat(cfg.estimation.fd.position, 3, 1)
        repmat(cfg.estimation.fd.velocity, 3, 1)
        cfg.estimation.fd.thrust
    ];

    n_parameter = numel(X_params);
    n_measurement = 3 * numel(t_obs);
    H = zeros(n_measurement, n_parameter);

    scheme = lower(cfg.estimation.fd.scheme);

    for j = 1:n_parameter
        h = steps(j);
        if h <= 0
            error('第%d个参数的数值差分步长必须为正。', j);
        end

        X_plus = X_params;
        X_plus(j) = X_plus(j) + h;
        y_plus = compute_all_observations(X_plus, t_reference, t_obs, cfg);

        if strcmp(scheme, 'central')
            X_minus = X_params;
            X_minus(j) = X_minus(j) - h;
            y_minus = compute_all_observations(X_minus, t_reference, t_obs, cfg);

            delta_y = observation_difference(y_plus, y_minus);
            derivative = delta_y / (2*h);
        else
            delta_y = observation_difference(y_plus, y0);
            derivative = delta_y / h;
        end

        H(:,j) = stack_observations(derivative);
    end
end

%% 子函数
function difference = observation_difference(y_a, y_b)
    difference = y_a - y_b;
    difference(:,2) = wrap_to_pi(difference(:,2));
end

function vector = stack_observations(matrix_Nx3)
    vector = reshape(matrix_Nx3.', [], 1);
end

function angle = wrap_to_pi(angle)
    angle = mod(angle + pi, 2*pi) - pi;
end
