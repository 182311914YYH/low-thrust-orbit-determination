function state_hist = propagate_orbit_j2_thrust(t0, state0, at, t_vec, cfg)
% 简化轨道传播：二体 + J2摄动 + 沿迹常值小推力
% 统一使用 m、m/s、m/s^2 单位，与 default_config 保持一致
% 输入：
%   t0     : 初始时刻 (s)
%   state0 : 6×1 初始状态 [r0(m); v0(m/s)]
%   at     : 沿迹加速度大小 (m/s^2)
%   t_vec  : M×1 待输出状态的时刻序列
%   cfg    : (可选) 全局配置结构体；不提供时使用内置默认值
% 输出：
%   state_hist : 6×M 状态历史，每列对应一个时刻

    if nargin < 5 || isempty(cfg)
        cfg = default_config();
    end

    mu = cfg.const.muEarth;   % m^3/s^2
    J2 = cfg.const.J2;
    Re = cfg.const.radiusEarth; % m

    state0 = state0(:);
    t_vec = double(t_vec(:));

    if numel(state0) ~= 6
        error('初始状态必须为6×1。');
    end
    if isempty(t_vec)
        state_hist = zeros(6,0);
        return;
    end
    if any(t_vec < t0)
        error('输出时刻不能早于初始时刻。');
    end

    % 去重并排序
    [t_sorted, sort_index] = sort(t_vec);
    [unique_times, ~, unique_map] = unique(t_sorted, 'stable');

    if unique_times(1) > t0
        integration_times = [t0; unique_times];
        remove_first = true;
    else
        integration_times = unique_times;
        remove_first = false;
    end

    % 微分方程右端
    function dstate = ode_right(~, state)
        r = state(1:3);
        v = state(4:6);
        r_norm = norm(r);

        if r_norm <= Re
            error('轨道状态进入地球内部。');
        end

        % 中心引力
        a_two = -mu / r_norm^3 * r;

        % J2摄动
        x = r(1); y = r(2); z = r(3);
        factor = 1.5 * J2 * mu * Re^2 / r_norm^5;
        a_j2 = factor * [
            x*(5*z^2/r_norm^2 - 1);
            y*(5*z^2/r_norm^2 - 1);
            z*(5*z^2/r_norm^2 - 3)
        ];

        % 沿迹推力
        v_norm = norm(v);
        if v_norm <= eps
            error('速度范数过小，无法确定推力方向。');
        end
        a_thrust = at * (v / v_norm);

        dstate = [v; a_two + a_j2 + a_thrust];
    end

    % 数值积分
    position_abs_tol = max(cfg.sim.absTol, 1e-3);
    velocity_abs_tol = max(cfg.sim.absTol, 1e-6);
    opts = odeset( ...
        'RelTol', cfg.sim.relTol, ...
        'AbsTol', [repmat(position_abs_tol,3,1); repmat(velocity_abs_tol,3,1)]);

    integrator_name = cfg.sim.integrator;
    if exist(integrator_name, 'file') ~= 2
        integrator_name = 'ode45';
    end
    integrator = str2func(integrator_name);

    [~, state_integrated] = integrator(@ode_right, integration_times, state0, opts);

    if remove_first
        state_unique = state_integrated(2:end,:);
    else
        state_unique = state_integrated;
    end

    state_sorted = state_unique(unique_map,:);
    state_original_order = zeros(size(state_sorted));
    state_original_order(sort_index,:) = state_sorted;
    state_hist = state_original_order';
end
