function metrics = compute_scenario_metrics(results, output_dir)
%COMPUTE_SCENARIO_METRICS  成员4：各场景误差计算
%
% 对每个场景的定轨结果进行12h和24h轨道预报，计算：
%   - ECI坐标系位置误差（RMS、最大值）
%   - RTN坐标系位置误差（RMS、最大值）
%   - 预报末端误差（12h和24h终点）
%   - 速度误差（RMS、最大值）
%
% 输入：
%   results    : run_robustness_experiments 返回的结果结构体数组
%   output_dir : (可选) 结果输出目录
% 输出：
%   metrics    : 1×N 结构体数组，每个元素包含12h和24h的误差指标

    if nargin < 2 || isempty(output_dir)
        output_dir = fullfile(fileparts(mfilename('fullpath')), 'results');
    end

    n = numel(results);
    metrics = struct('name', {}, 'label', {}, 'type', {}, ...
        'pred_12h', {}, 'pred_24h', {});

    fprintf('\n');
    fprintf('========================================================\n');
    fprintf('  误差计算：共 %d 个场景\n', n);
    fprintf('========================================================\n\n');

    for i = 1:n
        r = results(i);
        fprintf('--- [%d/%d] 计算场景: %s ---\n', i, n, r.label);

        m = struct('name', r.name, 'label', r.label, 'type', r.type);

        if ~r.converged || any(~isfinite(r.X_hat))
            fprintf('  [跳过] 场景未收敛或结果无效\n');
            m.pred_12h = make_empty_metrics();
            m.pred_24h = make_empty_metrics();
            metrics(i) = m;
            continue;
        end

        truth_data = r.truth_data;
        cfg = r.cfg;
        X_hat = r.X_hat;
        t_reference = truth_data.t_reference;
        t_od_end = truth_data.t_od_end;

        % 预报时段
        t_pred_start = t_od_end;
        t_pred_end_24h = t_od_end + 24 * 3600;

        % 检查真值数据是否覆盖预报时段
        if truth_data.t(end) < t_pred_end_24h
            warning('场景 ''%s'' 真值数据仅到 %.1f h，不足24h预报（需到 %.1f h）', ...
                r.name, (truth_data.t(end) - t_reference) / 3600, ...
                (t_pred_end_24h - t_reference) / 3600);
            t_pred_end_24h = truth_data.t(end);
        end

        % 生成预报时刻序列
        output_step = 60;  % 1分钟输出
        t_pred = (t_pred_start:output_step:t_pred_end_24h)';
        if t_pred(end) < t_pred_end_24h
            t_pred = [t_pred; t_pred_end_24h];
        end

        % 使用估计参数传播轨道（从参考历元到预报终点）
        t_prop = (t_reference:cfg.sim.integrationStep:t_pred_end_24h)';
        estimated_state = propagate_orbit_j2_thrust( ...
            t_reference, X_hat(1:6), X_hat(7), t_prop, cfg);

        % 提取预报时段的估计轨道
        est_pred_mask = t_prop >= t_pred_start & t_prop <= t_pred_end_24h;
        t_pred_est = t_prop(est_pred_mask);
        est_pred_full = estimated_state(:, est_pred_mask);

        % 使用 interp1 对齐真值轨道到预报时刻
        truth_pred = interp1(truth_data.t, truth_data.state', t_pred_est, 'pchip');

        % ---- 计算 12h 误差 ----
        mask_12h = t_pred_est <= t_pred_start + 12 * 3600;
        t_12h = t_pred_est(mask_12h);
        est_12h = est_pred_full(:, mask_12h);
        truth_12h = truth_pred(mask_12h, :)';

        m.pred_12h = compute_error_metrics(est_12h, truth_12h, t_12h, t_pred_start);

        % ---- 计算 24h 误差 ----
        m.pred_24h = compute_error_metrics(est_pred_full, truth_pred', ...
            t_pred_est, t_pred_start);

        metrics(i) = m;

        fprintf('  12h: 位置RMS=%.2f m, 最大=%.2f m, 末端=%.2f m\n', ...
            m.pred_12h.position_rms, m.pred_12h.position_max, ...
            m.pred_12h.end_error);
        fprintf('  24h: 位置RMS=%.2f m, 最大=%.2f m, 末端=%.2f m\n', ...
            m.pred_24h.position_rms, m.pred_24h.position_max, ...
            m.pred_24h.end_error);
        fprintf('\n');
    end

    % 保存结果
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    metrics_file = fullfile(output_dir, 'robustness_metrics.mat');
    save(metrics_file, 'metrics', '-v7.3');
    fprintf('[误差] 结果已保存至: %s\n', metrics_file);
end

%% ========================================================================
function m = compute_error_metrics(state_est, state_true, t_pred, t_start)
    % 计算 ECI 和 RTN 误差统计
    % 输入：
    %   state_est  : 6×M 估计轨道状态
    %   state_true : 6×M 真值轨道状态
    %   t_pred     : M×1 时间序列
    %   t_start    : 预报起始时刻

    n = size(state_est, 2);

    % ---- ECI 误差 ----
    eci_pos_err = state_est(1:3, :) - state_true(1:3, :);   % 3×M
    eci_vel_err = state_est(4:6, :) - state_true(4:6, :);   % 3×M

    pos_err_norm = vecnorm(eci_pos_err, 2, 1);  % M×1
    vel_err_norm = vecnorm(eci_vel_err, 2, 1);  % M×1

    % ---- RTN 误差 ----
    [err_rtn, ~] = rtn_error(state_est, state_true);  % 3×M

    % ---- 统计指标 ----
    m = struct();
    m.n_points = n;
    m.duration_h = (t_pred(end) - t_start) / 3600;

    % ECI 位置误差
    m.eci_pos_rms = rms(pos_err_norm);
    m.eci_pos_max = max(pos_err_norm);
    m.eci_pos_mean = mean(pos_err_norm);

    % ECI 速度误差
    m.eci_vel_rms = rms(vel_err_norm);
    m.eci_vel_max = max(vel_err_norm);

    % ECI 各分量RMS
    m.eci_x_rms = rms(eci_pos_err(1, :));
    m.eci_y_rms = rms(eci_pos_err(2, :));
    m.eci_z_rms = rms(eci_pos_err(3, :));

    % RTN 误差 RMS
    m.rtn_r_rms = rms(err_rtn(1, :));
    m.rtn_t_rms = rms(err_rtn(2, :));
    m.rtn_n_rms = rms(err_rtn(3, :));

    % RTN 误差最大值
    m.rtn_r_max = max(abs(err_rtn(1, :)));
    m.rtn_t_max = max(abs(err_rtn(2, :)));
    m.rtn_n_max = max(abs(err_rtn(3, :)));

    % 综合位置 RMS 和最大值
    m.position_rms = sqrt(mean(pos_err_norm.^2));
    m.position_max = max(pos_err_norm);

    % 预报末端误差
    m.end_error = pos_err_norm(end);
    m.end_vel_error = vel_err_norm(end);
    m.end_rtn_error = err_rtn(:, end);

    % 时间序列（供绘图使用）
    m.t_pred = t_pred;
    m.t_pred_h = (t_pred - t_start) / 3600;
    m.eci_pos_err = eci_pos_err;
    m.eci_pos_norm = pos_err_norm;
    m.rtn_err = err_rtn;
    m.rtn_norm = vecnorm(err_rtn, 2, 1);
end

%% ========================================================================
function m = make_empty_metrics()
    m = struct();
    m.n_points = 0;
    m.duration_h = 0;
    m.eci_pos_rms = nan;
    m.eci_pos_max = nan;
    m.eci_pos_mean = nan;
    m.eci_vel_rms = nan;
    m.eci_vel_max = nan;
    m.eci_x_rms = nan;
    m.eci_y_rms = nan;
    m.eci_z_rms = nan;
    m.rtn_r_rms = nan;
    m.rtn_t_rms = nan;
    m.rtn_n_rms = nan;
    m.rtn_r_max = nan;
    m.rtn_t_max = nan;
    m.rtn_n_max = nan;
    m.position_rms = nan;
    m.position_max = nan;
    m.end_error = nan;
    m.end_vel_error = nan;
    m.end_rtn_error = [nan; nan; nan];
    m.t_pred = [];
    m.t_pred_h = [];
    m.eci_pos_err = [];
    m.eci_pos_norm = [];
    m.rtn_err = [];
    m.rtn_norm = [];
end
