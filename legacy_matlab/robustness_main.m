% 轨道预报、RTN误差、RMS和最大误差计算测试脚本
% 成员4负责：轨道预报、RTN误差、RMS和最大误差计算
% 统一使用 m、m/s、m/s^2 单位

clear; clc; close all;

%% 1. 加载配置与数据
cfg = default_config();
if ~isfield(cfg.estimation, 'maxDampingTrials')
    cfg.estimation.maxDampingTrials = 6;
    cfg.estimation.maxNormalizedStep = 2.0;
end
if ~isfield(cfg.prediction, 'outputStep')
    cfg.prediction.outputStep = 60;
end

[data, t_obs, y_obs, t_reference, t_od_end] = generate_observation_data(cfg);

%% 2. 先执行完整定轨，获取估计参数
X0_guess = [
    cfg.estimation.x0Guess(:);
    cfg.thrust.estimate.initialGuessRTN(2)
];

fprintf('\n========== 执行完整定轨 ==========\n');
[X_hat, ~, ~, ~, ~] = batch_od_thrust_estimator( ...
    X0_guess, t_reference, t_obs, y_obs, cfg);

fprintf('\n定轨完成，估计沿迹加速度：%.6e m/s^2\n', X_hat(7));

%% 3. 12h 和 24h 轨道预报与误差评价
prediction = evaluate_predictions(X_hat, t_reference, t_od_end, data, cfg);
print_prediction_summary(prediction);

%% 4. 绘图（共5张）

% --- 图1：12小时预报RTN误差 ---
if isfield(prediction, 'h12')
    item = prediction.h12;
    figure('Name','12小时预报RTN误差','Color','w');
    subplot(3,1,1);
    plot(item.time_from_prediction_start/3600, item.position_error_rtn(:,1), 'LineWidth',1.1);
    ylabel('R方向误差 (m)'); grid on; title('12小时预报RTN误差');
    subplot(3,1,2);
    plot(item.time_from_prediction_start/3600, item.position_error_rtn(:,2), 'LineWidth',1.1);
    ylabel('T方向误差 (m)'); grid on;
    subplot(3,1,3);
    plot(item.time_from_prediction_start/3600, item.position_error_rtn(:,3), 'LineWidth',1.1);
    ylabel('N方向误差 (m)'); xlabel('预报时长 (h)'); grid on;
end

% --- 图2：24小时预报RTN误差 ---
if isfield(prediction, 'h24')
    item = prediction.h24;
    figure('Name','24小时预报RTN误差','Color','w');
    subplot(3,1,1);
    plot(item.time_from_prediction_start/3600, item.position_error_rtn(:,1), 'LineWidth',1.1);
    ylabel('R方向误差 (m)'); grid on; title('24小时预报RTN误差');
    subplot(3,1,2);
    plot(item.time_from_prediction_start/3600, item.position_error_rtn(:,2), 'LineWidth',1.1);
    ylabel('T方向误差 (m)'); grid on;
    subplot(3,1,3);
    plot(item.time_from_prediction_start/3600, item.position_error_rtn(:,3), 'LineWidth',1.1);
    ylabel('N方向误差 (m)'); xlabel('预报时长 (h)'); grid on;
end

% --- 图3：RTN误差对比（12h vs 24h） ---
if isfield(prediction, 'h12') && isfield(prediction, 'h24')
    p12 = prediction.h12;
    p24 = prediction.h24;
    figure('Name','RTN误差对比（12h vs 24h）','Color','w');
    subplot(3,1,1);
    plot(p12.time_from_prediction_start/3600, p12.position_error_rtn(:,1), 'b-', 'LineWidth',1.1); hold on;
    plot(p24.time_from_prediction_start/3600, p24.position_error_rtn(:,1), 'r--', 'LineWidth',1.1);
    ylabel('R方向误差 (m)'); legend('12h','24h'); grid on; title('RTN误差对比');
    subplot(3,1,2);
    plot(p12.time_from_prediction_start/3600, p12.position_error_rtn(:,2), 'b-', 'LineWidth',1.1); hold on;
    plot(p24.time_from_prediction_start/3600, p24.position_error_rtn(:,2), 'r--', 'LineWidth',1.1);
    ylabel('T方向误差 (m)'); grid on;
    subplot(3,1,3);
    plot(p12.time_from_prediction_start/3600, p12.position_error_rtn(:,3), 'b-', 'LineWidth',1.1); hold on;
    plot(p24.time_from_prediction_start/3600, p24.position_error_rtn(:,3), 'r--', 'LineWidth',1.1);
    ylabel('N方向误差 (m)'); xlabel('预报时长 (h)'); grid on;
end

% --- 图4：24小时惯性系位置误差 ---
if isfield(prediction, 'h24')
    item = prediction.h24;
    figure('Name','24小时惯性系位置误差','Color','w');
    subplot(3,1,1);
    plot(item.time_from_prediction_start/3600, item.position_error_eci(:,1), 'LineWidth',1.1);
    ylabel('x方向误差 (m)'); grid on; title('24小时惯性系位置误差');
    subplot(3,1,2);
    plot(item.time_from_prediction_start/3600, item.position_error_eci(:,2), 'LineWidth',1.1);
    ylabel('y方向误差 (m)'); grid on;
    subplot(3,1,3);
    plot(item.time_from_prediction_start/3600, item.position_error_eci(:,3), 'LineWidth',1.1);
    ylabel('z方向误差 (m)'); xlabel('预报时长 (h)'); grid on;
end

% --- 图5：位置误差RMS增长趋势 ---
fields = fieldnames(prediction);
figure('Name','位置误差RMS增长趋势','Color','w');
hold on;
colors = lines(numel(fields));
for i = 1:numel(fields)
    item = prediction.(fields{i});
    hours = item.horizon_seconds / 3600;
    n = size(item.position_error_eci, 1);
    trend = zeros(n, 1);
    for k = 1:n
        trend(k) = rms(item.position_error_eci(1:k,:), 'all');
    end
    plot(item.time_from_prediction_start/3600, trend, '-', ...
        'Color', colors(i,:), 'LineWidth', 1.2, ...
        'DisplayName', sprintf('%dh', hours));
end
xlabel('预报时长 (h)'); ylabel('累积位置误差 RMS (m)');
legend('Location','best'); grid on; title('位置误差RMS增长趋势');

fprintf('\n========== 成员4任务完成 ==========\n');

%% ========================================================================子函数
function prediction = evaluate_predictions(X_hat, t_reference, t_od_end, data, cfg)
    prediction = struct();

    for i = 1:numel(cfg.prediction.horizons)
        horizon = cfg.prediction.horizons(i);
        end_time = t_od_end + horizon;

        if end_time > data.t(end)
            warning('真值数据不足以验证%.1f小时预报，跳过。', horizon/3600);
            continue;
        end

        t_pred = (t_od_end:cfg.prediction.outputStep:end_time)';
        if t_pred(end) < end_time
            t_pred = [t_pred; end_time];
        end

        state_est_6xN = propagate_orbit_j2_thrust( ...
            t_reference, X_hat(1:6), X_hat(7), t_pred, cfg);
        state_est = state_est_6xN';

        state_truth = interp1(data.t, data.state, t_pred, 'pchip');

        position_error_eci = state_est(:,1:3) - state_truth(:,1:3);
        velocity_error_eci = state_est(:,4:6) - state_truth(:,4:6);
        position_error_rtn = rtn_position_error( ...
            state_truth(:,1:3), state_truth(:,4:6), position_error_eci);

        metric = struct();
        metric.horizon_seconds = horizon;
        metric.t = t_pred;
        metric.time_from_prediction_start = t_pred - t_od_end;
        metric.state_estimated = state_est;
        metric.state_truth = state_truth;
        metric.position_error_eci = position_error_eci;
        metric.velocity_error_eci = velocity_error_eci;
        metric.position_error_rtn = position_error_rtn;

        metric.position_rms_eci = sqrt(mean(sum(position_error_eci.^2,2)));
        metric.position_max_eci = max(vecnorm(position_error_eci,2,2));
        metric.velocity_rms_eci = sqrt(mean(sum(velocity_error_eci.^2,2)));
        metric.velocity_max_eci = max(vecnorm(velocity_error_eci,2,2));
        metric.rtn_component_rms = sqrt(mean(position_error_rtn.^2,1));
        metric.rtn_component_max = max(abs(position_error_rtn),[],1);
        metric.endpoint_position_error_eci = norm(position_error_eci(end,:));
        metric.endpoint_velocity_error_eci = norm(velocity_error_eci(end,:));
        metric.endpoint_position_error_rtn = position_error_rtn(end,:);

        field_name = sprintf('h%d', round(horizon/3600));
        prediction.(field_name) = metric;
    end
end

function error_rtn = rtn_position_error(r_truth, v_truth, error_eci)
    N = size(r_truth,1);
    error_rtn = zeros(N,3);

    for i = 1:N
        r = r_truth(i,:)';
        v = v_truth(i,:)';

        R_hat = r / norm(r);
        h = cross(r,v);
        N_hat = h / norm(h);
        T_hat = cross(N_hat,R_hat);
        T_hat = T_hat / norm(T_hat);

        C_rtn_to_eci = [R_hat, T_hat, N_hat];
        error_rtn(i,:) = (C_rtn_to_eci' * error_eci(i,:)')';
    end
end

function print_prediction_summary(prediction)
    fprintf('\n====================== 预报结果 ======================\n');
    fields = fieldnames(prediction);

    for i = 1:numel(fields)
        item = prediction.(fields{i});
        hours = item.horizon_seconds / 3600;

        fprintf('\n--- %.0f小时预报 ---\n', hours);
        fprintf('位置误差RMS：%.6f m\n', item.position_rms_eci);
        fprintf('位置最大误差：%.6f m\n', item.position_max_eci);
        fprintf('末端位置误差：%.6f m\n', item.endpoint_position_error_eci);
        fprintf('速度误差RMS：%.9f m/s\n', item.velocity_rms_eci);
        fprintf('RTN分量RMS [R,T,N]：[% .6f, % .6f, % .6f] m\n', ...
            item.rtn_component_rms(1), ...
            item.rtn_component_rms(2), ...
            item.rtn_component_rms(3));
        fprintf('RTN分量Max [R,T,N]：[% .6f, % .6f, % .6f] m\n', ...
            item.rtn_component_max(1), ...
            item.rtn_component_max(2), ...
            item.rtn_component_max(3));
        fprintf('末端RTN误差 [R,T,N]：[% .6f, % .6f, % .6f] m\n', ...
            item.endpoint_position_error_rtn(1), ...
            item.endpoint_position_error_rtn(2), ...
            item.endpoint_position_error_rtn(3));
    end
end
