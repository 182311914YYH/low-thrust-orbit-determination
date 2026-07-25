% 批处理定轨与小推力估计 单元测试：一次迭代验证
% 成员3负责：定轨与小推力估计框架、数值差分、一次迭代测试
% 统一使用 m、m/s、m/s^2 单位

clear; clc; close all;

%% 1. 加载配置与生成观测数据
cfg = default_config();

% 补齐 LM 参数（若 default_config 尚未更新）
if ~isfield(cfg.estimation, 'maxDampingTrials')
    cfg.estimation.maxDampingTrials = 6;
    cfg.estimation.maxNormalizedStep = 2.0;
end
if ~isfield(cfg.prediction, 'outputStep')
    cfg.prediction.outputStep = 60;
end

[data, t_obs, y_obs, t_reference, t_od_end] = generate_observation_data(cfg);

%% 2. 构造 7 维初值 [r0; v0; 沿迹加速度]
X0_guess = [
    cfg.estimation.x0Guess(:);
    cfg.thrust.estimate.initialGuessRTN(2)
];

truth_x0 = data.state(1, :)';
truth_at = cfg.thrust.truth.accelerationRTN(2);

fprintf('\n[初值] 位置误差猜测量级：%.3f m\n', ...
    norm(X0_guess(1:3) - truth_x0(1:3)));
fprintf('[初值] 速度误差猜测量级：%.6f m/s\n', ...
    norm(X0_guess(4:6) - truth_x0(4:6)));
fprintf('[初值] 沿迹加速度猜测：%.6e m/s^2\n', X0_guess(7));
fprintf('[初值] 沿迹加速度真值：%.6e m/s^2\n', truth_at);

%% 3. 执行批处理定轨（max_iter=1 做一次迭代测试）
cfg_od = cfg;
cfg_od.estimation.maxIterations = 1;
[X_hat, residuals, iter_num, converged, history] = ...
    batch_od_thrust_estimator(X0_guess, t_reference, t_obs, y_obs, cfg_od);

%% 4. 输出迭代记录
fprintf('\n========== 一次迭代测试记录 ==========\n');
fprintf('迭代次数：%d\n', iter_num);
fprintf('收敛标志：%d (1=收敛, 0=未收敛)\n\n', converged);

fprintf('【迭代前 6维初始状态+推力 分项误差】\n');
fprintf('x 位置误差：%.3f m\n', X0_guess(1) - truth_x0(1));
fprintf('y 位置误差：%.3f m\n', X0_guess(2) - truth_x0(2));
fprintf('z 位置误差：%.3f m\n', X0_guess(3) - truth_x0(3));
fprintf('vx速度误差：%.6f m/s\n', X0_guess(4) - truth_x0(4));
fprintf('vy速度误差：%.6f m/s\n', X0_guess(5) - truth_x0(5));
fprintf('vz速度误差：%.6f m/s\n', X0_guess(6) - truth_x0(6));
fprintf('推力加速度误差：%.8e m/s^2\n\n', X0_guess(7) - truth_at);

fprintf('【迭代后 6维初始状态+推力 分项误差】\n');
fprintf('x 位置误差：%.3f m\n', X_hat(1) - truth_x0(1));
fprintf('y 位置误差：%.3f m\n', X_hat(2) - truth_x0(2));
fprintf('z 位置误差：%.3f m\n', X_hat(3) - truth_x0(3));
fprintf('vx速度误差：%.6f m/s\n', X_hat(4) - truth_x0(4));
fprintf('vy速度误差：%.6f m/s\n', X_hat(5) - truth_x0(5));
fprintf('vz速度误差：%.6f m/s\n', X_hat(6) - truth_x0(6));
fprintf('推力加速度误差：%.8e m/s^2\n\n', X_hat(7) - truth_at);

fprintf('【整体误差范数统计】\n');
fprintf('迭代前位置误差范数：%.3f m\n', norm(X0_guess(1:3) - truth_x0(1:3)));
fprintf('迭代后位置误差范数：%.3f m\n', norm(X_hat(1:3) - truth_x0(1:3)));
fprintf('迭代前速度误差范数：%.6f m/s\n', norm(X0_guess(4:6) - truth_x0(4:6)));
fprintf('迭代后速度误差范数：%.6f m/s\n', norm(X_hat(4:6) - truth_x0(4:6)));
fprintf('迭代前推力误差：%.8e m/s^2\n', abs(X0_guess(7) - truth_at));
fprintf('迭代后推力误差：%.8e m/s^2\n\n', abs(X_hat(7) - truth_at));

fprintf('【残差统计】\n');
fprintf('距离残差 RMS：%.4f m\n', rms(residuals(:,1)));
fprintf('方位角残差 RMS：%.6f rad (%.4f deg)\n', ...
    rms(residuals(:,2)), rad2deg(rms(residuals(:,2))));
fprintf('俯仰角残差 RMS：%.6f rad (%.4f deg)\n\n', ...
    rms(residuals(:,3)), rad2deg(rms(residuals(:,3))));

% 验证设计矩阵维度
H_check = numerical_design_matrix(X0_guess, t_reference, t_obs, cfg_od, ...
    compute_all_observations(X0_guess, t_reference, t_obs, cfg_od));
fprintf('设计矩阵维度：%d × %d （预期：%d × 7）\n', ...
    size(H_check,1), size(H_check,2), 3*numel(t_obs));

%% 5. 基础结果图（共3张）

% 图1：观测残差序列
figure('Name','一次迭代后观测残差','Color','w');
subplot(3,1,1);
plot(t_obs/3600, residuals(:,1), 'LineWidth',1);
ylabel('距离残差 (m)'); grid on; title('观测残差序列');
subplot(3,1,2);
plot(t_obs/3600, rad2deg(residuals(:,2)), 'LineWidth',1);
ylabel('方位角残差 (deg)'); grid on;
subplot(3,1,3);
plot(t_obs/3600, rad2deg(residuals(:,3)), 'LineWidth',1);
ylabel('俯仰角残差 (deg)'); xlabel('时间 (h)'); grid on;

% 图2：观测值与理论计算值对比
y_calc = compute_all_observations(X_hat, t_reference, t_obs, cfg_od);
figure('Name','观测值与计算值对比','Color','w');
subplot(3,1,1);
plot(t_obs/3600, y_obs(:,1), '.', 'MarkerSize',3); hold on;
plot(t_obs/3600, y_calc(:,1), 'r-', 'LineWidth',1);
ylabel('斜距 (m)'); legend('带噪观测值','计算值');
grid on; title('观测值与理论计算值对比');
subplot(3,1,2);
plot(t_obs/3600, rad2deg(y_obs(:,2)), '.', 'MarkerSize',3); hold on;
plot(t_obs/3600, rad2deg(y_calc(:,2)), 'r-', 'LineWidth',1);
ylabel('方位角 (deg)'); grid on;
subplot(3,1,3);
plot(t_obs/3600, rad2deg(y_obs(:,3)), '.', 'MarkerSize',3); hold on;
plot(t_obs/3600, rad2deg(y_calc(:,3)), 'r-', 'LineWidth',1);
ylabel('俯仰角 (deg)'); xlabel('时间 (h)'); grid on;

% 图3：迭代前后轨道位置误差对比（ECI系）
state_before_6xN = propagate_orbit_j2_thrust( ...
    t_reference, X0_guess(1:6), X0_guess(7), t_obs, cfg_od);
state_after_6xN = propagate_orbit_j2_thrust( ...
    t_reference, X_hat(1:6), X_hat(7), t_obs, cfg_od);
state_true = interp1(data.t, data.state, t_obs, 'pchip');

err_before = state_before_6xN(1:3,:)' - state_true(:,1:3);
err_after = state_after_6xN(1:3,:)' - state_true(:,1:3);

figure('Name','迭代前后轨道位置误差对比','Color','w');
subplot(3,1,1);
plot(t_obs/3600, err_before(:,1), '--', 'LineWidth',1); hold on;
plot(t_obs/3600, err_after(:,1), '-', 'LineWidth',1);
ylabel('x方向误差 (m)'); legend('迭代前','迭代后');
grid on; title('轨道位置误差时序对比');
subplot(3,1,2);
plot(t_obs/3600, err_before(:,2), '--', 'LineWidth',1); hold on;
plot(t_obs/3600, err_after(:,2), '-', 'LineWidth',1);
ylabel('y方向误差 (m)'); grid on;
subplot(3,1,3);
plot(t_obs/3600, err_before(:,3), '--', 'LineWidth',1); hold on;
plot(t_obs/3600, err_after(:,3), '-', 'LineWidth',1);
ylabel('z方向误差 (m)'); xlabel('时间 (h)'); grid on;

fprintf('\n========== 成员3任务完成 ==========\n');
