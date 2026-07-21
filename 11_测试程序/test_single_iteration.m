% 批处理定轨与小推力估计 单元测试：一次迭代验证
% 功能：验证数值差分、残差计算、参数更新全链路，输出迭代记录与基础结果图

clear; clc; close all;

%% 1. 基础参数配置
% 测站位置（武汉站示例，联调时从default_config读取）
r_station = [1112.3; 4205.7; 4512.1]; % km

% 观测噪声协方差
sigma_rho = 0.01;   % km
sigma_az  = 0.001;  % rad
sigma_el  = 0.001;  % rad
R = diag([sigma_rho^2, sigma_az^2, sigma_el^2]);

% 迭代配置
max_iter = 1;           % 任务要求：先完成一次迭代测试
conv_thresh = 1e-6;

%% 2. 生成真值与模拟观测数据
mu = 398600.4418;
a = 7000;   % km 圆轨道半长轴
r0_true = [a; 0; 0];
v0_true = [0; sqrt(mu/a); 0];
at_true = 1e-5;         % km/s^2 沿迹小推力真值
X_true = [r0_true; v0_true; at_true];

% 观测弧段：1小时，间隔10s
t_obs = (0:10:3600)';
n_obs = length(t_obs);

% 传播函数句柄
prop_func = @propagate_orbit_j2_thrust;

% 真值观测值 + 噪声
y_true = compute_all_observations(X_true, t_obs, r_station, prop_func);
noise = [sigma_rho*randn(n_obs,1), sigma_az*randn(n_obs,1), sigma_el*randn(n_obs,1)];
y_obs = y_true + noise;

%% 3. 初始猜测值（带偏差）
r0_guess = r0_true + [0.2; 0.15; 0.1];
v0_guess = v0_true + [0.002; 0.0015; 0.001];
at_guess = 0.6e-5;
X0_guess = [r0_guess; v0_guess; at_guess];

%% 4. 执行一次迭代
[X_hat, residuals, iter_num, converged] = batch_od_thrust_estimator(...
    X0_guess, t_obs, y_obs, r_station, R, max_iter, conv_thresh, prop_func);

%% 5. 输出迭代记录（含6维状态分项误差）
fprintf('========== 一次迭代测试记录 ==========\n');
fprintf('迭代次数：%d\n', iter_num);
fprintf('收敛标志：%d (1=收敛, 0=未收敛)\n\n', converged);

fprintf('【迭代前 6维初始状态+推力 分项误差】\n');
fprintf('x 位置误差：%.4f km\n', X0_guess(1) - X_true(1));
fprintf('y 位置误差：%.4f km\n', X0_guess(2) - X_true(2));
fprintf('z 位置误差：%.4f km\n', X0_guess(3) - X_true(3));
fprintf('vx速度误差：%.6f km/s\n', X0_guess(4) - X_true(4));
fprintf('vy速度误差：%.6f km/s\n', X0_guess(5) - X_true(5));
fprintf('vz速度误差：%.6f km/s\n', X0_guess(6) - X_true(6));
fprintf('推力加速度误差：%.8f km/s^2\n\n', X0_guess(7) - X_true(7));

fprintf('【迭代后 6维初始状态+推力 分项误差】\n');
fprintf('x 位置误差：%.4f km\n', X_hat(1) - X_true(1));
fprintf('y 位置误差：%.4f km\n', X_hat(2) - X_true(2));
fprintf('z 位置误差：%.4f km\n', X_hat(3) - X_true(3));
fprintf('vx速度误差：%.6f km/s\n', X_hat(4) - X_true(4));
fprintf('vy速度误差：%.6f km/s\n', X_hat(5) - X_true(5));
fprintf('vz速度误差：%.6f km/s\n', X_hat(6) - X_true(6));
fprintf('推力加速度误差：%.8f km/s^2\n\n', X_hat(7) - X_true(7));

fprintf('【整体误差范数统计】\n');
fprintf('迭代前位置误差范数：%.4f km\n', norm(X0_guess(1:3) - X_true(1:3)));
fprintf('迭代后位置误差范数：%.4f km\n', norm(X_hat(1:3) - X_true(1:3)));
fprintf('迭代前速度误差范数：%.6f km/s\n', norm(X0_guess(4:6) - X_true(4:6)));
fprintf('迭代后速度误差范数：%.6f km/s\n', norm(X_hat(4:6) - X_true(4:6)));
fprintf('\n【残差统计】\n');
fprintf('距离残差 RMS：%.4f km\n', rms(residuals(:,1)));
fprintf('方位角残差 RMS：%.6f rad\n', rms(residuals(:,2)));
fprintf('俯仰角残差 RMS：%.6f rad\n', rms(residuals(:,3)));

% 验证设计矩阵维度
H_check = numerical_design_matrix(X0_guess, t_obs, r_station, prop_func);
fprintf('\n设计矩阵维度：%d × %d （预期：%d × 7）\n', ...
    size(H_check,1), size(H_check,2), 3*n_obs);

%% 6. 基础结果图（共3张）
% 图1：观测残差序列
figure('Name','一次迭代后观测残差');
subplot(3,1,1);
plot(t_obs, residuals(:,1), 'LineWidth',1);
ylabel('距离残差 (km)');
grid on; title('观测残差序列');
subplot(3,1,2);
plot(t_obs, residuals(:,2), 'LineWidth',1);
ylabel('方位角残差 (rad)');
grid on;
subplot(3,1,3);
plot(t_obs, residuals(:,3), 'LineWidth',1);
ylabel('俯仰角残差 (rad)');
xlabel('时间 (s)');
grid on;

% 图2：观测值与理论计算值对比
y_calc = compute_all_observations(X_hat, t_obs, r_station, prop_func);
figure('Name','观测值与计算值对比');
subplot(3,1,1);
plot(t_obs, y_obs(:,1), '.', 'MarkerSize',3); hold on;
plot(t_obs, y_calc(:,1), 'r-', 'LineWidth',1);
ylabel('斜距 (km)'); legend('带噪观测值','计算值');
grid on; title('观测值与理论计算值对比');
subplot(3,1,2);
plot(t_obs, y_obs(:,2), '.', 'MarkerSize',3); hold on;
plot(t_obs, y_calc(:,2), 'r-', 'LineWidth',1);
ylabel('方位角 (rad)');
grid on;
subplot(3,1,3);
plot(t_obs, y_obs(:,3), '.', 'MarkerSize',3); hold on;
plot(t_obs, y_calc(:,3), 'r-', 'LineWidth',1);
ylabel('俯仰角 (rad)'); xlabel('时间 (s)');
grid on;

% 图3：迭代前后轨道位置误差对比
state_true = prop_func(t_obs(1), X_true(1:6), X_true(7), t_obs);
state_before = prop_func(t_obs(1), X0_guess(1:6), X0_guess(7), t_obs);
state_after = prop_func(t_obs(1), X_hat(1:6), X_hat(7), t_obs);

err_before = state_before(1:3,:) - state_true(1:3,:);
err_after = state_after(1:3,:) - state_true(1:3,:);

figure('Name','迭代前后轨道位置误差对比');
subplot(3,1,1);
plot(t_obs, err_before(1,:), '--', 'LineWidth',1); hold on;
plot(t_obs, err_after(1,:), '-', 'LineWidth',1);
ylabel('x方向误差 (km)'); legend('迭代前','迭代后');
grid on; title('轨道位置误差时序对比');
subplot(3,1,2);
plot(t_obs, err_before(2,:), '--', 'LineWidth',1); hold on;
plot(t_obs, err_after(2,:), '-', 'LineWidth',1);
ylabel('y方向误差 (km)');
grid on;
subplot(3,1,3);
plot(t_obs, err_before(3,:), '--', 'LineWidth',1); hold on;
plot(t_obs, err_after(3,:), '-', 'LineWidth',1);
ylabel('z方向误差 (km)'); xlabel('时间 (s)');
grid on;