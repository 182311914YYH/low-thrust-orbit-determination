clear; clc; close all;

%% 1. 基础参数配置
mu = 398600.4418;
a = 7000;
r0_true = [a; 0; 0];
v0_true = [0; sqrt(mu/a); 0];
at_true = 1e-5;
X_true = [r0_true; v0_true; at_true];

t_final = 3600;

prop_func = @propagate_orbit_j2_thrust;

%% 2. 生成带偏差的估计参数（模拟定轨结果）
r0_est = r0_true + [0.1; 0.08; 0.05];
v0_est = v0_true + [0.001; 0.0008; 0.0005];
at_est = at_true * 0.95;
X_est = [r0_est; v0_est; at_est];

%% 3. 12小时轨道预报
fprintf('========== 12小时轨道预报测试 ==========\n');
[t_pred_12h, state_pred_12h] = orbit_prediction(X_est, t_final, '12h', prop_func);
[t_true_12h, state_true_12h] = orbit_prediction(X_true, t_final, '12h', prop_func);

[err_rtn_12h, err_inertial_12h] = rtn_error(state_pred_12h, state_true_12h);
stats_12h = error_statistics(err_rtn_12h, t_pred_12h);

fprintf('预报时长：%.1f 小时\n', stats_12h.Duration_h);
fprintf('R方向误差 RMS：%.4f km，Max：%.4f km\n', stats_12h.R_RMS, stats_12h.R_Max);
fprintf('T方向误差 RMS：%.4f km，Max：%.4f km\n', stats_12h.T_RMS, stats_12h.T_Max);
fprintf('N方向误差 RMS：%.4f km，Max：%.4f km\n', stats_12h.N_RMS, stats_12h.N_Max);
fprintf('综合误差 RMS：%.4f km，Max：%.4f km\n\n', stats_12h.Overall_RMS, stats_12h.Overall_Max);

%% 4. 24小时轨道预报
fprintf('========== 24小时轨道预报测试 ==========\n');
[t_pred_24h, state_pred_24h] = orbit_prediction(X_est, t_final, '24h', prop_func);
[t_true_24h, state_true_24h] = orbit_prediction(X_true, t_final, '24h', prop_func);

[err_rtn_24h, err_inertial_24h] = rtn_error(state_pred_24h, state_true_24h);
stats_24h = error_statistics(err_rtn_24h, t_pred_24h);

fprintf('预报时长：%.1f 小时\n', stats_24h.Duration_h);
fprintf('R方向误差 RMS：%.4f km，Max：%.4f km\n', stats_24h.R_RMS, stats_24h.R_Max);
fprintf('T方向误差 RMS：%.4f km，Max：%.4f km\n', stats_24h.T_RMS, stats_24h.T_Max);
fprintf('N方向误差 RMS：%.4f km，Max：%.4f km\n', stats_24h.N_RMS, stats_24h.N_Max);
fprintf('综合误差 RMS：%.4f km，Max：%.4f km\n\n', stats_24h.Overall_RMS, stats_24h.Overall_Max);

%% 5. 验证模块接口
fprintf('========== 模块接口验证 ==========\n');
fprintf('orbit_prediction 输出维度：\n');
fprintf('  t_pred: %d × %d\n', size(t_pred_24h, 1), size(t_pred_24h, 2));
fprintf('  state_pred: %d × %d\n', size(state_pred_24h, 1), size(state_pred_24h, 2));

fprintf('rtn_error 输出维度：\n');
fprintf('  err_rtn: %d × %d\n', size(err_rtn_24h, 1), size(err_rtn_24h, 2));
fprintf('  err_inertial: %d × %d\n', size(err_inertial_24h, 1), size(err_inertial_24h, 2));

fprintf('error_statistics 输出字段：\n');
fields = fieldnames(stats_24h);
for i = 1:length(fields)
    fprintf('  %s: ', fields{i});
    val = stats_24h.(fields{i});
    if isscalar(val)
        fprintf('%.6f\n', val);
    else
        fprintf('向量 (%d×1)\n', length(val));
    end
end

%% 6. 结果图

figure('Name','12小时预报RTN误差');
subplot(3,1,1);
plot((t_pred_12h - t_final)/3600, err_rtn_12h(1,:), 'LineWidth',1);
ylabel('R方向误差 (km)'); grid on; title('12小时预报RTN误差');
subplot(3,1,2);
plot((t_pred_12h - t_final)/3600, err_rtn_12h(2,:), 'LineWidth',1);
ylabel('T方向误差 (km)'); grid on;
subplot(3,1,3);
plot((t_pred_12h - t_final)/3600, err_rtn_12h(3,:), 'LineWidth',1);
ylabel('N方向误差 (km)'); xlabel('预报时长 (h)'); grid on;

figure('Name','24小时预报RTN误差');
subplot(3,1,1);
plot((t_pred_24h - t_final)/3600, err_rtn_24h(1,:), 'LineWidth',1);
ylabel('R方向误差 (km)'); grid on; title('24小时预报RTN误差');
subplot(3,1,2);
plot((t_pred_24h - t_final)/3600, err_rtn_24h(2,:), 'LineWidth',1);
ylabel('T方向误差 (km)'); grid on;
subplot(3,1,3);
plot((t_pred_24h - t_final)/3600, err_rtn_24h(3,:), 'LineWidth',1);
ylabel('N方向误差 (km)'); xlabel('预报时长 (h)'); grid on;

figure('Name','RTN误差对比（12h vs 24h）');
subplot(3,1,1);
plot((t_pred_12h - t_final)/3600, err_rtn_12h(1,:), 'b-', 'LineWidth',1); hold on;
plot((t_pred_24h - t_final)/3600, err_rtn_24h(1,:), 'r--', 'LineWidth',1);
ylabel('R方向误差 (km)'); legend('12h','24h'); grid on; title('RTN误差对比');
subplot(3,1,2);
plot((t_pred_12h - t_final)/3600, err_rtn_12h(2,:), 'b-', 'LineWidth',1); hold on;
plot((t_pred_24h - t_final)/3600, err_rtn_24h(2,:), 'r--', 'LineWidth',1);
ylabel('T方向误差 (km)'); grid on;
subplot(3,1,3);
plot((t_pred_12h - t_final)/3600, err_rtn_12h(3,:), 'b-', 'LineWidth',1); hold on;
plot((t_pred_24h - t_final)/3600, err_rtn_24h(3,:), 'r--', 'LineWidth',1);
ylabel('N方向误差 (km)'); xlabel('预报时长 (h)'); grid on;

figure('Name','惯性系位置误差');
subplot(3,1,1);
plot((t_pred_24h - t_final)/3600, err_inertial_24h(1,:), 'LineWidth',1);
ylabel('x方向误差 (km)'); grid on; title('24小时惯性系位置误差');
subplot(3,1,2);
plot((t_pred_24h - t_final)/3600, err_inertial_24h(2,:), 'LineWidth',1);
ylabel('y方向误差 (km)'); grid on;
subplot(3,1,3);
plot((t_pred_24h - t_final)/3600, err_inertial_24h(3,:), 'LineWidth',1);
ylabel('z方向误差 (km)'); xlabel('预报时长 (h)'); grid on;

fprintf('\n========== 测试完成 ==========\n');
