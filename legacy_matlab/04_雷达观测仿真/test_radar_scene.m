%% 雷达观测三组场景测试脚本
% 存放位置：11_测试程序/test_radar_scene.m
% 功能：
%   场景1：卫星正上空（俯仰角应接近90°，可见）
%   场景2：地平线以下（俯仰角应低于10°，不可见）
%   场景3：无噪声观测基准测试（关闭所有噪声和偏差）
% 输出：保存完整带噪观测数据至 09_数据与实验结果/radar_obs_full.mat

clear; clc; close all;

%% ========== 自动加载02_参数配置文件夹 ==========
script_full = mfilename('fullpath');
test_dir = fileparts(script_full);
root = fullfile(test_dir, '..');
cfg_path = fullfile(root, '02_参数配置');
addpath(cfg_path);

%% 加载全局配置
cfg = default_config();

%% ========== 读取动力学模块输出的真值轨道 ==========
orbit_mat = fullfile(root, '09_数据与实验结果', 'truth_orbit_for_radar.mat');
if ~exist(orbit_mat, 'file')
    error('真值轨道文件不存在，请先运行 test_dyn_switch.m 生成 truth_orbit_for_radar.mat');
end
load(orbit_mat, 't3', 'state3');
t_orbit = t3;
state_orbit = state3;

fprintf('\n========== 雷达观测单元测试开始 ==========\n\n');

%% ====================== 场景1：卫星正上空测试 ======================
fprintf('===== 场景1：卫星正上空测试 =====\n');

% 构造站址天顶方向：沿站址 ECEF 位置矢量径向向外 100 km
sta_ecef = cfg.station.ecef;          % 3×1 列向量
r_test = sta_ecef + 100000 * (sta_ecef / norm(sta_ecef));  % 3×1
v_test = [0; 0; 7000];                % 3×1

% ECEF → ECI 转换（t_test=0 时 R3 为单位阵）
t_test = 0;
theta = cfg.const.omegaEarth * t_test;
R3 = [cos(theta), sin(theta), 0;
     -sin(theta), cos(theta), 0;
      0,          0,         1];
r_eci_test = R3' * r_test;            % 3×1

% 确保 state_test 是 1×6 行向量
state_test = [r_eci_test(1), r_eci_test(2), r_eci_test(3), ...
              v_test(1), v_test(2), v_test(3)];   % 1×6

% 调用观测模型（t 必须为列向量，此处用标量也兼容）
[obs1, vis1] = enu_obs_model(cfg, t_test, state_test);

fprintf('俯仰角(deg)：%.2f（预期接近 90°），可见标记：%d（预期 1）\n', ...
    rad2deg(obs1(1,3)), vis1(1));

if abs(rad2deg(obs1(1,3)) - 90) > 5
    warning('场景1：俯仰角偏差较大，请检查站址经纬度单位或ECEF坐标');
end

%% ====================== 场景2：地平线以下不可见测试 ======================
fprintf('\n===== 场景2：地平线以下不可见测试 =====\n');

% 构造一个明显在地平线以下的点：水平偏移 2000 km，垂直向下 500 km
r_low = sta_ecef + [2000000; 2000000; -1500000];   % 3×1
r_eci_low = R3' * r_low;                          % 3×1
v_low = [7000; 0; 0];                             % 3×1

state_low = [r_eci_low(1), r_eci_low(2), r_eci_low(3), ...
             v_low(1), v_low(2), v_low(3)];        % 1×6

[obs2, vis2] = enu_obs_model(cfg, t_test, state_low);

fprintf('俯仰角(deg)：%.2f（预期 < 10°），可见标记：%d（预期 0）\n', ...
    rad2deg(obs2(1,3)), vis2(1));

if vis2(1) ~= 0
    warning('场景2：可见标记异常，请检查俯仰角是否确实低于最低仰角');
end

%% ====================== 场景3：无噪声观测基准测试 ======================
fprintf('\n===== 场景3：无噪声观测基准测试 =====\n');

% 复制 cfg 并关闭所有噪声和偏差
cfg_no_noise = cfg;
cfg_no_noise.measurement.sigmaRange   = 0;
cfg_no_noise.measurement.sigmaAzimuth = 0;
cfg_no_noise.measurement.sigmaElevation = 0;
cfg_no_noise.measurement.biasRange    = 0;
cfg_no_noise.measurement.biasAzimuth  = 0;
cfg_no_noise.measurement.biasElevation = 0;

[obs_no_noise, vis_all] = enu_obs_model(cfg_no_noise, t_orbit, state_orbit);
fprintf('无噪声观测生成完成，总数据量：%d 个历元\n', size(obs_no_noise, 1));

%% ====================== 生成标准带噪完整观测（联调交付） ======================
[obs_full, vis_full] = enu_obs_model(cfg, t_orbit, state_orbit);

% 统计可见弧段
vis_ratio = sum(vis_full) / length(vis_full) * 100;
fprintf('可见弧段占比：%.2f%%（%d / %d 个历元可见）\n', ...
    vis_ratio, sum(vis_full), length(vis_full));

% 保存数据
save(fullfile(root, '09_数据与实验结果', 'radar_obs_full.mat'), ...
    't_orbit', 'state_orbit', 'obs_full', 'vis_full', 'obs_no_noise');

fprintf('\n雷达全部测试完成，观测数据已保存至 09_数据与实验结果/radar_obs_full.mat\n');
fprintf('========================================\n');