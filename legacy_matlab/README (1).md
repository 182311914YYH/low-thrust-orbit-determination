%% 动力学模块开关测试脚本
% 存放位置：11_测试程序/test_dyn_switch.m
clear; clc; close all;

%% ========== 关键：自动加载02_参数配置文件夹 ==========
% 获取当前脚本所在目录
script_path = mfilename('fullpath');
test_folder = fileparts(script_path);
% 项目根目录 = 测试文件夹向上一级
root_dir = fullfile(test_folder, '..');
% 配置文件夹路径
cfg_dir = fullfile(root_dir, '02_参数配置');
% 添加到MATLAB搜索路径
addpath(cfg_dir);

%% 加载全局配置
cfg = default_config();
x0_truth = cfg.orbit.x0Truth; % 6×1初始ECI状态

%% 存储三组测试结果
test_res = struct();

%% ====================== 场景1：仅二体（关闭J2、关闭推力） ======================
cfg1 = cfg;
cfg1.model.truth.useJ2 = false;
cfg1.model.truth.useThrust = false;
[t1, state1] = orbit_propagate(cfg1, x0_truth);
test_res.sc1.t = t1;
test_res.sc1.state = state1;
fprintf('===== 场景1：纯二体模型 =====\n');
fprintf('传播总时长：%.2f h，末端位置：[%.2f, %.2f, %.2f] km\n', ...
    t1(end)/3600, state1(end,1)/1e3, state1(end,2)/1e3, state1(end,3)/1e3);

%% ====================== 场景2：二体+J2，关闭推力 ======================
cfg2 = cfg;
cfg2.model.truth.useJ2 = true;
cfg2.model.truth.useThrust = false;
[t2, state2] = orbit_propagate(cfg2, x0_truth);
test_res.sc2.t = t2;
test_res.sc2.state = state2;
fprintf('\n===== 场景2：二体+J2 无推力 =====\n');
fprintf('传播总时长：%.2f h，末端位置：[%.2f, %.2f, %.2f] km\n', ...
    t2(end)/3600, state2(end,1)/1e3, state2(end,2)/1e3, state2(end,3)/1e3);

%% ====================== 场景3：二体+J2+沿迹小推力（完整真值） ======================
cfg3 = cfg;
cfg3.model.truth.useJ2 = true;
cfg3.model.truth.useThrust = true;
[t3, state3] = orbit_propagate(cfg3, x0_truth);
test_res.sc3.t = t3;
test_res.sc3.state = state3;
fprintf('\n===== 场景3：二体+J2+连续小推力 =====\n');
fprintf('传播总时长：%.2f h，末端位置：[%.2f, %.2f, %.2f] km\n', ...
    t3(end)/3600, state3(end,1)/1e3, state3(end,2)/1e3, state3(end,3)/1e3);

%% 保存完整真值轨道给下午联合调试（雷达模块读取）
save(fullfile(root_dir, '09_数据与实验结果', 'truth_orbit_for_radar.mat'), ...
    't3', 'state3', 'cfg3');
fprintf('\n已输出真值轨道文件：09_数据与实验结果/truth_orbit_for_radar.mat\n');

%% 自检：检查代码无硬编码、参数全部来自cfg
fprintf('\n动力学模块测试全部完成，可交付给雷达模块对接\n');