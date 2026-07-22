%% robustness_main.m
%  鲁棒性测试主入口脚本
%  ============================================================
%  任务概要：
%    (1) 测试不同观测噪声下的定轨收敛情况
%    (2) 测试系统偏差、缺测和异常值对估计结果的影响
%    (3) 分析模型失配条件下的小推力估计误差
%    (4) 比较不同场景下12h、24h轨道预报误差
%
%  第二天任务：
%    成员3 - 运行各类鲁棒性定轨实验，记录收敛情况、迭代次数、
%            残差和小推力估计误差
%    成员4 - 计算各场景的ECI、RTN、RMS、最大误差和预报末端误差
%    统一对比 - 将各异常场景结果与基准场景进行统一对比
%
%  数据来源（第二组 11_测试程序 生成）：
%    基准数据：09_数据与实验结果/radar_obs_full.mat
%    异常数据：09_数据与实验结果/robustness/radar_obs_*.mat
%    失配真值：09_数据与实验结果/robustness/truth_*.mat
%
%  前置条件：先运行 11_测试程序/gen_radar_anomaly.m 和
%             11_测试程序/gen_dyn_mismatch.m 生成数据
%  ============================================================

clear; clc; close all;

%% ========== 0. 初始化 ==========

% 切换到项目根目录
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
cd(project_root);

% 添加所有子目录到路径
subdirs = dir(project_root);
for i = 1:numel(subdirs)
    if subdirs(i).isdir && ~startsWith(subdirs(i).name, '.') && ...
       ~strcmp(subdirs(i).name, '12_鲁棒性测试')
        addpath(fullfile(project_root, subdirs(i).name));
    end
end
addpath(script_dir);

% 加载基础配置
cfg = default_config();

% 输出目录
output_dir = fullfile(script_dir, 'results');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fprintf('============================================================\n');
fprintf('  低推力精密定轨鲁棒性测试\n');
fprintf('  日期: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf('  输出目录: %s\n', output_dir);
fprintf('============================================================\n\n');

%% ========== 0.5 检查前置数据 ==========

fprintf('>>> 步骤0: 检查第二组数据\n');
required_files = {
    fullfile('09_数据与实验结果', 'radar_obs_full.mat'), ...
    fullfile('09_数据与实验结果', 'robustness', 'radar_obs_noise1.mat'), ...
    fullfile('09_数据与实验结果', 'robustness', 'radar_obs_bias.mat'), ...
    fullfile('09_数据与实验结果', 'robustness', 'radar_obs_dropout10.mat'), ...
    fullfile('09_数据与实验结果', 'robustness', 'radar_obs_outlier.mat'), ...
    fullfile('09_数据与实验结果', 'robustness', 'truth_drag.mat')};

missing = false;
for i = 1:numel(required_files)
    fpath = fullfile(project_root, required_files{i});
    if exist(fpath, 'file') ~= 2
        fprintf('  [缺失] %s\n', required_files{i});
        missing = true;
    else
        fprintf('  [OK]    %s\n', required_files{i});
    end
end

if missing
    fprintf('\n[警告] 部分数据文件缺失，请先运行 11_测试程序 中的：\n');
    fprintf('  - gen_radar_anomaly.m\n');
    fprintf('  - gen_dyn_mismatch.m\n');
    fprintf('缺失场景将自动跳过。\n\n');
end

%% ========== 1. 生成测试场景配置 ==========

fprintf('>>> 步骤1: 生成测试场景配置\n');
scenarios = robustness_config(cfg);

fprintf('\n场景列表:\n');
for i = 1:numel(scenarios)
    fprintf('  %2d. [%-20s] %s\n', i, scenarios(i).name, scenarios(i).label);
end
fprintf('\n');

%% ========== 2. 运行鲁棒性定轨实验（成员3） ==========

fprintf('>>> 步骤2: 运行鲁棒性定轨实验（成员3）\n');
results = run_robustness_experiments(scenarios, cfg, output_dir);

%% ========== 3. 计算各场景误差指标（成员4） ==========

fprintf('>>> 步骤3: 计算各场景误差指标（成员4）\n');
metrics = compute_scenario_metrics(results, output_dir);

%% ========== 4. 统一对比分析 ==========

fprintf('>>> 步骤4: 统一对比分析\n');
comparison = compare_scenarios(results, metrics, output_dir);

%% ========== 5. 最终汇总 ==========

fprintf('\n============================================================\n');
fprintf('  鲁棒性测试完成\n');
fprintf('============================================================\n');
fprintf('  结果文件:\n');
fprintf('    - 实验结果: %s\n', fullfile(output_dir, 'robustness_experiment_results.mat'));
fprintf('    - 误差指标: %s\n', fullfile(output_dir, 'robustness_metrics.mat'));
fprintf('    - 对比分析: %s\n', fullfile(output_dir, 'robustness_comparison.mat'));
fprintf('    - 对比表格: %s\n', fullfile(output_dir, 'comparison_table.csv'));
fprintf('    - 对比图形: %s\n', fullfile(output_dir, 'figures'));
fprintf('============================================================\n');
