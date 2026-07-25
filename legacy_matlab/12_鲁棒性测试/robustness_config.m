function scenarios = robustness_config(base_cfg)
%ROBUSTNESS_CONFIG  鲁棒性测试场景配置生成器
%
% 场景定义与第二组（11_测试程序）生成的数据完全对齐：
%   gen_radar_anomaly.m  → 雷达异常场景（噪声/偏差/缺测/异常值）
%   gen_dyn_mismatch.m   → 动力学失配场景（阻力/时变推力/综合）
%
% 数据目录：09_数据与实验结果/robustness/
% 基准数据：09_数据与实验结果/radar_obs_full.mat
%
% 每个场景包含：
%   name        : 场景标识符（与数据文件名对应）
%   label       : 中文显示名称
%   type        : 场景类别 'baseline'|'noise'|'bias'|'dropout'|'outlier'|'model_mismatch'
%   data_file   : 数据文件相对路径（相对于项目根目录）
%   data_vars   : 数据文件中的变量名映射结构体
%   cfg         : 估计用配置（基于 base_cfg，覆盖噪声/偏差等参数）
%   truth_thrust: 真值推力加速度 (m/s^2)，用于计算推力估计误差

    if nargin < 1 || isempty(base_cfg)
        base_cfg = default_config();
    end

    scenarios = struct('name', {}, 'label', {}, 'type', {}, ...
        'data_file', {}, 'data_vars', {}, 'cfg', {}, 'truth_thrust', {});

    idx = 0;

    %% ========== 0. 基准场景 ==========
    % 数据文件：09_数据与实验结果/radar_obs_full.mat
    % 变量：t_orbit, state_orbit, obs_full, vis_full
    idx = idx + 1;
    scenarios(idx) = make_scenario( ...
        'baseline', '基准场景', 'baseline', ...
        fullfile('09_数据与实验结果', 'radar_obs_full.mat'), ...
        struct('t', 't_orbit', 'state', 'state_orbit', ...
               'obs', 'obs_full', 'vis', 'vis_full'), ...
        base_cfg, base_cfg.thrust.truth.accelerationRTN(2));

    %% ========== 1. 不同观测噪声水平 — 任务(1) ==========
    % 数据文件：radar_obs_noise1/2/3.mat
    % 变量：t_orbit, state_orbit, obs, vis, cfg_noise
    % 噪声等级：10m/0.01°, 100m/0.05°, 1000m/0.1°

    noise_specs = {
        'noise_low',  '低噪声 (10m, 0.01°)',   'radar_obs_noise1.mat', 10.0,  deg2rad(0.01);
        'noise_mid',  '中噪声 (100m, 0.05°)',  'radar_obs_noise2.mat', 100.0, deg2rad(0.05);
        'noise_high', '高噪声 (1000m, 0.1°)',  'radar_obs_noise3.mat', 1000.0, deg2rad(0.1)
    };

    for i = 1:size(noise_specs, 1)
        cfg_n = base_cfg;
        cfg_n.measurement.sigmaRange     = noise_specs{i, 4};
        cfg_n.measurement.sigmaAzimuth   = noise_specs{i, 5};
        cfg_n.measurement.sigmaElevation = noise_specs{i, 5};
        cfg_n.measurement.sigmaVector = [ ...
            cfg_n.measurement.sigmaRange; ...
            cfg_n.measurement.sigmaAzimuth; ...
            cfg_n.measurement.sigmaElevation];

        idx = idx + 1;
        scenarios(idx) = make_scenario( ...
            noise_specs{i, 1}, noise_specs{i, 2}, 'noise', ...
            fullfile('09_数据与实验结果', 'robustness', noise_specs{i, 3}), ...
            struct('t', 't_orbit', 'state', 'state_orbit', ...
                   'obs', 'obs', 'vis', 'vis', 'cfg', 'cfg_noise'), ...
            cfg_n, base_cfg.thrust.truth.accelerationRTN(2));
    end

    %% ========== 2. 系统偏差 — 任务(2) ==========
    % 数据文件：radar_obs_bias.mat
    % 变量：t_orbit, state_orbit, obs, vis, cfg_bias
    % 偏差：测距+50m，方位+0.5°，俯仰+0.05°
    % 注意：数据中观测无噪声（sigma=0），估计权重使用基准噪声
    idx = idx + 1;
    scenarios(idx) = make_scenario( ...
        'bias', '系统偏差 (50m, 0.5°, 0.05°)', 'bias', ...
        fullfile('09_数据与实验结果', 'robustness', 'radar_obs_bias.mat'), ...
        struct('t', 't_orbit', 'state', 'state_orbit', ...
               'obs', 'obs', 'vis', 'vis', 'cfg', 'cfg_bias'), ...
        base_cfg, base_cfg.thrust.truth.accelerationRTN(2));

    %% ========== 3. 随机缺测 — 任务(2) ==========
    % 数据文件：radar_obs_dropout10/30/50.mat
    % 变量：t_orbit, state_orbit, obs_drop, vis_drop, cfg_drop, drop_idx
    % 缺测率：10%, 30%, 50%

    dropout_specs = {
        'dropout_10', '缺测 10%', 'radar_obs_dropout10.mat';
        'dropout_30', '缺测 30%', 'radar_obs_dropout30.mat';
        'dropout_50', '缺测 50%', 'radar_obs_dropout50.mat'
    };

    for i = 1:size(dropout_specs, 1)
        idx = idx + 1;
        scenarios(idx) = make_scenario( ...
            dropout_specs{i, 1}, dropout_specs{i, 2}, 'dropout', ...
            fullfile('09_数据与实验结果', 'robustness', dropout_specs{i, 3}), ...
            struct('t', 't_orbit', 'state', 'state_orbit', ...
                   'obs', 'obs_drop', 'vis', 'vis_drop', 'cfg', 'cfg_drop'), ...
            base_cfg, base_cfg.thrust.truth.accelerationRTN(2));
    end

    %% ========== 4. 异常值 — 任务(2) ==========
    % 数据文件：radar_obs_outlier.mat
    % 变量：t_orbit, state_orbit, obs_out, vis_raw, cfg_out, outlier_idx
    % 5%可见观测测距+10km
    idx = idx + 1;
    scenarios(idx) = make_scenario( ...
        'outlier', '异常值 (5%, +10km)', 'outlier', ...
        fullfile('09_数据与实验结果', 'robustness', 'radar_obs_outlier.mat'), ...
        struct('t', 't_orbit', 'state', 'state_orbit', ...
               'obs', 'obs_out', 'vis', 'vis_raw', 'cfg', 'cfg_out'), ...
        base_cfg, base_cfg.thrust.truth.accelerationRTN(2));

    %% ========== 5. 动力学模型失配 — 任务(3) ==========
    % 数据文件：truth_drag.mat / truth_timevarying_thrust.mat / truth_drag_thrust.mat
    % 这些文件仅提供真值轨道，需用 enu_obs_model 生成观测
    % 变量名不同：t1/state1, t2/state2, t3/state3
    %
    % 真值推力：
    %   阻力场景：无推力 → truth_thrust = 0
    %   时变推力场景：amp*(1+0.5*sin) → 平均值 = amp = 5e-5
    %   综合场景：同上 = 5e-5

    % 5a. 大气阻力失配
    idx = idx + 1;
    scenarios(idx) = make_scenario( ...
        'mismatch_drag', '模型失配: 大气阻力', 'model_mismatch', ...
        fullfile('09_数据与实验结果', 'robustness', 'truth_drag.mat'), ...
        struct('t', 't1', 'state', 'state1', 'cfg', 'cfg1'), ...
        base_cfg, 0.0);

    % 5b. 时变推力失配
    idx = idx + 1;
    scenarios(idx) = make_scenario( ...
        'mismatch_thrust', '模型失配: 时变推力', 'model_mismatch', ...
        fullfile('09_数据与实验结果', 'robustness', 'truth_timevarying_thrust.mat'), ...
        struct('t', 't2', 'state', 'state2', 'cfg', 'cfg2'), ...
        base_cfg, 5.0e-5);

    % 5c. 综合：阻力+时变推力
    idx = idx + 1;
    scenarios(idx) = make_scenario( ...
        'mismatch_combined', '模型失配: 阻力+时变推力', 'model_mismatch', ...
        fullfile('09_数据与实验结果', 'robustness', 'truth_drag_thrust.mat'), ...
        struct('t', 't3', 'state', 'state3', 'cfg', 'cfg3'), ...
        base_cfg, 5.0e-5);

    fprintf('[robustness_config] 共定义 %d 个测试场景\n', numel(scenarios));
end

%% ========================================================================
function s = make_scenario(name, label, type, data_file, data_vars, cfg, truth_thrust)
    s.name         = name;
    s.label        = label;
    s.type         = type;
    s.data_file    = data_file;
    s.data_vars    = data_vars;
    s.cfg          = cfg;
    s.truth_thrust = truth_thrust;
end
