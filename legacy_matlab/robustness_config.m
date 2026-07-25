%% ============================================================
%  成员2：雷达异常场景生成
%  存放位置：11_测试程序/gen_radar_anomaly.m
%  功能：生成五种雷达异常场景的观测数据
%    场景1-3：高噪声（三档）
%    场景4：系统偏差
%    场景5-7：随机缺测（三档）
%    场景8：异常值
%  输出：保存至 09_数据与实验结果/robustness/
% ============================================================
function gen_radar_anomaly()
    
    %% 初始化
    clc; close all;
    script_path = mfilename('fullpath');
    test_dir = fileparts(script_path);
    root = fullfile(test_dir, '..');
    
    % 添加必要路径
    addpath(fullfile(root, '02_参数配置'));
    addpath(fullfile(root, '03_轨道动力学'));
    addpath(fullfile(root, '04_雷达观测仿真'));
    
    cfg = default_config();
    
    % 输出目录
    data_dir = fullfile(root, '09_数据与实验结果', 'robustness');
    if ~exist(data_dir, 'dir'), mkdir(data_dir); end
    
    % 读取基准真值轨道（优先使用成员1的失配轨道，若无则用昨天的基准）
    base_file = fullfile(root, '09_数据与实验结果', 'truth_orbit_for_radar.mat');
    if exist(base_file, 'file')
        load(base_file, 't3', 'state3');
        t_orbit = t3;
        state_orbit = state3;
    else
        error('基准真值轨道不存在，请先运行 test_dyn_switch.m');
    end
    
    fprintf('\n========== 成员2：雷达异常场景生成 ==========\n');
    fprintf('基准轨道数据点数：%d\n', length(t_orbit));
    
    %% ====== 场景1-3：不同等级高噪声 ======
    % 使用 cell 数组存储，避免维度不一致错误
    noise_levels = {
        10,   deg2rad(0.01), deg2rad(0.01), '低噪声 10m/0.01°';
        100,  deg2rad(0.05), deg2rad(0.05), '中噪声 100m/0.05°';
        1000, deg2rad(0.1),  deg2rad(0.1),  '高噪声 1000m/0.1°'
    };
    
    for i = 1:size(noise_levels, 1)
        cfg_noise = cfg;
        cfg_noise.measurement.sigmaRange = noise_levels{i, 1};
        cfg_noise.measurement.sigmaAzimuth = noise_levels{i, 2};
        cfg_noise.measurement.sigmaElevation = noise_levels{i, 3};
        cfg_noise.measurement.biasRange = 0;
        cfg_noise.measurement.biasAzimuth = 0;
        cfg_noise.measurement.biasElevation = 0;
        
        [obs, vis] = enu_obs_model(cfg_noise, t_orbit, state_orbit);
        obs(vis==0, :) = NaN;  % 不可见时刻观测值置为NaN
        
        save(fullfile(data_dir, sprintf('radar_obs_noise%d.mat', i)), ...
            't_orbit', 'state_orbit', 'obs', 'vis', 'cfg_noise');
        fprintf('场景%d：%s 已生成\n', i, noise_levels{i, 4});
    end
    
    %% ====== 场景4：系统偏差 ======
    fprintf('场景4：系统偏差（测距+50m，方位+0.5°）...\n');
    cfg_bias = cfg;
    cfg_bias.measurement.sigmaRange = 0;
    cfg_bias.measurement.sigmaAzimuth = 0;
    cfg_bias.measurement.sigmaElevation = 0;
    cfg_bias.measurement.biasRange = 50;
    cfg_bias.measurement.biasAzimuth = deg2rad(0.5);
    cfg_bias.measurement.biasElevation = deg2rad(0.05);
    
    [obs, vis] = enu_obs_model(cfg_bias, t_orbit, state_orbit);
    obs(vis==0, :) = NaN;
    save(fullfile(data_dir, 'radar_obs_bias.mat'), ...
        't_orbit', 'state_orbit', 'obs', 'vis', 'cfg_bias');
    fprintf('  系统偏差场景已保存\n');
    
    %% ====== 场景5-7：随机缺测 ======
    dropout_rates = [0.1, 0.3, 0.5];
    cfg_drop = cfg;
    cfg_drop.measurement.sigmaRange = 0;
    cfg_drop.measurement.sigmaAzimuth = 0;
    cfg_drop.measurement.sigmaElevation = 0;
    cfg_drop.measurement.biasRange = 0;
    cfg_drop.measurement.biasAzimuth = 0;
    cfg_drop.measurement.biasElevation = 0;
    
    for i = 1:length(dropout_rates)
        [obs_raw, vis_raw] = enu_obs_model(cfg_drop, t_orbit, state_orbit);
        
        % 在可见弧段内随机缺测
        visible_idx = find(vis_raw == 1);
        n_visible = length(visible_idx);
        n_drop = round(n_visible * dropout_rates(i));
        drop_idx = visible_idx(randperm(n_visible, n_drop));
        
        obs_drop = obs_raw;
        obs_drop(drop_idx, :) = NaN;
        vis_drop = vis_raw;
        vis_drop(drop_idx) = 0;
        
        save(fullfile(data_dir, sprintf('radar_obs_dropout%.0f.mat', dropout_rates(i)*100)), ...
            't_orbit', 'state_orbit', 'obs_drop', 'vis_drop', 'cfg_drop', 'drop_idx');
        fprintf('场景%d：%.0f%% 随机缺测已生成（实际缺测 %d 个点）\n', ...
            4+i, dropout_rates(i)*100, n_drop);
    end
    
    %% ====== 场景8：异常值 ======
    fprintf('场景8：异常值（5%% 测距偏差 +10km）...\n');
    cfg_out = cfg;
    cfg_out.measurement.sigmaRange = 0;
    cfg_out.measurement.sigmaAzimuth = 0;
    cfg_out.measurement.sigmaElevation = 0;
    cfg_out.measurement.biasRange = 0;
    cfg_out.measurement.biasAzimuth = 0;
    cfg_out.measurement.biasElevation = 0;
    
    [obs_raw, vis_raw] = enu_obs_model(cfg_out, t_orbit, state_orbit);
    visible_idx = find(vis_raw == 1);
    n_visible = length(visible_idx);
    n_outlier = round(n_visible * 0.05);
    outlier_idx = visible_idx(randperm(n_visible, n_outlier));
    
    obs_out = obs_raw;
    obs_out(outlier_idx, 1) = obs_out(outlier_idx, 1) + 10000;  % 测距 +10km
    obs_out(vis_raw==0, :) = NaN;
    
    save(fullfile(data_dir, 'radar_obs_outlier.mat'), ...
        't_orbit', 'state_orbit', 'obs_out', 'vis_raw', 'cfg_out', 'outlier_idx');
    fprintf('  异常值场景已保存（共 %d 个异常值）\n', n_outlier);
    
    %% 生成场景说明文件
    generate_readme(data_dir);
    
    fprintf('\n========== 成员2任务完成 ==========\n');
    fprintf('所有数据已保存至：%s\n', data_dir);
end

%% ====== 辅助函数：生成README说明 ======
function generate_readme(data_dir)
    fid = fopen(fullfile(data_dir, '场景说明.txt'), 'w');
    fprintf(fid, '============================================================\n');
    fprintf(fid, '  鲁棒性测试数据场景说明\n');
    fprintf(fid, '  生成时间：%s\n', datestr(now));
    fprintf(fid, '============================================================\n\n');
    
    fprintf(fid, '【基准场景】\n');
    fprintf(fid, '  文件：radar_obs_full.mat（位于 09_数据与实验结果/）\n');
    fprintf(fid, '  说明：原始无异常观测数据，作为对比基准\n\n');
    
    fprintf(fid, '【高噪声场景】\n');
    fprintf(fid, '  文件：radar_obs_noise1/2/3.mat\n');
    fprintf(fid, '  配置：sigmaRange = [10, 100, 1000] m\n');
    fprintf(fid, '        sigmaAzimuth/Elevation = [0.01°, 0.05°, 0.1°]\n\n');
    
    fprintf(fid, '【系统偏差场景】\n');
    fprintf(fid, '  文件：radar_obs_bias.mat\n');
    fprintf(fid, '  配置：测距偏差 +50m，方位角偏差 +0.5°，俯仰角偏差 +0.05°\n\n');
    
    fprintf(fid, '【随机缺测场景】\n');
    fprintf(fid, '  文件：radar_obs_dropout10/30/50.mat\n');
    fprintf(fid, '  配置：在可见弧段内随机删除 10%% / 30%% / 50%% 的观测点\n\n');
    
    fprintf(fid, '【异常值场景】\n');
    fprintf(fid, '  文件：radar_obs_outlier.mat\n');
    fprintf(fid, '  配置：在可见弧段内随机 5%% 的测距值增加 +10km\n\n');
    
    fprintf(fid, '【动力学失配场景（成员1生成）】\n');
    fprintf(fid, '  文件：truth_drag.mat / truth_timevarying_thrust.mat / truth_drag_thrust.mat\n');
    fprintf(fid, '  说明：真值轨道含阻力/时变推力，估计模型保持原始配置\n');
    fclose(fid);
end