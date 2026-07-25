function [truth_data, obs_data] = load_test_data(cfg, scenario)
%LOAD_TEST_DATA  测试数据加载器
%
% 从第二组（11_测试程序）生成的数据文件中加载测试数据。
%
% 数据来源：
%   基准场景：09_数据与实验结果/radar_obs_full.mat
%   雷达异常：09_数据与实验结果/robustness/radar_obs_*.mat
%   动力学失配：09_数据与实验结果/robustness/truth_*.mat
%
% 动力学失配场景仅提供真值轨道，本函数使用 enu_obs_model
% 以基准噪声配置生成观测数据。
%
% 输入：
%   cfg      : 全局配置结构体
%   scenario : 场景结构体（来自 robustness_config）
% 输出：
%   truth_data : 真值数据结构体
%       .t             : N×1 时间序列 (s)
%       .state         : 6×N 真值状态 (m, m/s)
%       .t_reference   : 参考历元 (s)
%       .t_od_end      : 定轨弧段终点 (s)
%       .thrust_true   : 真值推力加速度 (m/s^2)
%       .source        : 数据来源描述
%   obs_data : 观测数据结构体
%       .t_obs         : M×1 观测时刻 (s)
%       .y_obs         : M×3 观测值 [range, az, el]
%       .visible       : N×1 全时段可见性标记
%       .obs_full      : N×3 全时段观测值
%       .source        : 数据来源描述

    if nargin < 1 || isempty(cfg)
        cfg = default_config();
    end
    if nargin < 2
        error('必须提供 scenario 参数');
    end

    % 确定项目根目录
    script_dir = fileparts(mfilename('fullpath'));
    project_root = fileparts(script_dir);

    % 构建数据文件完整路径
    data_file = fullfile(project_root, scenario.data_file);

    if exist(data_file, 'file') ~= 2
        error('数据文件不存在: %s\n请先运行 11_测试程序 中的 gen_radar_anomaly.m 和 gen_dyn_mismatch.m', ...
              data_file);
    end

    fprintf('[load_test_data] 场景 ''%s''，加载数据文件: %s\n', ...
            scenario.name, data_file);

    raw = load(data_file);
    vars = scenario.data_vars;

    % 提取时间和状态
    t = double(raw.(vars.t));
    if iscolumn(t)
        t = t(:);
    else
        t = t(:);
    end

    state = double(raw.(vars.state));
    % 统一转为 6×N 格式
    if size(state, 1) == 6 && size(state, 2) > 6
        % 已经是 6×N
    elseif size(state, 2) == 6 && size(state, 1) > 6
        % N×6 → 6×N
        state = state';
    else
        error('状态矩阵维度异常: %dx%d，期望 6×N 或 N×6', size(state,1), size(state,2));
    end

    % 确保单位为 m（km → m 自动转换）
    pos_norm = median(vecnorm(state(1:3, :), 2, 1));
    if pos_norm < 1e5
        state(1:3, :) = state(1:3, :) * 1000;
        state(4:6, :) = state(4:6, :) * 1000;
    end

    % 排序（state 是 6×N，按行重排）
    [t, order] = sort(t);
    state = state(:, order);

    % 参考历元和定轨弧段
    t_reference = t(1);
    od_duration = min(2 * 86400, cfg.sim.truthDuration - 24 * 3600);
    t_od_end = t_reference + od_duration;

    % 真值推力
    thrust_true = scenario.truth_thrust;

    % 根据场景类型处理观测数据
    if strcmp(scenario.type, 'model_mismatch')
        % 动力学失配场景：仅有真值轨道，需生成观测
        [obs_full, vis_full] = generate_obs_from_truth(cfg, t, state);
        source_desc = sprintf('真值轨道+enu_obs_model生成 (来源: %s)', data_file);
    else
        % 雷达异常场景：直接读取观测数据
        obs_full = double(raw.(vars.obs));
        vis_full = logical(raw.(vars.vis));

        if size(obs_full, 2) ~= 3 && size(obs_full, 1) == 3
            obs_full = obs_full';
        end
        obs_full = obs_full(order, :);
        vis_full = vis_full(order);

        % 单位检查（km → m）
        vis_range = obs_full(vis_full, 1);
        if ~isempty(vis_range) && median(abs(vis_range)) < 1e5
            obs_full(:, 1) = obs_full(:, 1) * 1000;
        end

        source_desc = data_file;
    end

    % 选取定轨弧段内的有效观测
    valid = vis_full & ...
            t >= t_reference & ...
            t <= t_od_end & ...
            all(isfinite(obs_full), 2);

    sel_idx = find(valid);
    if numel(sel_idx) < 30
        error('场景 ''%s'' 定轨弧段内有效观测不足30组（仅%d组），无法稳定估计。', ...
              scenario.name, numel(sel_idx));
    end

    t_obs = t(sel_idx);
    y_obs = obs_full(sel_idx, :);
    [t_obs, uniq_idx] = unique(t_obs, 'stable');
    y_obs = y_obs(uniq_idx, :);

    % 组装输出
    truth_data.t           = t;
    truth_data.state       = state;
    truth_data.t_reference = t_reference;
    truth_data.t_od_end    = t_od_end;
    truth_data.thrust_true = thrust_true;
    truth_data.source      = source_desc;

    obs_data.t_obs    = t_obs;
    obs_data.y_obs    = y_obs;
    obs_data.visible  = vis_full;
    obs_data.obs_full = obs_full;
    obs_data.source   = source_desc;

    fprintf('[load_test_data] 总历元: %d, 可见历元: %d, 定轨观测: %d\n', ...
            numel(t), nnz(vis_full), numel(t_obs));
end

%% ========================================================================
function [obs, vis] = generate_obs_from_truth(cfg, t, state)
    % 使用 enu_obs_model 从真值轨道生成观测
    % 采用基准噪声配置（50m, 0.02°），无偏差
    obs_cfg = cfg;
    obs_cfg.measurement.sigmaRange     = cfg.measurement.sigmaRange;
    obs_cfg.measurement.sigmaAzimuth   = cfg.measurement.sigmaAzimuth;
    obs_cfg.measurement.sigmaElevation = cfg.measurement.sigmaElevation;
    obs_cfg.measurement.biasRange      = 0;
    obs_cfg.measurement.biasAzimuth    = 0;
    obs_cfg.measurement.biasElevation  = 0;

    state_Nx6 = state';  % 6×N → N×6
    [obs, vis] = enu_obs_model(obs_cfg, t, state_Nx6);
    obs(vis == 0, :) = NaN;
end
