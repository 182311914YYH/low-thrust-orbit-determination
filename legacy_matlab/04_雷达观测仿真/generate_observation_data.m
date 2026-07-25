function [data, t_obs, y_obs, t_reference, t_od_end] = generate_observation_data(cfg)
    if nargin < 1 || isempty(cfg)
        cfg = default_config();
    end

    %% 优先读取 radar_obs_full.mat
    base_dir = pwd;
    candidate_files = {
        fullfile(base_dir, 'radar_obs_full.mat')
        fullfile(pwd, 'radar_obs_full.mat')
    };

    radar_file = '';
    for i = 1:numel(candidate_files)
        if exist(candidate_files{i}, 'file') == 2
            radar_file = candidate_files{i};
            break;
        end
    end

    if ~isempty(radar_file)
        raw = load(radar_file);
        required = {'t_orbit', 'state_orbit', 'obs_full', 'vis_full'};
        for i = 1:numel(required)
            if ~isfield(raw, required{i})
                error('radar_obs_full.mat 缺少变量：%s', required{i});
            end
        end

        data.t = double(raw.t_orbit(:));
        data.state = double(raw.state_orbit);
        data.observation = double(raw.obs_full);
        data.visible = logical(raw.vis_full(:));
        data.source = radar_file;

        data = standardize_dataset(data);
        fprintf('[数据] 来源：%s\n', data.source);
    else
        fprintf('[数据] 未找到 radar_obs_full.mat，开始自动生成...\n');

        t = (0:cfg.sim.integrationStep:cfg.sim.truthDuration)';
        truth_at = cfg.thrust.truth.accelerationRTN(2);

        state_6xN = propagate_orbit_j2_thrust( ...
            0, cfg.orbit.x0Truth, truth_at, t, cfg);
        state_Nx6 = state_6xN';

        [obs, visible] = generate_radar_observations(cfg, t, state_Nx6);

        data.t = t;
        data.state = state_Nx6;
        data.observation = obs;
        data.visible = visible;
        data.source = '由 generate_observation_data 自动生成';

        t_orbit = data.t;
        state_orbit = data.state;
        obs_full = data.observation;
        vis_full = uint8(data.visible);
        save(fullfile(base_dir, 'radar_obs_full.mat'), ...
            't_orbit', 'state_orbit', 'obs_full', 'vis_full', '-v7.3');
    end

    %% 选取定轨弧段（前2天可见观测）
    t_reference = data.t(1);
    od_duration = min(2*86400, cfg.sim.truthDuration - 24*3600);
    if od_duration <= 0
        error('真值数据总时长不足，无法同时进行定轨和24小时预报。');
    end
    t_od_end = t_reference + od_duration;

    valid = data.visible & ...
            data.t >= t_reference & ...
            data.t <= t_od_end & ...
            all(isfinite(data.observation), 2);

    selected_index = find(valid);
    if numel(selected_index) < 30
        error('定轨弧段内有效观测少于30组，无法稳定估计7维参数。');
    end

    t_obs = data.t(selected_index);
    y_obs = data.observation(selected_index, :);

    [t_obs, unique_index] = unique(t_obs, 'stable');
    y_obs = y_obs(unique_index, :);

    fprintf('[数据] 总历元：%d，可见历元：%d\n', ...
        numel(data.t), nnz(data.visible));
    fprintf('[定轨] 参考历元：%.0f s，弧段终点：%.2f h\n', ...
        t_reference, t_od_end/3600);
    fprintf('[定轨] 使用可见观测：%d组\n', numel(t_obs));
end

%% 子函数
function data = standardize_dataset(data)
    if size(data.state,2) ~= 6 && size(data.state,1) == 6
        data.state = data.state';
    end
    if size(data.observation,2) ~= 3 && size(data.observation,1) == 3
        data.observation = data.observation';
    end

    N = numel(data.t);
    position_norm_median = median(vecnorm(data.state(:,1:3), 2, 2));
    if position_norm_median < 1e5
        data.state(:,1:3) = 1000 * data.state(:,1:3);
        data.state(:,4:6) = 1000 * data.state(:,4:6);
    end

    visible_range = data.observation(data.visible, 1);
    if ~isempty(visible_range) && median(abs(visible_range)) < 1e5
        data.observation(:,1) = 1000 * data.observation(:,1);
    end

    [data.t, order] = sort(data.t);
    data.state = data.state(order,:);
    data.observation = data.observation(order,:);
    data.visible = data.visible(order);
end

function [obs, visible] = generate_radar_observations(cfg, t, state_Nx6)
    theory = radar_observation_model(state_Nx6', cfg.station.ecef(:));
    visible = theory(:,3) >= cfg.station.minElevation;

    bias = [cfg.measurement.biasRange, ...
            cfg.measurement.biasAzimuth, ...
            cfg.measurement.biasElevation];
    sigma = [cfg.measurement.sigmaRange, ...
             cfg.measurement.sigmaAzimuth, ...
             cfg.measurement.sigmaElevation];

    obs = theory + repmat(bias, size(theory,1), 1) + ...
          randn(size(theory)) .* repmat(sigma, size(theory,1), 1);
    obs(:,2) = wrap_to_pi(obs(:,2));
end

function angle = wrap_to_pi(angle)
    angle = mod(angle + pi, 2*pi) - pi;
end