function results = run_robustness_experiments(scenarios, base_cfg, output_dir)
%RUN_ROBUSTNESS_EXPERIMENTS  成员3：鲁棒性定轨实验主程序
%
% 遍历全部测试场景，执行批处理定轨与小推力估计，记录：
%   - 收敛情况（是否收敛）
%   - 迭代次数
%   - 残差统计（RMS、最大值）
%   - 小推力估计值与估计误差
%   - 完整迭代历史
%
% 输入：
%   scenarios  : robustness_config() 返回的场景结构体数组
%   base_cfg   : default_config() 返回的基础配置
%   output_dir : (可选) 结果输出目录，默认为 12_鲁棒性测试/results
% 输出：
%   results    : 1×N 结构体数组，每个元素对应一个场景的实验结果

    if nargin < 1 || isempty(scenarios)
        base_cfg = default_config();
        scenarios = robustness_config(base_cfg);
    end
    if nargin < 2 || isempty(base_cfg)
        base_cfg = default_config();
    end
    if nargin < 3 || isempty(output_dir)
        output_dir = fullfile(fileparts(mfilename('fullpath')), 'results');
    end

    n_scenarios = numel(scenarios);
    results = struct('name', {}, 'label', {}, 'type', {}, ...
        'converged', {}, 'iterations', {}, 'residual_rms', {}, ...
        'residual_max', {}, 'thrust_estimate', {}, 'thrust_true', {}, ...
        'thrust_error', {}, 'thrust_rel_error', {}, ...
        'X_hat', {}, 'history', {}, ...
        'truth_data', {}, 'obs_data', {}, 'cfg', {}, 'elapsed_time', {}, ...
        'n_obs', {}, 't_obs', {}, 'y_obs', {}, 'residuals', {});

    fprintf('\n');
    fprintf('========================================================\n');
    fprintf('  鲁棒性定轨实验：共 %d 个场景\n', n_scenarios);
    fprintf('========================================================\n\n');

    for i = 1:n_scenarios
        scenario = scenarios(i);
        fprintf('--- [%d/%d] 场景: %s ---\n', i, n_scenarios, scenario.label);

        result = run_single_scenario(scenario, base_cfg);
        results(i) = result;

        print_scenario_summary(result);
        fprintf('\n');
    end

    % 保存结果
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    results_file = fullfile(output_dir, 'robustness_experiment_results.mat');
    save(results_file, 'results', 'scenarios', '-v7.3');
    fprintf('[实验] 结果已保存至: %s\n', results_file);

    print_overall_summary(results);
end

%% ========================================================================
function result = run_single_scenario(scenario, base_cfg)
    tic;

    cfg = scenario.cfg;

    % 0. 检查数据文件是否存在
    script_dir = fileparts(mfilename('fullpath'));
    project_root = fileparts(script_dir);
    data_file = fullfile(project_root, scenario.data_file);
    if exist(data_file, 'file') ~= 2
        fprintf('[跳过] 数据文件不存在: %s\n', data_file);
        result = make_failed_result(scenario, ...
            sprintf('数据文件缺失: %s', scenario.data_file));
        result.elapsed_time = toc;
        return;
    end

    % 1. 加载第二组提供的测试数据
    [truth_data, obs_data] = load_test_data(cfg, scenario);

    t_obs = obs_data.t_obs;
    y_obs = obs_data.y_obs;

    % 2. 准备初始猜测
    X0_guess = [cfg.estimation.x0Guess; ...
                cfg.thrust.estimate.initialGuessRTN(2)];

    % 3. 运行批处理定轨估计器
    fprintf('[定轨] 开始迭代估计（观测数: %d）...\n', numel(t_obs));

    try
        [X_hat, residuals, iter_num, converged, history] = ...
            batch_od_enu(X0_guess, truth_data.t_reference, ...
                         t_obs, y_obs, cfg);
        est_success = true;
    catch ME
        fprintf('[定轨] 估计失败: %s\n', ME.message);
        X_hat = nan(7, 1);
        residuals = nan(size(y_obs));
        iter_num = 0;
        converged = false;
        history = struct();
        est_success = false;
    end

    elapsed = toc;

    % 4. 计算残差统计
    if est_success && all(isfinite(residuals(:)))
        res_range = residuals(:, 1);
        res_az    = residuals(:, 2);
        res_el    = residuals(:, 3);

        residual_rms = [rms(res_range), rms(res_az), rms(res_el)];
        residual_max = [max(abs(res_range)), max(abs(res_az)), max(abs(res_el))];
    else
        residual_rms = [nan, nan, nan];
        residual_max = [nan, nan, nan];
    end

    % 5. 计算推力估计误差
    thrust_true = truth_data.thrust_true;
    if isfinite(X_hat(7))
        thrust_est = X_hat(7);
        thrust_err = thrust_est - thrust_true;
        if abs(thrust_true) > eps
            thrust_rel_err = thrust_err / thrust_true;
        else
            % 真值推力为0时，用绝对误差
            thrust_rel_err = nan;
        end
    else
        thrust_est = nan;
        thrust_err = nan;
        thrust_rel_err = nan;
    end

    % 6. 组装结果
    result.name             = scenario.name;
    result.label            = scenario.label;
    result.type             = scenario.type;
    result.converged        = converged;
    result.iterations       = iter_num;
    result.residual_rms     = residual_rms;
    result.residual_max     = residual_max;
    result.thrust_estimate  = thrust_est;
    result.thrust_true      = thrust_true;
    result.thrust_error     = thrust_err;
    result.thrust_rel_error = thrust_rel_err;
    result.X_hat            = X_hat;
    result.history          = history;
    result.truth_data       = truth_data;
    result.obs_data         = obs_data;
    result.cfg              = cfg;
    result.elapsed_time     = elapsed;
    result.n_obs            = numel(t_obs);
    result.t_obs            = t_obs;
    result.y_obs            = y_obs;
    result.residuals        = residuals;
end

%% ========================================================================
function result = make_failed_result(scenario, reason)
    result.name             = scenario.name;
    result.label            = scenario.label;
    result.type             = scenario.type;
    result.converged        = false;
    result.iterations       = 0;
    result.residual_rms     = [nan, nan, nan];
    result.residual_max     = [nan, nan, nan];
    result.thrust_estimate  = nan;
    result.thrust_true      = scenario.truth_thrust;
    result.thrust_error     = nan;
    result.thrust_rel_error = nan;
    result.X_hat            = nan(7, 1);
    result.history          = struct();
    result.truth_data       = struct();
    result.obs_data         = struct();
    result.cfg              = scenario.cfg;
    result.elapsed_time     = 0;
    result.n_obs            = 0;
    result.t_obs            = [];
    result.y_obs            = [];
    result.residuals        = [];
    fprintf('  [失败] %s\n', reason);
end

%% ========================================================================
function print_scenario_summary(result)
    fprintf('  收敛: %s\n', yesno(result.converged));
    fprintf('  迭代次数: %d\n', result.iterations);
    fprintf('  残差RMS: [距离 %.2f m, 方位 %.4f°, 俯仰 %.4f°]\n', ...
        result.residual_rms(1), rad2deg(result.residual_rms(2)), ...
        rad2deg(result.residual_rms(3)));
    fprintf('  残差最大: [距离 %.2f m, 方位 %.4f°, 俯仰 %.4f°]\n', ...
        result.residual_max(1), rad2deg(result.residual_max(2)), ...
        rad2deg(result.residual_max(3)));
    if isnan(result.thrust_rel_error)
        fprintf('  推力估计: %.6e m/s² (真值: %.6e, 绝对误差: %.2e)\n', ...
            result.thrust_estimate, result.thrust_true, ...
            result.thrust_error);
    else
        fprintf('  推力估计: %.6e m/s² (真值: %.6e, 误差: %.2e, 相对: %.2f%%)\n', ...
            result.thrust_estimate, result.thrust_true, ...
            result.thrust_error, result.thrust_rel_error * 100);
    end
    fprintf('  耗时: %.1f s\n', result.elapsed_time);
end

%% ========================================================================
function print_overall_summary(results)
    fprintf('\n========================================================\n');
    fprintf('  实验总体汇总\n');
    fprintf('========================================================\n');
    fprintf('%-28s %-6s %-4s %-12s %-14s %-10s\n', ...
        '场景', '收敛', '迭代', '距离残差RMS', '推力误差', '耗时(s)');
    fprintf('%s\n', repmat('-', 1, 80));

    for i = 1:numel(results)
        r = results(i);
        if isnan(r.thrust_rel_error)
            thrust_str = sprintf('%.2e', r.thrust_error);
        else
            thrust_str = sprintf('%.2f%%', r.thrust_rel_error * 100);
        end
        fprintf('%-28s %-6s %-4d %-12.2f %-14s %-10.1f\n', ...
            r.label, yesno(r.converged), r.iterations, ...
            r.residual_rms(1), thrust_str, r.elapsed_time);
    end
    fprintf('%s\n', repmat('-', 1, 80));

    n_conv = sum([results.converged]);
    fprintf('收敛场景: %d/%d\n', n_conv, numel(results));
end

%% ========================================================================
function s = yesno(flag)
    if flag
        s = '是';
    else
        s = '否';
    end
end
