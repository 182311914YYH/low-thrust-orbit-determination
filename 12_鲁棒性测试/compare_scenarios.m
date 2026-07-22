function comparison = compare_scenarios(results, metrics, output_dir)
%COMPARE_SCENARIOS  统一对比分析：各异常场景 vs 基准场景
%
% 将全部场景的定轨收敛、残差、推力估计误差和轨道预报误差
% 与基准场景进行统一对比，生成对比表格和可视化图形。
%
% 输入：
%   results    : run_robustness_experiments 返回的结果
%   metrics    : compute_scenario_metrics 返回的误差指标
%   output_dir : (可选) 输出目录
% 输出：
%   comparison : 对比分析结果结构体

    if nargin < 3 || isempty(output_dir)
        output_dir = fullfile(fileparts(mfilename('fullpath')), 'results');
    end
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    n = numel(results);

    % 找到基准场景索引
    baseline_idx = find(strcmp({results.name}, 'baseline'), 1);
    if isempty(baseline_idx)
        baseline_idx = 1;
        fprintf('[对比] 未找到基准场景，使用第一个场景作为参考。\n');
    end

    fprintf('\n========================================================\n');
    fprintf('  统一对比分析\n');
    fprintf('========================================================\n\n');

    %% 1. 构建对比表格
    comparison_table = build_comparison_table(results, metrics, baseline_idx);
    comparison.table = comparison_table;

    % 打印对比表格
    print_comparison_table(comparison_table, baseline_idx);

    %% 2. 按场景类别分组统计
    comparison.by_type = group_by_type(comparison_table);

    %% 3. 计算相对基准的变化倍数
    comparison.ratios = compute_ratios(comparison_table, baseline_idx);

    %% 4. 生成可视化图形
    generate_figures(results, metrics, comparison_table, baseline_idx, output_dir);

    %% 5. 保存对比结果
    comparison.results = results;
    comparison.metrics = metrics;
    comparison.baseline_idx = baseline_idx;

    comparison_file = fullfile(output_dir, 'robustness_comparison.mat');
    save(comparison_file, 'comparison', '-v7.3');
    fprintf('\n[对比] 结果已保存至: %s\n', comparison_file);

    % 导出 CSV 表格
    export_csv(comparison_table, fullfile(output_dir, 'comparison_table.csv'));
    fprintf('[对比] 对比表格已导出至: %s\n', ...
        fullfile(output_dir, 'comparison_table.csv'));
end

%% ========================================================================
function tbl = build_comparison_table(results, metrics, baseline_idx)
    n = numel(results);
    tbl = table();

    tbl.name        = {results.name}';
    tbl.label       = {results.label}';
    tbl.type        = {results.type}';
    tbl.converged   = [results.converged]';
    tbl.iterations  = [results.iterations]';
    tbl.n_obs       = [results.n_obs]';

    % 残差
    tbl.res_range_rms = arrayfun(@(r) r.residual_rms(1), results)';
    tbl.res_range_max = arrayfun(@(r) r.residual_max(1), results)';

    % 推力估计
    tbl.thrust_est     = [results.thrust_estimate]';
    tbl.thrust_true    = [results.thrust_true]';
    tbl.thrust_err_abs = [results.thrust_error]';       % 绝对误差 (m/s^2)
    tbl.thrust_rel_err = [results.thrust_rel_error]' * 100;  % %

    % 12h 预报误差
    tbl.pred12h_rms  = arrayfun(@(m) m.pred_12h.position_rms, metrics)';
    tbl.pred12h_max  = arrayfun(@(m) m.pred_12h.position_max, metrics)';
    tbl.pred12h_end  = arrayfun(@(m) m.pred_12h.end_error, metrics)';
    tbl.pred12h_r_rms = arrayfun(@(m) m.pred_12h.rtn_r_rms, metrics)';
    tbl.pred12h_t_rms = arrayfun(@(m) m.pred_12h.rtn_t_rms, metrics)';
    tbl.pred12h_n_rms = arrayfun(@(m) m.pred_12h.rtn_n_rms, metrics)';

    % 24h 预报误差
    tbl.pred24h_rms  = arrayfun(@(m) m.pred_24h.position_rms, metrics)';
    tbl.pred24h_max  = arrayfun(@(m) m.pred_24h.position_max, metrics)';
    tbl.pred24h_end  = arrayfun(@(m) m.pred_24h.end_error, metrics)';
    tbl.pred24h_r_rms = arrayfun(@(m) m.pred_24h.rtn_r_rms, metrics)';
    tbl.pred24h_t_rms = arrayfun(@(m) m.pred_24h.rtn_t_rms, metrics)';
    tbl.pred24h_n_rms = arrayfun(@(m) m.pred_24h.rtn_n_rms, metrics)';

    % 耗时
    tbl.elapsed = [results.elapsed_time]';
end

%% ========================================================================
function print_comparison_table(tbl, baseline_idx)
    fprintf('\n--- 场景对比总表 ---\n\n');
    fprintf('%-28s %-4s %-4s %-10s %-10s %-10s %-10s %-10s\n', ...
        '场景', '收敛', '迭代', '12h_RMS', '12h_MAX', '24h_RMS', '24h_MAX', '推力误差%');
    fprintf('%s\n', repmat('-', 1, 100));

    for i = 1:height(tbl)
        marker = '';
        if i == baseline_idx
            marker = ' *';
        end
        if isnan(tbl.thrust_rel_err(i))
            thrust_str = 'N/A';
        else
            thrust_str = sprintf('%.2f%%', tbl.thrust_rel_err(i));
        end
        fprintf('%-26s%-2s %-4s %-4d %-10.1f %-10.1f %-10.1f %-10.1f %-10s\n', ...
            tbl.label{i}, marker, yesno(tbl.converged(i)), tbl.iterations(i), ...
            tbl.pred12h_rms(i), tbl.pred12h_max(i), ...
            tbl.pred24h_rms(i), tbl.pred24h_max(i), ...
            thrust_str);
    end
    fprintf('%s\n', repmat('-', 1, 100));
    fprintf('* = 基准场景\n');
end

%% ========================================================================
function groups = group_by_type(tbl)
    types = unique(tbl.type);
    n_types = numel(types);

    groups = struct('type', {}, 'count', {}, 'scenarios', {}, ...
        'mean_12h_rms', {}, 'mean_24h_rms', {}, 'mean_thrust_err', {});

    for i = 1:n_types
        mask = strcmp(tbl.type, types{i});
        sub = tbl(mask, :);

        groups(i).type = types{i};
        groups(i).count = height(sub);
        groups(i).scenarios = sub.label;
        groups(i).mean_12h_rms = mean(sub.pred12h_rms, 'omitnan');
        groups(i).mean_24h_rms = mean(sub.pred24h_rms, 'omitnan');
        groups(i).mean_thrust_err = mean(sub.thrust_rel_err, 'omitnan');
    end

    fprintf('\n--- 按场景类别分组统计 ---\n\n');
    fprintf('%-16s %-6s %-12s %-12s %-12s\n', ...
        '类别', '数量', '12h_RMS均值', '24h_RMS均值', '推力误差均值%');
    fprintf('%s\n', repmat('-', 1, 60));
    for i = 1:n_types
        fprintf('%-16s %-6d %-12.1f %-12.1f %-12.2f\n', ...
            groups(i).type, groups(i).count, ...
            groups(i).mean_12h_rms, groups(i).mean_24h_rms, ...
            groups(i).mean_thrust_err);
    end
    fprintf('\n');
end

%% ========================================================================
function ratios = compute_ratios(tbl, baseline_idx)
    % 计算各场景相对基准的变化倍数
    n = height(tbl);
    ratios = table();

    ratios.name = tbl.name;
    ratios.label = tbl.label;

    base_12h_rms = tbl.pred12h_rms(baseline_idx);
    base_24h_rms = tbl.pred24h_rms(baseline_idx);
    base_thrust  = tbl.thrust_rel_err(baseline_idx);

    ratios.ratio_12h_rms = tbl.pred12h_rms / base_12h_rms;
    ratios.ratio_24h_rms = tbl.pred24h_rms / base_24h_rms;
    ratios.ratio_thrust  = tbl.thrust_rel_err / max(base_thrust, eps);

    fprintf('\n--- 相对基准场景变化倍数 ---\n\n');
    fprintf('%-28s %-10s %-10s %-10s\n', ...
        '场景', '12h_RMS倍数', '24h_RMS倍数', '推力误差倍数');
    fprintf('%s\n', repmat('-', 1, 60));
    for i = 1:n
        if isnan(ratios.ratio_thrust(i))
            thrust_ratio_str = 'N/A';
        else
            thrust_ratio_str = sprintf('%.2f', ratios.ratio_thrust(i));
        end
        fprintf('%-28s %-10.2f %-10.2f %-10s\n', ...
            ratios.label{i}, ratios.ratio_12h_rms(i), ...
            ratios.ratio_24h_rms(i), thrust_ratio_str);
    end
    fprintf('\n');
end

%% ========================================================================
function generate_figures(results, metrics, tbl, baseline_idx, output_dir)
    fig_dir = fullfile(output_dir, 'figures');
    if ~exist(fig_dir, 'dir')
        mkdir(fig_dir);
    end

    n = numel(results);

    % 图1：12h/24h 位置RMS对比柱状图
    fig1 = figure('Name', 'Position RMS Comparison', 'Position', [100 100 900 400]);
    data_rms = [tbl.pred12h_rms, tbl.pred24h_rms];
    b = bar(data_rms, 'grouped');
    b(1).FaceColor = [0.2 0.6 0.8];
    b(2).FaceColor = [0.8 0.4 0.2];
    set(gca, 'XTickLabel', tbl.label, 'XTickLabelRotation', 45);
    ylabel('位置RMS (m)');
    legend({'12h', '24h'}, 'Location', 'northwest');
    title('各场景轨道预报位置RMS对比');
    grid on;
    saveas(fig1, fullfile(fig_dir, 'fig01_position_rms_comparison.png'));

    % 图2：推力估计相对误差对比
    fig2 = figure('Name', 'Thrust Error', 'Position', [100 100 900 400]);
    bar(tbl.thrust_rel_err, 'FaceColor', [0.3 0.7 0.3]);
    set(gca, 'XTickLabel', tbl.label, 'XTickLabelRotation', 45);
    ylabel('推力估计相对误差 (%)');
    title('各场景小推力估计相对误差');
    grid on;
    saveas(fig2, fullfile(fig_dir, 'fig02_thrust_error_comparison.png'));

    % 图3：迭代次数对比
    fig3 = figure('Name', 'Iterations', 'Position', [100 100 900 400]);
    bar(tbl.iterations, 'FaceColor', [0.6 0.6 0.8]);
    set(gca, 'XTickLabel', tbl.label, 'XTickLabelRotation', 45);
    ylabel('迭代次数');
    title('各场景定轨迭代次数');
    grid on;
    saveas(fig3, fullfile(fig_dir, 'fig03_iterations_comparison.png'));

    % 图4：预报误差时间序列（选取部分场景）
    plot_types = {'baseline', 'noise', 'bias', 'dropout', 'outlier', 'model_mismatch'};
    fig4 = figure('Name', 'Error Time Series 24h', 'Position', [100 100 900 500]);
    hold on;
    colors = lines(n);
    legend_entries = {};
    legend_handles = [];
    for i = 1:n
        m = metrics(i);
        if isempty(m.pred_24h.t_pred_h)
            continue;
        end
        h = plot(m.pred_24h.t_pred_h, m.pred_24h.eci_pos_norm / 1000, ...
            'LineWidth', 1.2, 'Color', colors(i, :));
        legend_entries{end+1} = m.label;
        legend_handles(end+1) = h;
    end
    hold off;
    xlabel('预报时长 (h)');
    ylabel('位置误差 (km)');
    title('24h轨道预报位置误差时间序列');
    legend(legend_handles, legend_entries, 'Location', 'northwest');
    grid on;
    saveas(fig4, fullfile(fig_dir, 'fig04_error_timeseries_24h.png'));

    % 图5：RTN误差分量对比（24h，选取基准和典型异常场景）
    fig5 = figure('Name', 'RTN Error 24h', 'Position', [100 100 900 400]);
    hold on;
    for i = 1:n
        m = metrics(i);
        if isempty(m.pred_24h.t_pred_h)
            continue;
        end
        plot(m.pred_24h.t_pred_h, m.pred_24h.rtn_err(1, :) / 1000, ...
            'LineWidth', 1, 'Color', colors(i, :));
    end
    hold off;
    xlabel('预报时长 (h)');
    ylabel('R方向误差 (km)');
    title('24h RTN径向误差对比');
    grid on;
    saveas(fig5, fullfile(fig_dir, 'fig05_rtn_r_error_24h.png'));

    % 图6：12h vs 24h 末端误差散点图（带智能标注偏移避免重叠）
    fig6 = figure('Name', 'End Error Scatter', 'Position', [100 100 700 500]);
    scatter(tbl.pred12h_end / 1000, tbl.pred24h_end / 1000, 80, ...
        'filled', 'MarkerEdgeColor', 'k');

    % 智能标注：对每个点计算偏移方向，避免文字重叠
    offsets = compute_label_offsets(tbl.pred12h_end / 1000, tbl.pred24h_end / 1000);
    for i = 1:n
        text(tbl.pred12h_end(i)/1000 + offsets(i,1), ...
             tbl.pred24h_end(i)/1000 + offsets(i,2), ...
            tbl.label{i}, 'FontSize', 7, 'Rotation', 35, ...
            'HorizontalAlignment', 'left', 'BackgroundColor', [1 1 1 0.6]);
    end
    xlabel('12h末端误差 (km)');
    ylabel('24h末端误差 (km)');
    title('预报末端误差对比');
    grid on;
    hold on;
    lim = max([tbl.pred12h_end, tbl.pred24h_end]) / 1000 * 1.1;
    plot([0 lim], [0 lim], 'k--');
    hold off;
    saveas(fig6, fullfile(fig_dir, 'fig06_end_error_scatter.png'));

    fprintf('[对比] 已生成 %d 张对比图形\n', 6);
end

%% ========================================================================
function export_csv(tbl, filepath)
    % 将表格导出为 CSV
    T = tbl;
    % 将字符向量转为字符串
    T.name = string(T.name);
    T.label = string(T.label);
    T.type = string(T.type);
    writetable(T, filepath);
end

%% ========================================================================
function s = yesno(flag)
    if flag
        s = '是';
    else
        s = '否';
    end
end

%% ========================================================================
function offsets = compute_label_offsets(x, y)
    % 计算每个标注点的偏移量，避免文字重叠
    % 策略：按到原点的角度均匀分布在4个象限方向上偏移
    n = numel(x);
    offsets = zeros(n, 2);
    base_offset = 0.015 * max(max(x) - min(x), max(y) - min(y));
    if base_offset < 0.001
        base_offset = 0.001;
    end

    % 计算每个点到坐标轴原点的角度
    for i = 1:n
        angle = atan2(y(i), x(i));
        offsets(i, 1) = base_offset * cos(angle);
        offsets(i, 2) = base_offset * sin(angle);

        % 对于靠近的点（NaN除外），额外增加偏移
        for j = 1:i-1
            if ~isnan(x(i)) && ~isnan(x(j)) && ~isnan(y(i)) && ~isnan(y(j))
                dx = x(i) - x(j);
                dy = y(i) - y(j);
                dist = sqrt(dx^2 + dy^2);
                if dist < 2 * base_offset
                    % 靠近的点：往对角方向额外偏移
                    offsets(i, 1) = offsets(i, 1) + base_offset * 1.2 * sign(dx + 0.01);
                    offsets(i, 2) = offsets(i, 2) + base_offset * 1.2 * sign(dy + 0.01);
                end
            end
        end
    end
end
