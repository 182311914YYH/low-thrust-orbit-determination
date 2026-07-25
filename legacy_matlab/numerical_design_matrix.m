function [X_hat, residuals, iter_num, converged, history] = ...
    batch_od_thrust_estimator(X0_guess, t_reference, t_obs, y_obs, cfg)
% 批处理加权最小二乘精密定轨与沿迹小推力估计主程序
% 统一使用 m、m/s、m/s^2 单位
% 采用 Levenberg-Marquardt 阻尼 + 参数尺度归一化 + SVD 稳定求解

    X_curr = X0_guess(:);
    max_iter = cfg.estimation.maxIterations;
    conv_thresh = cfg.estimation.parameterTolerance;
    cost_thresh = cfg.estimation.costTolerance;

    % 参数尺度归一化
    parameter_scale = [
        repmat(cfg.estimation.scale.position, 3, 1)
        repmat(cfg.estimation.scale.velocity, 3, 1)
        cfg.estimation.scale.thrust
    ];
    S = diag(parameter_scale);

    % 观测标准差堆叠
    sigma_one_epoch = cfg.measurement.sigmaVector(:);
    if any(sigma_one_epoch <= 0)
        error('观测标准差必须全部大于0。');
    end
    sigma_stacked = repmat(sigma_one_epoch, numel(t_obs), 1);

    lambda = cfg.estimation.initialDamping;
    converged = false;
    iter_num = 0;

    % 预分配历史记录
    history.X = nan(7, max_iter + 1);
    history.X(:,1) = X_curr;
    history.cost = nan(max_iter, 1);
    history.trial_cost = nan(max_iter, 1);
    history.normalized_step = nan(max_iter, 1);
    history.lambda = nan(max_iter, 1);
    history.accepted = false(max_iter, 1);
    history.condition_number = nan(max_iter, 1);
    history.thrust = nan(max_iter + 1, 1);
    history.thrust(1) = X_curr(7);

    fprintf('\n====================== 迭代记录 ======================\n');

    for k = 1:max_iter
        iter_num = k;

        % 当前理论观测与残差
        y_calc = compute_all_observations(X_curr, t_reference, t_obs, cfg);
        residual_matrix = observation_residual(y_obs, y_calc);
        residual_vector = stack_observations(residual_matrix);
        weighted_residual = residual_vector ./ sigma_stacked;
        current_cost = 0.5 * (weighted_residual' * weighted_residual);

        % 数值差分设计矩阵
        H = numerical_design_matrix(X_curr, t_reference, t_obs, cfg, y_calc);

        % 参数尺度归一化：dx = S*du，因此 H_scaled = H*S
        H_scaled = H * S;
        H_weighted = H_scaled ./ sigma_stacked;

        normal_matrix = H_weighted' * H_weighted;
        right_hand = H_weighted' * weighted_residual;

        history.cost(k) = current_cost;
        history.condition_number(k) = safe_condition_number(normal_matrix);

        accepted = false;
        candidate_cost = inf;
        dx = zeros(7,1);
        X_candidate = X_curr;

        % Levenberg-Marquardt 内循环
        for trial = 1:cfg.estimation.maxDampingTrials
            diagonal_scale = max(diag(normal_matrix), 1e-12);
            damped_matrix = normal_matrix + lambda * diag(diagonal_scale);

            du = stable_linear_solve(damped_matrix, right_hand, cfg);
            dx_trial = S * du;

            % 限制单次归一化修正，避免非线性初期出现过大跳步
            trial_norm = norm(dx_trial ./ parameter_scale);
            if trial_norm > cfg.estimation.maxNormalizedStep
                dx_trial = dx_trial * ...
                    (cfg.estimation.maxNormalizedStep / trial_norm);
            end

            X_trial = enforce_parameter_bounds(X_curr + dx_trial, cfg);
            dx_trial = X_trial - X_curr;

            if any(~isfinite(X_trial))
                candidate_cost = inf;
                lambda = min(lambda * cfg.estimation.dampingIncrease, 1e16);
                continue;
            end

            try
                y_trial = compute_all_observations(X_trial, t_reference, t_obs, cfg);
                residual_trial = observation_residual(y_obs, y_trial);
                residual_trial_vector = stack_observations(residual_trial);
                weighted_trial = residual_trial_vector ./ sigma_stacked;
                candidate_cost = 0.5 * (weighted_trial' * weighted_trial);
            catch
                candidate_cost = inf;
            end

            if candidate_cost < current_cost
                accepted = true;
                X_candidate = X_trial;
                dx = dx_trial;
                lambda = max(lambda * cfg.estimation.dampingDecrease, 1e-12);
                break;
            end

            lambda = min(lambda * cfg.estimation.dampingIncrease, 1e16);
        end

        history.trial_cost(k) = candidate_cost;
        history.lambda(k) = lambda;
        history.accepted(k) = accepted;

        if ~accepted
            history.X(:,k+1) = X_curr;
            history.thrust(k+1) = X_curr(7);
            warning('第%d次迭代未找到降低代价的参数修正，迭代提前结束。', k);
            break;
        end

        normalized_step = norm(dx ./ parameter_scale);
        relative_cost_change = abs(current_cost - candidate_cost) / ...
            max(current_cost, 1);

        X_curr = X_candidate;
        history.X(:,k+1) = X_curr;
        history.thrust(k+1) = X_curr(7);
        history.normalized_step(k) = normalized_step;

        fprintf(['迭代 %2d：代价 %.6e -> %.6e，', ...
                 '归一化修正 %.3e，aT = %.9e，cond = %.3e\n'], ...
            k, current_cost, candidate_cost, normalized_step, ...
            X_curr(7), history.condition_number(k));

        correction_converged = normalized_step < conv_thresh;
        cost_converged = relative_cost_change < cost_thresh;

        if correction_converged || cost_converged
            converged = true;
            break;
        end
    end

    X_hat = X_curr;
    y_final = compute_all_observations(X_hat, t_reference, t_obs, cfg);
    residuals = observation_residual(y_obs, y_final);

    % 截断预分配的历史数组
    history.X = history.X(:,1:iter_num+1);
    history.cost = history.cost(1:iter_num);
    history.trial_cost = history.trial_cost(1:iter_num);
    history.normalized_step = history.normalized_step(1:iter_num);
    history.lambda = history.lambda(1:iter_num);
    history.accepted = history.accepted(1:iter_num);
    history.condition_number = history.condition_number(1:iter_num);
    history.thrust = history.thrust(1:iter_num+1);
end

%% ========================================================================子函数
function residual = observation_residual(y_obs, y_calc)
    residual = y_obs - y_calc;
    residual(:,2) = wrap_to_pi(residual(:,2));
end

function difference = observation_difference(y_a, y_b)
    difference = y_a - y_b;
    difference(:,2) = wrap_to_pi(difference(:,2));
end

function vector = stack_observations(matrix_Nx3)
    vector = reshape(matrix_Nx3.', [], 1);
end

function X = enforce_parameter_bounds(X, cfg)
    lower_at = cfg.thrust.estimate.lowerBoundRTN(2);
    upper_at = cfg.thrust.estimate.upperBoundRTN(2);
    X(7) = min(max(X(7), lower_at), upper_at);
end

function x = stable_linear_solve(A, b, cfg)
    if cfg.estimation.useSVD || rcond(A) < 1e-12
        [U, S_svd, V] = svd(A, 'econ');
        s = diag(S_svd);
        if isempty(s)
            x = zeros(size(b));
            return;
        end
        tolerance = max(size(A)) * eps(max(s));
        inverse_s = zeros(size(s));
        valid = s > tolerance;
        inverse_s(valid) = 1 ./ s(valid);
        x = V * (inverse_s .* (U' * b));
    else
        x = A \ b;
    end
end

function c = safe_condition_number(A)
    try
        c = cond(A);
    catch
        c = inf;
    end
end

function angle = wrap_to_pi(angle)
    angle = mod(angle + pi, 2*pi) - pi;
end
