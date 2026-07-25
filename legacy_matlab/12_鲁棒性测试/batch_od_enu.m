function [X_hat, residuals, iter_num, converged, history] = ...
    batch_od_enu(X0_guess, t_reference, t_obs, y_obs, cfg)
%BATCH_OD_ENU  使用ENU观测模型的批处理加权最小二乘定轨
%
% 与原始 batch_od_thrust_estimator 逻辑完全一致，
% 唯一区别：使用 compute_obs_enu 替代 compute_all_observations，
% 确保估计端与数据生成端（enu_obs_model）观测模型一致。
%
% 输入/输出与 batch_od_thrust_estimator 相同。

    X_curr = X0_guess(:);
    max_iter = cfg.estimation.maxIterations;
    conv_thresh = cfg.estimation.parameterTolerance;
    cost_thresh = cfg.estimation.costTolerance;

    parameter_scale = [
        repmat(cfg.estimation.scale.position, 3, 1)
        repmat(cfg.estimation.scale.velocity, 3, 1)
        cfg.estimation.scale.thrust
    ];
    S = diag(parameter_scale);

    sigma_one_epoch = cfg.measurement.sigmaVector(:);
    if any(sigma_one_epoch <= 0)
        error('观测标准差必须全部大于0。');
    end
    sigma_stacked = repmat(sigma_one_epoch, numel(t_obs), 1);

    lambda = cfg.estimation.initialDamping;
    converged = false;
    iter_num = 0;

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

        y_calc = compute_obs_enu(X_curr, t_reference, t_obs, cfg);
        residual_matrix = obs_residual(y_obs, y_calc);
        residual_vector = stack_obs(residual_matrix);
        weighted_residual = residual_vector ./ sigma_stacked;
        current_cost = 0.5 * (weighted_residual' * weighted_residual);

        H = design_matrix_enu(X_curr, t_reference, t_obs, cfg, y_calc);

        H_scaled = H * S;
        H_weighted = H_scaled ./ sigma_stacked;

        normal_matrix = H_weighted' * H_weighted;
        right_hand = H_weighted' * weighted_residual;

        history.cost(k) = current_cost;
        history.condition_number(k) = safe_cond(normal_matrix);

        accepted = false;
        candidate_cost = inf;
        dx = zeros(7,1);
        X_candidate = X_curr;

        for trial = 1:cfg.estimation.maxDampingTrials
            diagonal_scale = max(diag(normal_matrix), 1e-12);
            damped_matrix = normal_matrix + lambda * diag(diagonal_scale);

            du = stable_solve(damped_matrix, right_hand, cfg);
            dx_trial = S * du;

            trial_norm = norm(dx_trial ./ parameter_scale);
            if trial_norm > cfg.estimation.maxNormalizedStep
                dx_trial = dx_trial * ...
                    (cfg.estimation.maxNormalizedStep / trial_norm);
            end

            X_trial = enforce_bounds(X_curr + dx_trial, cfg);
            dx_trial = X_trial - X_curr;

            if any(~isfinite(X_trial))
                candidate_cost = inf;
                lambda = min(lambda * cfg.estimation.dampingIncrease, 1e16);
                continue;
            end

            try
                y_trial = compute_obs_enu(X_trial, t_reference, t_obs, cfg);
                residual_trial = obs_residual(y_obs, y_trial);
                residual_trial_vector = stack_obs(residual_trial);
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

        if normalized_step < conv_thresh || relative_cost_change < cost_thresh
            converged = true;
            break;
        end
    end

    X_hat = X_curr;
    y_final = compute_obs_enu(X_hat, t_reference, t_obs, cfg);
    residuals = obs_residual(y_obs, y_final);

    history.X = history.X(:,1:iter_num+1);
    history.cost = history.cost(1:iter_num);
    history.trial_cost = history.trial_cost(1:iter_num);
    history.normalized_step = history.normalized_step(1:iter_num);
    history.lambda = history.lambda(1:iter_num);
    history.accepted = history.accepted(1:iter_num);
    history.condition_number = history.condition_number(1:iter_num);
    history.thrust = history.thrust(1:iter_num+1);
end

%% ========================================================================
function H = design_matrix_enu(X_params, t_reference, t_obs, cfg, y0)
    steps = [
        repmat(cfg.estimation.fd.position, 3, 1)
        repmat(cfg.estimation.fd.velocity, 3, 1)
        cfg.estimation.fd.thrust
    ];

    n_parameter = numel(X_params);
    n_measurement = 3 * numel(t_obs);
    H = zeros(n_measurement, n_parameter);

    scheme = lower(cfg.estimation.fd.scheme);

    for j = 1:n_parameter
        h = steps(j);

        X_plus = X_params;
        X_plus(j) = X_plus(j) + h;
        y_plus = compute_obs_enu(X_plus, t_reference, t_obs, cfg);

        if strcmp(scheme, 'central')
            X_minus = X_params;
            X_minus(j) = X_minus(j) - h;
            y_minus = compute_obs_enu(X_minus, t_reference, t_obs, cfg);
            delta_y = obs_diff(y_plus, y_minus);
            derivative = delta_y / (2*h);
        else
            delta_y = obs_diff(y_plus, y0);
            derivative = delta_y / h;
        end

        H(:,j) = stack_obs(derivative);
    end
end

%% ========================================================================
function r = obs_residual(y_obs, y_calc)
    r = y_obs - y_calc;
    r(:,2) = wrap_angle(r(:,2));
end

function d = obs_diff(y_a, y_b)
    d = y_a - y_b;
    d(:,2) = wrap_angle(d(:,2));
end

function v = stack_obs(matrix_Nx3)
    v = reshape(matrix_Nx3.', [], 1);
end

function X = enforce_bounds(X, cfg)
    X(7) = min(max(X(7), cfg.thrust.estimate.lowerBoundRTN(2)), ...
                  cfg.thrust.estimate.upperBoundRTN(2));
end

function x = stable_solve(A, b, cfg)
    if cfg.estimation.useSVD || rcond(A) < 1e-12
        [U, S_svd, V] = svd(A, 'econ');
        s = diag(S_svd);
        if isempty(s)
            x = zeros(size(b,1), 1);
            return;
        end
        tol = max(size(A)) * eps(max(s));
        inv_s = zeros(size(s));
        valid = s > tol;
        inv_s(valid) = 1 ./ s(valid);
        x = V * (inv_s .* (U' * b));
    else
        x = A \ b;
    end
end

function c = safe_cond(A)
    try
        c = cond(A);
    catch
        c = inf;
    end
end

function angle = wrap_angle(angle)
    angle = mod(angle + pi, 2*pi) - pi;
end
