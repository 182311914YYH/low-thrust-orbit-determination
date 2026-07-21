function H = numerical_design_matrix(X_params, t_obs, r_station, prop_func, epsilon)
% 前向数值差分法计算观测设计矩阵 H
% 输入：
%   X_params  : 7×1 当前参数值
%   t_obs     : N×1 观测时刻
%   r_station : 3×1 测站位置
%   prop_func : 轨道传播函数句柄
%   epsilon   : 差分扰动量（可选，默认1e-6）
% 输出：
%   H         : (3N)×7 设计矩阵，每行对应1个观测分量，每列对应1个待估参数

    if nargin < 5
        epsilon = 1e-6;
    end
    
    n_param = length(X_params);
    n_obs = length(t_obs);
    n_obs_comp = 3 * n_obs;
    H = zeros(n_obs_comp, n_param);
    
    % 基准观测值
    y0 = compute_all_observations(X_params, t_obs, r_station, prop_func);
    y0_vec = y0(:);
    
    % 逐参数加扰动，计算偏导列
    for j = 1:n_param
        X_pert = X_params;
        X_pert(j) = X_pert(j) + epsilon;
        
        y_pert = compute_all_observations(X_pert, t_obs, r_station, prop_func);
        y_pert_vec = y_pert(:);
        
        H(:, j) = (y_pert_vec - y0_vec) / epsilon;
    end
end