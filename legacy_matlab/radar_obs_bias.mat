function stats = error_statistics(err_rtn, t_pred)
% 误差统计评价模块：计算RTN误差的RMS和最大误差
% 输入：
%   err_rtn   : 3×M RTN误差矩阵 [err_R; err_T; err_N]
%   t_pred    : M×1 预报时刻序列 (s)
% 输出：
%   stats     : 结构体，包含各项统计指标

    stats.R_RMS = rms(err_rtn(1, :));
    stats.T_RMS = rms(err_rtn(2, :));
    stats.N_RMS = rms(err_rtn(3, :));
    stats.Overall_RMS = rms(err_rtn(:));
    
    stats.R_Max = max(abs(err_rtn(1, :)));
    stats.T_Max = max(abs(err_rtn(2, :)));
    stats.N_Max = max(abs(err_rtn(3, :)));
    stats.Overall_Max = max(err_rtn(:));
    
    [stats.R_Max, idx_R] = max(abs(err_rtn(1, :)));
    [stats.T_Max, idx_T] = max(abs(err_rtn(2, :)));
    [stats.N_Max, idx_N] = max(abs(err_rtn(3, :)));
    
    stats.R_Max_Time = t_pred(idx_R);
    stats.T_Max_Time = t_pred(idx_T);
    stats.N_Max_Time = t_pred(idx_N);
    
    stats.Overall_Max = max([stats.R_Max, stats.T_Max, stats.N_Max]);
    
    stats.Duration_h = (t_pred(end) - t_pred(1)) / 3600;
end
