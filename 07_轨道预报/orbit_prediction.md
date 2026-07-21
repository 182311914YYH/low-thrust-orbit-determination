function [t_pred, state_pred] = orbit_prediction(X_params, t_final, pred_duration, prop_func)
% 轨道预报模块：基于估计参数进行指定时长的轨道预报
% 支持12小时、24小时及任意时长预报
% 输入：
%   X_params    : 7×1 估计参数 [r0(km); v0(km/s); at(km/s^2)]
%   t_final     : 定轨弧段结束时刻 (s)，作为预报起始时刻
%   pred_duration: 预报时长，支持字符或数值
%                  字符选项: '12h' (12小时), '24h' (24小时)
%                  数值: 预报时长(s)
%   prop_func   : 轨道传播函数句柄
% 输出：
%   t_pred      : M×1 预报时刻序列 (s)
%   state_pred  : 6×M 预报状态历史，每列对应一个时刻 [r; v]

    r0 = X_params(1:3);
    v0 = X_params(4:6);
    at = X_params(7);
    
    if ischar(pred_duration)
        switch pred_duration
            case '12h'
                dt_pred = 12 * 3600;
            case '24h'
                dt_pred = 24 * 3600;
            otherwise
                error('不支持的预报时长选项，支持 ''12h'' 或 ''24h''');
        end
    else
        dt_pred = pred_duration;
    end
    
    dt_step = 60;
    t_pred = (t_final : dt_step : t_final + dt_pred)';
    
    state_pred = prop_func(t_final, [r0; v0], at, t_pred);
end
