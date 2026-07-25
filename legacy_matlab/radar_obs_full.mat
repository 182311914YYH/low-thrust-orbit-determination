function [t_pred, state_pred] = orbit_prediction(X_params, t_start, pred_duration, cfg)
% 轨道预报模块：基于估计参数进行指定时长的轨道预报
% 统一使用 m、m/s、m/s^2 单位
% 输入：
%   X_params     : 7×1 估计参数 [r0(m); v0(m/s); at(m/s^2)]
%   t_start      : 预报起始时刻 (s)
%   pred_duration: 预报时长，支持字符或数值
%                  字符选项: '12h' (12小时), '24h' (24小时)
%                  数值: 预报时长(s)
%   cfg          : 全局配置结构体
% 输出：
%   t_pred       : M×1 预报时刻序列 (s)
%   state_pred   : 6×M 预报状态历史，每列对应一个时刻 [r; v]

    r0 = X_params(1:3);
    v0 = X_params(4:6);
    at = X_params(7);

    if ischar(pred_duration) || isstring(pred_duration)
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

    dt_step = cfg.prediction.outputStep;
    if isempty(dt_step) || dt_step <= 0
        dt_step = 60;
    end

    t_pred = (t_start : dt_step : t_start + dt_pred)';
    if t_pred(end) < t_start + dt_pred
        t_pred = [t_pred; t_start + dt_pred];
    end

    state_pred = propagate_orbit_j2_thrust( ...
        t_start, [r0; v0], at, t_pred, cfg);
end
