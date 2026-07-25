function [t_out, state_out] = orbit_propagate(cfg, x0)
%ORBIT_PROPAGATE 轨道数值传播统一接口
% 输入：
%   cfg 全局配置
%   x0  6×1 ECI初始轨道状态 [r;v]
% 输出：
%   t_out     N×1 时间序列(s)
%   state_out N×6 状态 [x,y,z,vx,vy,vz]

%% 读取仿真参数
t_total = cfg.sim.truthDuration;
dt_step = cfg.sim.integrationStep;
abs_tol = cfg.sim.absTol;
rel_tol = cfg.sim.relTol;
integrator = cfg.sim.integrator;

%% 生成均匀时间节点
t_span = 0 : dt_step : t_total;

%% 积分选项
opts = odeset(...
    'AbsTol', abs_tol, ...
    'RelTol', rel_tol);

%% 数值积分
ode_fun = @(t,x) dyn_equation(t, x, cfg);
[t_raw, x_raw] = feval(integrator, ode_fun, t_span, x0, opts);

%% 转换为标准输出格式 N×1、N×6
t_out = t_raw(:);
state_out = x_raw;

end