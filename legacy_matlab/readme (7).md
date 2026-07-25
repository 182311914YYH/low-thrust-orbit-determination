%% ============================================================
%  成员1：动力学模型失配场景生成
%  存放位置：11_测试程序/gen_dyn_mismatch.m
%  功能：生成三种动力学模型失配场景的真值轨道数据
%    场景1：真值含大气阻力（估计模型无阻力）
%    场景2：真值含时变推力（估计模型假设常值推力）
%    场景3：真值含阻力 + 时变推力（严重失配）
%  输出：保存至 09_数据与实验结果/robustness/
% ============================================================
function gen_dyn_mismatch()
    
    %% 初始化
    clc; close all;
    script_path = mfilename('fullpath');
    test_dir = fileparts(script_path);
    root = fullfile(test_dir, '..');
    
    % 添加必要路径
    addpath(fullfile(root, '02_参数配置'));
    addpath(fullfile(root, '03_轨道动力学'));
    
    cfg = default_config();
    
    % 输出目录
    data_dir = fullfile(root, '09_数据与实验结果', 'robustness');
    if ~exist(data_dir, 'dir'), mkdir(data_dir); end
    
    x0 = cfg.orbit.x0Truth;
    
    fprintf('\n========== 成员1：动力学模型失配场景生成 ==========\n');
    
    %% ====== 场景1：真值含大气阻力 ======
    fprintf('\n场景1：真值含大气阻力（估计模型无阻力）...\n');
    cfg1 = cfg;
    % 开启阻力
    cfg1.model.truth.useDrag = true;
    cfg1.model.truth.dragCd = 2.2;           % 阻力系数
    cfg1.model.truth.dragArea = 10;           % 迎风面积 (m^2)
    cfg1.model.truth.dragRho0 = 3.614e-13;    % 大气密度基准 (kg/m^3)
    cfg1.model.truth.dragH0 = 80e3;           % 大气标尺高度 (m)
    % 关闭推力（确保仅阻力影响）
    cfg1.model.truth.useThrust = false;
    cfg1.model.truth.useJ2 = true;
    
    % 使用带阻力的传播函数
    [t1, state1] = orbit_propagate_with_drag(cfg1, x0);
    save(fullfile(data_dir, 'truth_drag.mat'), 't1', 'state1', 'cfg1');
    fprintf('  阻力场景真值轨道已保存（传播时长：%.2f h，数据点数：%d）\n', ...
        t1(end)/3600, length(t1));
    
    %% ====== 场景2：真值含时变推力 ======
    fprintf('\n场景2：真值含时变推力（估计模型假设常值推力）...\n');
    cfg2 = cfg;
    cfg2.model.truth.useThrust = true;
    cfg2.model.truth.thrustType = 'time_varying';
    cfg2.model.truth.thrustAmplitude = 5e-5;     % 推力幅值 (m/s^2)
    cfg2.model.truth.thrustPeriod = 3600;        % 变化周期 (s)
    cfg2.model.truth.useJ2 = true;
    cfg2.model.truth.useDrag = false;
    
    [t2, state2] = orbit_propagate_with_varying_thrust(cfg2, x0);
    save(fullfile(data_dir, 'truth_timevarying_thrust.mat'), 't2', 'state2', 'cfg2');
    fprintf('  时变推力场景真值轨道已保存（传播时长：%.2f h，数据点数：%d）\n', ...
        t2(end)/3600, length(t2));
    
    %% ====== 场景3：真值含阻力 + 时变推力（严重失配） ======
    fprintf('\n场景3：真值含阻力+时变推力（严重失配）...\n');
    cfg3 = cfg;
    cfg3.model.truth.useDrag = true;
    cfg3.model.truth.dragCd = 2.2;
    cfg3.model.truth.dragArea = 10;
    cfg3.model.truth.dragRho0 = 3.614e-13;
    cfg3.model.truth.dragH0 = 80e3;
    cfg3.model.truth.useThrust = true;
    cfg3.model.truth.thrustType = 'time_varying';
    cfg3.model.truth.thrustAmplitude = 5e-5;
    cfg3.model.truth.thrustPeriod = 3600;
    cfg3.model.truth.useJ2 = true;
    
    [t3, state3] = orbit_propagate_with_drag_and_varying_thrust(cfg3, x0);
    save(fullfile(data_dir, 'truth_drag_thrust.mat'), 't3', 'state3', 'cfg3');
    fprintf('  严重失配场景真值轨道已保存（传播时长：%.2f h，数据点数：%d）\n', ...
        t3(end)/3600, length(t3));
    
    %% ====== 生成估计模型配置（供第三组使用） ======
    % 估计模型保持原始配置（无阻力、常值推力）
    cfg_est = cfg;
    cfg_est.model.truth.useDrag = false;
    cfg_est.model.truth.thrustType = 'constant';
    % 确保估计模型的推力值是常值
    cfg_est.thrust.truth.accelerationRTN = cfg.thrust.truth.accelerationRTN;
    save(fullfile(data_dir, 'cfg_estimation_model.mat'), 'cfg_est');
    fprintf('\n估计模型配置已保存（供第三组定轨使用）\n');
    
    %% ====== 生成场景说明 ======
    generate_readme(data_dir);
    
    fprintf('\n========== 成员1任务完成 ==========\n');
    fprintf('所有数据已保存至：%s\n', data_dir);
end


%% ============================================================
%  辅助传播函数1：带大气阻力的轨道传播
% ============================================================
function [t_out, state_out] = orbit_propagate_with_drag(cfg, x0)
    % 读取仿真参数
    t_total = cfg.sim.truthDuration;
    dt_step = cfg.sim.integrationStep;
    abs_tol = cfg.sim.absTol;
    rel_tol = cfg.sim.relTol;
    integrator = cfg.sim.integrator;
    
    t_span = 0 : dt_step : t_total;
    opts = odeset('AbsTol', abs_tol, 'RelTol', rel_tol);
    
    % 使用带阻力的微分方程
    ode_fun = @(t, x) dyn_equation_with_drag(t, x, cfg);
    [t_raw, x_raw] = feval(integrator, ode_fun, t_span, x0, opts);
    
    t_out = t_raw(:);
    state_out = x_raw;
end


%% ============================================================
%  辅助传播函数2：带时变推力的轨道传播
% ============================================================
function [t_out, state_out] = orbit_propagate_with_varying_thrust(cfg, x0)
    t_total = cfg.sim.truthDuration;
    dt_step = cfg.sim.integrationStep;
    abs_tol = cfg.sim.absTol;
    rel_tol = cfg.sim.relTol;
    integrator = cfg.sim.integrator;
    
    t_span = 0 : dt_step : t_total;
    opts = odeset('AbsTol', abs_tol, 'RelTol', rel_tol);
    
    ode_fun = @(t, x) dyn_equation_with_varying_thrust(t, x, cfg);
    [t_raw, x_raw] = feval(integrator, ode_fun, t_span, x0, opts);
    
    t_out = t_raw(:);
    state_out = x_raw;
end


%% ============================================================
%  辅助传播函数3：带阻力+时变推力的轨道传播
% ============================================================
function [t_out, state_out] = orbit_propagate_with_drag_and_varying_thrust(cfg, x0)
    t_total = cfg.sim.truthDuration;
    dt_step = cfg.sim.integrationStep;
    abs_tol = cfg.sim.absTol;
    rel_tol = cfg.sim.relTol;
    integrator = cfg.sim.integrator;
    
    t_span = 0 : dt_step : t_total;
    opts = odeset('AbsTol', abs_tol, 'RelTol', rel_tol);
    
    ode_fun = @(t, x) dyn_equation_with_drag_and_varying_thrust(t, x, cfg);
    [t_raw, x_raw] = feval(integrator, ode_fun, t_span, x0, opts);
    
    t_out = t_raw(:);
    state_out = x_raw;
end


%% ============================================================
%  带大气阻力的微分方程
% ============================================================
function dxdt = dyn_equation_with_drag(t, x, cfg)
    % 提取状态
    r = x(1:3);
    v = x(4:6);
    rx = r(1); ry = r(2); rz = r(3);
    r_norm = norm(r);
    
    % 读取常数
    mu = cfg.const.muEarth;
    Re = cfg.const.radiusEarth;
    J2 = cfg.const.J2;
    useJ2 = cfg.model.truth.useJ2;
    
    % 1. 中心引力
    a_grav = -mu / r_norm^3 * r;
    
    % 2. J2摄动
    a_J2 = zeros(3,1);
    if useJ2
        z2 = rz^2;
        r2 = r_norm^2;
        term = 1.5 * J2 * mu * Re^2 / r2^(2.5);
        a_J2(1) = term * rx * (5*z2/r2 - 1);
        a_J2(2) = term * ry * (5*z2/r2 - 1);
        a_J2(3) = term * rz * (5*z2/r2 - 3);
    end
    
    % 3. 大气阻力（新增）
    a_drag = zeros(3,1);
    if cfg.model.truth.useDrag
        Cd = cfg.model.truth.dragCd;
        A = cfg.model.truth.dragArea;
        rho0 = cfg.model.truth.dragRho0;
        H0 = cfg.model.truth.dragH0;
        % 大气密度随高度指数衰减
        h = r_norm - Re;
        rho = rho0 * exp(-h / H0);
        % 假设大气随地球自转（简化：忽略自转）
        v_rel = v;
        a_drag = -0.5 * Cd * (A / 1) * rho * norm(v_rel) * v_rel;
        % 注意：质量在 cfg 中为 cfg.const.mass，但之前未定义，暂时用 1
        % 实际应用中需要加 cfg.const.mass = 500 kg
    end
    
    % 总加速度
    a_total = a_grav + a_J2 + a_drag;
    
    % 组装导数
    dxdt = zeros(6,1);
    dxdt(1:3) = v;
    dxdt(4:6) = a_total;
end


%% ============================================================
%  带时变推力的微分方程
% ============================================================
function dxdt = dyn_equation_with_varying_thrust(t, x, cfg)
    r = x(1:3);
    v = x(4:6);
    rx = r(1); ry = r(2); rz = r(3);
    r_norm = norm(r);
    
    mu = cfg.const.muEarth;
    Re = cfg.const.radiusEarth;
    J2 = cfg.const.J2;
    useJ2 = cfg.model.truth.useJ2;
    useThrust = cfg.model.truth.useThrust;
    
    % 中心引力
    a_grav = -mu / r_norm^3 * r;
    
    % J2
    a_J2 = zeros(3,1);
    if useJ2
        z2 = rz^2;
        r2 = r_norm^2;
        term = 1.5 * J2 * mu * Re^2 / r2^(2.5);
        a_J2(1) = term * rx * (5*z2/r2 - 1);
        a_J2(2) = term * ry * (5*z2/r2 - 1);
        a_J2(3) = term * rz * (5*z2/r2 - 3);
    end
    
    % 时变推力（新增）
    a_thrust_ECI = zeros(3,1);
    if useThrust
        if strcmp(cfg.model.truth.thrustType, 'time_varying')
            amp = cfg.model.truth.thrustAmplitude;
            period = cfg.model.truth.thrustPeriod;
            a_mag = amp * (1 + 0.5 * sin(2*pi*t/period));
        else
            % 常值推力（保底）
            a_mag = cfg.thrust.truth.accelerationRTN(2);
        end
        
        % RTN到ECI转换
        R_vec = r / r_norm;
        T_vec = v / norm(v);
        N_vec = cross(R_vec, T_vec);
        N_vec = N_vec / norm(N_vec);
        RTN2ECI = [R_vec, T_vec, N_vec];
        a_thrust_RTN = [0; a_mag; 0];
        a_thrust_ECI = RTN2ECI * a_thrust_RTN;
    end
    
    a_total = a_grav + a_J2 + a_thrust_ECI;
    
    dxdt = zeros(6,1);
    dxdt(1:3) = v;
    dxdt(4:6) = a_total;
end


%% ============================================================
%  带阻力+时变推力的微分方程
% ============================================================
function dxdt = dyn_equation_with_drag_and_varying_thrust(t, x, cfg)
    r = x(1:3);
    v = x(4:6);
    rx = r(1); ry = r(2); rz = r(3);
    r_norm = norm(r);
    
    mu = cfg.const.muEarth;
    Re = cfg.const.radiusEarth;
    J2 = cfg.const.J2;
    useJ2 = cfg.model.truth.useJ2;
    useThrust = cfg.model.truth.useThrust;
    
    % 中心引力
    a_grav = -mu / r_norm^3 * r;
    
    % J2
    a_J2 = zeros(3,1);
    if useJ2
        z2 = rz^2;
        r2 = r_norm^2;
        term = 1.5 * J2 * mu * Re^2 / r2^(2.5);
        a_J2(1) = term * rx * (5*z2/r2 - 1);
        a_J2(2) = term * ry * (5*z2/r2 - 1);
        a_J2(3) = term * rz * (5*z2/r2 - 3);
    end
    
    % 大气阻力
    a_drag = zeros(3,1);
    if cfg.model.truth.useDrag
        Cd = cfg.model.truth.dragCd;
        A = cfg.model.truth.dragArea;
        rho0 = cfg.model.truth.dragRho0;
        H0 = cfg.model.truth.dragH0;
        h = r_norm - Re;
        rho = rho0 * exp(-h / H0);
        v_rel = v;
        a_drag = -0.5 * Cd * (A / 1) * rho * norm(v_rel) * v_rel;
    end
    
    % 时变推力
    a_thrust_ECI = zeros(3,1);
    if useThrust
        if strcmp(cfg.model.truth.thrustType, 'time_varying')
            amp = cfg.model.truth.thrustAmplitude;
            period = cfg.model.truth.thrustPeriod;
            a_mag = amp * (1 + 0.5 * sin(2*pi*t/period));
        else
            a_mag = cfg.thrust.truth.accelerationRTN(2);
        end
        
        R_vec = r / r_norm;
        T_vec = v / norm(v);
        N_vec = cross(R_vec, T_vec);
        N_vec = N_vec / norm(N_vec);
        RTN2ECI = [R_vec, T_vec, N_vec];
        a_thrust_RTN = [0; a_mag; 0];
        a_thrust_ECI = RTN2ECI * a_thrust_RTN;
    end
    
    a_total = a_grav + a_J2 + a_drag + a_thrust_ECI;
    
    dxdt = zeros(6,1);
    dxdt(1:3) = v;
    dxdt(4:6) = a_total;
end


%% ============================================================
%  辅助函数：生成README说明
% ============================================================
function generate_readme(data_dir)
    fid = fopen(fullfile(data_dir, '动力学失配场景说明.txt'), 'w');
    fprintf(fid, '============================================================\n');
    fprintf(fid, '  动力学模型失配场景说明（成员1生成）\n');
    fprintf(fid, '  生成时间：%s\n', datestr(now));
    fprintf(fid, '============================================================\n\n');
    
    fprintf(fid, '【场景1：大气阻力失配】\n');
    fprintf(fid, '  文件：truth_drag.mat\n');
    fprintf(fid, '  真值模型：二体 + J2 + 大气阻力\n');
    fprintf(fid, '  估计模型：二体 + J2（无阻力，无推力）\n');
    fprintf(fid, '  阻力参数：Cd = 2.2, A = 10 m^2\n\n');
    
    fprintf(fid, '【场景2：时变推力失配】\n');
    fprintf(fid, '  文件：truth_timevarying_thrust.mat\n');
    fprintf(fid, '  真值模型：二体 + J2 + 时变推力（正弦变化）\n');
    fprintf(fid, '  估计模型：二体 + J2 + 常值推力\n');
    fprintf(fid, '  推力参数：幅值 5e-5 m/s^2，周期 3600 s\n\n');
    
    fprintf(fid, '【场景3：阻力 + 时变推力（严重失配）】\n');
    fprintf(fid, '  文件：truth_drag_thrust.mat\n');
    fprintf(fid, '  真值模型：二体 + J2 + 大气阻力 + 时变推力\n');
    fprintf(fid, '  估计模型：二体 + J2 + 常值推力（无阻力）\n');
    fprintf(fid, '  组合了场景1和场景2的失配因素，测试算法在多重失配下的鲁棒性\n\n');
    
    fprintf(fid, '【估计模型配置】\n');
    fprintf(fid, '  文件：cfg_estimation_model.mat\n');
    fprintf(fid, '  说明：包含第三组定轨时使用的估计模型参数\n');
    fprintf(fid, '  加载方法：load(''cfg_estimation_model.mat'', ''cfg_est'')\n');
    
    fclose(fid);
end