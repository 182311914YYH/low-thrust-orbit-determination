function dxdt = dyn_equation(t, x, cfg)
%DYN_EQUATION 轨道动力学微分方程右端
% 输入：
%   t       当前时刻s
%   x       6×1 ECI状态 [rx,ry,rz,vx,vy,vz]'
%   cfg     全局配置结构体
% 输出：
%   dxdt    6×1 状态导数

%% 提取状态
r = x(1:3);
v = x(4:6);
rx = r(1); ry = r(2); rz = r(3);
r_norm = norm(r);

%% 读取全局常数（全部来自cfg，无硬编码）
mu = cfg.const.muEarth;
Re = cfg.const.radiusEarth;
J2 = cfg.const.J2;
useJ2 = cfg.model.truth.useJ2;
useThrust = cfg.model.truth.useThrust;
a_thrust_RTN = cfg.thrust.truth.accelerationRTN; % [R;T;N]

%% 1. 中心引力加速度
a_grav = -mu / r_norm^3 * r;

%% 2. J2摄动加速度
a_J2 = zeros(3,1);
if useJ2
    z2 = rz^2;
    r2 = r_norm^2;
    term = 1.5 * J2 * mu * Re^2 / r2^(2.5);
    a_J2(1) = term * rx * (5*z2/r2 - 1);
    a_J2(2) = term * ry * (5*z2/r2 - 1);
    a_J2(3) = term * rz * (5*z2/r2 - 3);
end

%% 3. RTN沿迹推力转换到ECI加速度
a_thrust_ECI = zeros(3,1);
if useThrust
    % RTN坐标系单位矢量
    R_vec = r / r_norm;
    T_vec = v / norm(v);
    N_vec = cross(R_vec, T_vec);
    N_vec = N_vec / norm(N_vec);

    % RTN到ECI转换矩阵
    RTN2ECI = [R_vec, T_vec, N_vec];
    a_thrust_ECI = RTN2ECI * a_thrust_RTN;
end

%% 总加速度
a_total = a_grav + a_J2 + a_thrust_ECI;

%% 组装导数 [vx,vy,vz,ax,ay,az]
dxdt = zeros(6,1);
dxdt(1:3) = v;
dxdt(4:6) = a_total;

end