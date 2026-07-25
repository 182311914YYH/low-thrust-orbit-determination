function [obs_out, vis_flag] = enu_obs_model(cfg, t, state)
%ENU_OBS_MODEL 武汉单站雷达观测仿真主函数
% 输入：
%   cfg     全局配置结构体
%   t       N×1 时间序列 (s)
%   state   N×6 ECI轨道状态 [x,y,z,vx,vy,vz] (m, m/s)
% 输出：
%   obs_out N×3 观测 [range(m), azimuth(rad), elevation(rad)]
%   vis_flag N×1 可见标记 1=可见，0=不可见

%% 输入维度检查
if size(state,2) ~= 6
    error('state 必须是 N×6 矩阵 [x,y,z,vx,vy,vz]');
end
if length(t) ~= size(state,1)
    error('时间序列长度必须与状态行数一致');
end

N = size(state,1);
obs_out = zeros(N,3);
vis_flag = zeros(N,1);

%% 读取配置全部参数（无硬编码）
sta_ecef = cfg.station.ecef;          % 武汉站 ECEF 坐标 [x,y,z] (m)
min_elev = cfg.station.minElevation;  % 最低俯仰角 (rad)
sig_r   = cfg.measurement.sigmaRange;
sig_az  = cfg.measurement.sigmaAzimuth;
sig_el  = cfg.measurement.sigmaElevation;
bias_r  = cfg.measurement.biasRange;
bias_az = cfg.measurement.biasAzimuth;
bias_el = cfg.measurement.biasElevation;
omega_e = cfg.const.omegaEarth;

% 确保经纬度为弧度（从 default_config 读取时已为弧度，此处保留）
lat0 = cfg.station.latitude;
lon0 = cfg.station.longitude;

%% 预计算三角函数（提高效率，固定值）
sin_lat = sin(lat0);
cos_lat = cos(lat0);
sin_lon = sin(lon0);
cos_lon = cos(lon0);

%% ENU 转换矩阵（ECEF → ENU），方向为 [E; N; U]
R_enu = [-sin_lon,  cos_lon, 0;
         -sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat;
          cos_lat*cos_lon,  cos_lat*sin_lon, sin_lat];

%% 逐时刻计算观测
for i = 1:N
    % 当前 ECI 位置（6维状态的前3个）
    r_eci = state(i, 1:3)';
    ti = t(i);
    
    % ECI → ECEF 旋转矩阵（考虑地球自转，角度为 -omega_e * ti）
    theta = -omega_e * ti;
    cos_t = cos(theta);
    sin_t = sin(theta);
    R3 = [cos_t, sin_t, 0;
         -sin_t, cos_t, 0;
          0,     0,     1];
    r_sat_ecef = R3 * r_eci;
    
    % 计算站心矢量 ECEF
    rho_ecef = r_sat_ecef - sta_ecef;
    
    % ECEF → ENU
    rho_enu = R_enu * rho_ecef;
    east  = rho_enu(1);
    north = rho_enu(2);
    up    = rho_enu(3);
    
    % 理论观测值
    range_true = norm(rho_enu);
    az_true    = atan2(east, north);                     % 方位角 [-π, π]
    el_true    = atan2(up, sqrt(east^2 + north^2) + eps); % 俯仰角，+eps防除零
    
    % 可见性判断（俯仰角 >= 最低仰角）
    if el_true >= min_elev
        vis_flag(i) = 1;
    end
    
    % 叠加系统偏差
    range_obs = range_true + bias_r;
    az_obs    = az_true + bias_az;
    el_obs    = el_true + bias_el;
    
    % 叠加高斯白噪声
    range_obs = range_obs + sig_r * randn();
    az_obs    = az_obs + sig_az * randn();
    el_obs    = el_obs + sig_el * randn();
    
    % 存储观测值
    obs_out(i, :) = [range_obs, az_obs, el_obs];
end

end