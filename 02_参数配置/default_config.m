function cfg = default_config()
%DEFAULT_CONFIG  非合作航天器连续小推力精密定轨与预报项目统一配置
%
% 用途：
%   1. 统一全组的物理常数、坐标系、单位、仿真场景和算法边界；
%   2. 所有模块只读取 cfg，不在其他函数中重复硬编码参数；
%   3. 同时区分“真值模型”和“估计模型”，便于开展非同源仿真。
%
% 约定：
%   - 长度：m
%   - 速度：m/s
%   - 加速度：m/s^2
%   - 时间：s
%   - 角度：rad
%   - 主状态坐标系：J2000/ECI
%   - 地面站坐标系：ECEF
%   - 测角坐标系：ENU
%
% 调用：
%   cfg = default_config();
%
% 第一阶段基准场景：
%   550 km近圆轨道、53 deg倾角、沿迹连续小推力、
%   武汉单站雷达、3天定轨弧、12 h与24 h预报。
%
% 注意：
%   本文件确定的是项目第一版“统一边界”，后续若修改公共参数，
%   应商讨后统一修改本文件并更新版本号。

%% 0. 配置版本与随机数
cfg.meta.projectName = 'Noncooperative Spacecraft Low-Thrust OD and Prediction';
cfg.meta.version = 'V1.0-Day1';
cfg.meta.owner = 'Project Team';
cfg.meta.created = '2026-07';
cfg.random.seed = 20260701;
rng(cfg.random.seed, 'twister');

%% 1. 单位与坐标系约定
cfg.units.length = 'm';
cfg.units.velocity = 'm/s';
cfg.units.acceleration = 'm/s^2';
cfg.units.time = 's';
cfg.units.angle = 'rad';

cfg.frame.propagation = 'ECI_J2000';
cfg.frame.earthFixed = 'ECEF';
cfg.frame.localTopocentric = 'ENU';
cfg.frame.orbital = 'RTN';

%% 2. 地球与环境常数
% WGS-84 / 常用轨道动力学参数
cfg.const.muEarth = 3.986004418e14;       % m^3/s^2
cfg.const.radiusEarth = 6378137.0;        % m
cfg.const.flatteningEarth = 1/298.257223563;
cfg.const.omegaEarth = 7.2921150e-5;      % rad/s
cfg.const.J2 = 1.08262668e-3;
cfg.const.J3 = -2.5324105e-6;
cfg.const.J4 = -1.6198976e-6;

cfg.const.au = 149597870700.0;            % m
cfg.const.cLight = 299792458.0;           % m/s

%% 3. 时间设置
% 场景历元用于ECEF/ECI转换。第一阶段允许采用简化GMST模型。
cfg.time.epochUTC = datetime(2026,7,1,0,0,0,'TimeZone','UTC');

cfg.sim.truthDuration = 3 * 86400;         % 3天真值/定轨区间
cfg.sim.predictionDuration = 24 * 3600;    % 最长24 h预报
cfg.sim.integrationStep = 10;              % 动力学输出步长
cfg.sim.measurementStep = 10;              % 可见弧内观测采样间隔
cfg.sim.absTol = 1e-10;
cfg.sim.relTol = 1e-10;
cfg.sim.integrator = 'ode113';             % 无工具箱时可改为ode45

cfg.prediction.horizons = [12, 24] * 3600; % 12 h、24 h

%% 4. 基准轨道初值
% 经典轨道根数：
% [a, e, i, RAAN, argPerigee, trueAnomaly]
cfg.orbit.altitude = 550e3;
cfg.orbit.a = cfg.const.radiusEarth + cfg.orbit.altitude;
cfg.orbit.e = 0.001;
cfg.orbit.i = deg2rad(53.0);
cfg.orbit.RAAN = deg2rad(20.0);
cfg.orbit.argPerigee = deg2rad(0.0);
cfg.orbit.trueAnomaly = deg2rad(0.0);

cfg.orbit.elements = [ ...
    cfg.orbit.a;
    cfg.orbit.e;
    cfg.orbit.i;
    cfg.orbit.RAAN;
    cfg.orbit.argPerigee;
    cfg.orbit.trueAnomaly];

[cfg.orbit.r0ECI, cfg.orbit.v0ECI] = localKep2Cart( ...
    cfg.orbit.elements, cfg.const.muEarth);

cfg.orbit.x0Truth = [cfg.orbit.r0ECI; cfg.orbit.v0ECI];

% 初始轨道周期，仅用于结果检查
cfg.orbit.meanMotion = sqrt(cfg.const.muEarth / cfg.orbit.a^3);
cfg.orbit.period = 2*pi / cfg.orbit.meanMotion;

%% 5. 真值动力学模型
% 一级验证可令truth与estimate尽量一致；
% 二级验证再打开J3/J4、SRP、第三体等形成模型失配。
cfg.model.truth.useCentralGravity = true;
cfg.model.truth.useJ2 = true;
cfg.model.truth.useJ3 = false;
cfg.model.truth.useJ4 = false;
cfg.model.truth.useDrag = true;
cfg.model.truth.useSRP = false;
cfg.model.truth.useThirdBody = false;
cfg.model.truth.useThrust = true;

%% 6. 估计动力学模型
cfg.model.estimate.useCentralGravity = true;
cfg.model.estimate.useJ2 = true;
cfg.model.estimate.useJ3 = false;
cfg.model.estimate.useJ4 = false;
cfg.model.estimate.useDrag = true;
cfg.model.estimate.useSRP = false;
cfg.model.estimate.useThirdBody = false;
cfg.model.estimate.useThrust = true;

%% 7. 航天器物理参数
% 取适合低轨小卫星/星链类目标的量级，第一阶段只用于仿真验证。
cfg.spacecraft.mass = 260.0;               % kg
cfg.spacecraft.areaDrag = 4.0;             % m^2
cfg.spacecraft.Cd = 2.2;
cfg.spacecraft.areaSRP = 4.0;              % m^2
cfg.spacecraft.Cr = 1.3;

%% 8. 简化大气阻力参数
% 第一版采用指数大气模型；后续可替换为NRLMSISE-00。
cfg.atmosphere.model = 'exponential';
cfg.atmosphere.referenceAltitude = 550e3;  % m
cfg.atmosphere.referenceDensity = 2.0e-13; % kg/m^3，基准量级
cfg.atmosphere.scaleHeight = 60e3;          % m
cfg.atmosphere.rotateWithEarth = true;

%% 9. 连续小推力真值模型
% 第一阶段固定为沿迹正向连续小推力，代表缓慢升轨。
% 5.0e-5 m/s^2 对应260 kg目标约13 mN推力。
cfg.thrust.truth.enabled = true;
cfg.thrust.truth.frame = 'RTN';
cfg.thrust.truth.mode = 'piecewise_constant';
cfg.thrust.truth.direction = 'along_track';
cfg.thrust.truth.accelerationRTN = [0; 5.0e-5; 0]; % [R;T;N] m/s^2
cfg.thrust.truth.startTime = 0;
cfg.thrust.truth.stopTime = cfg.sim.truthDuration + cfg.sim.predictionDuration;

% 为后续非同源仿真预留缓慢变化与周期项
cfg.thrust.truth.linearRateRTN = [0; 0; 0];     % m/s^3
cfg.thrust.truth.periodicAmplitudeRTN = [0; 0; 0];
cfg.thrust.truth.periodicPeriod = cfg.orbit.period;

%% 10. 估计端推力模型
% 第一版先估计沿迹常值加速度。
cfg.thrust.estimate.enabled = true;
cfg.thrust.estimate.frame = 'RTN';
cfg.thrust.estimate.model = 'constant_along_track';
cfg.thrust.estimate.estimateRadial = false;
cfg.thrust.estimate.estimateAlongTrack = true;
cfg.thrust.estimate.estimateNormal = false;

% 初始猜测故意与真值存在20%偏差
cfg.thrust.estimate.initialGuessRTN = [0; 4.0e-5; 0];
cfg.thrust.estimate.lowerBoundRTN = [-1e-4; -2e-4; -1e-4];
cfg.thrust.estimate.upperBoundRTN = [ 1e-4;  2e-4;  1e-4];

%% 11. 雷达站参数：武汉站
cfg.station.name = 'Wuhan';
cfg.station.latitude = deg2rad(30.5);
cfg.station.longitude = deg2rad(114.3);
cfg.station.altitude = 39.3;               % m
cfg.station.minElevation = deg2rad(10.0);

cfg.station.ecef = localGeodetic2ECEF( ...
    cfg.station.latitude, ...
    cfg.station.longitude, ...
    cfg.station.altitude, ...
    cfg.const.radiusEarth, ...
    cfg.const.flatteningEarth);

%% 12. 雷达观测类型与噪声
cfg.measurement.types = {'range','azimuth','elevation'};
cfg.measurement.useRange = true;
cfg.measurement.useAzimuth = true;
cfg.measurement.useElevation = true;

% 一级验证噪声
cfg.measurement.sigmaRange = 50.0;          % m
cfg.measurement.sigmaAzimuth = deg2rad(0.02);
cfg.measurement.sigmaElevation = deg2rad(0.02);

% 系统偏差，一级验证先置零
cfg.measurement.biasRange = 0.0;
cfg.measurement.biasAzimuth = 0.0;
cfg.measurement.biasElevation = 0.0;

% 二级验证接口
cfg.measurement.outlier.enabled = false;
cfg.measurement.outlier.probability = 0.01;
cfg.measurement.outlier.scale = 8.0;

cfg.measurement.dropout.enabled = false;
cfg.measurement.dropout.probability = 0.05;

% 观测权阵中的标准差向量
cfg.measurement.sigmaVector = [ ...
    cfg.measurement.sigmaRange;
    cfg.measurement.sigmaAzimuth;
    cfg.measurement.sigmaElevation];

%% 13. 初始状态猜测
% 定轨初值不直接使用真值，模拟初轨误差。
cfg.estimation.initialPositionErrorECI = [ ...
     1000;
     -800;
      600];                                % m

cfg.estimation.initialVelocityErrorECI = [ ...
     0.8;
    -0.5;
     0.3];                                 % m/s

cfg.estimation.x0Guess = cfg.orbit.x0Truth + [ ...
    cfg.estimation.initialPositionErrorECI;
    cfg.estimation.initialVelocityErrorECI];

%% 14. 批处理最小二乘设置
cfg.estimation.method = 'batch_weighted_least_squares';
cfg.estimation.maxIterations = 15;
cfg.estimation.parameterTolerance = 1e-4;
cfg.estimation.costTolerance = 1e-3;
cfg.estimation.useLevenbergMarquardt = true;
cfg.estimation.initialDamping = 1e-3;
cfg.estimation.dampingIncrease = 10;
cfg.estimation.dampingDecrease = 0.3;
cfg.estimation.maxConditionNumber = 1e12;
cfg.estimation.useSVD = true;
cfg.estimation.rejectOutliers = false;
cfg.estimation.outlierThresholdSigma = 4.0;

% LM 阻尼内循环参数
cfg.estimation.maxDampingTrials = 6;
cfg.estimation.maxNormalizedStep = 2.0;

% 数值差分步长
cfg.estimation.fd.position = 1.0;           % m
cfg.estimation.fd.velocity = 1e-3;          % m/s
cfg.estimation.fd.thrust = 1e-7;            % m/s^2
cfg.estimation.fd.scheme = 'central';

% 参数尺度，用于归一化
cfg.estimation.scale.position = 1e6;
cfg.estimation.scale.velocity = 1e3;
cfg.estimation.scale.thrust = 1e-5;

%% 15. 分段估计设置
cfg.segmentation.enabled = true;

% 先比较三类方案
cfg.segmentation.methods = { ...
    'none', ...
    'fixed_time', ...
    'fixed_arcs'};

cfg.segmentation.fixedTimeWindows = [ ...
    0.5, 1.0, 1.5] * 86400;                % s

cfg.segmentation.fixedArcCounts = [1, 2, 3];
cfg.segmentation.minimumMeasurementsPerSegment = 30;
cfg.segmentation.minimumSegmentDuration = 20 * 60;

%% 16. 轨道预报设置
cfg.prediction.useEstimatedFinalState = true;
cfg.prediction.futureThrustModel = 'hold_last_segment';
cfg.prediction.compareNoThrustBaseline = true;
cfg.prediction.compareTruthThrust = true;
cfg.prediction.outputStep = 60;

%% 17. 评价指标
cfg.evaluation.computeECIError = true;
cfg.evaluation.computeRTNError = true;
cfg.evaluation.computePositionRMS = true;
cfg.evaluation.computeVelocityRMS = true;
cfg.evaluation.computeMaximumError = true;
cfg.evaluation.computeThrustError = true;
cfg.evaluation.computeResidualStatistics = true;
cfg.evaluation.computeConvergenceStatistics = true;

cfg.evaluation.primaryMetric = 'position_RMS_24h';
cfg.evaluation.successThrustRelativeError = 0.05;
cfg.evaluation.successImprovementRatio = 0.20;

%% 18. 蒙特卡洛与鲁棒性实验
cfg.monteCarlo.enabled = false;
cfg.monteCarlo.numberOfRuns = 50;

cfg.robustness.enableModelMismatch = false;
cfg.robustness.enableMeasurementBias = false;
cfg.robustness.enableDropout = false;
cfg.robustness.enableOutliers = false;
cfg.robustness.enableThrustDirectionError = false;

%% 19. 绘图与数据保存
cfg.output.saveResults = true;
cfg.output.saveFigures = true;
cfg.output.figureFormat = 'png';
cfg.output.figureDPI = 200;
cfg.output.outputRoot = fullfile(pwd, 'results');
cfg.output.dataDirectory = fullfile(cfg.output.outputRoot, 'data');
cfg.output.figureDirectory = fullfile(cfg.output.outputRoot, 'figures');
cfg.output.logDirectory = fullfile(cfg.output.outputRoot, 'logs');

cfg.plot.showOrbit3D = true;
cfg.plot.showVisibility = true;
cfg.plot.showResiduals = true;
cfg.plot.showRTNErrors = true;
cfg.plot.showThrustEstimate = true;
cfg.plot.showPredictionComparison = true;

%% 20. 项目公共接口边界
cfg.interface.truthStateShape = 'N_by_6';
cfg.interface.timeShape = 'N_by_1';
cfg.interface.observationShape = 'N_by_3';
cfg.interface.stateOrder = {'x','y','z','vx','vy','vz'};
cfg.interface.measurementOrder = {'range','azimuth','elevation'};
cfg.interface.rtnOrder = {'radial','along_track','normal'};

%% 21. 自动检查
localValidateConfig(cfg);

end

%% ========================================================================
function [rECI, vECI] = localKep2Cart(elements, mu)
%LOCAlKEP2CART 经典轨道根数转ECI位置速度
a = elements(1);
e = elements(2);
inc = elements(3);
raan = elements(4);
argp = elements(5);
nu = elements(6);

p = a * (1 - e^2);

rPQW = (p / (1 + e*cos(nu))) * [cos(nu); sin(nu); 0];
vPQW = sqrt(mu/p) * [-sin(nu); e + cos(nu); 0];

R3_W = [ cos(raan), -sin(raan), 0;
         sin(raan),  cos(raan), 0;
         0,          0,         1];

R1_i = [1, 0,         0;
        0, cos(inc), -sin(inc);
        0, sin(inc),  cos(inc)];

R3_w = [ cos(argp), -sin(argp), 0;
         sin(argp),  cos(argp), 0;
         0,          0,         1];

Q_PQW_to_ECI = R3_W * R1_i * R3_w;

rECI = Q_PQW_to_ECI * rPQW;
vECI = Q_PQW_to_ECI * vPQW;
end

%% ========================================================================
function rECEF = localGeodetic2ECEF(lat, lon, h, a, f)
%LOCALGEODETIC2ECEF 大地经纬高转ECEF
e2 = f * (2 - f);
N = a / sqrt(1 - e2 * sin(lat)^2);

rECEF = [ ...
    (N + h) * cos(lat) * cos(lon);
    (N + h) * cos(lat) * sin(lon);
    (N * (1 - e2) + h) * sin(lat)];
end

%% ========================================================================
function localValidateConfig(cfg)
%LOCALVALIDATECONFIG 检查关键配置是否自洽

assert(cfg.const.muEarth > 0, 'muEarth必须为正。');
assert(cfg.const.radiusEarth > 6.0e6, '地球半径设置异常。');
assert(cfg.orbit.a > cfg.const.radiusEarth, '轨道半长轴必须大于地球半径。');
assert(cfg.orbit.e >= 0 && cfg.orbit.e < 1, '当前项目只支持椭圆轨道。');
assert(cfg.sim.integrationStep > 0, '积分步长必须为正。');
assert(cfg.sim.measurementStep >= cfg.sim.integrationStep, ...
    '建议观测步长不小于动力学输出步长。');
assert(cfg.station.minElevation >= 0 && cfg.station.minElevation < pi/2, ...
    '截止高度角应位于[0, pi/2)范围。');
assert(all(cfg.measurement.sigmaVector > 0), '观测标准差必须为正。');
assert(cfg.estimation.maxIterations >= 1, '最大迭代次数必须大于0。');
assert(cfg.thrust.truth.stopTime > cfg.thrust.truth.startTime, ...
    '推力结束时刻必须晚于开始时刻。');
assert(numel(cfg.orbit.x0Truth) == 6, '初始轨道状态必须为6维。');
assert(numel(cfg.estimation.x0Guess) == 6, '定轨初值必须为6维。');

fprintf('[default_config] 配置加载成功：%s\n', cfg.meta.version);
fprintf('[default_config] 基准轨道高度：%.1f km\n', cfg.orbit.altitude/1e3);
fprintf('[default_config] 轨道周期：%.2f min\n', cfg.orbit.period/60);
fprintf('[default_config] 真值沿迹加速度：%.3e m/s^2\n', ...
    cfg.thrust.truth.accelerationRTN(2));
fprintf('[default_config] 地面站：%s，最低高度角：%.1f deg\n', ...
    cfg.station.name, rad2deg(cfg.station.minElevation));
end
