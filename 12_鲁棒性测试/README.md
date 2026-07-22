# 12_鲁棒性测试

## 概述

本模块利用第二组（`11_测试程序`）生成的不同测试数据，分析精密定轨、小推力估计和轨道预报算法在异常条件下的稳定性。

## 前置条件

运行本模块前，必须先执行第二组程序生成测试数据：

```matlab
% 在项目根目录下执行
cd 11_测试程序
gen_radar_anomaly;   % 生成雷达异常场景数据
gen_dyn_mismatch;    % 生成动力学失配真值数据
```

生成的数据文件位于 `09_数据与实验结果/robustness/`。

## 任务清单

| 编号 | 任务 | 负责人 |
|------|------|--------|
| (1) | 测试不同观测噪声下的定轨收敛情况 | 成员3 |
| (2) | 测试系统偏差、缺测和异常值对估计结果的影响 | 成员3 |
| (3) | 分析模型失配条件下的小推力估计误差 | 成员3 |
| (4) | 比较不同场景下12h、24h轨道预报误差 | 成员4 |
| - | 运行各类鲁棒性定轨实验，记录收敛情况、迭代次数、残差和小推力估计误差 | 成员3 |
| - | 计算各场景的ECI、RTN、RMS、最大误差和预报末端误差 | 成员4 |
| - | 将各异常场景结果与基准场景进行统一对比 | 全组 |

## 文件结构

```
12_鲁棒性测试/
├── robustness_main.m              # 主入口脚本
├── robustness_config.m            # 鲁棒性测试场景配置（对齐第二组数据）
├── load_test_data.m               # 测试数据加载器（从第二组数据文件加载）
├── run_robustness_experiments.m   # 成员3：鲁棒性定轨实验主程序
├── compute_scenario_metrics.m     # 成员4：各场景误差计算（ECI/RTN/RMS/最大/末端）
├── compare_scenarios.m            # 统一对比分析
├── README.md                      # 本文档
└── results/                       # 结果输出目录
    ├── robustness_experiment_results.mat
    ├── robustness_metrics.mat
    ├── robustness_comparison.mat
    ├── comparison_table.csv
    └── figures/
```

## 运行方式

### 完整运行

```matlab
cd low-thrust-orbit-determination-main
addpath(genpath('12_鲁棒性测试'));
robustness_main;
```

### 分步运行

```matlab
cfg = default_config();
scenarios = robustness_config(cfg);

% 成员3 - 运行定轨实验
results = run_robustness_experiments(scenarios, cfg);

% 成员4 - 计算误差指标
metrics = compute_scenario_metrics(results);

% 统一对比
comparison = compare_scenarios(results, metrics);
```

## 测试场景定义

共定义 12 个测试场景，与第二组数据完全对齐：

### 0. 基准场景
| 名称 | 数据文件 | 说明 |
|------|----------|------|
| baseline | `radar_obs_full.mat` | 基准场景（标准噪声50m/0.02°，无异常） |

### 1. 不同观测噪声水平 — 任务(1)
| 名称 | 数据文件 | 距离噪声 | 角噪声 |
|------|----------|----------|--------|
| noise_low | `radar_obs_noise1.mat` | 10 m | 0.01° |
| noise_mid | `radar_obs_noise2.mat` | 100 m | 0.05° |
| noise_high | `radar_obs_noise3.mat` | 1000 m | 0.1° |

### 2. 系统偏差 — 任务(2)
| 名称 | 数据文件 | 距离偏差 | 方位偏差 | 俯仰偏差 |
|------|----------|----------|----------|----------|
| bias | `radar_obs_bias.mat` | +50 m | +0.5° | +0.05° |

### 3. 随机缺测 — 任务(2)
| 名称 | 数据文件 | 缺测率 |
|------|----------|--------|
| dropout_10 | `radar_obs_dropout10.mat` | 10% |
| dropout_30 | `radar_obs_dropout30.mat` | 30% |
| dropout_50 | `radar_obs_dropout50.mat` | 50% |

### 4. 异常值 — 任务(2)
| 名称 | 数据文件 | 异常值比例 | 幅度 |
|------|----------|------------|------|
| outlier | `radar_obs_outlier.mat` | 5%可见观测 | +10km测距 |

### 5. 动力学模型失配 — 任务(3)
| 名称 | 数据文件 | 失配类型 | 真值推力 |
|------|----------|----------|----------|
| mismatch_drag | `truth_drag.mat` | 大气阻力未建模 | 0 m/s² |
| mismatch_thrust | `truth_timevarying_thrust.mat` | 时变推力（常推力估计） | 5e-5 m/s² |
| mismatch_combined | `truth_drag_thrust.mat` | 阻力+时变推力 | 5e-5 m/s² |

> **注**：动力学失配场景仅有真值轨道，观测数据由 `enu_obs_model` 以基准噪声配置生成。真值推力为时变推力的平均值。

## 第二组数据格式

### 雷达异常场景（gen_radar_anomaly.m 生成）

每个 `.mat` 文件包含：

| 变量名 | 维度 | 单位 | 说明 |
|--------|------|------|------|
| `t_orbit` | N×1 | s | 时间序列 |
| `state_orbit` | N×6 | m, m/s | 真值状态 [x,y,z,vx,vy,vz] |
| `obs` / `obs_drop` / `obs_out` | N×3 | m, rad, rad | 观测值 [距离, 方位角, 俯仰角] |
| `vis` / `vis_drop` / `vis_raw` | N×1 | - | 可见性标记 (1=可见, 0=不可见) |
| `cfg_noise` / `cfg_bias` / ... | struct | - | 场景配置 |

> **注意**：不同场景类型的观测/可见性变量名不同。

### 动力学失配场景（gen_dyn_mismatch.m 生成）

| 变量名 | 维度 | 单位 | 说明 |
|--------|------|------|------|
| `t1` / `t2` / `t3` | N×1 | s | 时间序列 |
| `state1` / `state2` / `state3` | N×6 | m, m/s | 真值状态 |
| `cfg1` / `cfg2` / `cfg3` | struct | - | 真值模型配置 |

> **注意**：仅提供真值轨道，观测数据由本模块自动生成。

## 输出说明

### 结果文件

| 文件 | 内容 |
|------|------|
| `robustness_experiment_results.mat` | 全部场景的定轨结果（收敛、迭代、残差、推力估计） |
| `robustness_metrics.mat` | 全部场景的误差指标（ECI/RTN/RMS/最大/末端） |
| `robustness_comparison.mat` | 统一对比分析结果 |
| `comparison_table.csv` | 对比表格（CSV格式） |
| `figures/` | 对比图形（PNG格式） |

### 对比图形

| 图号 | 内容 |
|------|------|
| fig01 | 12h/24h 位置RMS对比柱状图 |
| fig02 | 推力估计相对误差对比 |
| fig03 | 迭代次数对比 |
| fig04 | 24h预报误差时间序列 |
| fig05 | 24h RTN径向误差对比 |
| fig06 | 12h vs 24h 末端误差散点图 |

## 成员分工

### 成员3：鲁棒性定轨实验

**负责文件**：`run_robustness_experiments.m`

**记录指标**：
- 收敛情况（是/否）
- 迭代次数
- 残差RMS和最大值（距离、方位角、俯仰角）
- 小推力估计值和估计误差（绝对误差、相对误差）

### 成员4：误差计算

**负责文件**：`compute_scenario_metrics.m`

**计算指标**：
- ECI坐标系位置误差（RMS、最大值、均值）
- ECI坐标系速度误差（RMS、最大值）
- RTN坐标系各分量误差（RMS、最大值）
- 12h和24h预报末端误差
- 误差时间序列（供绘图）

### 统一对比

**负责文件**：`compare_scenarios.m`

**对比内容**：
- 全部场景 vs 基准场景的对比表格
- 按场景类别分组统计
- 相对基准的变化倍数
- 可视化图形
