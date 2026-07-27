# 数据与实验预处理结果 — 数据说明文档

## 概览

本目录存放星链卫星轨道数据集从原始下载到机器学习就绪的全流程数据，按 **Medallion 分层架构** 组织：

```
raw → bronze → silver → gold
```

| 层级 | 含义 | 操作 |
|------|------|------|
| **raw** | 原始数据，不可修改 | Space-Track API 下载的原始 JSON/CSV |
| **bronze** | 清洗后数据 | 字段统一、时间转换、标准化格式 |
| **silver** | 衍生数据 | SGP4 传播的统一时间网格轨道状态 |
| **gold** | 模型就绪数据 | Informer 时间窗口、GNN 图边表、RTN 残差、数据集划分 |

**数据集版本：** `starlink_v1`  
**规模：** 60 颗卫星 × 6 个月（2026-01-01 ~ 2026-07-01）  
**采样间隔：** 10 分钟  

---

## 目录结构

```
数据与实验预处理结果/
├── raw/                          # 原始层
│   └── space_track/
│       ├── catalog/              # 卫星目录
│       ├── gp_history/           # 历史轨道根数
│       ├── current_gp/           # 当前最新 GP（预留）
│       ├── manifests/            # 下载元数据
│       └── pilot_backup/         # 试运行备份
├── bronze/                       # 清洗层
│   └── omm_history/              # 标准化 OMM
├── silver/                       # 衍生层
│   ├── orbit_states/             # SGP4 统一时间网格
│   ├── physics_prediction/       # 物理传播器预测（预留）
│   └── graph_snapshots/          # 图快照（预留，边表在 gold）
├── gold/                         # 模型就绪层
│   ├── informer_samples/         # Informer 时间序列样本
│   ├── gnn_samples/              # GNN 图边表
│   ├── residual_acceleration/    # RTN 残差加速度
│   └── dataset_split/            # 训练/验证/测试划分
├── database/                     # 数据库
│   └── starlink.duckdb           # DuckDB 主库
├── reports/                      # 报告
│   ├── data_quality/             # 质量报告
│   ├── download_logs/            # 下载日志（预留）
│   └── statistics/               # 统计报告
└── README.md                     # 本文档
```

---

## 1. raw/ — 原始数据层

### 1.1 catalog/ — 卫星目录

| 文件 | 说明 |
|------|------|
| `starlink_gp_latest.csv` | 从 Space-Track API 获取的当前在轨星链卫星目录（10,836 颗），包含轨道根数、TLE 等 |
| `starlink_gp_latest.json` | 同上，JSON 格式 |
| `starlink_selected.csv` | 经筛选后的 60 颗目标卫星，同一轨道壳层（倾角 52°~54°，高度 500~570km） |

**字段说明（CSV）：**

```
NORAD_CAT_ID       - NORAD 卫星编号（唯一标识）
OBJECT_NAME        - 卫星名称，如 STARLINK-1184
OBJECT_ID          - 国际编号，如 2019-074B
EPOCH              - 轨道根数历元（UTC）
INCLINATION        - 轨道倾角 (°)
RA_OF_ASC_NODE     - 升交点赤经 (°)
ECCENTRICITY       - 偏心率
ARG_OF_PERICENTER  - 近地点幅角 (°)
MEAN_ANOMALY       - 平近点角 (°)
MEAN_MOTION        - 平均运动 (rev/day)
SEMIMAJOR_AXIS     - 半长轴 (km)
PERIOD             - 轨道周期 (min)
APOGEE             - 远地点 (km)
PERIGEE            - 近地点 (km)
BSTAR              - 大气阻力系数
TLE_LINE1          - TLE 第一行
TLE_LINE2          - TLE 第二行
DECAY_DATE         - 衰减日期（空=在轨）
```

**使用方式：**

```python
import pandas as pd

# 读取所有在轨星链卫星
catalog = pd.read_csv("12_数据与实验结果/raw/space_track/catalog/starlink_gp_latest.csv")

# 读取目标卫星
selected = pd.read_csv("12_数据与实验结果/raw/space_track/catalog/starlink_selected.csv")
norad_ids = selected["NORAD_CAT_ID"].tolist()
```

### 1.2 gp_history/ — 历史轨道根数

| 文件 | 说明 |
|------|------|
| `gp_history_2026-01-01_2026-07-01_batch_00X.json` | 3 批次下载的历史 OMM 数据，每批 20 颗卫星 |

每条记录为完整的 OMM (Orbit Mean-Elements Message) 格式 JSON，包含 `EPOCH`、`MEAN_MOTION`、`ECCENTRICITY`、`INCLINATION`、`TLE_LINE1`、`TLE_LINE2` 等字段。

**使用方式：**

```python
import json

with open("12_数据与实验结果/raw/space_track/gp_history/gp_history_2026-01-01_2026-07-01_batch_001.json") as f:
    records = json.load(f)
# records 是 list[dict]，每个 dict 是一条 OMM 记录
```

### 1.3 manifests/ — 下载元数据

每个下载批次对应一个 Manifest，记录 SHA256 哈希值保证数据完整性。

```json
{
  "download_id": "uuid",
  "download_time_utc": "ISO 8601",
  "source": "Space-Track",
  "source_class": "gp_history",
  "start_date": "2026-01-01",
  "end_date": "2026-07-01",
  "norad_ids": [45754, 45668, ...],
  "record_count": 10800,
  "sha256": "abc123..."
}
```

---

## 2. bronze/ — 清洗层

### 2.1 omm_history/ — 标准化 OMM

| 文件 | 说明 |
|------|------|
| `omm_history_bronze.parquet` | 所有原始 JSON 合并、字段统一、时间转换后的 Parquet 文件 |

**字段：**

```
norad_id              - NORAD 编号
object_name           - 卫星名称
object_id             - 国际编号
epoch_utc             - 历元 (UTC, 带时区)
creation_date_utc     - 创建时间 (UTC)
mean_motion_rev_day   - 平均运动 (rev/day)
eccentricity          - 偏心率
inclination_deg       - 倾角 (°)
raan_deg              - 升交点赤经 (°)
arg_perigee_deg       - 近地点幅角 (°)
mean_anomaly_deg      - 平近点角 (°)
bstar                 - 大气阻力系数
mean_motion_dot       - 平均运动一阶导数
mean_motion_ddot      - 平均运动二阶导数
semimajor_axis_km     - 半长轴 (km)
period_min            - 周期 (min)
apogee_km             - 远地点 (km)
perigee_km            - 近地点 (km)
element_set_no        - 根数批次号
revolution_at_epoch   - 历元时累计圈数
tle_line1             - TLE 第一行
tle_line2             - TLE 第二行
source_file           - 来源文件路径
quality_flag          - 质量标记
```

**使用方式：**

```python
import pandas as pd

omm = pd.read_parquet("12_数据与实验结果/bronze/omm_history/omm_history_bronze.parquet")
# 按卫星分组
for norad_id, group in omm.groupby("norad_id"):
    print(f"NORAD {norad_id}: {len(group)} 条记录")
```

---

## 3. database/ — DuckDB 数据库

| 文件 | 说明 |
|------|------|
| `starlink.duckdb` | DuckDB 主数据库（7.8MB） |

### 表结构

#### omm_history — OMM 历史记录表（27,848 行）

主表，字段与 bronze Parquet 一致。唯一键组合：`(norad_id, epoch_utc, element_set_no)`。

```sql
-- 统计概览
SELECT
    COUNT(*) AS total_records,
    COUNT(DISTINCT norad_id) AS satellite_count,
    MIN(epoch_utc) AS first_epoch,
    MAX(epoch_utc) AS last_epoch
FROM omm_history;
```

#### satellite_catalog — 卫星目录表（60 行）

```sql
SELECT * FROM satellite_catalog;
-- norad_id, object_name, object_id, inclination_deg, mean_altitude_km,
-- raan_deg, record_count, start_epoch, end_epoch, history_days
```

#### satellite_topology — 拓扑表（60 行）

包含壳层和轨道面标记结果。

```sql
SELECT shell_id, plane_id, COUNT(*) as sat_count
FROM satellite_topology
GROUP BY shell_id, plane_id
ORDER BY shell_id, plane_id;
```

#### download_manifest — 下载清单表

```sql
SELECT * FROM download_manifest;
```

### 使用方式：

```python
import duckdb

conn = duckdb.connect("12_数据与实验结果/database/starlink.duckdb")

# 查询某颗卫星的所有 OMM
df = conn.execute("""
    SELECT epoch_utc, semimajor_axis_km, inclination_deg, tle_line1, tle_line2
    FROM omm_history
    WHERE norad_id = 45098 AND quality_flag = 'good'
    ORDER BY epoch_utc
""").fetchdf()

conn.close()
```

---

## 4. silver/ — 衍生层

### 4.1 orbit_states/ — SGP4 统一时间网格

| 格式 | 说明 |
|------|------|
| 180 个 Parquet 文件 | 按天分区：`orbit_states_YYYYMMDD.parquet` |
| 全局统一时刻 | 25,756 个同步时间点，每点约 60 颗卫星 |
| 总状态数 | 1,545,360 |

**字段：**

```
norad_id                - NORAD 编号
timestamp_utc           - 统一时刻 (UTC)
x_teme_m, y_teme_m, z_teme_m      - TEME 位置 (m)
vx_teme_mps, vy_teme_mps, vz_teme_mps - TEME 速度 (m/s)
x_eci_m, y_eci_m, z_eci_m         - ECI 位置 (m, TEME 近似)
vx_eci_mps, vy_eci_mps, vz_eci_mps - ECI 速度 (m/s)
semimajor_axis_m        - 半长轴 (m)
eccentricity            - 偏心率
inclination_rad         - 倾角 (rad)
raan_rad                - 升交点赤经 (rad)
arg_perigee_rad         - 近地点幅角 (rad)
mean_anomaly_rad        - 平近点角 (rad)
source_epoch_utc        - 源 TLE 历元 (UTC)
source_creation_date_utc - 源 TLE 创建时间 (UTC)
tle_age_seconds         - TLE 年龄 (s, 传播量)
bstar                   - 大气阻力系数
shell_id                - 壳层 ID
plane_id                - 轨道面 ID
quality_flag            - 质量标记
source_file             - 来源文件
```

**关键设计：**

- **全局统一网格：** 所有卫星在同一组 UTC 时间点上传播，确保 GNN 图快照同步
- **防未来泄漏：** 每个时刻只能使用 `EPOCH <= t` 的轨道根数
- **SGP4 传播器：** 使用 `sgp4` Python 库，统一的 WGS84 地球模型

**使用方式：**

```python
import pandas as pd

# 读取某一天的状态
states = pd.read_parquet("12_数据与实验结果/silver/orbit_states/orbit_states_20260115.parquet")

# 加载全部（慎用，1.5M 行）
import glob
all_states = pd.concat([
    pd.read_parquet(f) for f in sorted(glob.glob("12_数据与实验结果/silver/orbit_states/*.parquet"))
], ignore_index=True)

# 获取某个时刻所有卫星的状态（用于 GNN）
snapshot = all_states[all_states["timestamp_utc"] == "2026-03-15T12:00:00+00:00"]
```

---

## 5. gold/ — 模型就绪层

### 5.1 informer_samples/ — Informer 时间窗口样本

| 文件 | 大小 | 说明 |
|------|------|------|
| `informer_samples_preview.json` | 931MB | 前 5,000 个样本的 JSON（可读） |
| `informer_samples.parquet` | 1,032MB | 全部 21,120 个样本的 Parquet |
| `informer_samples.json` | 188MB | v1 旧版，可删除 |

**样本结构（JSON）：**

```json
{
  "norad_id": 45098,
  "start_time_utc": "2026-01-02T12:00:00+00:00",
  "end_input_time_utc": "2026-01-04T12:00:00+00:00",
  "input_window_h": 48,
  "input_steps": 288,
  "input_features": [[x,y,z,vx,vy,vz,a,e,i,raan,ap,ma,bstar,age], ...],  // 288×14
  "label_12h": [[...], ...],   // 72×14
  "label_24h": [[...], ...]    // 144×14
}
```

**参数：**

| 参数 | 值 |
|------|-----|
| 输入窗口 | 48 小时 |
| 输入步长 | 10 分钟 |
| 输入序列长度 | 288 步 |
| 预测目标 | 12 小时 / 24 小时 |
| 特征维度 | 14（位置3 + 速度3 + 轨道根数6 + BSTAR + TLE年龄） |
| 滑动步长 | 输入窗口的 1/4 |

**使用方式：**

```python
import pandas as pd
import json

# Parquet 格式（推荐，高效）
samples = pd.read_parquet(
    "12_数据与实验结果/gold/informer_samples/informer_samples.parquet"
)
# 每行的 input_features 和 label_features 是 JSON 字符串，需解析
import json
row = samples.iloc[0]
inputs = json.loads(row["input_features"])   # list[list[float]], 288×14
labels = json.loads(row["label_features"])   # list[list[float]], 72×14 或 144×14

# JSON 格式（前5000条，可直接读）
with open("12_数据与实验结果/gold/informer_samples/informer_samples_preview.json") as f:
    preview_samples = json.load(f)
```

### 5.2 gnn_samples/ — GNN 图边表

| 文件 | 大小 | 说明 |
|------|------|------|
| `gnn_edges.parquet` | 142MB | 25,756 个同步图快照的全部边（7,367,506 条） |

**边表字段：**

```
graph_time_utc          - 图快照时间
source_norad_id         - 源卫星 NORAD 编号
target_norad_id         - 目标卫星 NORAD 编号
relative_distance_m     - 相对距离 (m)
relative_speed_mps      - 相对速度 (m/s)
delta_raan_rad          - RAAN 差 (rad)
delta_mean_anomaly_rad  - 平近点角差 (rad)
same_plane              - 是否同轨道面
same_shell              - 是否同壳层
edge_type               - 边类型: same_plane / same_shell / knn
```

**边构建策略：**

```
同一轨道壳层 + 同一轨道面相邻卫星
  +
空间距离最近的 4 颗卫星 (KNN)
```

**使用方式：**

```python
import pandas as pd

edges = pd.read_parquet("12_数据与实验结果/gold/gnn_samples/gnn_edges.parquet")

# 获取某个时刻的图
graph = edges[edges["graph_time_utc"] == "2026-03-15T12:00:00+00:00"]
print(f"节点数: {graph['source_norad_id'].nunique() + graph['target_norad_id'].nunique()}")
print(f"边数: {len(graph)}")

# 构建 PyTorch Geometric 图
import torch
from torch_geometric.data import Data

# edge_index: [2, num_edges]
unique_nodes = sorted(set(graph["source_norad_id"].unique()) | set(graph["target_norad_id"].unique()))
node_to_idx = {n: i for i, n in enumerate(unique_nodes)}

edge_index = torch.tensor([
    [node_to_idx[s] for s in graph["source_norad_id"]],
    [node_to_idx[t] for t in graph["target_norad_id"]],
], dtype=torch.long)

edge_attr = torch.tensor(graph[[
    "relative_distance_m", "relative_speed_mps",
    "delta_raan_rad", "delta_mean_anomaly_rad"
]].values, dtype=torch.float32)

# 节点特征需从 silver/orbit_states 中加载对应时刻的状态
```

**获取节点特征 + 边的完整示例：**

```python
import pandas as pd
import torch

# 加载边
edges = pd.read_parquet("12_数据与实验结果/gold/gnn_samples/gnn_edges.parquet")

# 加载对应时刻的状态
target_time = "2026-03-15T12:00:00+00:00"
date_str = target_time[:10].replace("-", "")
states = pd.read_parquet(f"12_数据与实验结果/silver/orbit_states/orbit_states_{date_str}.parquet")
snapshot = states[states["timestamp_utc"] == target_time]

# 节点特征：位置 + 速度 + 轨道根数
node_feature_cols = [
    "x_eci_m", "y_eci_m", "z_eci_m",
    "vx_eci_mps", "vy_eci_mps", "vz_eci_mps",
    "semimajor_axis_m", "eccentricity", "inclination_rad",
    "raan_rad", "arg_perigee_rad", "mean_anomaly_rad",
    "bstar", "tle_age_seconds",
]
node_features = torch.tensor(snapshot[node_feature_cols].values, dtype=torch.float32)

# 该时刻的边
graph_edges = edges[edges["graph_time_utc"] == target_time]
```

### 5.3 residual_acceleration/ — RTN 残差加速度

| 文件 | 大小 | 说明 |
|------|------|------|
| `rtn_residuals.parquet` | 40MB | 1,545,240 条 RTN 残差 |

**字段：**

```
norad_id            - NORAD 编号
timestamp_utc       - 时刻 (UTC)
residual_R_mps2     - 径向残差加速度 (m/s²)
residual_T_mps2     - 切向/沿轨残差加速度 (m/s²)
residual_N_mps2     - 法向/跨轨残差加速度 (m/s²)
```

**计算方式：**

```
RTN 残差 = 数值加速度（中心差分）- 中心引力加速度（二体问题）

R (Radial)         = 地心到卫星方向
T (Transverse)     = 速度方向在轨道面投影
N (Normal)         = 轨道面法向
```

**使用方式：**

```python
import pandas as pd

residuals = pd.read_parquet("12_数据与实验结果/gold/residual_acceleration/rtn_residuals.parquet")

# 合并到状态数据作为训练标签
states = pd.read_parquet("12_数据与实验结果/silver/orbit_states/orbit_states_20260115.parquet")
merged = states.merge(residuals, on=["norad_id", "timestamp_utc"], how="left")
```

### 5.4 dataset_split/ — 数据集划分

| 文件 | 说明 |
|------|------|
| `dataset_split.json` | 训练/验证/测试集卫星分配 |

```json
{
  "creation_time_utc": "2026-07-27T...",
  "split_method": "satellite_based_with_unseen_plane",
  "train_satellites": [45098, 45668, ...],   // 29 颗
  "val_satellites": [45754, 48472, ...],      // 9 颗
  "test_satellites": [51115, 51121, ...],     // 23 颗 (含 4 颗 unseen plane)
  "unseen_plane_id": 0,
  "n_train": 29,
  "n_val": 9,
  "n_test": 23
}
```

**划分策略：**

| 维度 | 方法 |
|------|------|
| 卫星划分 | 60% 训练 / 15% 验证 / 25% 测试 |
| 未见轨道面 | 至少 1 个轨道面的全部卫星不在训练集 |
| 时间划分 | 前 70% 训练 / 中 15% 验证 / 后 15% 测试（通过样本时间戳隐式实现） |

**使用方式：**

```python
import json

with open("12_数据与实验结果/gold/dataset_split/dataset_split.json") as f:
    split = json.load(f)

train_sats = split["train_satellites"]
val_sats = split["val_satellites"]
test_sats = split["test_satellites"]

# 按划分筛选数据
states = pd.read_parquet("...")
train_data = states[states["norad_id"].isin(train_sats)]
val_data = states[states["norad_id"].isin(val_sats)]
test_data = states[states["norad_id"].isin(test_sats)]
```

---

## 6. reports/ — 报告

| 目录 | 说明 |
|------|------|
| `data_quality/` | 质量检查报告（完整性、合法性、时间间隙、轨道跳变） |
| `statistics/` | 数据库构建统计（卫星数、记录数、时间跨度） |

---

## 典型使用流程

### 加载完整数据管道

```python
import pandas as pd
import duckdb
import json
from pathlib import Path

BASE = Path("12_数据与实验结果")

# 1. 从数据库查询卫星目录
conn = duckdb.connect(str(BASE / "database/starlink.duckdb"))
catalog = conn.execute("SELECT * FROM satellite_catalog").fetchdf()
topology = conn.execute("SELECT * FROM satellite_topology").fetchdf()
conn.close()

# 2. 读取数据集划分
with open(BASE / "gold/dataset_split/dataset_split.json") as f:
    split = json.load(f)

# 3. 加载统一状态（按需加载特定日期）
states = pd.concat([
    pd.read_parquet(f)
    for f in BASE.glob("silver/orbit_states/orbit_states_202603*.parquet")
], ignore_index=True)

# 4. 加载 RTN 残差
residuals = pd.read_parquet(BASE / "gold/residual_acceleration/rtn_residuals.parquet")

# 5. 合并 → 训练就绪
data = states.merge(residuals, on=["norad_id", "timestamp_utc"], how="left")
train = data[data["norad_id"].isin(split["train_satellites"])]
val = data[data["norad_id"].isin(split["val_satellites"])]
test = data[data["norad_id"].isin(split["test_satellites"])]
```

### Informer 训练数据加载

```python
import pandas as pd

samples = pd.read_parquet(BASE / "gold/informer_samples/informer_samples.parquet")
# 按划分筛选
train_samples = samples[samples["norad_id"].isin(split["train_satellites"])]
```

### GNN + Informer 联合训练

```python
# 1. 加载边表
edges = pd.read_parquet(BASE / "gold/gnn_samples/gnn_edges.parquet")

# 2. 对每个训练时间点：
#    - 从 silver/orbit_states 加载节点特征
#    - 从 gnn_edges 筛选对应时刻的边
#    - 从 rtn_residuals 获取残差标签

# 3. 对每个 Informer 样本：
#    - 输入 288 步状态序列
#    - 预测 72/144 步残差
```

---

## 数据来源与许可

- 数据来源：Space-Track.org（需注册账号）
- 数据产品：GP / GP_History（公开通用摄动轨道产品）
- 使用限制：见 Space-Track 用户协议
- 坐标系：TEME（True Equator Mean Equinox），ECI 为 TEME 近似

---

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| `starlink_v1` | 2026-07-27 | 60 颗卫星，6 个月历史，10min 网格 |

---

## 相关脚本

所有数据采集与预处理脚本位于 `03_星链数据处理/`：

```
select_satellites.py         # 卫星筛选
download_catalog.py          # GP 目录下载
download_gp_history.py       # 历史数据下载
build_database.py            # DuckDB 建库
quality_check.py             # 数据质量检查
shell_plane_labeling.py      # 壳层与轨道面标记
generate_unified_grid.py     # SGP4 统一时间网格
prepare_ml_samples.py        # Informer/GNN/RTN 样本生成
```

配置文件：`02_参数配置/dataset_config.yaml`
