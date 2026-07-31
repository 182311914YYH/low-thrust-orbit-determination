# -*- coding: utf-8 -*-
"""
非合作航天器连续小推力精密定轨与智能预报
PINN+LSTM 模型评估 — 运行说明

快速开始
--------
  conda activate aipi
  cd 模型与结果/
  python run.py

产出文件 (评估图表/)
--------
  fig1_sma.png          半长轴预报精度 (真实 QK-1, 12h)
  fig2_detection.png    机动检测性能 (仿真验证)
  fig3_pod.png          雷达精密定轨与机动融合

文件夹说明
----------
  AI融合模型/     PINN+LSTM 模型定义与训练
  物理力模型/     高精度轨道力模型 (NRLMSISE-00+EGM20+SRP+III)
  坐标时间系统/   坐标系变换与时间系统
  系统配置/       统一配置入口
  数据文件/       QK-1 OEM 数据 + STK 验证数据集
  评估图表/       三图输出 (PNG + PDF)

验证结果
--------
  仿真 (degree-2 自洽):   T 方向 R2 = 0.995
  STK HPOP (dv_RTN):     AUC = 0.998, T 方向 R2 = 0.884
  真实 QK-1 (524 天):     模型正确输出 Δa ≈ 0 (无机动)

注意
----
  模型推理特征 (= degree-2 物理残差) 与轨道传播物理模型
  (= NRLMSISE-00+EGM20+SRP+III) 使用不同精度等级。特征提取用
  degree-2 以与训练保持自洽, 轨道传播用高精度以评估真实物理
  基线。这是有意设计, 不是不一致。

环境
----
  Python 3.12, PyTorch 2.x, CUDA 12.6
  numpy, scipy, matplotlib, scikit-learn, pandas, astropy, nrlmsise00
"""
print("运行: python run.py")
