# -*- coding: utf-8 -*-
"""
非合作航天器连续小推力精密定轨与智能预报
PINN+LSTM 模型 — 运行说明

使用方式:
  1. 确保在项目根目录下运行 (e:/项目实习/完整项目（claude）/)
  2. 激活 aipi 虚拟环境
  3. python draw_figures.py                 # 生成三张评估图
  4. python 核心模型代码/draw_figures.py     # 同等效果

三张评估图:
  图1  fig1_sma.png/pdf        半长轴预报精度 (真实 QK-1 OEM 数据)
  图2  fig2_detection.png/pdf  机动检测性能 (物理残差特征)
  图3  fig3_pod.png/pdf        雷达精密定轨与机动融合 (仿真武汉站雷达)
"""
print("请从项目根目录运行: python draw_figures.py")
