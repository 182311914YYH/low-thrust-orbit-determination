# -*- coding: utf-8 -*-
"""非合作航天器连续小推力精密定轨与智能预报
PINN+LSTM 模型评估 — 一键运行

用法: python run.py
"""
import sys, os
# 确保当前文件夹在搜索路径最前面
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from draw_figures import figure1, figure2, figure3

print("=" * 55)
print("  PINN+LSTM 模型评估")
print("  物理模型: NRLMSISE-00 + EGM96x20 + SRP + 三体引力")
print("  数据: QK-1 OEM 精密星历 (8天)")
print("=" * 55)

figure1()
figure2()
figure3()

print("\n三图已生成: 评估结果图表/")
print("  fig1_sma.png          半长轴预报精度")
print("  fig2_detection.png    机动检测性能")
print("  fig3_pod.png          雷达精密定轨与机动融合")
