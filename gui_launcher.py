# -*- coding: utf-8 -*-
"""
非合作航天器连续小推力精密定轨与智能预报 - 桌面软件 v7.0

多界面架构：
  1. 软件主界面（登录 + 软件说明）
  2. 操作主界面（数据载入 + 任务场景选择 + 参数选择 + 定轨与预报仿真）
  3. 仿真结果分析与存储（多标签图表 + 导出功能）
"""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import sys
import os
import json
import threading
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 导入统一图表样式
from chart_style import apply_style
apply_style()

# 导入核心仿真与绘图逻辑
# 优先尝试真实流水线（需要 torch），失败则回退到演示流水线
PIPELINE_MODE = "none"
_pipeline = None
try:
    import generate_software_figures as _pipeline
    PIPELINE_READY = True
    PIPELINE_MODE = "full"
except Exception as e:
    print(f"⚠ 完整仿真引擎未就绪 ({e})，尝试演示模式...")
    try:
        import demo_pipeline as _pipeline
        PIPELINE_READY = True
        PIPELINE_MODE = "demo"
        print("✓ 演示模式已就绪 — 使用模拟数据生成图表")
    except Exception as e2:
        PIPELINE_READY = False
        print(f"⚠ 演示模式也不可用: {e2}")

def _get_fn(name):
    """从流水线模块安全获取函数，不存在则返回 None"""
    if _pipeline is None:
        return None
    return getattr(_pipeline, name, None)

run_pipeline = _get_fn('run_pipeline')

# ============================================================
# 颜色主题
# ============================================================
COLOR_BG          = '#f0f2f5'
COLOR_CARD        = '#ffffff'
COLOR_PRIMARY      = '#2b6cb0'
COLOR_PRIMARY_DARK = '#1a4f82'
COLOR_PRIMARY_LIGHT= '#ebf4ff'
COLOR_ACCENT       = '#3182ce'
COLOR_TEXT         = '#1a202c'
COLOR_TEXT_GRAY    = '#718096'
COLOR_TEXT_LIGHT   = '#a0aec0'
COLOR_SUCCESS      = '#38a169'
COLOR_ERROR        = '#e53e3e'
COLOR_BORDER       = '#e2e8f0'
COLOR_TOPBAR       = '#1a365d'
COLOR_STATUSBAR    = '#f7fafc'
COLOR_SIDEBAR_BG   = '#ffffff'

# 字体常量
F_LOGIN_TITLE  = ('Microsoft YaHei', 26, 'bold')
F_LOGIN_SUB    = ('Microsoft YaHei', 16, 'bold')
F_LOGIN_DESC   = ('Microsoft YaHei', 11)
F_TOPBAR       = ('Microsoft YaHei', 14, 'bold')
F_HEADING      = ('Microsoft YaHei', 12, 'bold')
F_BODY         = ('Microsoft YaHei', 10)
F_BODY_BOLD    = ('Microsoft YaHei', 10, 'bold')
F_SMALL        = ('Microsoft YaHei', 9)
F_BUTTON       = ('Microsoft YaHei', 11, 'bold')
F_BUTTON_LG    = ('Microsoft YaHei', 13, 'bold')
F_STATUS       = ('Microsoft YaHei', 9)

SOFTWARE_NAME    = "非合作航天器连续小推力精密定轨与智能预报系统"
SOFTWARE_VERSION = "v7.0"
SOFTWARE_DESC = [
    "基于 PINN+LSTM 物理信息神经网络的智能定轨与预报平台",
    "支持 QK-1 OEM 精密星历数据处理与轨道预报",
    "集成高精度轨道力学模型（NRLMSISE-00 + EGM96 + SRP + 三体引力）",
    "AI 融合机动检测、推力估计与雷达精密定轨",
]

# 场景图表配置：{场景名: [(显示名称, 内部key, 图表类型, 函数名), ...]}
# 图表类型: "3d" = 3D轴, "2d" = 2D轴, "multi" = 多子图(用fig), "table" = 表格
SCENARIO_CONFIG = {
    "精密定轨与预报": [
        ("轨道三维可视化",    "s1_3d",         "3d",    "figure_3d_orbit"),
        ("预报误差分析",      "s1_error",      "2d",    "figure_error_curve"),
        ("轨道根数变化",      "s1_elements",   "multi", "figure_orbital_elements"),
        ("定轨收敛曲线",      "s1_convergence","2d",    "figure_od_convergence"),
    ],
    "机动检测与估计": [
        ("机动检测ROC曲线",   "s2_roc",        "2d",    "figure_roc_curve"),
        ("推力强度热图",      "s2_thrust",     "2d",    "figure_thrust_heatmap"),
        ("推力估计精度",      "s2_scatter",    "2d",    "figure_thrust_scatter"),
        ("机动检测时间线",    "s2_timeline",   "2d",    "figure_detection_timeline"),
    ],
    "雷达定轨融合": [
        ("位置误差对比",      "s3_pos",        "2d",    "figure_position_error"),
        ("速度误差CDF",       "s3_vel",        "2d",    "figure_velocity_cdf"),
        ("雷达观测残差",      "s3_resid",      "multi", "figure_radar_residuals"),
        ("融合精度对比表",    "s3_table",      "table", "figure_fusion_table"),
    ],
    "综合评估": [
        ("轨道三维可视化",    "s4_3d",         "3d",    "figure_3d_orbit"),
        ("预报误差分析",      "s4_error",      "2d",    "figure_error_curve"),
        ("推力强度热图",      "s4_thrust",     "2d",    "figure_thrust_heatmap"),
        ("统计评估报告",      "s4_stats",      "table", "figure_statistics_report"),
    ],
}


# ============================================================
# 辅助函数
# ============================================================
def styled_button(parent, text, command, bg=COLOR_PRIMARY, fg='white',
                  font=F_BUTTON, padx=20, pady=10, state=tk.NORMAL):
    """创建统一样式的扁平按钮"""
    return tk.Button(
        parent, text=text, command=command,
        bg=bg, fg=fg, font=font,
        relief=tk.FLAT, padx=padx, pady=pady,
        activebackground=COLOR_PRIMARY_DARK if bg == COLOR_PRIMARY else bg,
        activeforeground=fg,
        cursor='hand2', bd=0, state=state,
    )


def setup_ttk_style():
    """配置 ttk 控件样式"""
    style = ttk.Style()
    style.theme_use('clam')

    # Notebook 标签页
    style.configure('TNotebook', background=COLOR_BG, borderwidth=0)
    style.configure('TNotebook.Tab',
                    background=COLOR_CARD, foreground=COLOR_TEXT_GRAY,
                    padding=[18, 8], font=F_BODY,
                    borderwidth=0)
    style.map('TNotebook.Tab',
              background=[('selected', COLOR_PRIMARY)],
              foreground=[('selected', 'white')])

    # Combobox
    style.configure('TCombobox',
                    fieldbackground='white', background='white',
                    foreground=COLOR_TEXT, borderwidth=1,
                    arrowcolor=COLOR_PRIMARY, font=F_SMALL)
    style.map('TCombobox',
              fieldbackground=[('readonly', 'white')],
              selectbackground=[('readonly', COLOR_PRIMARY_LIGHT)],
              selectforeground=[('readonly', COLOR_TEXT)])

    # Progressbar
    style.configure('TProgressbar',
                    background=COLOR_PRIMARY, troughcolor=COLOR_BORDER,
                    borderwidth=0, thickness=8)

    # Scrollbar
    style.configure('TScrollbar', background=COLOR_BORDER,
                    troughcolor=COLOR_CARD, arrowcolor=COLOR_TEXT_GRAY,
                    borderwidth=0)

    # Separator
    style.configure('TSeparator', background=COLOR_BORDER)


# ============================================================
# 用户管理 — 简单 JSON 存储
# ============================================================
USER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")

def _load_users():
    """从 JSON 文件加载已注册用户，默认包含 admin"""
    default_users = {"admin": "123456"}
    if os.path.exists(USER_FILE):
        try:
            with open(USER_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return default_users

def _save_users(users):
    """保存用户到 JSON 文件"""
    try:
        with open(USER_FILE, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ============================================================
# 1. 软件主界面 — 登录
# ============================================================
class LoginScreen(tk.Frame):
    """软件主界面：展示软件名称，提供登录与注册入口"""

    def __init__(self, parent, on_login):
        super().__init__(parent)
        self.on_login = on_login
        self.configure(bg=COLOR_BG)
        self._build_ui()

    def _build_ui(self):
        # ---- 居中白色卡片 ----
        card = tk.Frame(self, bg=COLOR_CARD, bd=0, relief=tk.FLAT)
        card.place(relx=0.5, rely=0.5, anchor='center', width=520, height=520)

        # 卡片阴影效果（用边框模拟）
        shadow = tk.Frame(self, bg='#d1d9e6', bd=0)
        shadow.place(relx=0.5, rely=0.5, anchor='center', width=524, height=524)
        card.lift(shadow)

        # ---- Logo ----
        tk.Label(card, text="🛰️", font=('Microsoft YaHei', 44),
                 bg=COLOR_CARD, fg=COLOR_PRIMARY).pack(pady=(35, 4))

        # ---- 软件名称 ----
        tk.Label(card, text="非合作航天器", font=F_LOGIN_TITLE,
                 bg=COLOR_CARD, fg=COLOR_TEXT).pack(pady=(4, 0))
        tk.Label(card, text="连续小推力精密定轨与智能预报系统",
                 font=F_LOGIN_SUB, bg=COLOR_CARD, fg=COLOR_PRIMARY).pack(pady=(2, 12))

        # 分隔线
        ttk.Separator(card, orient='horizontal').pack(fill=tk.X, padx=80, pady=4)

        # ---- 登录表单 ----
        form = tk.Frame(card, bg=COLOR_CARD)
        form.pack(pady=(12, 5))

        tk.Label(form, text="用户名", font=F_BODY,
                 bg=COLOR_CARD, fg=COLOR_TEXT_GRAY).grid(
                     row=0, column=0, sticky='w', pady=(0, 3))
        self.username_var = tk.StringVar(value="admin")
        entry_user = tk.Entry(form, textvariable=self.username_var,
                              font=F_BODY, width=30, bd=1, relief=tk.SOLID,
                              highlightthickness=1,
                              highlightcolor=COLOR_PRIMARY,
                              highlightbackground=COLOR_BORDER)
        entry_user.grid(row=1, column=0, ipady=7, pady=(0, 10))

        tk.Label(form, text="密码", font=F_BODY,
                 bg=COLOR_CARD, fg=COLOR_TEXT_GRAY).grid(
                     row=2, column=0, sticky='w', pady=(0, 3))
        self.password_var = tk.StringVar(value="123456")
        entry_pwd = tk.Entry(form, textvariable=self.password_var,
                             font=F_BODY, width=30, bd=1, relief=tk.SOLID,
                             show='●', highlightthickness=1,
                             highlightcolor=COLOR_PRIMARY,
                             highlightbackground=COLOR_BORDER)
        entry_pwd.grid(row=3, column=0, ipady=7, pady=(0, 12))

        login_btn = styled_button(form, "进 入 系 统", self._do_login,
                                  padx=55, pady=9, font=F_BUTTON_LG)
        login_btn.grid(row=4, column=0, pady=(4, 6))

        # ---- 注册入口 ----
        reg_frame = tk.Frame(card, bg=COLOR_CARD)
        reg_frame.pack(pady=(0, 14))
        tk.Label(reg_frame, text="还没有账号？", font=F_SMALL,
                 bg=COLOR_CARD, fg=COLOR_TEXT_GRAY).pack(side=tk.LEFT)
        reg_link = tk.Button(reg_frame, text="注册新用户", command=self._open_register,
                             font=F_SMALL, bg=COLOR_CARD, fg=COLOR_PRIMARY,
                             relief=tk.FLAT, bd=0, cursor='hand2',
                             activebackground=COLOR_CARD,
                             activeforeground=COLOR_PRIMARY_DARK)
        reg_link.pack(side=tk.LEFT)

        # ---- 绑定回车键 ----
        for w in (entry_user, entry_pwd):
            w.bind('<Return>', lambda e: self._do_login())
        self.bind('<Return>', lambda e: self._do_login())

    def _do_login(self):
        username = self.username_var.get().strip()
        password = self.password_var.get().strip()
        if not username or not password:
            messagebox.showwarning("提示", "请输入用户名和密码！")
            return
        users = _load_users()
        if username not in users:
            messagebox.showerror("登录失败", f"用户「{username}」不存在，请先注册。")
            return
        if users[username] != password:
            messagebox.showerror("登录失败", "密码错误，请重试。")
            return
        self.on_login(username)

    def _open_register(self):
        """打开注册对话框"""
        RegisterDialog(self, self._on_register_done)

    def _on_register_done(self, username, password):
        """注册成功后自动填充登录表单"""
        self.username_var.set(username)
        self.password_var.set(password)
        messagebox.showinfo("注册成功", f"用户「{username}」注册成功，请点击登录。")


# ============================================================
# 注册对话框
# ============================================================
class RegisterDialog(tk.Toplevel):
    """用户注册对话框：录入用户名和密码"""

    def __init__(self, parent, on_success):
        super().__init__(parent)
        self.on_success = on_success
        self.title("用户注册")
        self.geometry("420x480")
        self.resizable(False, False)
        self.configure(bg=COLOR_BG)
        self.transient(parent)
        self.grab_set()

        # 居中显示
        self.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() - 420) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - 480) // 2
        self.geometry(f"+{max(x,0)}+{max(y,0)}")

        self._build_ui()

    def _build_ui(self):
        card = tk.Frame(self, bg=COLOR_CARD, bd=0, relief=tk.FLAT)
        card.place(relx=0.5, rely=0.5, anchor='center', width=360, height=420)

        tk.Label(card, text="📝 用户注册", font=F_LOGIN_SUB,
                 bg=COLOR_CARD, fg=COLOR_PRIMARY).pack(pady=(24, 4))
        ttk.Separator(card, orient='horizontal').pack(fill=tk.X, padx=50, pady=6)

        form = tk.Frame(card, bg=COLOR_CARD)
        form.pack(pady=(10, 8))

        tk.Label(form, text="用户名", font=F_BODY,
                 bg=COLOR_CARD, fg=COLOR_TEXT_GRAY).grid(row=0, column=0, sticky='w', pady=(0, 3))
        self.reg_user_var = tk.StringVar()
        entry_user = tk.Entry(form, textvariable=self.reg_user_var,
                              font=F_BODY, width=28, bd=1, relief=tk.SOLID,
                              highlightthickness=1,
                              highlightcolor=COLOR_PRIMARY,
                              highlightbackground=COLOR_BORDER)
        entry_user.grid(row=1, column=0, ipady=7, pady=(0, 10))

        tk.Label(form, text="密码", font=F_BODY,
                 bg=COLOR_CARD, fg=COLOR_TEXT_GRAY).grid(row=2, column=0, sticky='w', pady=(0, 3))
        self.reg_pwd_var = tk.StringVar()
        entry_pwd = tk.Entry(form, textvariable=self.reg_pwd_var,
                             font=F_BODY, width=28, bd=1, relief=tk.SOLID,
                             show='●', highlightthickness=1,
                             highlightcolor=COLOR_PRIMARY,
                             highlightbackground=COLOR_BORDER)
        entry_pwd.grid(row=3, column=0, ipady=7, pady=(0, 10))

        tk.Label(form, text="确认密码", font=F_BODY,
                 bg=COLOR_CARD, fg=COLOR_TEXT_GRAY).grid(row=4, column=0, sticky='w', pady=(0, 3))
        self.reg_pwd2_var = tk.StringVar()
        entry_pwd2 = tk.Entry(form, textvariable=self.reg_pwd2_var,
                              font=F_BODY, width=28, bd=1, relief=tk.SOLID,
                              show='●', highlightthickness=1,
                              highlightcolor=COLOR_PRIMARY,
                              highlightbackground=COLOR_BORDER)
        entry_pwd2.grid(row=5, column=0, ipady=7, pady=(0, 14))

        btn_frame = tk.Frame(card, bg=COLOR_CARD)
        btn_frame.pack(pady=(4, 20))
        styled_button(btn_frame, "注 册", self._do_register,
                      padx=30, pady=8, font=F_BUTTON).pack(side=tk.LEFT, padx=(0, 10))
        tk.Button(btn_frame, text="取 消", command=self.destroy,
                  font=F_BUTTON, bg=COLOR_BORDER, fg=COLOR_TEXT,
                  relief=tk.FLAT, padx=30, pady=8, cursor='hand2',
                  activebackground='#cbd5e0').pack(side=tk.LEFT)

        for w in (entry_user, entry_pwd, entry_pwd2):
            w.bind('<Return>', lambda e: self._do_register())

    def _do_register(self):
        username = self.reg_user_var.get().strip()
        pwd = self.reg_pwd_var.get().strip()
        pwd2 = self.reg_pwd2_var.get().strip()

        if not username or not pwd:
            messagebox.showwarning("提示", "用户名和密码不能为空！", parent=self)
            return
        if len(username) < 2:
            messagebox.showwarning("提示", "用户名至少需要2个字符！", parent=self)
            return
        if len(pwd) < 4:
            messagebox.showwarning("提示", "密码至少需要4个字符！", parent=self)
            return
        if pwd != pwd2:
            messagebox.showerror("错误", "两次输入的密码不一致！", parent=self)
            return

        users = _load_users()
        if username in users:
            messagebox.showerror("错误", f"用户「{username}」已存在，请更换用户名！", parent=self)
            return

        users[username] = pwd
        _save_users(users)
        self.on_success(username, pwd)
        self.destroy()


# ============================================================
# 2. 操作主界面 — 数据载入 / 任务场景 / 参数选择 / 仿真
# ============================================================
class MainScreen(tk.Frame):
    """操作主界面：左侧控制面板 + 右侧结果分析区"""

    def __init__(self, parent, username, on_logout):
        super().__init__(parent)
        self.username = username
        self.on_logout = on_logout
        self.configure(bg=COLOR_BG)

        # ---- 状态变量 ----
        self.data_folder = None
        self.scenario_var = tk.StringVar(value="综合评估")
        self.gravity_var  = tk.StringVar(value="20")
        self.srp_var       = tk.BooleanVar(value=True)
        self.third_body_var = tk.BooleanVar(value=True)
        self.nrlmsise_var   = tk.BooleanVar(value=True)
        self.prediction_var = tk.StringVar(value="12h")
        self.od_arc_var     = tk.StringVar(value="2天")
        self.status_var     = tk.StringVar(
            value=f"就绪 — {'演示模式' if PIPELINE_MODE == 'demo' else '完整模式'} | 请加载数据并开始仿真")
        self.progress_text_var = tk.StringVar(value="")
        self.data_info_var  = tk.StringVar(value="未加载数据")
        self.is_running     = False
        self.has_results    = False

        # 图表对象
        self.figures = {}
        self.canvases = {}
        self.axes = {}
        self.current_tab_config = []  # 当前场景的图表配置列表

        self._build_ui()

    # ============================================================
    # 界面构建
    # ============================================================

    def _build_ui(self):
        self._build_topbar()
        content = tk.Frame(self, bg=COLOR_BG)
        content.pack(fill=tk.BOTH, expand=True)
        self._build_sidebar(content)
        self._build_results_area(content)
        self._build_statusbar()

    def _build_topbar(self):
        topbar = tk.Frame(self, bg=COLOR_TOPBAR, height=54)
        topbar.pack(fill=tk.X)
        topbar.pack_propagate(False)

        tk.Label(topbar, text="🛰️  非合作航天器定轨与智能预报系统",
                 font=F_TOPBAR, bg=COLOR_TOPBAR, fg='white').pack(
                     side=tk.LEFT, padx=20)

        right = tk.Frame(topbar, bg=COLOR_TOPBAR)
        right.pack(side=tk.RIGHT, padx=20)

        tk.Label(right, text=f"👤 {self.username}",
                 font=F_BODY, bg=COLOR_TOPBAR, fg='#cbd5e0').pack(
                     side=tk.LEFT, padx=(0, 15))

        tk.Button(right, text="退出登录", command=self._do_logout,
                  font=F_SMALL, bg=COLOR_TOPBAR, fg='#cbd5e0',
                  relief=tk.FLAT, bd=0, cursor='hand2',
                  activebackground=COLOR_TOPBAR,
                  activeforeground='white').pack(side=tk.LEFT)

    def _build_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg=COLOR_SIDEBAR_BG, width=340)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 1))
        sidebar.pack_propagate(False)

        # 可滚动内容
        canvas = tk.Canvas(sidebar, bg=COLOR_SIDEBAR_BG,
                           highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(sidebar, orient='vertical',
                                  command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=COLOR_SIDEBAR_BG)

        scroll_frame.bind('<Configure>',
                          lambda e: canvas.configure(
                              scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=scroll_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 鼠标滚轮
        def _on_wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        canvas.bind('<Enter>',
                     lambda e: canvas.bind_all('<MouseWheel>', _on_wheel))
        canvas.bind('<Leave>',
                     lambda e: canvas.unbind_all('<MouseWheel>'))

        self._build_data_section(scroll_frame)
        self._build_scenario_section(scroll_frame)
        self._build_parameter_section(scroll_frame)
        self._build_control_section(scroll_frame)

    def _build_data_section(self, parent):
        tk.Label(parent, text="📊 数据载入", font=F_HEADING,
                 bg=COLOR_SIDEBAR_BG, fg=COLOR_TEXT).pack(
                     anchor='w', padx=20, pady=(20, 8))

        path_frame = tk.Frame(parent, bg=COLOR_SIDEBAR_BG)
        path_frame.pack(fill=tk.X, padx=20, pady=(0, 5))

        self.path_var = tk.StringVar()
        entry = tk.Entry(path_frame, textvariable=self.path_var, font=F_SMALL,
                         bd=1, relief=tk.SOLID, highlightthickness=1,
                         highlightcolor=COLOR_PRIMARY,
                         highlightbackground=COLOR_BORDER)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6, padx=(0, 5))

        tk.Button(path_frame, text="浏览", command=self._browse_folder,
                  font=F_SMALL, bg=COLOR_PRIMARY, fg='white',
                  relief=tk.FLAT, padx=12, pady=5, cursor='hand2',
                  activebackground=COLOR_PRIMARY_DARK).pack(side=tk.RIGHT)

        tk.Label(parent, textvariable=self.data_info_var, font=F_SMALL,
                 bg=COLOR_SIDEBAR_BG, fg=COLOR_TEXT_GRAY, wraplength=295,
                 justify=tk.LEFT).pack(anchor='w', padx=20, pady=(0, 8))

        ttk.Separator(parent, orient='horizontal').pack(
            fill=tk.X, padx=20, pady=5)

    def _build_scenario_section(self, parent):
        tk.Label(parent, text="🎯 任务场景选择", font=F_HEADING,
                 bg=COLOR_SIDEBAR_BG, fg=COLOR_TEXT).pack(
                     anchor='w', padx=20, pady=(8, 8))

        scenarios = [
            ("精密定轨与预报", "QK-1 OEM 数据精密定轨与轨道预报"),
            ("机动检测与估计", "检测航天器机动并估计推力参数"),
            ("雷达定轨融合",   "雷达观测与 AI 机动信息融合定轨"),
            ("综合评估",       "全流程仿真：定轨+预报+检测+统计"),
        ]

        for name, desc in scenarios:
            tk.Radiobutton(parent, text=name, value=name,
                           variable=self.scenario_var, font=F_BODY,
                           bg=COLOR_SIDEBAR_BG, fg=COLOR_TEXT,
                           selectcolor=COLOR_SIDEBAR_BG,
                           activebackground=COLOR_SIDEBAR_BG,
                           activeforeground=COLOR_TEXT,
                           anchor='w', cursor='hand2',
                           command=self._on_scenario_change).pack(
                               anchor='w', padx=20, pady=(2, 0))
            tk.Label(parent, text=f"    {desc}", font=('Microsoft YaHei', 8),
                     bg=COLOR_SIDEBAR_BG, fg=COLOR_TEXT_GRAY,
                     wraplength=285, justify=tk.LEFT).pack(
                         anchor='w', padx=20)

        ttk.Separator(parent, orient='horizontal').pack(
            fill=tk.X, padx=20, pady=5)

    def _build_parameter_section(self, parent):
        tk.Label(parent, text="⚙️ 参数选择", font=F_HEADING,
                 bg=COLOR_SIDEBAR_BG, fg=COLOR_TEXT).pack(
                     anchor='w', padx=20, pady=(8, 8))

        # 引力场阶数
        row1 = tk.Frame(parent, bg=COLOR_SIDEBAR_BG)
        row1.pack(fill=tk.X, padx=20, pady=(0, 6))
        tk.Label(row1, text="引力场阶数", font=F_BODY,
                 bg=COLOR_SIDEBAR_BG, fg=COLOR_TEXT_GRAY).pack(side=tk.LEFT)
        ttk.Combobox(row1, textvariable=self.gravity_var,
                     values=['2', '10', '20'], width=8,
                     state='readonly', font=F_SMALL).pack(side=tk.RIGHT)

        # 复选框组
        for label, var in [
            ("太阳光压 (SRP)",          self.srp_var),
            ("三体引力 (日月)",          self.third_body_var),
            ("大气阻力 (NRLMSISE-00)",  self.nrlmsise_var),
        ]:
            tk.Checkbutton(parent, text=label, variable=var, font=F_BODY,
                           bg=COLOR_SIDEBAR_BG, fg=COLOR_TEXT,
                           selectcolor=COLOR_PRIMARY_LIGHT,
                           activebackground=COLOR_SIDEBAR_BG,
                           activeforeground=COLOR_TEXT,
                           anchor='w', cursor='hand2').pack(
                               anchor='w', padx=20, pady=(2, 0))

        # 预报时长
        row2 = tk.Frame(parent, bg=COLOR_SIDEBAR_BG)
        row2.pack(fill=tk.X, padx=20, pady=(8, 0))
        tk.Label(row2, text="预报时长", font=F_BODY,
                 bg=COLOR_SIDEBAR_BG, fg=COLOR_TEXT_GRAY).pack(side=tk.LEFT)
        ttk.Combobox(row2, textvariable=self.prediction_var,
                     values=['12h', '24h', '48h'], width=8,
                     state='readonly', font=F_SMALL).pack(side=tk.RIGHT)

        # 定轨弧段
        row3 = tk.Frame(parent, bg=COLOR_SIDEBAR_BG)
        row3.pack(fill=tk.X, padx=20, pady=(6, 0))
        tk.Label(row3, text="定轨弧段", font=F_BODY,
                 bg=COLOR_SIDEBAR_BG, fg=COLOR_TEXT_GRAY).pack(side=tk.LEFT)
        ttk.Combobox(row3, textvariable=self.od_arc_var,
                     values=['1天', '2天', '3天'], width=8,
                     state='readonly', font=F_SMALL).pack(side=tk.RIGHT)

        ttk.Separator(parent, orient='horizontal').pack(
            fill=tk.X, padx=20, pady=10)

    def _build_control_section(self, parent):
        self.run_btn = styled_button(
            parent, "▶  开始仿真", self._start_simulation,
            font=F_BUTTON_LG, padx=30, pady=12)
        self.run_btn.pack(fill=tk.X, padx=20, pady=(5, 8))

        self.progress = ttk.Progressbar(parent, mode='indeterminate', length=300)
        self.progress.pack(fill=tk.X, padx=20, pady=(0, 4))

        tk.Label(parent, textvariable=self.progress_text_var, font=F_SMALL,
                 bg=COLOR_SIDEBAR_BG, fg=COLOR_TEXT_GRAY).pack(pady=(0, 20))

    def _build_results_area(self, parent):
        results = tk.Frame(parent, bg=COLOR_BG)
        results.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ---- 工具栏 ----
        toolbar = tk.Frame(results, bg=COLOR_BG)
        toolbar.pack(fill=tk.X, pady=(0, 8))

        self.results_title_var = tk.StringVar(value="📈 仿真结果分析 — 综合评估")
        tk.Label(toolbar, textvariable=self.results_title_var, font=F_HEADING,
                 bg=COLOR_BG, fg=COLOR_TEXT).pack(side=tk.LEFT)

        export_frame = tk.Frame(toolbar, bg=COLOR_BG)
        export_frame.pack(side=tk.RIGHT)

        self.btn_export_png = tk.Button(
            export_frame, text="导出 PNG", command=self._export_png,
            font=F_SMALL, bg=COLOR_CARD, fg=COLOR_PRIMARY,
            relief=tk.SOLID, bd=1, padx=12, pady=4, cursor='hand2',
            state=tk.DISABLED)
        self.btn_export_png.pack(side=tk.LEFT, padx=(0, 5))

        self.btn_export_pdf = tk.Button(
            export_frame, text="导出 PDF", command=self._export_pdf,
            font=F_SMALL, bg=COLOR_CARD, fg=COLOR_PRIMARY,
            relief=tk.SOLID, bd=1, padx=12, pady=4, cursor='hand2',
            state=tk.DISABLED)
        self.btn_export_pdf.pack(side=tk.LEFT, padx=(0, 5))

        self.btn_export_all = tk.Button(
            export_frame, text="导出全部", command=self._export_all,
            font=F_SMALL, bg=COLOR_PRIMARY, fg='white',
            relief=tk.FLAT, bd=0, padx=12, pady=4, cursor='hand2',
            state=tk.DISABLED)
        self.btn_export_all.pack(side=tk.LEFT)

        # ---- 标签页 ----
        self.notebook = ttk.Notebook(results)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 根据当前场景构建标签页
        self._rebuild_tabs()

    def _build_statusbar(self):
        bar = tk.Frame(self, bg=COLOR_STATUSBAR, height=26)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        bar.pack_propagate(False)

        tk.Label(bar, textvariable=self.status_var, font=F_STATUS,
                 bg=COLOR_STATUSBAR, fg=COLOR_TEXT_GRAY, anchor='w').pack(
                     side=tk.LEFT, padx=15)
        tk.Label(bar, text=SOFTWARE_VERSION, font=F_STATUS,
                 bg=COLOR_STATUSBAR, fg=COLOR_TEXT_LIGHT).pack(
                     side=tk.RIGHT, padx=15)
        mode_text = "演示模式" if PIPELINE_MODE == "demo" else (
            "完整模式" if PIPELINE_MODE == "full" else "引擎未就绪")
        mode_color = COLOR_TEXT_GRAY if PIPELINE_MODE == "demo" else (
            COLOR_SUCCESS if PIPELINE_MODE == "full" else COLOR_ERROR)
        tk.Label(bar, text=f"  |  {mode_text}", font=F_STATUS,
                 bg=COLOR_STATUSBAR, fg=mode_color).pack(
                     side=tk.RIGHT)

    # ============================================================
    # 场景切换 & 标签页重建
    # ============================================================
    def _on_scenario_change(self):
        """场景选择变化时重建结果分析区的标签页"""
        scenario = self.scenario_var.get()
        self.results_title_var.set(f"📈 仿真结果分析 — {scenario}")
        self.has_results = False
        self._set_export_buttons(tk.DISABLED)
        self._rebuild_tabs()
        self.status_var.set(f"已切换至「{scenario}」— 请开始仿真")

    def _rebuild_tabs(self):
        """根据当前场景重建 Notebook 标签页"""
        scenario = self.scenario_var.get()
        config = SCENARIO_CONFIG.get(scenario, SCENARIO_CONFIG["综合评估"])
        self.current_tab_config = config

        # 清空旧的图表对象和标签页
        for key in list(self.figures.keys()):
            fig = self.figures.pop(key)
            plt.close(fig)
        self.canvases.clear()
        self.axes.clear()

        # 移除所有现有标签页
        for tab_id in self.notebook.tabs():
            self.notebook.forget(tab_id)

        # 重新创建标签页
        for display_name, key, chart_type, fn_name in config:
            tab = tk.Frame(self.notebook, bg='white')
            self.notebook.add(tab, text=f"  {display_name}  ")

            fig = plt.Figure(figsize=(8, 6), dpi=100, facecolor='white')
            if chart_type == "3d":
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)
            fig.subplots_adjust(left=0.06, right=0.96, top=0.94, bottom=0.08)

            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.figures[key] = fig
            self.canvases[key] = canvas
            self.axes[key] = ax

            self._show_placeholder(ax, display_name, chart_type)
            canvas.draw()

    # ============================================================
    # 占位图 & 加载状态
    # ============================================================
    @staticmethod
    def _is_3d_axes(ax):
        """判断是否为 3D 坐标轴"""
        return hasattr(ax, 'zaxis')

    def _axes_text(self, ax, x, y, text, **kwargs):
        """在 2D 或 3D 坐标轴上放置 2D 文字（兼容两种轴类型）"""
        if self._is_3d_axes(ax):
            ax.text2D(x, y, text, **kwargs)
        else:
            ax.text(x, y, text, **kwargs)

    def _show_placeholder(self, ax, display_name, chart_type="2d"):
        ax.clear()
        ax.set_axis_off()
        self._axes_text(ax, 0.5, 0.55, display_name,
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=16, color='#cbd5e0', fontweight='bold')
        self._axes_text(ax, 0.5, 0.42, '等待仿真开始...',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=12, color='#e2e8f0')

    def _show_loading(self, ax):
        ax.clear()
        ax.set_axis_off()
        self._axes_text(ax, 0.5, 0.5, '🔄 计算中...',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=14, color='#a0aec0', fontweight='bold')

    # ============================================================
    # 事件处理
    # ============================================================
    def _browse_folder(self):
        default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "数据文件", "QK1原始OEM数据")
        initial_dir = default_dir if os.path.isdir(default_dir) else os.getcwd()
        folder = filedialog.askdirectory(title="选择数据文件夹",
                                         initialdir=initial_dir)
        if folder:
            self.path_var.set(folder)
            self.data_folder = folder
            self._update_data_info(folder)

    def _update_data_info(self, folder):
        try:
            dat_files  = list(Path(folder).rglob("*.dat"))
            csv_files  = list(Path(folder).rglob("*.csv"))
            e_files    = list(Path(folder).rglob("*.e"))
            total = len(dat_files) + len(csv_files) + len(e_files)
            if total > 0:
                self.data_info_var.set(
                    f"✓ 已加载: {total} 个数据文件\n"
                    f"  路径: {folder}")
                self.status_var.set(
                    f"数据已加载 — {total} 个文件，可开始仿真")
            else:
                self.data_info_var.set(
                    f"⚠ 未找到数据文件 (.dat/.csv/.e)\n  路径: {folder}")
        except Exception:
            self.data_info_var.set(f"路径: {folder}")

    def _start_simulation(self):
        if self.is_running:
            return

        folder = self.path_var.get().strip()
        if not folder:
            messagebox.showwarning("提示", "请先选择数据文件夹！")
            return
        if not Path(folder).exists():
            messagebox.showerror("错误", "所选路径不存在！")
            return
        if not PIPELINE_READY:
            messagebox.showerror("错误",
                                 "仿真引擎未就绪，请检查依赖安装。\n"
                                 "确保 generate_software_figures.py 可正常导入。")
            return

        # 收集仿真参数
        sim_params = {
            'gravity_order': int(self.gravity_var.get()),
            'srp': self.srp_var.get(),
            'third_body': self.third_body_var.get(),
            'nrlmsise': self.nrlmsise_var.get(),
            'prediction_hours': int(self.prediction_var.get().replace('h', '')),
            'od_arc_days': int(self.od_arc_var.get().replace('天', '')),
        }
        if _pipeline and hasattr(_pipeline, 'set_params'):
            _pipeline.set_params(sim_params)

        # 锁定界面
        self.is_running = True
        self.run_btn.config(state=tk.DISABLED, text="⏳ 仿真计算中...")
        self.progress.start(12)
        self.status_var.set(
            f"正在运行仿真 — 引力{sim_params['gravity_order']}阶 / "
            f"预报{sim_params['prediction_hours']}h / 弧段{sim_params['od_arc_days']}天...")
        self.progress_text_var.set("加载数据 → 构建特征 → 模型推理 → 生成图表")
        self._set_export_buttons(tk.DISABLED)

        # 显示加载状态
        for key in self.axes:
            self._show_loading(self.axes[key])
            self.canvases[key].draw()

        # 后台线程执行仿真

        def run_task():
            try:
                os.environ["CUSTOM_DATA_PATH"] = folder
                try:
                    run_pipeline(data_folder=folder, params=sim_params)
                except TypeError:
                    run_pipeline(data_folder=folder)
                self.after(0, self._on_simulation_done)
            except Exception as e:
                self.after(0, self._on_simulation_error, str(e))

        threading.Thread(target=run_task, daemon=True).start()

    def _on_simulation_done(self):
        self.progress.stop()
        self.is_running = False
        self.run_btn.config(state=tk.NORMAL, text="▶  开始仿真")
        self.progress_text_var.set("仿真完成")
        self.status_var.set("✓ 仿真完成 — 结果已生成，可查看图表或导出")
        self.has_results = True

        try:
            # 根据当前场景渲染对应图表
            for display_name, key, chart_type, fn_name in self.current_tab_config:
                fn = _get_fn(fn_name)
                if fn is None:
                    self._show_placeholder(
                        self.axes[key], f"{display_name} (函数未就绪)", chart_type)
                    self.canvases[key].draw()
                    continue

                fig = self.figures[key]
                ax = self.axes[key]

                if chart_type == "multi":
                    # 多子图：传入 fig，函数内部管理子图布局
                    fig.clear()
                    fn(fig=fig)
                else:
                    # 单轴图表（2d / 3d / table）
                    ax.clear()
                    fn(ax=ax)

                self.canvases[key].draw()

            self._set_export_buttons(tk.NORMAL)
        except Exception as e:
            self.status_var.set(f"✗ 图表渲染失败: {str(e)[:60]}")
            messagebox.showerror("渲染错误", str(e))

    def _on_simulation_error(self, error_msg):
        self.progress.stop()
        self.is_running = False
        self.run_btn.config(state=tk.NORMAL, text="▶  开始仿真")
        self.progress_text_var.set("仿真失败")
        self.status_var.set(f"✗ 仿真失败: {error_msg[:60]}")
        messagebox.showerror("仿真失败", error_msg)

    def _set_export_buttons(self, state):
        self.btn_export_png.config(state=state)
        self.btn_export_pdf.config(state=state)
        self.btn_export_all.config(state=state)

    # ============================================================
    # 导出功能
    # ============================================================
    def _get_current_tab_info(self):
        """返回当前选中标签页的 (display_name, key, chart_type, fn_name)"""
        idx = self.notebook.index(self.notebook.select())
        if idx < len(self.current_tab_config):
            return self.current_tab_config[idx]
        return None

    def _export_png(self):
        info = self._get_current_tab_info()
        if not info:
            return
        display_name, key, _, _ = info
        filepath = filedialog.asksaveasfilename(
            title="导出 PNG 图片",
            defaultextension=".png",
            initialfile=f"{display_name}.png",
            filetypes=[("PNG 图片", "*.png")])
        if filepath:
            self.figures[key].savefig(filepath, dpi=300, bbox_inches='tight',
                                       facecolor='white')
            self.status_var.set(f"✓ 已导出: {filepath}")

    def _export_pdf(self):
        info = self._get_current_tab_info()
        if not info:
            return
        display_name, key, _, _ = info
        filepath = filedialog.asksaveasfilename(
            title="导出 PDF 文档",
            defaultextension=".pdf",
            initialfile=f"{display_name}.pdf",
            filetypes=[("PDF 文档", "*.pdf")])
        if filepath:
            self.figures[key].savefig(filepath, dpi=300, bbox_inches='tight',
                                       facecolor='white')
            self.status_var.set(f"✓ 已导出: {filepath}")

    def _export_all(self):
        folder = filedialog.askdirectory(title="选择导出目录")
        if not folder:
            return

        scenario = self.scenario_var.get()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(folder, f"仿真结果_{scenario}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        for display_name, key, _, _ in self.current_tab_config:
            fig = self.figures[key]
            fig.savefig(os.path.join(save_dir, f"{display_name}.png"),
                        dpi=300, bbox_inches='tight', facecolor='white')
            fig.savefig(os.path.join(save_dir, f"{display_name}.pdf"),
                        dpi=300, bbox_inches='tight', facecolor='white')

        # 导出参数摘要
        params = {
            "软件": SOFTWARE_NAME,
            "版本": SOFTWARE_VERSION,
            "导出时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "用户": self.username,
            "数据路径": self.data_folder or "",
            "任务场景": scenario,
            "参数": {
                "引力场阶数": self.gravity_var.get(),
                "太阳光压": self.srp_var.get(),
                "三体引力": self.third_body_var.get(),
                "大气阻力NRLMSISE": self.nrlmsise_var.get(),
                "预报时长": self.prediction_var.get(),
                "定轨弧段": self.od_arc_var.get(),
            },
        }
        import json
        with open(os.path.join(save_dir, "仿真参数.json"), 'w',
                  encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False, indent=2)

        n = len(self.current_tab_config)
        self.status_var.set(f"✓ 全部结果已导出至: {save_dir}")
        messagebox.showinfo(
            "导出完成",
            f"{n} 张图表 (PNG+PDF) 及参数摘要已导出至:\n{save_dir}")

    def _do_logout(self):
        if messagebox.askyesno("确认", "确定要退出登录吗？"):
            self.on_logout()


# ============================================================
# 3. 主应用控制器 — 管理界面切换
# ============================================================
class OrbitDeterminationApp:
    def __init__(self, root):
        self.root = root
        root.title(SOFTWARE_NAME)
        root.geometry("1440x900")
        root.minsize(1200, 750)
        root.configure(bg=COLOR_BG)

        setup_ttk_style()

        self.container = tk.Frame(root, bg=COLOR_BG)
        self.container.pack(fill=tk.BOTH, expand=True)

        self.current_screen = None
        self.show_login()

    def show_login(self):
        if self.current_screen:
            self.current_screen.destroy()
        self.current_screen = LoginScreen(self.container, self.show_main)
        self.current_screen.pack(fill=tk.BOTH, expand=True)
        self.root.title(f"{SOFTWARE_NAME} — 登录")

    def show_main(self, username):
        self.current_screen.destroy()
        self.current_screen = MainScreen(
            self.container, username, self.show_login)
        self.current_screen.pack(fill=tk.BOTH, expand=True)
        self.root.title(f"{SOFTWARE_NAME} — {username}")

    def run(self):
        self.root.mainloop()


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = OrbitDeterminationApp(root)
    app.run()
