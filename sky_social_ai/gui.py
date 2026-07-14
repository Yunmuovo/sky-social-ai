"""
光遇社交AI - 悬浮窗控制面板

tkinter悬浮窗，控制AI的启动/停止，实时查看状态和日志。
新增：区域预览面板，用不同颜色标定各识别区域。
无需命令行，双击运行即可。
"""

import sys
import os
import ctypes

# ===== 高DPI适配（必须在导入tkinter之前） =====
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import queue
import time
import logging
import datetime
from io import BytesIO

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ========== 区域定义：颜色对照表 ==========
# 每个区域的 (颜色名, 颜色BGR用于opencv, 颜色HEX用于tkinter, 中文名)
REGION_COLORS = {
    "chat_area":      ("#ff4444", "聊天识别区"),
    "chat_input":     ("#44ff44", "输入区域"),
    "friend_ui":      ("#4488ff", "好友操作区"),
    "speech_bubble":  ("#ffaa00", "头顶气泡"),
    "chat_button":    ("#ff44ff", "聊天按钮"),
}


class LogHandler(logging.Handler):
    """将 logging 日志重定向到 tkinter 队列"""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        msg = self.format(record)
        self.log_queue.put(("log", msg))


class FileLogHandler(logging.Handler):
    """将日志持久化写入文件"""

    def __init__(self, log_dir: str = None):
        super().__init__()
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        self._log_path = os.path.join(log_dir, f"sky_ai_{date_str}.log")
        self._file = open(self._log_path, "a", encoding="utf-8")

    def emit(self, record):
        try:
            msg = self.format(record)
            self._file.write(msg + "\n")
            self._file.flush()
        except Exception:
            pass

    def close_file(self):
        try:
            self._file.close()
        except Exception:
            pass


class StatusHandler:
    """通过队列发送状态更新到 GUI"""

    def __init__(self, status_queue: queue.Queue):
        self.queue = status_queue

    def update(self, state: str, detail: str = ""):
        self.queue.put(("status", {"state": state, "detail": detail}))

    def chat(self, role: str, text: str, name: str = ""):
        self.queue.put(("chat", {"role": role, "text": text, "name": name}))


class SkyGUI:
    """悬浮窗主界面"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("光遇社交AI")

        # 根据DPI自动缩放窗口大小
        scale = self._get_dpi_scale()
        base_w, base_h = 460, 660
        self.root.geometry(f"{int(base_w * scale)}x{int(base_h * scale)}")
        self.root.resizable(True, True)
        self.root.minsize(int(380 * scale), int(500 * scale))

        # 置顶
        self._always_on_top = tk.BooleanVar(value=True)
        self.root.attributes("-topmost", True)

        # 配色
        self.colors = {
            "bg": "#1e1e2e",
            "fg": "#cdd6f4",
            "accent": "#89b4fa",
            "accent2": "#a6e3a1",
            "danger": "#f38ba8",
            "warn": "#fab387",
            "input_bg": "#313244",
            "frame_bg": "#181825",
            "border": "#45475a",
            "btn_text": "#1a1a2e",
            "region_chat": "#ff4444",
            "region_input": "#44ff44",
            "region_friend": "#4488ff",
            "region_bubble": "#ffaa00",
            "region_button": "#ff44ff",
        }

        self.root.configure(bg=self.colors["bg"])

        # 字体缩放
        self._scale = scale
        self.fonts = {
            "title": ("Microsoft YaHei UI", int(12 * scale), "bold"),
            "status": ("Microsoft YaHei UI", int(11 * scale), "bold"),
            "detail": ("Microsoft YaHei UI", int(9 * scale)),
            "detail_bold": ("Microsoft YaHei UI", int(9 * scale), "bold"),
            "btn": ("Microsoft YaHei UI", int(10 * scale), "bold"),
            "btn_small": ("Microsoft YaHei UI", int(9 * scale)),
            "log": ("Consolas", int(9 * scale)),
            "stats": ("Microsoft YaHei UI", int(8 * scale)),
            "legend": ("Microsoft YaHei UI", int(8 * scale)),
        }

        # AI 实例
        self.ai = None
        self.ai_thread = None
        self._running = False

        # 文件日志处理器（关机时 close）
        self._file_log_handler = None

        # 队列通信
        self.log_queue = queue.Queue()
        self.status_queue = queue.Queue()

        # 区域预览相关
        self._preview_visible = False
        self._preview_photo = None
        self._config_regions = {}

        # 构建UI
        self._build_header()
        self._build_status_panel()
        self._build_control_panel()
        self._build_region_preview_panel()
        self._build_chat_panel()
        self._build_log_panel()
        self._build_status_bar()

        # 加载配置中的区域坐标
        self._load_region_config()

        # 定时轮询队列
        self._poll_queue()

        # 关闭处理
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    @staticmethod
    def _get_dpi_scale() -> float:
        """获取 DPI 缩放比例"""
        try:
            import tkinter as tk
            temp = tk.Tk()
            screen_width = temp.winfo_screenwidth()
            screen_height = temp.winfo_screenheight()
            temp.destroy()
            scale = max(screen_width / 1920, screen_height / 1080, 1.0)
            return min(max(scale, 1.0), 2.5)
        except Exception:
            return 1.0

    # ========== 配置加载 ==========

    def _load_region_config(self):
        """从 config.yaml 读取各区域坐标"""
        try:
            import yaml
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

            self._config_regions = {}

            # 聊天显示区
            ca = cfg.get("chat_area", {})
            if ca.get("region"):
                self._config_regions["chat_area"] = ca["region"]
                self._config_regions["chat_area_bottom"] = ca.get("max_ocr_height", 400)

            # 输入区
            ci = cfg.get("chat_input", {})
            if ci.get("region"):
                self._config_regions["chat_input"] = ci["region"]

            # 好友UI
            fu = cfg.get("friend_ui", {})
            friend_regions = {}
            for key in ["light_candle_region", "add_friend_region",
                         "name_input_region", "confirm_region", "cancel_region"]:
                if fu.get(key):
                    friend_regions[key.replace("_region", "")] = fu[key]
            if friend_regions:
                self._config_regions["friend_ui"] = friend_regions

            # 头顶气泡搜索区
            sb = cfg.get("speech_bubble", {})
            if sb.get("search_region"):
                self._config_regions["speech_bubble"] = sb["search_region"]

            # 聊天按钮搜索区
            cb = cfg.get("chat_button", {})
            if cb.get("search_region"):
                self._config_regions["chat_button"] = cb["search_region"]

        except Exception as e:
            logging.getLogger(__name__).debug(f"加载区域配置失败: {e}")

    # ========== UI 构建 ==========

    def _redraw_toggle(self):
        """根据置顶状态重绘Toggle开关"""
        on = self._always_on_top.get()
        if on:
            self._ontop_toggle.coords(self._tgl_bg, 2, 3, 52, 19)
            self._ontop_toggle.itemconfig(self._tgl_bg, fill=self.colors["accent2"])
            self._ontop_toggle.coords(self._tgl_knob, 32, 1, 52, 21)
            self._ontop_toggle.coords(self._tgl_label, 36, 11)
            self._ontop_toggle.itemconfig(self._tgl_label, text="ON", fill=self.colors["btn_text"])
        else:
            self._ontop_toggle.coords(self._tgl_bg, 2, 3, 52, 19)
            self._ontop_toggle.itemconfig(self._tgl_bg, fill=self.colors["border"])
            self._ontop_toggle.coords(self._tgl_knob, 2, 1, 22, 21)
            self._ontop_toggle.coords(self._tgl_label, 8, 11)
            self._ontop_toggle.itemconfig(self._tgl_label, text="OFF", fill=self.colors["border"])

    def _build_header(self):
        """顶部标题栏（自定义Toggle替代Checkbutton）"""
        frame = tk.Frame(self.root, bg=self.colors["frame_bg"], height=42)
        frame.pack(fill=tk.X)
        frame.pack_propagate(False)

        title = tk.Label(
            frame, text="光遇社交AI",
            font=self.fonts["title"],
            fg=self.colors["accent"],
            bg=self.colors["frame_bg"],
        )
        title.pack(side=tk.LEFT, padx=12, pady=8)

        # 自定义置顶开关：Canvas画矩形替代系统Checkbutton（深色主题下不可见）
        self._ontop_toggle = tk.Canvas(
            frame, width=54, height=22,
            bg=self.colors["frame_bg"], highlightthickness=0,
            cursor="hand2",
        )
        self._ontop_toggle.pack(side=tk.RIGHT, padx=10)

        # 开关背景圆角矩形
        self._tgl_bg = self._ontop_toggle.create_rectangle(
            2, 3, 52, 19, fill=self.colors["accent2"], outline="", tags="bg"
        )
        # 开关滑块
        self._tgl_knob = self._ontop_toggle.create_oval(
            30, 1, 52, 21, fill="#ffffff", outline="", tags="knob"
        )
        # 标签文字
        self._tgl_label = self._ontop_toggle.create_text(
            32, 11, text="ON", fill=self.colors["btn_text"],
            font=("Microsoft YaHei UI", int(7 * self._scale), "bold"),
            anchor="w", tags="label"
        )
        self._ontop_toggle.bind("<Button-1>", self._toggle_ontop)

        sep = tk.Frame(self.root, height=1, bg=self.colors["border"])
        sep.pack(fill=tk.X)

    def _build_status_panel(self):
        """状态显示区"""
        frame = tk.Frame(self.root, bg=self.colors["bg"], padx=12, pady=8)
        frame.pack(fill=tk.X)

        status_frame = tk.Frame(frame, bg=self.colors["bg"])
        status_frame.pack(fill=tk.X)

        # 更大的状态指示灯
        self.status_dot = tk.Canvas(
            status_frame, width=18, height=18,
            bg=self.colors["bg"], highlightthickness=0,
        )
        self.status_dot.pack(side=tk.LEFT, padx=(0, 10))
        self._dot = self.status_dot.create_oval(3, 3, 15, 15, fill="#6c7086", outline="")

        self.status_label = tk.Label(
            status_frame, text="未启动",
            font=self.fonts["status"],
            fg="#6c7086", bg=self.colors["bg"],
        )
        self.status_label.pack(side=tk.LEFT)

        # 分隔线
        sep = tk.Frame(frame, height=1, bg=self.colors["border"])
        sep.pack(fill=tk.X, pady=(6, 4))

        self.detail_label = tk.Label(
            frame, text="点击 [启动] 开始AI",
            font=self.fonts["detail"],
            fg=self.colors["border"], bg=self.colors["bg"],
            anchor="w", justify=tk.LEFT,
        )
        self.detail_label.pack(fill=tk.X)

    def _build_control_panel(self):
        """控制按钮区（分组排列，避免文字溢出）"""
        outer = tk.Frame(self.root, bg=self.colors["bg"], padx=10, pady=4)
        outer.pack(fill=tk.X)

        # --- 第1行：启动/停止（主操作） ---
        self.start_btn = tk.Button(
            outer, text="启动 AI",
            command=self._toggle_run,
            font=self.fonts["btn"],
            bg=self.colors["accent2"], fg=self.colors["btn_text"],
            activebackground="#94e2d5", activeforeground=self.colors["btn_text"],
            relief=tk.FLAT, cursor="hand2",
            padx=12, pady=6,
        )
        self.start_btn.pack(fill=tk.X, pady=(0, 4))

        # --- 第2行：标定 + 配置 + 重置（3等分） ---
        row2 = tk.Frame(outer, bg=self.colors["bg"])
        row2.pack(fill=tk.X, pady=(0, 2))

        self._make_btn(row2, "标 定", self._run_calibrate).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self._make_btn(row2, "配 置", self._open_config).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        self._make_btn(row2, "重置对话", self._reset_chat).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        # --- 第3行：安装OCR + 校准窗口 + 区域预览（3等分） ---
        row3 = tk.Frame(outer, bg=self.colors["bg"])
        row3.pack(fill=tk.X)

        self._make_btn(row3, "安装OCR", self._install_ocr, accent=True).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self._make_btn(row3, "校准窗口", self._calibrate_window, highlight=True).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        # 区域预览开关按钮
        self._preview_btn = tk.Button(
            row3, text="区域预览",
            command=self._toggle_region_preview,
            font=self.fonts["btn_small"],
            bg=self.colors["input_bg"], fg=self.colors["accent"],
            activebackground=self.colors["border"], activeforeground=self.colors["accent2"],
            relief=tk.FLAT, cursor="hand2", padx=8, pady=5,
        )
        self._preview_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

    def _make_btn(self, parent, text, command, *, accent=False, highlight=False):
        """快捷创建统一风格按钮"""
        if highlight:
            bg = self.colors["accent"]
            fg = self.colors["btn_text"]
            abg = "#74c7ec"
        elif accent:
            bg = self.colors["warn"]
            fg = self.colors["btn_text"]
            abg = "#f9e2af"
        else:
            bg = self.colors["input_bg"]
            fg = self.colors["fg"]
            abg = self.colors["border"]
        return tk.Button(
            parent, text=text, command=command,
            font=self.fonts["btn_small"],
            bg=bg, fg=fg,
            activebackground=abg, activeforeground=self.colors["btn_text"],
            relief=tk.FLAT, cursor="hand2", padx=6, pady=5,
        )

    def _build_region_preview_panel(self):
        """区域预览面板（可折叠）"""
        # 外层容器（用于显示/隐藏）
        self._preview_container = tk.Frame(self.root, bg=self.colors["bg"])

        # 分隔线
        sep = tk.Frame(self._preview_container, height=1, bg=self.colors["border"])
        sep.pack(fill=tk.X)

        # 内容区
        inner = tk.Frame(self._preview_container, bg=self.colors["bg"], padx=10, pady=6)
        inner.pack(fill=tk.X)

        # 标题行 + 刷新按钮
        title_row = tk.Frame(inner, bg=self.colors["bg"])
        title_row.pack(fill=tk.X, pady=(0, 4))

        tk.Label(
            title_row, text="区域预览",
            font=self.fonts["detail_bold"],
            fg=self.colors["accent"], bg=self.colors["bg"],
        ).pack(side=tk.LEFT)

        self._refresh_btn = tk.Button(
            title_row, text="截图刷新",
            command=self._refresh_preview,
            font=self.fonts["legend"],
            bg=self.colors["input_bg"], fg=self.colors["accent2"],
            activebackground=self.colors["border"],
            relief=tk.FLAT, cursor="hand2", padx=6, pady=2,
        )
        self._refresh_btn.pack(side=tk.RIGHT)

        # 画布
        canvas_frame = tk.Frame(inner, bg=self.colors["border"], padx=1, pady=1)
        canvas_frame.pack(fill=tk.X, pady=(0, 4))

        canvas_w = 400
        canvas_h = 200
        self._preview_canvas = tk.Canvas(
            canvas_frame,
            width=canvas_w, height=canvas_h,
            bg=self.colors["input_bg"],
            highlightthickness=0,
        )
        self._preview_canvas.pack(fill=tk.X)
        # 初始提示文字
        self._preview_canvas.create_text(
            canvas_w // 2, canvas_h // 2,
            text="点击 [截图刷新] 加载屏幕预览",
            fill=self.colors["border"],
            font=self.fonts["detail"],
        )

        # 图例行
        legend_frame = tk.Frame(inner, bg=self.colors["bg"])
        legend_frame.pack(fill=tk.X)

        legend_items = [
            ("#ff4444", "聊天区"), ("#44ff44", "输入区"),
            ("#4488ff", "好友UI"), ("#ffaa00", "气泡"),
        ]
        for hex_color, name in legend_items:
            item = tk.Frame(legend_frame, bg=self.colors["bg"])
            item.pack(side=tk.LEFT, padx=(0, 12))

            dot = tk.Canvas(item, width=10, height=10,
                            bg=self.colors["bg"], highlightthickness=0)
            dot.create_rectangle(0, 0, 10, 10, fill=hex_color, outline="")
            dot.pack(side=tk.LEFT, padx=(0, 4))

            tk.Label(item, text=name,
                     font=self.fonts["legend"],
                     fg=self.colors["fg"], bg=self.colors["bg"],
                     ).pack(side=tk.LEFT)

        # 默认不显示
        self._preview_container.pack_forget()

    def _build_chat_panel(self):
        """对话区面板（仅显示聊天文字）"""
        # 保存引用用于锚定
        self._chat_sep_frame = tk.Frame(self.root, bg=self.colors["bg"])
        self._chat_sep_frame.pack(fill=tk.X, padx=10)

        tk.Frame(self._chat_sep_frame, height=2, bg="#5a5599").pack(fill=tk.X, pady=(6, 0))
        tk.Frame(self._chat_sep_frame, height=1, bg=self.colors["border"]).pack(fill=tk.X, pady=(1, 2))

        title_row = tk.Frame(self._chat_sep_frame, bg=self.colors["bg"])
        title_row.pack(fill=tk.X)

        tk.Label(
            title_row, text="对话区",
            font=self.fonts["detail_bold"],
            fg=self.colors["accent"], bg=self.colors["bg"],
        ).pack(side=tk.LEFT)

        # 对话文本框
        chat_frame = tk.Frame(self.root, bg=self.colors["border"], padx=1, pady=1)
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(2, 0))

        self.chat_text = scrolledtext.ScrolledText(
            chat_frame,
            font=self.fonts["log"],
            bg="#1a1826",
            fg=self.colors["fg"],
            insertbackground=self.colors["fg"],
            relief=tk.FLAT,
            wrap=tk.WORD,
            state=tk.NORMAL,
            height=6,
        )
        self.chat_text.pack(fill=tk.BOTH, expand=True)
        # 初始占位
        self.chat_text.insert(tk.END, "等待聊天记录...\n", "placeholder")
        self.chat_text.tag_config("placeholder", foreground="#585b70")
        # 聊天角色颜色标签
        self.chat_text.tag_config("msg_other", foreground="#f9e2af")
        self.chat_text.tag_config("msg_ai", foreground="#cba6f7")
        self.chat_text.configure(state=tk.DISABLED)

    def _build_log_panel(self):
        """日志面板（系统运行日志）"""
        # 双线分隔（视觉更强）
        self._log_sep_frame = tk.Frame(self.root, bg=self.colors["bg"])
        self._log_sep_frame.pack(fill=tk.X, padx=10)

        tk.Frame(self._log_sep_frame, height=2, bg="#5a5599").pack(fill=tk.X, pady=(6, 0))
        tk.Frame(self._log_sep_frame, height=1, bg=self.colors["border"]).pack(fill=tk.X, pady=(1, 2))

        title_row = tk.Frame(self._log_sep_frame, bg=self.colors["bg"])
        title_row.pack(fill=tk.X)

        tk.Label(
            title_row, text="日志",
            font=self.fonts["detail_bold"],
            fg=self.colors["border"], bg=self.colors["bg"],
        ).pack(side=tk.LEFT)

        tk.Label(
            title_row, text="系统运行记录",
            font=self.fonts["detail"],
            fg=self.colors["border"], bg=self.colors["bg"],
        ).pack(side=tk.RIGHT)

        # 日志文本框（固定高度120px，始终可见）
        text_frame = tk.Frame(self.root, bg=self.colors["border"], padx=1, pady=1)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(2, 6))

        self.log_text = scrolledtext.ScrolledText(
            text_frame,
            font=self.fonts["log"],
            bg=self.colors["input_bg"],
            fg=self.colors["fg"],
            insertbackground=self.colors["fg"],
            relief=tk.FLAT,
            wrap=tk.WORD,
            state=tk.DISABLED,
            height=6,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _build_status_bar(self):
        """底部状态栏"""
        frame = tk.Frame(self.root, bg=self.colors["frame_bg"], height=26)
        frame.pack(fill=tk.X, side=tk.BOTTOM)
        frame.pack_propagate(False)

        self.stats_label = tk.Label(
            frame, text="对话: 0轮 | 状态: 空闲",
            font=self.fonts["stats"],
            fg=self.colors["border"], bg=self.colors["frame_bg"],
        )
        self.stats_label.pack(side=tk.LEFT, padx=10, pady=3)

        tip_label = tk.Label(
            frame, text="光遇社交AI v1.0",
            font=self.fonts["stats"],
            fg=self.colors["border"], bg=self.colors["frame_bg"],
        )
        tip_label.pack(side=tk.RIGHT, padx=10, pady=3)

    # ========== 区域预览逻辑 ==========

    def _toggle_region_preview(self):
        """切换区域预览面板显示/隐藏"""
        self._preview_visible = not self._preview_visible
        if self._preview_visible:
            self._preview_container.pack(
                fill=tk.X, before=self._chat_sep_frame
            )
            self._preview_btn.configure(fg=self.colors["accent2"])
            # 自动加载配置并截图
            self._load_region_config()
            self.root.after(200, self._refresh_preview)
        else:
            self._preview_container.pack_forget()
            self._preview_btn.configure(fg=self.colors["accent"])

    def _refresh_preview(self):
        """截图并在画布上绘制所有区域（使用 mss，与 AI 截图一致）"""
        if not self._preview_visible:
            return

        try:
            import mss
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np
        except ImportError:
            self._append_log("[WARN] 需要安装 pillow 和 mss")
            return

        self._refresh_btn.configure(text="截图中...", state=tk.DISABLED)
        self.root.update_idletasks()

        try:
            # 使用 mss 截图（与 AI 主程序截图方式一致）
            sct = mss.mss()
            raw = sct.grab(sct.monitors[0])
            screenshot = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
            sw, sh = screenshot.size

            # 计算缩放到画布宽度
            canvas_w = self._preview_canvas.winfo_width()
            if canvas_w < 50:
                canvas_w = 400
            scale = canvas_w / sw
            canvas_h = int(sh * scale)

            # 缩放截图
            thumbnail = screenshot.resize((canvas_w, canvas_h), Image.LANCZOS)
            draw = ImageDraw.Draw(thumbnail, "RGBA")

            # 图例颜色（RGBA，半透明填充 + 不透明边框）
            color_map = {
                "chat_area":      ((255, 68, 68, 60),   (255, 68, 68, 220)),
                "chat_input":     ((68, 255, 68, 60),   (68, 255, 68, 220)),
                "friend_ui":      ((68, 136, 255, 60),  (68, 136, 255, 220)),
                "speech_bubble":  ((255, 170, 0, 60),   (255, 170, 0, 220)),
                "chat_button":    ((255, 68, 255, 60),  (255, 68, 255, 220)),
            }
            border_width = max(2, int(3 * scale * 2))

            def draw_rect(region, color_key, label=""):
                fill_c, border_c = color_map.get(color_key, ((255,255,255,60),(255,255,255,220)))
                l, t, r, b = [int(v * scale) for v in region]
                # 填充
                draw.rectangle([l, t, r, b], fill=fill_c)
                # 边框
                for i in range(border_width):
                    draw.rectangle(
                        [l - i, t - i, r + i, b + i],
                        outline=border_c
                    )
                # 标签
                if label:
                    try:
                        font = ImageFont.truetype("msyh.ttc", max(10, int(11 * scale)))
                    except Exception:
                        font = ImageFont.load_default()
                    # 文字背景
                    tw = draw.textlength(label, font=font) if hasattr(draw, 'textlength') else len(label) * 7
                    th = 16
                    draw.rectangle(
                        [l + 2, t - th - 2, l + 2 + tw + 6, t],
                        fill=border_c
                    )
                    draw.text((l + 4, t - th), label, fill=(255, 255, 255, 255), font=font)

            # --- 画 chat_area ---
            if "chat_area" in self._config_regions:
                region = self._config_regions["chat_area"]
                draw_rect(region, "chat_area", "聊天识别区")

                # 底部 OCR 子区域（黄色虚线风格：用半透明黄条表示）
                bottom_h = self._config_regions.get("chat_area_bottom", 400)
                l, t, r, b = [int(v * scale) for v in region]
                crop_top = max(t, b - int(bottom_h * scale))
                if crop_top > t:
                    draw.rectangle(
                        [l, crop_top, r, b],
                        fill=(255, 220, 50, 40),
                        outline=(255, 220, 50, 180)
                    )

            # --- 画 chat_input ---
            if "chat_input" in self._config_regions:
                draw_rect(self._config_regions["chat_input"], "chat_input", "输入区")

            # --- 画 friend_ui 子区域 ---
            if "friend_ui" in self._config_regions:
                for sub_name, sub_region in self._config_regions["friend_ui"].items():
                    draw_rect(sub_region, "friend_ui")

            # --- 画 speech_bubble ---
            if "speech_bubble" in self._config_regions:
                draw_rect(self._config_regions["speech_bubble"], "speech_bubble", "气泡")

            # --- 画 chat_button ---
            if "chat_button" in self._config_regions:
                draw_rect(self._config_regions["chat_button"], "chat_button", "聊天按钮")

            # 转为 tkinter PhotoImage
            bio = BytesIO()
            thumbnail.save(bio, format="PNG")
            self._preview_photo = tk.PhotoImage(data=bio.getvalue())

            # 更新画布
            self._preview_canvas.delete("all")
            self._preview_canvas.config(height=canvas_h)
            self._preview_canvas.create_image(0, 0, anchor=tk.NW, image=self._preview_photo)
            # 右下角分辨率标注
            self._preview_canvas.create_text(
                canvas_w - 6, canvas_h - 6,
                text=f"{sw}x{sh}",
                anchor=tk.SE,
                fill="#ffffff",
                font=self.fonts["legend"],
            )

            self._append_log(f"[预览] 区域预览已更新 ({sw}x{sh})")

        except Exception as e:
            self._append_log(f"[WARN] 截图失败: {e}")
        finally:
            self._refresh_btn.configure(text="截图刷新", state=tk.NORMAL)

    # ========== 核心逻辑 ==========

    def _toggle_run(self):
        """启动/停止 AI"""
        if self._running:
            self._stop_ai()
        else:
            self._start_ai()

    def _start_ai(self):
        """启动 AI"""
        self._running = True
        self.start_btn.configure(
            text="停止 AI",
            bg=self.colors["danger"],
            fg=self.colors["btn_text"],
            activebackground="#eba0ac",
        )
        self._set_status("starting", "正在初始化...")

        self.ai_thread = threading.Thread(target=self._ai_run, daemon=True)
        self.ai_thread.start()

    def _stop_ai(self):
        """停止 AI"""
        self._running = False
        if self.ai:
            self.ai._running = False
        self.start_btn.configure(
            text="启动 AI",
            bg=self.colors["accent2"],
            fg=self.colors["btn_text"],
            activebackground="#94e2d5",
        )
        self._set_status("stopped", "已停止")

    def _ai_run(self):
        """AI 运行线程"""
        try:
            from main import SkySocialAI
            self.ai = SkySocialAI()
            self.ai._running = True

            self._setup_gui_logging()

            self.ai._gui_status = StatusHandler(self.status_queue)

            self.status_queue.put(("status", {"state": "running", "detail": "扫描椅子中..."}))

            while self._running and self.ai._running:
                try:
                    self.ai._tick()
                    time.sleep(self.ai.scan_interval)
                except Exception as e:
                    self.log_queue.put(("log", f"[ERROR] {e}"))

            self.status_queue.put(("status", {"state": "stopped", "detail": "已停止"}))

        except Exception as e:
            self.log_queue.put(("log", f"[FATAL] 启动失败: {e}"))
            self.status_queue.put(("status", {"state": "error", "detail": str(e)}))
            self._running = False
            self.log_queue.put(("action", "reset_button"))

    def _setup_gui_logging(self):
        """将 logging 重定向到 GUI + 持久化到文件"""
        root_logger = logging.getLogger()
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)

        formatter = logging.Formatter(
            "%(asctime)s [%(name)-6s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )

        gui_handler = LogHandler(self.log_queue)
        gui_handler.setFormatter(formatter)
        gui_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(gui_handler)

        # 文件日志持久化
        self._file_log_handler = FileLogHandler()
        file_formatter = logging.Formatter(
            "%(asctime)s [%(name)-6s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._file_log_handler.setFormatter(file_formatter)
        self._file_log_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(self._file_log_handler)

        root_logger.setLevel(logging.DEBUG)

    def _poll_queue(self):
        """定时轮询队列，更新UI"""
        try:
            while True:
                msg_type, data = self.log_queue.get_nowait()

                if msg_type == "log":
                    self._append_log(data)
                elif msg_type == "action":
                    if data == "reset_button":
                        self.start_btn.configure(
                            text="启动 AI",
                            bg=self.colors["accent2"],
                            fg=self.colors["btn_text"],
                        )
                        self._running = False

        except queue.Empty:
            pass

        try:
            while True:
                msg_type, data = self.status_queue.get_nowait()

                if msg_type == "status":
                    self._set_status(data["state"], data.get("detail", ""))
                elif msg_type == "chat":
                    self._append_chat(data["role"], data["text"], data.get("name", ""))

        except queue.Empty:
            pass

        self.root.after(100, self._poll_queue)

    def _set_status(self, state: str, detail: str):
        """更新状态显示"""
        colors_map = {
            "running": ("#a6e3a1", "运行中"),
            "starting": ("#f9e2af", "启动中"),
            "stopped": ("#6c7086", "已停止"),
            "error": ("#f38ba8", "异常"),
            "thinking": ("#89b4fa", "思考中"),
            "responding": ("#cba6f7", "回复中"),
            "seated": ("#94e2d5", "已坐下"),
            "idle": ("#6c7086", "空闲"),
        }

        color, text = colors_map.get(state, ("#6c7086", state))
        self.status_dot.itemconfig(self._dot, fill=color)
        self.status_label.configure(text=text, fg=color)
        self.detail_label.configure(text=detail)

    def _append_log(self, text: str):
        """追加日志"""
        self.log_text.configure(state=tk.NORMAL)

        if "[ERROR]" in text or "[FATAL]" in text:
            tag = "error"
            self.log_text.tag_config(tag, foreground=self.colors["danger"])
        elif "[WARNING]" in text:
            tag = "warn"
            self.log_text.tag_config(tag, foreground=self.colors["warn"])
        elif "新消息" in text or "AI回复" in text:
            tag = "chat"
            self.log_text.tag_config(tag, foreground=self.colors["accent2"])
        elif "发现椅子" in text:
            tag = "chair"
            self.log_text.tag_config(tag, foreground=self.colors["accent"])
        else:
            tag = "normal"
            self.log_text.tag_config(tag, foreground=self.colors["fg"])

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {text}\n", tag)
        self.log_text.configure(state=tk.DISABLED)
        self.log_text.see(tk.END)

    def _append_chat(self, role: str, text: str, name: str = ""):
        """追加聊天记录到对话区"""
        self.chat_text.configure(state=tk.NORMAL)
        # 清除占位文字
        content = self.chat_text.get("1.0", tk.END).strip()
        if content == "等待聊天记录...":
            self.chat_text.delete("1.0", tk.END)
        if role == "other":
            tag = "msg_other"
            prefix = f"[{name}]" if name else "[对方]"
        else:
            tag = "msg_ai"
            prefix = "[AI]  "
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.chat_text.insert(tk.END, f"[{timestamp}] {prefix} {text}\n", tag)
        self.chat_text.configure(state=tk.DISABLED)
        self.chat_text.see(tk.END)

    def _toggle_ontop(self, event=None):
        """切换置顶（带自定义Toggle动画）"""
        new_val = not self._always_on_top.get()
        self._always_on_top.set(new_val)
        self.root.attributes("-topmost", new_val)
        self._redraw_toggle()

    def _run_calibrate(self):
        """启动标定工具"""
        self._append_log("启动标定工具...")
        import subprocess
        calibrate_path = os.path.join(os.path.dirname(__file__), "calibrate.py")
        threading.Thread(
            target=lambda: subprocess.Popen(
                [sys.executable, calibrate_path],
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            ),
            daemon=True,
        ).start()
        self._append_log("标定窗口已打开（全屏模式），请切换到标定窗口进行框选")

    def _open_config(self):
        """打开配置文件"""
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        os.startfile(config_path)
        self._append_log("已打开配置文件")

    def _reset_chat(self):
        """重置对话历史"""
        if self.ai and self.ai.dialogue:
            self.ai.dialogue.reset_history()
        self.chat_text.configure(state=tk.NORMAL)
        self.chat_text.delete("1.0", tk.END)
        self.chat_text.insert(tk.END, "等待聊天记录...\n", "placeholder")
        self.chat_text.configure(state=tk.DISABLED)
        self._append_log("对话历史已重置")

    def _install_ocr(self):
        """在后台安装 PaddleOCR 依赖"""
        self._append_log("开始安装 PaddleOCR，请耐心等待...")

        def do_install():
            import subprocess
            try:
                self.log_queue.put(("log", "[INSTALL] 安装 paddlepaddle..."))
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "paddlepaddle", "paddleocr",
                     "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"],
                    capture_output=True, text=True, check=False,
                )
                for line in result.stdout.splitlines()[-30:]:
                    self.log_queue.put(("log", f"[INSTALL] {line}"))
                if result.returncode == 0:
                    self.log_queue.put(("log", "[INSTALL] PaddleOCR 安装完成，请重启悬浮窗"))
                else:
                    self.log_queue.put(("log", f"[INSTALL] 安装失败: {result.stderr[-500:]}"))
            except Exception as e:
                self.log_queue.put(("log", f"[INSTALL] 异常: {e}"))

        threading.Thread(target=do_install, daemon=True).start()

    def _calibrate_window(self):
        """校准窗口偏移：实时显示鼠标位置，一键记录"""
        import pyautogui

        popup = tk.Toplevel(self.root)
        popup.title("窗口校准")
        popup.geometry("380x180")
        popup.resizable(False, False)
        popup.attributes("-topmost", True)
        popup.configure(bg=self.colors["bg"])
        popup.transient(self.root)

        info = tk.Label(popup, text="将鼠标移到游戏画面左上角，然后点\"记录\"",
                        font=self.fonts["status"], fg=self.colors["fg"], bg=self.colors["bg"])
        info.pack(pady=(15, 10))

        pos_label = tk.Label(popup, text="当前位置: (--, --)",
                             font=self.fonts["detail_bold"], fg=self.colors["accent2"],
                             bg=self.colors["bg"])
        pos_label.pack(pady=5)

        btn_frame = tk.Frame(popup, bg=self.colors["bg"])
        btn_frame.pack(pady=15)

        def update_pos():
            if popup.winfo_exists():
                try:
                    x, y = pyautogui.position()
                    pos_label.config(text=f"当前位置: ({x}, {y})")
                except Exception:
                    pass
                popup.after(100, update_pos)

        def record():
            import re
            x, y = pyautogui.position()
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
            with open(config_path, "r", encoding="utf-8") as f:
                content = f.read()
            content = re.sub(r'window_offset:\s*\[.*?\]',
                             f'window_offset: [{x}, {y}]', content)
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(content)
            self._append_log(f"窗口偏移已更新: ({x}, {y})")
            popup.destroy()

        record_btn = tk.Button(btn_frame, text="记录", command=record,
                               font=self.fonts["btn_small"], bg=self.colors["accent2"],
                               fg=self.colors["btn_text"], relief=tk.FLAT,
                               cursor="hand2", padx=12, pady=4)
        record_btn.pack(side=tk.LEFT, padx=5)

        cancel_btn = tk.Button(btn_frame, text="取消", command=popup.destroy,
                               font=self.fonts["btn_small"], bg=self.colors["input_bg"],
                               fg=self.colors["fg"], relief=tk.FLAT,
                               cursor="hand2", padx=12, pady=4)
        cancel_btn.pack(side=tk.LEFT, padx=5)

        update_pos()

    def _on_close(self):
        """关闭窗口"""
        self._running = False
        if self.ai:
            self.ai._running = False
        if self._file_log_handler:
            self._file_log_handler.close_file()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    app = SkyGUI()
    app.run()


if __name__ == "__main__":
    main()
