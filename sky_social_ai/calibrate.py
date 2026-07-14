"""
光遇社交AI - 可视化标定工具 (v2)

全屏覆盖式标定：在游戏画面之上叠加半透明蒙版，
鼠标拖拽直接框选各区域，所有坐标自动保存到 config.yaml。

用法：
  - 启动后会自动截屏并全屏显示
  - 左上角显示当前步骤说明
  - 鼠标拖拽框选区域（从左上角拖到右下角）
  - 滚轮 / 上下方向键：调整区域位置微调
  - Enter：确认当前框选 → 下一步
  - Backspace：删除当前框选，重新画
  - Escape：跳过当前步骤（不标定这个区域）
  - 全部完成后自动保存到 config.yaml
  - 退出：关闭窗口 或 按 Q
"""

import sys
import os
import ctypes

# ===== DPI 适配 =====
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import yaml
import logging
from io import BytesIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("calibrate")


# ========== 标定步骤定义 ==========
# (配置键, 显示名称, 提示文字, 颜色, 是否为单点)
CALIBRATION_STEPS = [
    {
        "key": "window_offset",
        "name": "游戏窗口偏移（全屏填0即可）",
        "hint": "如果游戏是全屏占满整个屏幕 → 直接按 Enter 跳过\n如果是窗口模式 → 把鼠标放在游戏窗口左上角，点击记录",
        "color": "#ffaa00",
        "is_point": True,
    },
    {
        "key": "focus_point",
        "name": "游戏窗口焦点点击位",
        "hint": "在游戏画面中找一个【不会误操作的角落】点击一下\n打字前会先点这里确保游戏窗口获得焦点\n建议：左上角或右上角的空白区域",
        "color": "#ffaa00",
        "is_point": True,
    },
    {
        "key": "chat_area",
        "name": "聊天显示区（左侧对话框）",
        "hint": "从对话框【左上角】拖到【右下角】，框住整个左侧聊天文字区域",
        "color": "#ff4444",
        "is_point": False,
    },
    {
        "key": "chat_input",
        "name": "聊天输入区域（底部输入条）",
        "hint": "框住底部的聊天输入条（包含输入框+发送按钮）",
        "color": "#44ff44",
        "is_point": False,
    },
    {
        "key": "friend_light_candle",
        "name": "「点亮」按钮",
        "hint": "框住好友面板中的「点亮/点火」按钮区域",
        "color": "#4488ff",
        "is_point": False,
    },
    {
        "key": "friend_add",
        "name": "「添加好友」按钮",
        "hint": "框住好友面板中的「添加好友」按钮区域",
        "color": "#4488ff",
        "is_point": False,
    },
    {
        "key": "friend_name_input",
        "name": "好友「命名输入框」",
        "hint": "框住输入好友昵称的文本框区域",
        "color": "#4488ff",
        "is_point": False,
    },
    {
        "key": "friend_confirm",
        "name": "「确认」按钮",
        "hint": "框住确认/✓按钮区域",
        "color": "#4488ff",
        "is_point": False,
    },
    {
        "key": "friend_cancel",
        "name": "「取消」按钮",
        "hint": "框住取消/✗按钮区域",
        "color": "#4488ff",
        "is_point": False,
    },
    {
        "key": "back_button",
        "name": "「返回」按钮",
        "hint": "发送消息后，聊天输入框附近出现的「返回/收起」按钮，框住按钮区域",
        "color": "#ff44aa",
        "is_point": False,
    },
]


class Calibrator:
    """全屏标定覆盖层"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("标定工具 - 光遇社交AI")

        # 获取屏幕分辨率
        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()

        self.root.geometry(f"{self.screen_w}x{self.screen_h}+0+0")
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        self.root.configure(bg="black", cursor="crosshair")

        # 截屏作为背景
        self._background_image = None
        self._bg_photo = None
        self._bg_id = None
        self._capture_screen()

        # 画布（覆盖在背景上，用于绘制框选）
        self.canvas = tk.Canvas(
            self.root,
            width=self.screen_w,
            height=self.screen_h,
            bg="black",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.place(x=0, y=0)
        # 背景图
        self._bg_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self._bg_photo)

        # 当前步骤
        self._step_index = 0
        # 标定结果 {key: [x1, y1, x2, y2] 或 point: [x, y]}
        self._results = {}

        # 当前拖拽状态
        self._dragging = False
        self._start_x = 0
        self._start_y = 0
        self._current_rect = None  # canvas rect id
        self._force_offset = False  # window_offset 安全确认标志

        # 已完成的框（步骤对应的 canvas item ids）
        self._result_rects = {}  # key -> canvas rect id
        self._result_labels = {}  # key -> canvas text id

        # UI 元素
        self._build_info_panel()

        # 绑定事件
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.root.bind("<Key>", self._on_key)
        self.root.bind("<Escape>", lambda e: self._skip_step())
        self.root.bind("<Return>", lambda e: self._confirm_step())
        self.root.bind("<BackSpace>", lambda e: self._clear_current())
        self.root.bind("<q>", lambda e: self._quit())
        self.root.bind("<Q>", lambda e: self._quit())
        self.root.bind("<r>", lambda e: self._refresh_screenshot())
        self.root.bind("<R>", lambda e: self._refresh_screenshot())

        # 关闭窗口
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

        # 显示第一步
        self._update_display()

        logger.info("标定工具已启动")

    def _refresh_screenshot(self):
        """重新截屏更新背景（按R触发），方便先切到游戏布置好场景再截图"""
        self._capture_screen()
        if self._bg_id:
            self.canvas.itemconfig(self._bg_id, image=self._bg_photo)
        # 实时坐标提示
        self.canvas.itemconfig(
            self._info_coords,
            text="✅ 截图已刷新",
            fill="#44ff44",
        )
        self.root.after(1500, lambda: self._update_coord_display(None, None))

    def _capture_screen(self):
        """使用 mss 截屏（与 AI 主程序一致），转为 tk 可用格式"""
        try:
            import mss
            sct = mss.mss()
            monitor = sct.monitors[0]  # 全屏
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            # mss 返回 BGRA，用 PIL 读取
            pil_img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            # 缩放适配屏幕
            if pil_img.size != (self.screen_w, self.screen_h):
                pil_img = pil_img.resize((self.screen_w, self.screen_h), Image.LANCZOS)
            self._bg_photo = ImageTk.PhotoImage(pil_img)
            self._background_image = pil_img
        except Exception as e:
            logger.error(f"截屏失败: {e}")
            # 创建黑色背景
            self._background_image = Image.new("RGB", (self.screen_w, self.screen_h), (30, 30, 30))
            self._bg_photo = ImageTk.PhotoImage(self._background_image)

    def _build_info_panel(self):
        """构建顶部信息面板"""
        panel_h = 70

        # 半透明背景条
        self._info_bg = self.canvas.create_rectangle(
            0, 0, self.screen_w, panel_h,
            fill="#000000",
            stipple="gray50",
            outline="",
        )

        # 步骤名称
        self._info_title = self.canvas.create_text(
            20, 14,
            text="",
            anchor=tk.NW,
            fill="#ffffff",
            font=("Microsoft YaHei UI", 16, "bold"),
        )

        # 提示文字
        self._info_hint = self.canvas.create_text(
            20, 40,
            text="",
            anchor=tk.NW,
            fill="#cccccc",
            font=("Microsoft YaHei UI", 12),
        )

        # 当前坐标（右下角）
        self._info_coords = self.canvas.create_text(
            self.screen_w - 20, self.screen_h - 30,
            text="",
            anchor=tk.SE,
            fill="#aaffaa",
            font=("Consolas", 14, "bold"),
        )

        # 底部快捷键提示
        shortcuts = "Enter=确认 | Backspace=重画 | Esc=跳过 | Q=退出"
        self.canvas.create_text(
            self.screen_w // 2, self.screen_h - 10,
            text=shortcuts,
            anchor=tk.S,
            fill="#888888",
            font=("Microsoft YaHei UI", 9),
        )

    def _update_display(self):
        """刷新当前步骤的显示"""
        if self._step_index >= len(CALIBRATION_STEPS):
            self._show_summary()
            return

        step = CALIBRATION_STEPS[self._step_index]
        total = len(CALIBRATION_STEPS)

        # 更新信息面板
        title = f"步骤 {self._step_index + 1}/{total}: {step['name']}"
        self.canvas.itemconfig(self._info_title, text=title)
        self.canvas.itemconfig(self._info_hint, text=step["hint"])

        # 高亮当前步骤的颜色指示
        # 在左上角画一个小色块
        if hasattr(self, "_color_indicator"):
            self.canvas.delete(self._color_indicator)
        self._color_indicator = self.canvas.create_rectangle(
            10, self.screen_h - 50, 30, self.screen_h - 30,
            fill=step["color"], outline="#ffffff", width=1,
        )

        self._update_coord_display(None, None)

    def _update_coord_display(self, x, y):
        """更新坐标显示"""
        if x is not None and y is not None:
            text = f"鼠标: ({x}, {y})"
            self.canvas.itemconfig(self._info_coords, text=text, fill="#aaffaa")
        else:
            # 显示当前步骤的已保存坐标
            step = CALIBRATION_STEPS[self._step_index]
            key = step["key"]
            if key in self._results:
                val = self._results[key]
                if step["is_point"]:
                    text = f"已记录: ({val[0]}, {val[1]})"
                else:
                    text = f"已框选: [{val[0]}, {val[1]}, {val[2]}, {val[3]}]  尺寸: {val[2]-val[0]}x{val[3]-val[1]}"
                self.canvas.itemconfig(self._info_coords, text=text, fill="#44ff44")
            else:
                self.canvas.itemconfig(self._info_coords, text="未标定", fill="#ff8844")

    # ========== 鼠标事件 ==========

    def _on_press(self, event):
        self._dragging = True
        self._start_x = event.x
        self._start_y = event.y

        # 清除当前拖拽矩形
        if self._current_rect:
            self.canvas.delete(self._current_rect)
            self._current_rect = None

        step = CALIBRATION_STEPS[self._step_index]

        if step["is_point"]:
            # 单点模式：画十字
            self._current_rect = self.canvas.create_line(
                event.x - 10, event.y, event.x + 10, event.y,
                fill="#ffff00", width=2,
            )
            self.canvas.create_line(
                event.x, event.y - 10, event.x, event.y + 10,
                fill="#ffff00", width=2,
            )
        else:
            # 矩形模式：画虚线框
            color = step["color"]
            self._current_rect = self.canvas.create_rectangle(
                event.x, event.y, event.x, event.y,
                outline=color, width=3, dash=(8, 4),
            )

    def _on_drag(self, event):
        if not self._dragging:
            return

        step = CALIBRATION_STEPS[self._step_index]

        if step["is_point"]:
            return  # 单点不拖拽

        # 更新拖拽矩形
        x1, y1 = min(self._start_x, event.x), min(self._start_y, event.y)
        x2, y2 = max(self._start_x, event.x), max(self._start_y, event.y)
        if self._current_rect:
            self.canvas.coords(self._current_rect, x1, y1, x2, y2)

        # 实时显示尺寸
        w = x2 - x1
        h = y2 - y1
        self.canvas.itemconfig(
            self._info_coords,
            text=f"区域: [{x1}, {y1}, {x2}, {y2}]  尺寸: {w}x{h}",
            fill="#ffff00",
        )

    def _on_release(self, event):
        if not self._dragging:
            return
        self._dragging = False

        step = CALIBRATION_STEPS[self._step_index]
        key = step["key"]

        if step["is_point"]:
            # 单点：记录点击位置
            self._results[key] = [event.x, event.y]
            self._draw_result_point(key, event.x, event.y, step["color"])
            self._update_coord_display(None, None)
        else:
            # 矩形：记录框选区域
            x1, y1 = min(self._start_x, event.x), min(self._start_y, event.y)
            x2, y2 = max(self._start_x, event.x), max(self._start_y, event.y)
            w, h = x2 - x1, y2 - y1

            if w < 5 or h < 5:
                # 太小了，忽略
                if self._current_rect:
                    self.canvas.delete(self._current_rect)
                    self._current_rect = None
                return

            self._results[key] = [x1, y1, x2, y2]

            # 删除拖拽虚线框，画实线框
            if self._current_rect:
                self.canvas.delete(self._current_rect)
                self._current_rect = None

            self._draw_result_rect(key, x1, y1, x2, y2, step["color"])
            self._update_coord_display(None, None)

    # ========== 绘制结果 ==========

    def _draw_result_rect(self, key, x1, y1, x2, y2, color):
        """在画布上绘制已确认的框"""
        # 删除旧的
        if key in self._result_rects:
            self.canvas.delete(self._result_rects[key])
        if key in self._result_labels:
            self.canvas.delete(self._result_labels[key])

        # 半透明填充（用 stipple 模拟）
        fill_id = self.canvas.create_rectangle(
            x1, y1, x2, y2,
            fill=color, stipple="gray25", outline="",
        )
        # 实线边框
        rect_id = self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline=color, width=3,
        )
        self._result_rects[key] = rect_id

        # 标签
        step = CALIBRATION_STEPS[self._step_index]
        label_id = self.canvas.create_text(
            x1 + 4, y1 - 12,
            text=step["name"],
            anchor=tk.SW,
            fill=color,
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        self._result_labels[key] = label_id

        # 角落尺寸标注
        size_text = f"{x2-x1}×{y2-y1}"
        self.canvas.create_text(
            x2 - 4, y2 - 4,
            text=size_text,
            anchor=tk.SE,
            fill=color,
            font=("Consolas", 9),
        )

    def _draw_result_point(self, key, x, y, color):
        """在画布上绘制已确认的点"""
        if key in self._result_rects:
            self.canvas.delete(self._result_rects[key])
        if key in self._result_labels:
            self.canvas.delete(self._result_labels[key])

        # 靶心标记
        rect_id = self.canvas.create_oval(
            x - 8, y - 8, x + 8, y + 8,
            outline=color, width=3,
        )
        self.canvas.create_line(x - 15, y, x + 15, y, fill=color, width=2)
        self.canvas.create_line(x, y - 15, x, y + 15, fill=color, width=2)
        self._result_rects[key] = rect_id

        step = CALIBRATION_STEPS[self._step_index]
        label_id = self.canvas.create_text(
            x + 12, y - 12,
            text=f"{step['name']} ({x},{y})",
            anchor=tk.SW,
            fill=color,
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        self._result_labels[key] = label_id

    # ========== 键盘事件 ==========

    def _on_key(self, event):
        step = CALIBRATION_STEPS[self._step_index]
        if step["is_point"]:
            return

        key = step["key"]
        if key not in self._results:
            return

        r = self._results[key]
        delta = 1

        if event.keysym == "Up":
            r[1] -= delta
            r[3] -= delta
        elif event.keysym == "Down":
            r[1] += delta
            r[3] += delta
        elif event.keysym == "Left":
            r[0] -= delta
            r[2] -= delta
        elif event.keysym == "Right":
            r[0] += delta
            r[2] += delta
        else:
            return

        # 重绘
        if key in self._result_rects:
            self.canvas.delete(self._result_rects[key])
        if key in self._result_labels:
            self.canvas.delete(self._result_labels[key])
        self._draw_result_rect(key, r[0], r[1], r[2], r[3], step["color"])
        self._update_coord_display(None, None)

    def _confirm_step(self, event=None):
        """确认当前步骤，进入下一步"""
        step = CALIBRATION_STEPS[self._step_index]
        key = step["key"]

        if key not in self._results:
            # 特殊处理 window_offset：没点过直接设为 [0, 0]
            if key == "window_offset":
                self._results[key] = [0, 0]
                self._draw_result_point(key, 0, 0, step["color"])
                self._update_coord_display(None, None)
            else:
                # 还没标定，提示
                self.canvas.itemconfig(
                    self._info_coords,
                    text="⚠ 请先框选区域再按 Enter！",
                    fill="#ff4444",
                )
                self.root.after(1500, lambda: self._update_coord_display(None, None))
                return

        # 安全检查：window_offset 如果过大，警告
        if key == "window_offset":
            ox, oy = self._results[key]
            if abs(ox) > 200 or abs(oy) > 200:
                if not self._force_offset:
                    self.canvas.itemconfig(
                        self._info_coords,
                        text=f"⚠ 偏移 ({ox},{oy}) 过大！全屏游戏应设为 [0,0]。再次按 Enter 强制确认，Backspace 重设",
                        fill="#ff4444",
                    )
                    self._force_offset = True
                    return
                # 用户二次确认，强制通过
                self._force_offset = False
            elif abs(ox) == 0 and abs(oy) == 0:
                self.canvas.itemconfig(
                    self._info_coords,
                    text="✅ 全屏模式，窗口偏移设为 [0, 0]",
                    fill="#44ff44",
                )

        self._step_index += 1
        self._current_rect = None
        self._update_display()

    def _skip_step(self, event=None):
        """跳过当前步骤"""
        self._step_index += 1
        self._current_rect = None
        self._update_display()

    def _clear_current(self):
        """清除当前步骤的框选"""
        step = CALIBRATION_STEPS[self._step_index]
        key = step["key"]
        if key in self._results:
            del self._results[key]
        if key in self._result_rects:
            self.canvas.delete(self._result_rects[key])
            del self._result_rects[key]
        if key in self._result_labels:
            self.canvas.delete(self._result_labels[key])
            del self._result_labels[key]
        if self._current_rect:
            self.canvas.delete(self._current_rect)
            self._current_rect = None
        self._update_coord_display(None, None)

    # ========== 完成 / 保存 ==========

    def _show_summary(self):
        """全部标定完成，显示摘要"""
        # 清空画布重绘总结
        self.canvas.delete("all")

        # 标题
        self.canvas.create_text(
            self.screen_w // 2, 60,
            text="标定完成！",
            fill="#44ff44",
            font=("Microsoft YaHei UI", 28, "bold"),
            anchor=tk.CENTER,
        )

        y = 120
        for step in CALIBRATION_STEPS:
            key = step["key"]
            if key in self._results:
                val = self._results[key]
                if step["is_point"]:
                    text = f"✅ {step['name']}: ({val[0]}, {val[1]})"
                else:
                    text = f"✅ {step['name']}: [{val[0]}, {val[1]}, {val[2]}, {val[3]}]  ({val[2]-val[0]}×{val[3]-val[1]})"
                color = step["color"]
            else:
                text = f"⏭ {step['name']}: 已跳过"
                color = "#666666"

            self.canvas.create_text(
                self.screen_w // 2, y,
                text=text,
                fill=color,
                font=("Consolas", 12),
                anchor=tk.CENTER,
            )
            y += 30

        # 保存按钮
        btn_x = self.screen_w // 2
        btn_y = y + 40
        self.canvas.create_rectangle(
            btn_x - 100, btn_y - 20, btn_x + 100, btn_y + 20,
            fill="#44ff44", outline="#22cc22", width=2,
        )
        self.canvas.create_text(
            btn_x, btn_y,
            text="保存到 config.yaml",
            fill="#000000",
            font=("Microsoft YaHei UI", 14, "bold"),
            anchor=tk.CENTER,
        )
        self.canvas.tag_bind(
            self.canvas.create_rectangle(btn_x - 100, btn_y - 20, btn_x + 100, btn_y + 20,
                                          fill="", outline="", width=0),
            "<Button-1>", lambda e: self._save_and_exit(),
        )

        # 不保存直接退出
        self.canvas.create_text(
            btn_x, btn_y + 50,
            text="按 Q 退出（不保存）",
            fill="#888888",
            font=("Microsoft YaHei UI", 10),
            anchor=tk.CENTER,
        )
        # 也响应按钮点击
        self.canvas.create_rectangle(
            btn_x - 90, btn_y + 35, btn_x + 90, btn_y + 55,
            fill="", outline="", width=0,
            tags=("save_btn",)
        )
        self.canvas.tag_bind("save_btn", "<Button-1>", lambda e: self._save_and_exit())

    def _save_and_exit(self):
        """保存到 config.yaml 并退出"""
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        except Exception:
            cfg = {}

        # 写入 window_offset
        if "window_offset" in self._results:
            cfg["window_offset"] = self._results["window_offset"]

        # 写入 focus_point
        if "focus_point" in self._results:
            cfg["focus_point"] = self._results["focus_point"]

        # 写入 chat_area
        if "chat_area" in self._results:
            if "chat_area" not in cfg:
                cfg["chat_area"] = {}
            cfg["chat_area"]["region"] = self._results["chat_area"]

        # 写入 chat_input
        if "chat_input" in self._results:
            if "chat_input" not in cfg:
                cfg["chat_input"] = {}
            cfg["chat_input"]["region"] = self._results["chat_input"]

        # 写入 back_button
        if "back_button" in self._results:
            if "back_button" not in cfg:
                cfg["back_button"] = {}
            cfg["back_button"]["region"] = self._results["back_button"]

        # 写入 friend_ui
        friend_keys = {
            "friend_light_candle": "light_candle_region",
            "friend_add": "add_friend_region",
            "friend_name_input": "name_input_region",
            "friend_confirm": "confirm_region",
            "friend_cancel": "cancel_region",
        }
        for calib_key, cfg_key in friend_keys.items():
            if calib_key in self._results:
                if "friend_ui" not in cfg:
                    cfg["friend_ui"] = {}
                cfg["friend_ui"][cfg_key] = self._results[calib_key]

        # 写回文件
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=None, sort_keys=False)

        logger.info(f"标定结果已保存到 {config_path}")
        self._quit()

    def _quit(self):
        """退出"""
        self.root.destroy()
        sys.exit(0)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    Calibrator().run()
