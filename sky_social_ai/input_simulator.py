"""
输入模拟模块 - 支持 PC 和 Android 平台

PC端: Win32 SendInput API（驱动级，兼容 DirectInput 游戏）
Android端(后续): adb input 模拟触屏操作
"""

from abc import ABC, abstractmethod
from typing import Tuple
import time
import logging

logger = logging.getLogger(__name__)


class InputSimulator(ABC):
    """输入模拟抽象基类"""

    @abstractmethod
    def click(self, x: int, y: int, button: str = "left"):
        """点击指定坐标"""
        ...

    @abstractmethod
    def double_click(self, x: int, y: int):
        """双击"""
        ...

    @abstractmethod
    def long_press(self, x: int, y: int, duration: float = 1.0):
        """长按"""
        ...

    @abstractmethod
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.5):
        """滑动"""
        ...

    @abstractmethod
    def type_text(self, text: str, interval: float = 0.05):
        """键入文字"""
        ...

    @abstractmethod
    def press_key(self, key: str):
        """按下按键"""
        ...

    @abstractmethod
    def move_to(self, x: int, y: int, duration: float = 0.2):
        """移动鼠标到指定位置"""
        ...

    @abstractmethod
    def hold_key(self, key: str):
        """按住按键不松开"""
        ...

    @abstractmethod
    def release_key(self, key: str):
        """松开按键"""
        ...

    def click_raw(self, x: int, y: int, button: str = "left"):
        """点击屏幕绝对坐标（不叠加窗口偏移），默认等同于 click"""
        self.click(x, y, button=button)


class PCInput(InputSimulator):
    """PC 端输入模拟
    - 鼠标: SendInput (ctypes)
    - 键盘: pydirectinput（专为 DirectInput 游戏设计）
    """

    # ========== Win32 常量（鼠标用）==========
    _INPUT_MOUSE = 0
    _MOUSEEVENTF_LEFTDOWN = 0x0002
    _MOUSEEVENTF_LEFTUP = 0x0004
    _MOUSEEVENTF_RIGHTDOWN = 0x0008
    _MOUSEEVENTF_RIGHTUP = 0x0010
    _MOUSEEVENTF_MOVE = 0x0001

    def __init__(self, offset: Tuple[int, int] = (0, 0)):
        try:
            import ctypes
            from ctypes import wintypes
            self.ct = ctypes
            self.wt = wintypes
        except ImportError:
            raise ImportError("ctypes 不可用")

        try:
            import pyperclip
            self.pyperclip = pyperclip
        except ImportError:
            self.pyperclip = None
            logger.warning("pyperclip 未安装，中文输入不可用: pip install pyperclip")

        # pydirectinput: 专为 DirectInput 游戏设计的键盘输入
        try:
            import pydirectinput
            self.pydi = pydirectinput
            # 关闭 fail-safe（不需要鼠标移到角落触发异常）
            self.pydi.FAILSAFE = False
        except ImportError:
            self.pydi = None
            logger.error("pydirectinput 未安装！请执行: pip install pydirectinput")

        # 构建 INPUT 结构体（仅鼠标 SendInput 用）
        class _MOUSEINPUT(self.ct.Structure):
            _fields_ = [
                ("dx", self.wt.LONG),
                ("dy", self.wt.LONG),
                ("mouseData", self.wt.DWORD),
                ("dwFlags", self.wt.DWORD),
                ("time", self.wt.DWORD),
                ("dwExtraInfo", self.ct.c_ulong),
            ]

        class _DUMMY(self.ct.Structure):
            _fields_ = [("dummy", self.ct.c_ulonglong * 2)]

        class _INPUT_UNION(self.ct.Union):
            _fields_ = [("mi", _MOUSEINPUT), ("dummy", _DUMMY)]

        class _INPUT(self.ct.Structure):
            _fields_ = [("type", self.wt.DWORD), ("union", _INPUT_UNION)]

        self._INPUT = _INPUT

        # 窗口偏移
        self.offset_x, self.offset_y = offset

        logger.info("PC输入模拟器就绪 (鼠标:SendInput, 键盘:pydirectinput)")

    def _apply_offset(self, x: int, y: int) -> Tuple[int, int]:
        return x + self.offset_x, y + self.offset_y

    # ========== 鼠标操作 (SendInput) ==========

    def _send_mouse_input(self, dx: int, dy: int, flags: int, mouse_data: int = 0):
        inp = self._INPUT()
        inp.type = self._INPUT_MOUSE
        inp.union.mi.dx = dx
        inp.union.mi.dy = dy
        inp.union.mi.mouseData = mouse_data
        inp.union.mi.dwFlags = flags
        inp.union.mi.time = 0
        inp.union.mi.dwExtraInfo = 0
        self.ct.windll.user32.SendInput(1, self.ct.byref(inp), self.ct.sizeof(inp))

    def move_to(self, x: int, y: int, duration: float = 0.2):
        tx, ty = self._apply_offset(x, y)
        self.ct.windll.user32.SetCursorPos(tx, ty)
        if duration > 0:
            time.sleep(duration * 0.3)

    def click(self, x: int, y: int, button: str = "left"):
        tx, ty = self._apply_offset(x, y)
        self.ct.windll.user32.SetCursorPos(tx, ty)
        time.sleep(0.03)
        if button == "right":
            self._send_mouse_input(0, 0, self._MOUSEEVENTF_RIGHTDOWN)
            time.sleep(0.04)
            self._send_mouse_input(0, 0, self._MOUSEEVENTF_RIGHTUP)
        else:
            self._send_mouse_input(0, 0, self._MOUSEEVENTF_LEFTDOWN)
            time.sleep(0.04)
            self._send_mouse_input(0, 0, self._MOUSEEVENTF_LEFTUP)
        time.sleep(0.03)
        logger.debug(f"点击 ({tx}, {ty})")

    def click_raw(self, x: int, y: int, button: str = "left"):
        """点击屏幕绝对坐标，不叠加窗口偏移"""
        self.ct.windll.user32.SetCursorPos(x, y)
        time.sleep(0.03)
        if button == "right":
            self._send_mouse_input(0, 0, self._MOUSEEVENTF_RIGHTDOWN)
            time.sleep(0.04)
            self._send_mouse_input(0, 0, self._MOUSEEVENTF_RIGHTUP)
        else:
            self._send_mouse_input(0, 0, self._MOUSEEVENTF_LEFTDOWN)
            time.sleep(0.04)
            self._send_mouse_input(0, 0, self._MOUSEEVENTF_LEFTUP)
        time.sleep(0.03)
        logger.debug(f"点击(原始) ({x}, {y})")

    def double_click(self, x: int, y: int):
        tx, ty = self._apply_offset(x, y)
        self.click(x, y)
        time.sleep(0.08)
        self.click(x, y)
        logger.debug(f"双击 ({tx}, {ty})")

    def long_press(self, x: int, y: int, duration: float = 1.0):
        tx, ty = self._apply_offset(x, y)
        self.ct.windll.user32.SetCursorPos(tx, ty)
        time.sleep(0.03)
        self._send_mouse_input(0, 0, self._MOUSEEVENTF_LEFTDOWN)
        time.sleep(duration)
        self._send_mouse_input(0, 0, self._MOUSEEVENTF_LEFTUP)
        logger.debug(f"长按 ({tx}, {ty}) {duration}s")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.5):
        tx1, ty1 = self._apply_offset(x1, y1)
        tx2, ty2 = self._apply_offset(x2, y2)
        self.ct.windll.user32.SetCursorPos(tx1, ty1)
        time.sleep(0.02)
        self.ct.windll.user32.mouse_event(self._MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        steps = max(5, int(duration * 20))
        dx = (tx2 - tx1) / steps
        dy = (ty2 - ty1) / steps
        step_t = duration / steps
        for i in range(steps):
            cur_x = int(tx1 + dx * (i + 1))
            cur_y = int(ty1 + dy * (i + 1))
            self.ct.windll.user32.SetCursorPos(cur_x, cur_y)
            time.sleep(step_t)
        self.ct.windll.user32.mouse_event(self._MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        logger.debug(f"滑动 ({tx1},{ty1}) -> ({tx2},{ty2})")

    # ========== 键盘操作 (pydirectinput) ==========

    def press_key(self, key: str):
        """按下一个键"""
        if self.pydi is None:
            logger.error("pydirectinput 未安装")
            return
        self.pydi.press(key)
        logger.debug(f"按键: {key}")

    def hold_key(self, key: str):
        """按住不松开"""
        if self.pydi is None:
            return
        self.pydi.keyDown(key)
        logger.debug(f"按住: {key}")

    def release_key(self, key: str):
        """松开"""
        if self.pydi is None:
            return
        self.pydi.keyUp(key)
        logger.debug(f"松开: {key}")

    def press_hotkey(self, *keys: str):
        """组合键（如 Ctrl+V）"""
        if self.pydi is None:
            logger.error("pydirectinput 未安装")
            return
        # pydirectinput 新版本没有 hotkey，手动实现
        for k in keys[:-1]:
            self.pydi.keyDown(k)
            time.sleep(0.04)
        self.pydi.press(keys[-1])
        time.sleep(0.05)
        for k in reversed(keys[:-1]):
            self.pydi.keyUp(k)
            time.sleep(0.03)
        logger.debug(f"组合键: {'+'.join(keys)}")

    def type_text(self, text: str, interval: float = 0.05):
        """键入文字，中文走剪贴板粘贴"""
        if not text:
            return

        has_non_ascii = any(ord(c) > 127 for c in text)

        if has_non_ascii and self.pyperclip is not None:
            original = ""
            try:
                original = self.pyperclip.paste()
            except Exception:
                original = ""

            try:
                self.pyperclip.copy("")
                time.sleep(0.02)
                self.pyperclip.copy(text)
                time.sleep(0.15)
                self.press_hotkey("ctrl", "v")
                time.sleep(0.10)
                logger.debug(f"粘贴: {text[:30]}...")
            finally:
                try:
                    if original:
                        self.pyperclip.copy(original)
                except Exception:
                    pass
        else:
            # 纯英文逐字输入
            for ch in text:
                if self.pydi:
                    self.pydi.write(ch, interval=interval)
                time.sleep(interval)
            logger.debug(f"键入: {text[:30]}...")


class AndroidInput(InputSimulator):
    """Android 端输入模拟 (基于 adb)"""

    def __init__(self):
        import subprocess
        try:
            subprocess.run(["adb", "version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("adb 不可用")
        logger.info("Android输入模拟器就绪")

    def click(self, x: int, y: int, button: str = "left"):
        self._adb(f"shell input tap {x} {y}")
        logger.debug(f"点击 ({x}, {y})")

    def double_click(self, x: int, y: int):
        self.click(x, y)
        time.sleep(0.1)
        self.click(x, y)
        logger.debug(f"双击 ({x}, {y})")

    def long_press(self, x: int, y: int, duration: float = 1.0):
        ms = int(duration * 1000)
        self._adb(f"shell input swipe {x} {y} {x} {y} {ms}")
        logger.debug(f"长按 ({x}, {y}) {duration}s")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.5):
        ms = int(duration * 1000)
        self._adb(f"shell input swipe {x1} {y1} {x2} {y2} {ms}")
        logger.debug(f"滑动 ({x1},{y1}) -> ({x2},{y2})")

    def type_text(self, text: str, interval: float = 0.05):
        # 替换特殊字符，adb input text 不支持空格和部分特殊字符
        text = text.replace(" ", "%s")
        self._adb(f'shell input text "{text}"')
        logger.debug(f"键入: {text[:30]}...")

    def press_key(self, key: str):
        self._adb(f"shell input keyevent {key}")
        logger.debug(f"按键: {key}")

    def move_to(self, x: int, y: int, duration: float = 0.2):
        # Android 不需要移动鼠标
        pass

    def hold_key(self, key: str):
        pass

    def release_key(self, key: str):
        pass

    def _adb(self, command: str):
        import subprocess
        subprocess.run(f"adb {command}", shell=True, check=True, capture_output=True)


def create_input(platform: str = "pc", offset: Tuple[int, int] = (0, 0)) -> InputSimulator:
    """工厂函数：根据平台创建输入模拟实例"""
    if platform == "pc":
        return PCInput(offset=offset)
    elif platform == "android":
        return AndroidInput()
    else:
        raise ValueError(f"不支持的平台: {platform}")
