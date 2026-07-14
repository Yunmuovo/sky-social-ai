"""
屏幕截图模块 - 支持 PC 和 Android 平台

PC端: 使用 mss (快速截屏)
Android端(后续): 使用 adb screencap
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import time
import logging

logger = logging.getLogger(__name__)


class ScreenCapture(ABC):
    """屏幕截图抽象基类，PC/Android 各自实现"""

    @abstractmethod
    def capture(self) -> np.ndarray:
        """截取全屏，返回 BGR 格式的 numpy 数组"""
        ...

    @abstractmethod
    def capture_region(self, region: Tuple[int, int, int, int]) -> np.ndarray:
        """截取指定区域 (left, top, right, bottom)"""
        ...

    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """返回当前分辨率 (width, height)"""
        ...


class PCCapture(ScreenCapture):
    """PC 端屏幕截图 (Windows)"""

    def __init__(self, monitor_index: int = 0):
        try:
            import mss
            self.mss = mss.mss
        except ImportError:
            raise ImportError("请安装 mss: pip install mss")

        self.sct = self.mss()
        self.monitor = self.sct.monitors[monitor_index]
        # monitor 格式: {"left": 0, "top": 0, "width": 1920, "height": 1080}
        self._width = self.monitor["width"]
        self._height = self.monitor["height"]
        logger.info(f"PC屏幕: {self._width}x{self._height}")

    def capture(self) -> np.ndarray:
        screenshot = self.sct.grab(self.monitor)
        img = np.array(screenshot)
        # mss 返回 BGRA，转 BGR
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def capture_region(self, region: Tuple[int, int, int, int]) -> np.ndarray:
        left, top, right, bottom = region
        monitor = {
            "left": left,
            "top": top,
            "width": right - left,
            "height": bottom - top,
        }
        screenshot = self.sct.grab(monitor)
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def get_resolution(self) -> Tuple[int, int]:
        return self._width, self._height


class AndroidCapture(ScreenCapture):
    """Android 端屏幕截图 (通过 adb)"""

    def __init__(self):
        import subprocess
        # 检查 adb 是否可用
        try:
            subprocess.run(["adb", "version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("adb 不可用，请安装 Android SDK Platform Tools")

        self._width, self._height = self._get_resolution_adb()
        logger.info(f"Android屏幕: {self._width}x{self._height}")

    def _get_resolution_adb(self) -> Tuple[int, int]:
        import subprocess
        import re
        result = subprocess.run(
            ["adb", "shell", "wm", "size"],
            capture_output=True, text=True, check=True
        )
        # 输出格式: "Physical size: 1080x2400"
        match = re.search(r"(\d+)x(\d+)", result.stdout)
        if match:
            return int(match.group(1)), int(match.group(2))
        return 1080, 1920  # 默认

    def capture(self) -> np.ndarray:
        import subprocess
        import os
        import tempfile

        tmp_path = os.path.join(tempfile.gettempdir(), "sky_screenshot.png")
        # 截图到手机
        subprocess.run(
            ["adb", "shell", "screencap", "-p", "/sdcard/sky_screenshot.png"],
            capture_output=True, check=True
        )
        # 拉取到电脑
        subprocess.run(
            ["adb", "pull", "/sdcard/sky_screenshot.png", tmp_path],
            capture_output=True, check=True
        )
        img = cv2.imread(tmp_path)
        os.remove(tmp_path)
        return img

    def capture_region(self, region: Tuple[int, int, int, int]) -> np.ndarray:
        full = self.capture()
        left, top, right, bottom = region
        return full[top:bottom, left:right]

    def get_resolution(self) -> Tuple[int, int]:
        return self._width, self._height


def create_capture(platform: str = "pc") -> ScreenCapture:
    """工厂函数：根据平台创建截图实例"""
    if platform == "pc":
        return PCCapture()
    elif platform == "android":
        return AndroidCapture()
    else:
        raise ValueError(f"不支持的平台: {platform}")
