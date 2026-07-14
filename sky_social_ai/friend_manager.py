"""
好友管理模块 - 处理光遇好友相关操作

操作流程：靠近 → 点亮 → 加好友 → 命名 → 确认
"""

import time
import logging
from typing import Tuple, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FriendManager:
    """好友操作管理器"""

    def __init__(self, input_simulator, config: dict):
        self.input = input_simulator
        self.cfg = config.get("friend_ui", {})
        self.click_interval = config.get("actions", {}).get("click_interval", 0.5)

    def light_candle(self):
        """
        点亮蜡烛（打招呼）
        流程：靠近其他玩家 → 点击"点亮"按钮
        """
        region = self.cfg.get("light_candle_region", [750, 550, 1170, 620])
        center = self._region_center(region)
        logger.info(f"点亮蜡烛: 点击 ({center[0]}, {center[1]})")
        self.input.click(*center)
        time.sleep(self.click_interval)

    def add_friend(self, name: Optional[str] = None):
        """
        添加好友
        流程：点击添加好友 → 输入名字 → 确认

        Args:
            name: 好友名称，None则使用默认名
        """
        # 步骤1: 点击"添加好友"
        add_region = self.cfg.get("add_friend_region", [750, 640, 1170, 710])
        add_center = self._region_center(add_region)
        logger.info(f"添加好友: 点击添加按钮 ({add_center[0]}, {add_center[1]})")
        self.input.click(*add_center)
        time.sleep(self.click_interval * 2)

        # 步骤2: 点击命名输入框并输入名字
        if name is None:
            from datetime import datetime
            name = f"旅人{datetime.now().strftime('%H%M')}"

        name_region = self.cfg.get("name_input_region", [600, 480, 1320, 560])
        name_center = self._region_center(name_region)
        logger.info(f"输入好友名称: {name}")
        self.input.click(*name_center)
        time.sleep(self.click_interval)

        # 清空可能存在的默认文本 (Ctrl+A + Backspace)
        self.input.press_key("ctrl")
        self.input.press_key("a")
        time.sleep(0.1)
        self.input.press_key("backspace")
        time.sleep(0.1)

        self.input.type_text(name, interval=0.05)
        time.sleep(self.click_interval)

        # 步骤3: 点击确认
        confirm_region = self.cfg.get("confirm_region", [750, 700, 1170, 760])
        confirm_center = self._region_center(confirm_region)
        logger.info(f"确认添加好友: ({confirm_center[0]}, {confirm_center[1]})")
        self.input.click(*confirm_center)
        time.sleep(self.click_interval)

        logger.info(f"好友添加完成: {name}")
        return name

    def cancel_action(self):
        """取消当前操作"""
        cancel_region = self.cfg.get("cancel_region", [750, 770, 1170, 830])
        cancel_center = self._region_center(cancel_region)
        self.input.click(*cancel_center)
        logger.info("取消操作")

    def send_friend_request(self, name: Optional[str] = None):
        """
        完整的加好友流程（点亮 + 添加）

        先尝试点亮，等待一下，再添加好友。
        在某些情况下可能不需要点亮步骤。
        """
        logger.info("开始完整加好友流程...")
        self.light_candle()
        time.sleep(self.click_interval * 3)  # 等待动画
        return self.add_friend(name)

    @staticmethod
    def _region_center(region: list) -> Tuple[int, int]:
        """计算区域中心坐标"""
        x = (region[0] + region[2]) // 2
        y = (region[1] + region[3]) // 2
        return x, y
