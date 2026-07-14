"""
聊天按钮检测模块 - 识别左下角聊天按钮

光遇中点击左下角的气泡按钮才会弹出输入框，
本模块通过模板匹配检测该按钮。
"""

import cv2
import numpy as np
import os
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChatButtonResult:
    """聊天按钮检测结果"""
    found: bool
    center: Tuple[int, int]
    confidence: float
    rect: Tuple[int, int, int, int]


class ChatButtonDetector:
    """左下角聊天按钮检测器"""

    def __init__(
        self,
        template_path: str = "templates/chat_button.png",
        threshold: float = 0.75,
        search_region: Optional[Tuple[int, int, int, int]] = None,
    ):
        """
        Args:
            template_path: 聊天按钮模板图片路径
            threshold: 匹配阈值
            search_region: 可选，只在屏幕左下角区域搜索 (left, top, right, bottom)
        """
        self.template_path = template_path
        self.threshold = threshold
        self.search_region = search_region
        self.template = self._load_template()

    def _load_template(self) -> Optional[np.ndarray]:
        if not os.path.exists(self.template_path):
            logger.warning(f"聊天按钮模板不存在: {self.template_path}")
            return None
        img = cv2.imread(self.template_path)
        if img is None:
            logger.warning(f"无法读取聊天按钮模板: {self.template_path}")
        return img

    def detect(self, screen: np.ndarray) -> Optional[ChatButtonResult]:
        """
        检测屏幕左下角的聊天按钮
        """
        if self.template is None:
            return None

        # 裁剪搜索区域（左下角）
        if self.search_region:
            left, top, right, bottom = self.search_region
            left = max(0, left)
            top = max(0, top)
            right = min(screen.shape[1], right)
            bottom = min(screen.shape[0], bottom)
            search_img = screen[top:bottom, left:right]
            offset_x, offset_y = left, top
        else:
            h, w = screen.shape[:2]
            # 默认只搜索左下角 1/4 区域
            search_img = screen[h * 3 // 4:, :w // 3]
            offset_x, offset_y = 0, h * 3 // 4

        if search_img.size == 0:
            return None

        screen_gray = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)

        best_result = None
        best_conf = 0.0

        # 多尺度匹配
        scales = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]
        for scale in scales:
            scaled_w = int(template_gray.shape[1] * scale)
            scaled_h = int(template_gray.shape[0] * scale)
            if (scaled_w > screen_gray.shape[1] or
                    scaled_h > screen_gray.shape[0] or
                    scaled_w < 10 or scaled_h < 10):
                continue

            scaled = cv2.resize(template_gray, (scaled_w, scaled_h))
            res = cv2.matchTemplate(screen_gray, scaled, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > best_conf:
                best_conf = max_val
                center_x = offset_x + max_loc[0] + scaled_w // 2
                center_y = offset_y + max_loc[1] + scaled_h // 2
                best_result = ChatButtonResult(
                    found=True,
                    center=(center_x, center_y),
                    confidence=max_val,
                    rect=(
                        offset_x + max_loc[0],
                        offset_y + max_loc[1],
                        offset_x + max_loc[0] + scaled_w,
                        offset_y + max_loc[1] + scaled_h,
                    ),
                )

        if best_result and best_conf >= self.threshold:
            logger.info(f"检测到聊天按钮: 置信度={best_conf:.3f}")
            return best_result

        logger.debug(f"未检测到聊天按钮 (最高置信度={best_conf:.3f})")
        return None
