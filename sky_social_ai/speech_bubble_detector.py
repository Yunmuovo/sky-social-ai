"""
对话气泡检测模块 - 识别人物头顶上的聊天气泡

光遇中别人说的话会先显示在人物头顶的气泡里，
点击气泡后才会在左侧展开完整对话框。
本模块通过模板匹配+颜色检测寻找头顶气泡。
"""

import cv2
import numpy as np
import os
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SpeechBubbleResult:
    """对话气泡检测结果"""
    found: bool
    center: Tuple[int, int]
    confidence: float
    rect: Tuple[int, int, int, int]
    is_self: bool = False  # 是否是自己发的气泡


class SpeechBubbleDetector:
    """人物头顶对话气泡检测器"""

    def __init__(
        self,
        template_path: str = "templates/speech_bubble.png",
        threshold: float = 0.75,
        search_region: Optional[Tuple[int, int, int, int]] = None,
    ):
        """
        Args:
            template_path: 气泡模板图片路径（默认头顶白色气泡）
            threshold: 匹配阈值
            search_region: 可选搜索区域，默认全屏
        """
        self.template_path = template_path
        self.threshold = threshold
        self.search_region = search_region
        self.template = self._load_template()

    def _load_template(self) -> Optional[np.ndarray]:
        if not os.path.exists(self.template_path):
            logger.warning(f"气泡模板不存在: {self.template_path}")
            return None
        img = cv2.imread(self.template_path)
        if img is None:
            logger.warning(f"无法读取气泡模板: {self.template_path}")
        return img

    def detect(self, screen: np.ndarray) -> Optional[SpeechBubbleResult]:
        """
        检测屏幕中人物头顶的对话气泡

        Returns:
            返回最可能的一个气泡（暂只支持单个对话对象）
        """
        if self.template is None:
            return None

        # 裁剪搜索区域
        if self.search_region:
            left, top, right, bottom = self.search_region
            left = max(0, left)
            top = max(0, top)
            right = min(screen.shape[1], right)
            bottom = min(screen.shape[0], bottom)
            search_img = screen[top:bottom, left:right]
            offset_x, offset_y = left, top
        else:
            search_img = screen
            offset_x, offset_y = 0, 0

        screen_gray = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)

        best_result = None
        best_conf = 0.0

        scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6]
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
                best_result = SpeechBubbleResult(
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
            logger.info(f"检测到头顶气泡: 置信度={best_conf:.3f}")
            return best_result

        logger.debug(f"未检测到头顶气泡 (最高置信度={best_conf:.3f})")
        return None

    def detect_all(self, screen: np.ndarray) -> List[SpeechBubbleResult]:
        """
        检测所有可见的对话气泡
        """
        if self.template is None:
            return []

        if self.search_region:
            left, top, right, bottom = self.search_region
            left = max(0, left)
            top = max(0, top)
            right = min(screen.shape[1], right)
            bottom = min(screen.shape[0], bottom)
            search_img = screen[top:bottom, left:right]
            offset_x, offset_y = left, top
        else:
            search_img = screen
            offset_x, offset_y = 0, 0

        screen_gray = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)

        results = []
        scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6]
        for scale in scales:
            scaled_w = int(template_gray.shape[1] * scale)
            scaled_h = int(template_gray.shape[0] * scale)
            if (scaled_w > screen_gray.shape[1] or
                    scaled_h > screen_gray.shape[0] or
                    scaled_w < 10 or scaled_h < 10):
                continue

            scaled = cv2.resize(template_gray, (scaled_w, scaled_h))
            res = cv2.matchTemplate(screen_gray, scaled, cv2.TM_CCOEFF_NORMED)

            # 找到所有超过阈值的位置
            loc = np.where(res >= self.threshold)
            for y, x in zip(*loc):
                conf = res[y, x]
                center_x = offset_x + x + scaled_w // 2
                center_y = offset_y + y + scaled_h // 2
                results.append(SpeechBubbleResult(
                    found=True,
                    center=(center_x, center_y),
                    confidence=float(conf),
                    rect=(
                        offset_x + x,
                        offset_y + y,
                        offset_x + x + scaled_w,
                        offset_y + y + scaled_h,
                    ),
                ))

        # 简单按置信度排序，去重（重叠区域只保留一个）
        results.sort(key=lambda r: r.confidence, reverse=True)
        unique = []
        for r in results:
            too_close = False
            for u in unique:
                dx = r.center[0] - u.center[0]
                dy = r.center[1] - u.center[1]
                if np.sqrt(dx * dx + dy * dy) < 50:
                    too_close = True
                    break
            if not too_close:
                unique.append(r)

        return unique
