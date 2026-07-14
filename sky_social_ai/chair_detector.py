"""
椅子检测模块 - 使用 OpenCV 模板匹配

检测屏幕中的椅子，返回椅子位置，支持点击坐下
"""

import cv2
import numpy as np
import os
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """椅子检测结果"""
    found: bool
    center: Tuple[int, int]  # 椅子中心坐标 (x, y)
    confidence: float        # 匹配置信度
    template_name: str       # 匹配到的模板名称
    rect: Tuple[int, int, int, int]  # 检测区域 (left, top, right, bottom)


class ChairDetector:
    """椅子检测器 - 多模板匹配"""

    def __init__(
        self,
        template_dir: str = "templates",
        template_names: Optional[List[str]] = None,
        threshold: float = 0.75,
    ):
        self.template_dir = template_dir
        self.threshold = threshold
        self.templates: dict[str, np.ndarray] = {}

        # 加载模板
        if template_names:
            for name in template_names:
                self._load_template(name)

        logger.info(f"已加载 {len(self.templates)} 个椅子模板，阈值={threshold}")

    def _load_template(self, name: str):
        """加载单个模板图片"""
        path = os.path.join(self.template_dir, name)
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                self.templates[name] = img
                logger.debug(f"加载模板: {name} ({img.shape[1]}x{img.shape[0]})")
            else:
                logger.warning(f"无法读取模板图片: {path}")
        else:
            logger.warning(f"模板文件不存在: {path} (请将椅子截图放入此目录)")

    def add_template(self, name: str, image: np.ndarray):
        """动态添加模板"""
        self.templates[name] = image
        logger.info(f"动态添加模板: {name}")

    def detect(self, screen: np.ndarray) -> Optional[DetectionResult]:
        """
        在屏幕中检测椅子

        Args:
            screen: 截屏图像 (BGR格式)

        Returns:
            检测结果，未找到返回 None
        """
        best_result = None
        best_confidence = 0.0

        screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        for name, template in self.templates.items():
            # 如果模板比屏幕大，跳过
            if (template.shape[0] > screen.shape[0] or
                    template.shape[1] > screen.shape[1]):
                logger.debug(f"模板 {name} 比屏幕大，跳过")
                continue

            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            # 多尺度匹配（适配不同分辨率）
            results = self._multi_scale_match(screen_gray, template_gray, name)

            for result in results:
                if result.confidence > best_confidence:
                    best_confidence = result.confidence
                    best_result = result

        if best_result and best_confidence >= self.threshold:
            logger.info(
                f"检测到椅子: {best_result.template_name} "
                f"位置={best_result.center} 置信度={best_confidence:.3f}"
            )
            return best_result
        else:
            logger.debug(f"未检测到椅子 (最高置信度={best_confidence:.3f})")
            return None

    def _multi_scale_match(
        self, screen_gray: np.ndarray, template_gray: np.ndarray, name: str
    ) -> List[DetectionResult]:
        """多尺度模板匹配"""
        results = []
        scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]

        for scale in scales:
            scaled_w = int(template_gray.shape[1] * scale)
            scaled_h = int(template_gray.shape[0] * scale)

            if (scaled_w > screen_gray.shape[1] or
                    scaled_h > screen_gray.shape[0] or
                    scaled_w < 10 or scaled_h < 10):
                continue

            scaled_template = cv2.resize(template_gray, (scaled_w, scaled_h))
            res = cv2.matchTemplate(
                screen_gray, scaled_template, cv2.TM_CCOEFF_NORMED
            )
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_val >= self.threshold:
                center_x = max_loc[0] + scaled_w // 2
                center_y = max_loc[1] + scaled_h // 2
                results.append(DetectionResult(
                    found=True,
                    center=(center_x, center_y),
                    confidence=max_val,
                    template_name=name,
                    rect=(
                        max_loc[0], max_loc[1],
                        max_loc[0] + scaled_w, max_loc[1] + scaled_h,
                    ),
                ))

        return results

    def detect_all(self, screen: np.ndarray) -> List[DetectionResult]:
        """检测所有椅子（不限于最佳匹配）"""
        all_results = []
        screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        for name, template in self.templates.items():
            if (template.shape[0] > screen.shape[0] or
                    template.shape[1] > screen.shape[1]):
                continue

            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            results = self._multi_scale_match(screen_gray, template_gray, name)
            all_results.extend(results)

        # 按置信度排序
        all_results.sort(key=lambda r: r.confidence, reverse=True)

        # NMS 去重
        return self._nms(all_results, iou_threshold=0.3)

    def _nms(
        self, results: List[DetectionResult], iou_threshold: float = 0.3
    ) -> List[DetectionResult]:
        """非极大值抑制，去除重复检测"""
        if len(results) <= 1:
            return results

        keep = []
        while results:
            best = results.pop(0)
            keep.append(best)
            results = [
                r for r in results
                if self._iou(best.rect, r.rect) < iou_threshold
            ]
        return keep

    @staticmethod
    def _iou(
        rect1: Tuple[int, int, int, int],
        rect2: Tuple[int, int, int, int]
    ) -> float:
        """计算两个矩形的 IoU"""
        x1 = max(rect1[0], rect2[0])
        y1 = max(rect1[1], rect2[1])
        x2 = min(rect1[2], rect2[2])
        y2 = min(rect1[3], rect2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
        area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0
