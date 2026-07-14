"""
聊天OCR模块 - 识别光遇聊天区域的文字，结构化解析对话

优先 PaddleOCR，失败自动降级为盲聊模式
"""

import os
import datetime

# ===== 必须在 import paddle 之前设置 =====
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
# 加速 PaddleOCR 模型下载：使用 Hugging Face 国内镜像
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 跳过模型源连接检查（模型已在本地缓存）
if not os.environ.get("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"):
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# ===== 修复 paddle 污染 DLL 搜索路径导致 torch 加载 shm.dll 失败 =====
# paddle 加载后会干扰 DLL 搜索顺序，必须确保 torch 在 paddle 之前初始化
import torch  # noqa: F401

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import logging
import time
import hashlib
import difflib

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """单条结构化聊天消息"""
    content: str                # 消息内容（已去除发送者名）
    sender: str = ""            # 发送者名字（可能为空）
    raw_text: str = ""          # OCR 原始文本
    y: int = 0                  # 在截图中的 y 坐标（越大越靠下）
    is_new: bool = False        # 是否为本次扫描新出现的消息

    @property
    def display(self) -> str:
        """用于 LLM 上下文的展示格式"""
        if self.sender:
            return f"[{self.sender}] {self.content}"
        return f"[某人] {self.content}"


@dataclass
class ConversationSnapshot:
    """一次扫描产生的完整对话快照"""
    messages: List[ChatMessage] = field(default_factory=list)
    has_new: bool = False       # 是否有新消息
    sender_name: str = ""       # 最新消息的发送者（用于日志/GUI）

    @property
    def latest_message(self) -> Optional[str]:
        """最新一条消息的纯文本内容"""
        new_msgs = [m for m in self.messages if m.is_new]
        if new_msgs:
            return new_msgs[-1].content
        return None

    @property
    def conversation_text(self) -> str:
        """格式化为 LLM 可读的对话上下文"""
        if not self.messages:
            return ""
        lines = [m.display for m in self.messages]
        return "\n".join(lines)


class ChatOCR:
    """聊天文字识别器"""

    def __init__(self, config: dict):
        chat_cfg = config.get("chat_area", {})
        persona_cfg = config.get("persona", {})
        self.chat_region: Tuple[int, int, int, int] = tuple(chat_cfg.get(
            "region", [300, 650, 1620, 850]
        ))
        # AI 角色名字（用于过滤自己的消息）
        self._ai_name: str = persona_cfg.get("name", "小光")
        self.lang = chat_cfg.get("lang", "ch")
        # 只识别聊天区域底部 N 像素（最新消息通常在最下面）
        self.max_ocr_height = int(chat_cfg.get("max_ocr_height", 400))
        # 是否启用预处理：黑底白字 → 白底黑字
        self.preprocess = chat_cfg.get("preprocess", True)
        # 是否保存预处理后的调试图
        self.save_preprocessed = chat_cfg.get("save_preprocessed", False)
        self.preprocessed_dir = chat_cfg.get("preprocessed_dir", "debug_preprocessed")
        self._preprocessed_max_keep = 100  # 滚动保留数量
        self._preprocessed_cleanup_counter = 0  # 每 N 次保存才清理一次，避免频繁 IO
        if self.save_preprocessed:
            os.makedirs(self.preprocessed_dir, exist_ok=True)

        self.ocr = None
        self._ocr_engine = None        # "paddle" 或 "easyocr"
        self._ocr_ok = True            # OCR 当前是否健康
        self._last_init_attempt = 0.0   # 上次尝试初始化的时间
        self._init_retry_cooldown = 30  # 初始化失败后重试冷却（秒）
        self._init_fail_count = 0       # 连续初始化失败次数
        self._max_init_fails = 3        # 最大初始化失败次数，超过后永久放弃
        self._last_text = ""           # 上一轮识别的文字，用于去重
        self._last_sender_name = ""    # 上一轮消息的发送者名字
        self._last_timestamp = 0.0
        self._last_chat_hash = ""      # 上一帧聊天区域图像哈希，用于跳过无变化的 OCR
        self._region_debug_saved = False  # 区域可视化调试图是否已保存（只存一次）
        # 默认关闭左半边裁剪；颜色过滤（黑底气泡）才是通用方案
        self._left_only = chat_cfg.get("left_only", False)
        self._left_ratio = float(chat_cfg.get("left_ratio", 0.5))  # 左侧比例，默认 50%
        # ===== 对话历史跟踪 =====
        self._known_messages: Dict[str, ChatMessage] = {}  # key → 已见过的消息
        self._conversation_context: List[ChatMessage] = []  # 当前可见的完整对话
        # 纯内容哈希环：y 坐标无关，解决输入框弹出时聊天区滚动导致去重失效
        # 用 OrderedDict 实现确定性 FIFO（set 无序，裁切不可靠）
        from collections import OrderedDict
        self._seen_content_hashes: OrderedDict = OrderedDict()
        self._seen_ring_max = 500  # 最多记住 500 条消息的内容哈希

        # 延迟初始化 OCR（首次调用时加载）
        logger.info(f"ChatOCR 就绪，聊天区域: {self.chat_region}, 最大识别高度: {self.max_ocr_height}")


    def _init_ocr(self) -> bool:
        """延迟初始化 OCR，按优先级：PaddleOCR → EasyOCR → 放弃"""
        if self.ocr is not None:
            return True

        # 永久放弃：连续失败超过上限
        if self._init_fail_count >= self._max_init_fails:
            return False

        # 之前失败过，检查冷却时间
        if not self._ocr_ok:
            if time.time() - self._last_init_attempt < self._init_retry_cooldown:
                return False
            logger.info("🔄 重新尝试初始化 OCR...")

        self._last_init_attempt = time.time()

        # ---- 方案1: PaddleOCR ----
        try:
            from paddleocr import PaddleOCR
            import paddle
            # 强制 CPU 模式，避免 GPU 推理异常（could not execute a primitive）
            try:
                paddle.set_device('cpu')
            except Exception:
                pass
            self.ocr = PaddleOCR(
                lang=self.lang,
                use_gpu=False,
                use_angle_cls=False,          # 游戏截图文字都是水平方向，跳过角度分类省时间
                det_db_thresh=0.3,             # 降低检测阈值加速
                det_db_box_thresh=0.5,         # 框阈值
                rec_batch_num=6,               # 批量识别6行，减少推理调用次数
            )
            self._ocr_ok = True
            self._ocr_engine = "paddle"
            self._init_fail_count = 0
            logger.info("✅ PaddleOCR 初始化完成 (CPU模式)")
            # 预热
            try:
                _dummy = np.full((120, 300, 3), 255, dtype=np.uint8)
                self.ocr.ocr(_dummy)
                logger.debug("OCR 预热完成")
            except Exception as e:
                logger.debug(f"OCR 预热跳过: {e}")
            return True
        except ImportError:
            logger.warning("⚠️ PaddleOCR 未安装，尝试 EasyOCR 降级...")
        except Exception as e:
            logger.warning(f"⚠️ PaddleOCR 初始化异常: {e}，尝试 EasyOCR 降级...")

        # ---- 方案2: EasyOCR 降级 ----
        try:
            import easyocr
            lang_list = ["ch_sim", "en"] if self.lang == "ch" else ["en"]
            self.ocr = easyocr.Reader(
                lang_list, gpu=False,
                verbose=False,
            )
            self._ocr_ok = True
            self._ocr_engine = "easyocr"
            self._init_fail_count = 0
            logger.info("✅ EasyOCR (降级) 初始化完成")
            return True
        except ImportError:
            logger.warning("⚠️ EasyOCR 也未安装: pip install easyocr")
        except Exception as e:
            logger.warning(f"⚠️ EasyOCR 初始化异常: {e}")

        # ---- 全部失败 ----
        self._ocr_ok = False
        self._init_fail_count += 1
        if self._init_fail_count >= self._max_init_fails:
            logger.error(
                f"❌ OCR 初始化连续失败 {self._init_fail_count} 次，永久放弃。"
                f"请安装: pip install paddlepaddle paddleocr  或  pip install easyocr"
            )
        else:
            logger.warning(
                f"⚠️ OCR 初始化失败 ({self._init_fail_count}/{self._max_init_fails})，"
                f"{self._init_retry_cooldown}s 后重试"
            )
        return False

    def _filter_dark_bg_text(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        只保留黑底气泡区域（别人发的消息），屏蔽白底气泡（自己发的消息）。

        关键区别：
        - 别人消息：大面积黑底 + 白字 → 连通区域大 → 保留
        - 自己消息：白底 + 小黑字 → 文字笔画细，连通区域小 → 丢弃

        策略：模糊→阈值→膨胀→连通域面积过滤

        Returns:
            (过滤后的图像, 是否有黑底内容)
        """
        h, w = img_bgr.shape[:2]
        if h < 20 or w < 50:
            return img_bgr, False

        if len(img_bgr.shape) == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_bgr.copy()

        # 模糊获取背景色估计（核太大→边缘被模糊掉→右侧白字漏检）
        blur_ksize = max(9, min(w, h) // 10)
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

        # 自适应暗阈值：亮度环境越亮，阈值相应提高
        # 暗场景均值~40 → 阈值 60；亮场景均值~150 → 阈值 110
        mean_brightness = float(np.mean(gray))
        dark_threshold = int(np.clip(mean_brightness * 0.75, 50, 130))
        dark_mask = (blurred < dark_threshold)
        dark_u8 = dark_mask.astype(np.uint8) * 255

        if dark_u8.max() == 0:
            logger.debug(f"未检测到暗色像素(阈值={dark_threshold}, 亮度均值={mean_brightness:.0f})，无别人发言")
            return img_bgr, False

        # 膨胀：把黑底+白字连成一个大连通区域
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        dilated = cv2.dilate(dark_u8, dilate_kernel, iterations=2)

        # 连通域分析：按面积过滤
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            dilated, connectivity=8
        )
        # 面积阈值：至少占图像 2% 才算黑底气泡
        min_area = max(200, int(h * w * 0.015))

        keep_mask = np.zeros_like(dilated, dtype=np.uint8)
        kept_count = 0
        for i in range(1, num_labels):  # label 0 = 背景
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                keep_mask[labels == i] = 255
                kept_count += 1

        if keep_mask.max() == 0:
            logger.debug(f"所有暗区面积均<{min_area}px，判定为文字笔画面非气泡，跳过")
            return img_bgr, False

        # 生成结果：只保留大黑底气泡区域，其余涂白
        # 关键：白字可能超出黑底气泡边界，需要向四周扩展掩膜
        keep_bool = keep_mask > 0
        # 椭圆核膨胀：比矩形更贴近气泡形状，向上下左右都扩展
        pad_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 21))
        keep_mask_padded = cv2.dilate(keep_mask, pad_kernel, iterations=1)
        # 重点向右扩展：聊天内容/玩家名常在气泡最右侧，150px 水平膨胀兜底
        h_pad_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 1))
        keep_mask_padded = cv2.dilate(keep_mask_padded, h_pad_kernel, iterations=1)
        keep_bool = keep_mask_padded > 0

        result = img_bgr.copy()
        if len(result.shape) == 3:
            result[~keep_bool] = [255, 255, 255]
        else:
            result[~keep_bool] = 255

        logger.debug(f"检测到 {kept_count} 个黑底气泡区域（面积≥{min_area}），已过滤白底内容")
        return result, True


    def _preprocess_for_ocr(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        轻量预处理：只做对比度增强，不二值化、不锐化、不形态学操作。

        核心原则：保留所有文字细节，让 OCR 引擎自己处理。
        之前的 adaptiveThreshold + unsharp mask 会把灰度过渡区的
        细笔画（如名字末尾小字、半透明气泡边缘文字）直接毁掉。
        """
        if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_bgr

        h, w = gray.shape[:2]
        if h < 20 or w < 50:
            if len(gray.shape) == 2:
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return gray

        # 轻微中值滤波去噪（3x3 小块，保留笔画边缘）
        denoised = cv2.medianBlur(gray, 3)

        # 判断整体亮度，暗底翻转为亮底（OCR 对白底黑字效果更好）
        mean_brightness = float(np.mean(denoised))
        if mean_brightness < 100:
            # 整体偏暗 → 全局翻转：白字变黑字，黑底变白底
            result = 255 - denoised
        else:
            # 画面偏亮（有白底有黑底混合）→ 不翻转，直接增强
            result = denoised

        # CLAHE 局部对比度增强（保留灰度层次，不做二值化）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        result = clahe.apply(result)

        # 转 BGR 三通道（PaddleOCR 需要）
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        return result

    def _save_preprocessed_debug(self, img: np.ndarray, prefix: str = "ocr"):
        """保存预处理调试图，方便用户检查"""
        if not self.save_preprocessed:
            return
        try:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = os.path.join(self.preprocessed_dir, f"{prefix}_{ts}.png")
            cv2.imwrite(path, img)
            # 每 20 次保存清理一次，降低 IO 开销
            self._preprocessed_cleanup_counter += 1
            if self._preprocessed_cleanup_counter % 20 == 0:
                self._cleanup_preprocessed()
        except Exception as e:
            logger.debug(f"保存预处理调试图失败: {e}")

    def _cleanup_preprocessed(self):
        """滚动清理：只保留最近 N 张预处理图"""
        try:
            files = sorted(
                [f for f in os.listdir(self.preprocessed_dir) if f.endswith(".png")],
                key=lambda f: os.path.getmtime(os.path.join(self.preprocessed_dir, f)),
                reverse=True,
            )
            for old in files[self._preprocessed_max_keep:]:
                os.remove(os.path.join(self.preprocessed_dir, old))
        except Exception:
            pass

    def save_region_debug(self, screen: np.ndarray, force: bool = False):
        """
        在全屏截图上画出 chat_area 区域框，保存为调试图片。
        只在第一次调用时保存（除非 force=True），避免刷屏。
        
        - 红色粗线框 = 完整 chat_area 区域
        - 黄色虚线框 = 底部 max_ocr_height 实际识别区域
        """
        if self._region_debug_saved and not force:
            return
        if not self.save_preprocessed:
            return

        try:
            # 复制一份全屏截图，以免修改原始数据
            vis = screen.copy()
            left, top, right, bottom = self.chat_region
            h_full, w_full = vis.shape[:2]

            # 裁剪坐标到屏幕范围内
            l = max(0, left)
            t = max(0, top)
            r = min(w_full, right)
            b = min(h_full, bottom)

            # 红色粗线：完整 chat_area
            cv2.rectangle(vis, (l, t), (r, b), (0, 0, 255), 4)

            # 黄色虚线：底部 max_ocr_height 实际 OCR 识别区域
            crop_top = max(t, b - self.max_ocr_height)
            if crop_top > t:
                for y in range(crop_top, b, 8):
                    cv2.line(vis, (l, y), (r, y), (0, 255, 255), 1)

            # 左上角标注文字
            label = f"chat_area: ({l},{t}) -> ({r},{b})  size={r-l}x{b-t}"
            cv2.putText(vis, label, (l, max(t - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            ocr_label = f"OCR zone: bottom {self.max_ocr_height}px"
            cv2.putText(vis, ocr_label, (l, max(t - 30, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.preprocessed_dir, f"REGION_DEBUG_{ts}.png")
            cv2.imwrite(path, vis)
            self._region_debug_saved = True
            logger.info(f"📸 区域调试截图已保存: {path}")
            logger.info(f"   红色框 = chat_area ({l},{t},{r},{b}), 黄色虚线 = 底部OCR识别区 ({self.max_ocr_height}px)")
        except Exception as e:
            logger.debug(f"保存区域调试图失败: {e}")

    def read_message(self, screen: np.ndarray) -> Optional[ConversationSnapshot]:
        """
        从屏幕截图中读取聊天区域的完整对话快照。

        核心策略：用 OCR 检测框的 x 坐标区分左右。
        - 左半边（x_center < 图像宽度×55%）：别人发的黑底白字 → 保留
        - 右半边（x_center > 图像宽度×55%）：自己发的白底黑字 → 丢弃

        长消息自动换行 → 相邻行（y 间距<30px）合并为同一条消息再解析。

        Args:
            screen: 全屏截图或聊天区域截图 (BGR格式)

        Returns:
            ConversationSnapshot，无变化或无有效内容返回 None
        """
        if not self._init_ocr():
            return None

        # 首次调用时保存一张带区域标注的全屏截图
        self.save_region_debug(screen)

        # 裁剪聊天区域
        l, t, r, b = self.chat_region
        screen_h, screen_w = screen.shape[:2]
        if screen_w > (r - l) * 2:
            try:
                chat_img = screen[t:b, l:r]
            except Exception:
                logger.debug("截取聊天区域失败（屏幕尺寸不匹配？）")
                return None
        else:
            chat_img = screen

        if chat_img.size == 0:
            return None

        # ===== 快速变化检测 =====
        img_hash = self._img_hash(chat_img)
        if img_hash == self._last_chat_hash:
            return None
        self._last_chat_hash = img_hash

        # max_ocr_height=0 表示 OCR 整个聊天区域（不做裁剪）
        h, w = chat_img.shape[:2]
        if self.max_ocr_height > 0:
            crop_h = min(h, self.max_ocr_height)
            if crop_h < h:
                chat_img = chat_img[h - crop_h:h, 0:w]

        img_h, img_w = chat_img.shape[:2]

        # ===== 关键：屏蔽自己发的消息（右侧白底气泡）=====
        # 光遇聊天布局：别人=左侧黑底气泡+白字，自己=右侧白底气泡+黑字
        # _filter_dark_bg_text 只保留黑底区域，将白底区域涂白。
        # 必须在预处理之前调用（预处理可能翻转颜色，干扰暗色检测）。
        chat_img, has_dark = self._filter_dark_bg_text(chat_img)
        if not has_dark:
            logger.debug("聊天区无黑底气泡（无非自己消息），跳过")
            return None

        # 预处理：轻量对比度增强（只在 preprocess=True 时）
        if self.preprocess:
            chat_img = self._preprocess_for_ocr(chat_img)

        # 缩放：只在宽度 > 1200 时缩放，保留更多细节
        # 之前 800px 对小字（如名字）压缩太多导致 OCR 漏字
        max_width = 1200
        if img_w > max_width:
            scale = max_width / img_w
            new_w, new_h = max_width, int(img_h * scale)
            chat_img = cv2.resize(chat_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            img_h, img_w = new_h, new_w

        self._save_preprocessed_debug(chat_img, "read_message")
        if len(chat_img.shape) == 2:
            chat_img = cv2.cvtColor(chat_img, cv2.COLOR_GRAY2BGR)

        # ===== OCR 全部文字（含 x 坐标） =====
        results, elapsed = self._ocr_infer(chat_img)
        if not results:
            return None

        # ===== x 坐标过滤：只保留左侧气泡（别人的消息）=====
        # 关键：用气泡左边缘判断，不用 x_center。
        # 因为左侧气泡内容一长，x_center 可能过中线，导致末尾文字和名字被误切。
        left_results = [
            (text, y, xl, xr, xc)
            for text, y, xl, xr, xc in results
            if self._is_left_bubble(xl, xr, img_w)
        ]
        right_count = len(results) - len(left_results)

        all_raw = [r[0] for r in results[:20]]
        left_raw = [r[0] for r in left_results[:15]]
        logger.info(f"OCR 全部 {len(results)} 条（左{len(left_results)} + 右{right_count}）")
        logger.info(f"  左侧(别人): {left_raw}")

        if not left_results:
            logger.debug("左侧没有别人的消息，跳过")
            return None

        # ===== 多行合并：相邻 y 间距 < 30px 的是同一人同一条消息 =====
        merged_texts = self._merge_multiline_texts(left_results)

        # ===== 解析每条合并后的消息 → ChatMessage =====
        all_messages: List[ChatMessage] = []
        for merged_text, avg_y in merged_texts:
            stripped = merged_text.strip()
            if not stripped or self._is_garbage_text(stripped):
                continue

            content = ""
            sender = ""

            # 策略1: 分隔符匹配 "内容 - 名字"
            sep_result = self._extract_msg_by_separator(stripped)
            if sep_result is not None:
                content, sender = sep_result
            else:
                # 策略2: 启发式匹配（末尾短名字）
                heuristic = self._extract_msg_heuristic(stripped)
                if heuristic is not None:
                    content, sender = heuristic
                else:
                    # 策略3: 整行当内容
                    content = stripped
                    sender = ""

            if content:
                all_messages.append(ChatMessage(
                    content=content,
                    sender=sender,
                    raw_text=stripped,
                    y=int(avg_y),
                    is_new=False,
                ))

        # 按 y 从上到下排序
        all_messages.sort(key=lambda m: m.y)

        if not all_messages:
            logger.debug("OCR: 未解析到有效消息")
            return None

        # 日志
        for i, msg in enumerate(all_messages):
            sender_tag = f" [{msg.sender}]" if msg.sender else ""
            logger.debug(f"  #{i+1} y={msg.y}{sender_tag}: {msg.content}")

        # ===== 标记新消息 =====
        has_new = self._mark_new_messages(all_messages)

        # 找最新消息的发送者
        new_msgs = [m for m in all_messages if m.is_new]
        sender_name = new_msgs[-1].sender if new_msgs else all_messages[-1].sender
        self._last_sender_name = sender_name

        if has_new:
            new_lines = "\n".join(m.display for m in new_msgs)
            logger.info(f"📩 新消息 ({len(new_msgs)}条):\n{new_lines}")

        # 构建对话上下文（去重+只保留最近N条）
        self._build_context(all_messages)

        snapshot = ConversationSnapshot(
            messages=list(self._conversation_context),
            has_new=has_new,
            sender_name=sender_name,
        )
        return snapshot

    @property
    def is_available(self) -> bool:
        """OCR 是否当前可用（已初始化成功）"""
        return self.ocr is not None and self._ocr_ok

    def _ocr_infer(self, img: np.ndarray):
        """
        统一 OCR 推理接口，屏蔽 PaddleOCR / EasyOCR 差异。
        返回 (results, elapsed) 或 (None, 0)
        results: [(文字, y底部坐标, x左边缘, x右边缘, x中心), ...]
        """
        if self.ocr is None:
            return None, 0
        _t0 = time.time()
        try:
            if self._ocr_engine == "easyocr":
                raw = self.ocr.readtext(img)
            else:
                raw = self.ocr.ocr(img)
        except Exception as e:
            err_msg = str(e)
            logger.error(f"OCR 识别异常: {err_msg}")
            # GPU 崩溃 → 标记 OCR 不可用，下次自动重初始化
            if "could not execute a primitive" in err_msg or "CUDNN" in err_msg or "CUDA" in err_msg:
                logger.warning("⚠️ GPU 推理异常，重置 OCR 引擎以便下次重试...")
                self.ocr = None
                self._ocr_ok = False
                self._last_init_attempt = 0.0  # 允许立即重试
            return None, 0
        elapsed = time.time() - _t0

        if not raw or (isinstance(raw, list) and len(raw) == 0):
            return None, elapsed

        results = []
        min_confidence = 0.3  # 降低阈值：游戏字体笔画细，名字/末尾小字容易低于 0.5
        if self._ocr_engine == "easyocr":
            # EasyOCR: [(bbox, text, confidence), ...]
            for item in raw:
                if item and len(item) >= 3:
                    text = item[1]
                    conf = float(item[2]) if len(item) >= 3 else 1.0
                    bbox = item[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    if text and text.strip() and conf >= min_confidence:
                        y_bottom = max(pt[1] for pt in bbox) if bbox else 0
                        x_left = min(pt[0] for pt in bbox) if bbox else 0
                        x_right = max(pt[0] for pt in bbox) if bbox else 0
                        x_center = (x_left + x_right) / 2.0
                        results.append((text.strip(), y_bottom, x_left, x_right, x_center))
        else:
            # PaddleOCR: [[bbox, (text, confidence)], ...]
            lines = raw[0] if isinstance(raw, (list, tuple)) and len(raw) > 0 and raw[0] else []
            for line in (lines or []):
                if line and len(line) >= 2:
                    text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                    conf = float(line[1][1]) if isinstance(line[1], (list, tuple)) and len(line[1]) >= 2 else 1.0
                    bbox = line[0] if line[0] else None
                    if text and text.strip() and conf >= min_confidence:
                        y_bottom = max(p[1] for p in bbox) if bbox else 0
                        x_left = min(p[0] for p in bbox) if bbox else 0
                        x_right = max(p[0] for p in bbox) if bbox else 0
                        x_center = (x_left + x_right) / 2.0
                        results.append((text.strip(), y_bottom, x_left, x_right, x_center))

        # 按 y 坐标从下到上排序（底部最新消息排前面）
        results.sort(key=lambda x: -x[1])

        return results, elapsed

    # ==================== 左右气泡判断 ====================

    @staticmethod
    def _is_left_bubble(x_left: float, x_right: float, img_width: float) -> bool:
        """
        判断一个 OCR 检测框是否属于左侧气泡（别人的消息）。

        微信/聊天界面布局：
        - 左侧气泡：左边缘贴近左边距，右边缘可变
        - 右侧气泡：右边缘贴近右边距，左边缘可变

        不依赖 x_center，因为左侧气泡如果内容很长，x_center 可能靠近中线。
        正确做法：看哪个边缘离对应边距更近。

        Returns:
            True  → 左侧气泡（别人的消息，保留）
            False → 右侧气泡（自己的消息，丢弃）
        """
        if img_width <= 0:
            return False
        left_margin_dist = x_left
        right_margin_dist = img_width - x_right
        # 左边缘离左边界更近 → 左侧气泡；反之右侧气泡
        return left_margin_dist <= right_margin_dist

    # ==================== 多行合并 ====================

    @staticmethod
    def _merge_multiline_texts(
        ocr_results: List[Tuple[str, float, float, float, float]],
        y_gap_threshold: float = 30.0,
    ) -> List[Tuple[str, float]]:
        """
        合并相邻的 OCR 文本行，解决长消息自动换行问题。

        游戏中别人的发言格式是：黑底气泡中，"内容 - 名字"。
        当内容太长时，游戏自动换行成多行，只有最后一行带 "- 名字"。
        比如："今天天气真好我想出去\n走走买点东西 - 小明"

        策略：
        1. 按 y 坐标从小到大排序（从上到下）
        2. 自动计算每行文字高度，动态阈值 = 行高 × 1.8
        3. y 间距 < 动态阈值 → 视为同一人的同一消息，合并
        4. y 间距 >= 动态阈值 → 新消息开始

        Returns:
            [(合并后的完整文本, 平均y坐标), ...]
        """
        if not ocr_results:
            return []

        # 按 y 从小到大（从上到下）
        sorted_results = sorted(ocr_results, key=lambda r: r[1])

        # 估算每行文字高度（用 y_bottom 和 y_top 的差）
        # bbox 里有 y_bottom，用相邻行差值估算行高
        line_heights = []
        for i in range(len(sorted_results)):
            text, y_bottom, xl, xr, xc = sorted_results[i]
            # 用左右边缘差估算文字宽度 → 反推行高（通常高宽比 1:4~1:8）
            text_width = xr - xl
            if text_width > 0:
                est_height = text_width / max(len(text), 1) * 1.5  # 中文字符高宽比
                line_heights.append(est_height)

        # 平均行高
        avg_line_height = (sum(line_heights) / len(line_heights)) if line_heights else 20.0
        # 动态阈值：行高的 1.8 倍，夹在 15~45px 之间
        dyn_threshold = max(15.0, min(45.0, avg_line_height * 1.8))

        groups: List[List[Tuple[str, float, float, float, float]]] = []
        current_group = [sorted_results[0]]

        for item in sorted_results[1:]:
            prev_y = current_group[-1][1]
            curr_y = item[1]
            if (curr_y - prev_y) < dyn_threshold:
                current_group.append(item)
            else:
                groups.append(current_group)
                current_group = [item]
        groups.append(current_group)

        # 每组合并文本：按 y 排序后拼接
        merged = []
        for group in groups:
            group_sorted = sorted(group, key=lambda r: r[1])
            joined = " ".join(r[0] for r in group_sorted)
            avg_y = sum(r[1] for r in group_sorted) / len(group_sorted)
            merged.append((joined, avg_y))

        return merged


    # ==================== 对话结构化解析 ====================

    def _parse_all_messages(self, ocr_results: List[Tuple[str, int]]) -> List[ChatMessage]:
        """
        将 OCR 结果解析为结构化的 ChatMessage 列表。

        每行尝试：
        1. 分隔符匹配 → 拆出 (content, sender)
        2. 启发式匹配 → 裁掉末尾短名
        3. 都失败 → 保留整行作为 content（可能是一条没有名字的独立消息）

        然后合并相邻同发送者的消息（同一人的多行发言）。
        """
        raw_messages: List[ChatMessage] = []

        for text, y in ocr_results:
            stripped = text.strip()
            if not stripped or self._is_garbage_text(stripped):
                continue

            content = ""
            sender = ""

            # 策略1: 分隔符匹配
            sep_result = self._extract_msg_by_separator(stripped)
            if sep_result is not None:
                content, sender = sep_result
            else:
                # 策略2: 启发式匹配
                heuristic = self._extract_msg_heuristic(stripped)
                if heuristic is not None:
                    content, sender = heuristic
                else:
                    # 策略3: 直接当消息内容（可能是没有名字的纯文本）
                    content = stripped
                    sender = ""

            if content:
                raw_messages.append(ChatMessage(
                    content=content,
                    sender=sender,
                    raw_text=stripped,
                    y=y,
                    is_new=False,
                ))

        # 按 y 从上到下排序（时间先后顺序）
        raw_messages.sort(key=lambda m: m.y)

        if not raw_messages:
            return []

        # ===== 合并同行消息（去重） =====
        # 同一 y 区域内（±15px）高度相似的文本 → 只保留一条
        merged: List[ChatMessage] = [raw_messages[0]]
        for m in raw_messages[1:]:
            prev = merged[-1]
            y_gap = abs(m.y - prev.y)
            sim = difflib.SequenceMatcher(None, m.content, prev.content).ratio()
            if y_gap <= 15 and sim >= 0.7:
                # 同一行的重复识别，保留更长的
                if len(m.content) > len(prev.content):
                    merged[-1] = m
                continue
            merged.append(m)

        logger.debug(f"解析消息: {len(raw_messages)} 原始 → {len(merged)} 合并后")
        for i, msg in enumerate(merged):
            sender_tag = f" [{msg.sender}]" if msg.sender else ""
            logger.debug(f"  #{i+1} y={msg.y}{sender_tag}: {msg.content}")

        return merged

    @staticmethod
    def _message_key(msg: ChatMessage) -> str:
        """生成消息的稳定追踪 key"""
        y_zone = (msg.y // 40) * 40  # 40px 区间
        content_hash = hashlib.md5(msg.content.encode("utf-8", errors="ignore")).hexdigest()[:8]
        return f"{y_zone}_{content_hash}"

    @staticmethod
    def _content_hash(content: str, sender: str) -> str:
        """生成 y 坐标无关的纯内容哈希，用于跨滚动去重"""
        raw = f"{content}|{sender}"
        return hashlib.md5(raw.encode("utf-8", errors="ignore")).hexdigest()

    def _mark_new_messages(self, messages: List[ChatMessage]) -> bool:
        """
        两阶段去重标记新消息：

        阶段 1（内容哈希环）：y 坐标无关，只要 (内容, 发送者) 曾经见过，
        无论滚动多少像素都判定为旧消息。解决输入框弹出导致聊天区滚动时
        已见消息被误判为新消息的问题。

        阶段 2（已知消息字典）：y 坐标相关的精细匹配，处理 OCR 抖动/变异。
        用 difflib 相似度比对同 y 区域的候选消息。

        返回是否有至少一条新消息。
        """
        has_new = False
        for msg in messages:
            ch = self._content_hash(msg.content, msg.sender)

            # ---- 阶段 1: 纯内容去重（y 无关）----
            # 多重 hash 回退：OCR 可能读丢分隔符/名字
            # 1. 标准 hash: content|sender
            # 2. 裸文本 hash: raw_text|"" （OCR 整行原文）
            # 3. 纯内容 hash: content|"" （OCR 丢了名字）
            raw_ch = self._content_hash(msg.raw_text, "")
            content_ch = self._content_hash(msg.content, "")
            if ch in self._seen_content_hashes or raw_ch in self._seen_content_hashes or content_ch in self._seen_content_hashes:
                msg.is_new = False
                # 仍然更新 _known_messages 里的 y 坐标供上下文排序
                key = self._message_key(msg)
                if key in self._known_messages:
                    self._known_messages[key].y = msg.y
                else:
                    self._known_messages[key] = msg
                continue

            # ---- 阶段 2: y 区域相似度匹配（OCR 抖动补偿）----
            is_known = False
            for known_key, known_msg in self._known_messages.items():
                y_gap = abs(msg.y - known_msg.y)
                if y_gap > 50:
                    continue
                sim = difflib.SequenceMatcher(None, msg.content, known_msg.content).ratio()
                if sim >= 0.65:
                    is_known = True
                    # 用更完整的内容更新已知消息
                    if len(msg.content) > len(known_msg.content):
                        del self._known_messages[known_key]
                        key = self._message_key(msg)
                        self._known_messages[key] = msg
                    # 也加入内容哈希环
                    self._seen_content_hashes[ch] = None
                    break

            if not is_known:
                # 真正的新消息
                msg.is_new = True
                has_new = True
                self._known_messages[self._message_key(msg)] = msg
                self._seen_content_hashes[ch] = None
                logger.debug(f"  ✨ 新消息: {msg.display}")
            else:
                msg.is_new = False

        # ---- 限制 known_messages 大小 ----
        if len(self._known_messages) > 200:
            sorted_keys = sorted(
                self._known_messages.keys(),
                key=lambda k: self._known_messages[k].y,
                reverse=True,
            )
            for old_key in sorted_keys[100:]:
                del self._known_messages[old_key]

        # ---- 限制内容哈希环大小（OrderedDict FIFO，确定性）----
        excess = len(self._seen_content_hashes) - self._seen_ring_max
        for _ in range(excess):
            self._seen_content_hashes.popitem(last=False)  # FIFO: 弹出最早插入的

        return has_new

    def _build_context(self, messages: List[ChatMessage], max_len: int = 15):
        """
        构建对话上下文：按时间顺序保留最近 N 条可见消息。
        用作 LLM 的对话历史输入。

        自动过滤 AI 自己的消息（_ai_name），避免 LLM 看到自己说过的话产生困惑。
        """
        # 按 y 从上到下（时间先后）
        sorted_msgs = sorted(messages, key=lambda m: m.y)

        # 去重 + 过滤自己的消息
        deduped = []
        for m in sorted_msgs:
            # 跳过 AI 自己发的消息（通过发送者名匹配）
            if self._ai_name and m.sender == self._ai_name:
                continue
            if deduped and m.content == deduped[-1].content and m.sender == deduped[-1].sender:
                continue
            deduped.append(m)

        # 保留最近 max_len 条
        if len(deduped) > max_len:
            deduped = deduped[-max_len:]

        self._conversation_context = deduped
        logger.debug(f"对话上下文 ({len(deduped)}条):")
        for m in deduped:
            logger.debug(f"  {m.display}")

    def mark_sent_message(self, text: str):
        """
        记录一条 AI 自己发送的消息，OCR 扫描到时不会误判为新消息。

        原理：AI 发消息后，聊天区域会显示这条消息。
        如果不记录，下次 OCR 会把它当成"别人发的新消息"，导致 AI 回复自己。

        防御策略（多重 hash 注入）：
        1. 按分隔符拆出 (content, sender) → 标准 hash
        2. 原始文本 raw_text → 裸文本 hash（OCR 可能读不出分隔符）
        3. 纯内容（不带 sender）→ content-only hash（OCR 可能丢掉名字）
        三层都注入 _seen_content_hashes，无论 OCR 怎么读都能兜住。
        """
        if not text or not text.strip():
            return
        stripped = text.strip()
        # 尝试拆分内容+名字（与 OCR 解析保持一致）
        content = stripped
        sender = ""
        sep_result = self._extract_msg_by_separator(stripped)
        if sep_result is not None:
            content, sender = sep_result
        else:
            heuristic = self._extract_msg_heuristic(stripped)
            if heuristic is not None:
                content, sender = heuristic

        # 用很大 y 值，确保在对话上下文中排在底部（最新）
        msg = ChatMessage(
            content=content,
            sender=sender or self._ai_name,
            raw_text=stripped,
            y=999999,
            is_new=False,
        )
        key = self._message_key(msg)
        self._known_messages[key] = msg

        # ===== 多重 hash 注入：无论 OCR 怎么拆分/读变体都能匹配 =====
        # 1. 标准 hash：content + sender
        std_ch = self._content_hash(content, msg.sender)
        self._seen_content_hashes[std_ch] = None
        # 2. 裸文本 hash：OCR 没读分隔符时的整行原文
        raw_ch = self._content_hash(stripped, "")
        self._seen_content_hashes[raw_ch] = None
        # 3. 纯内容 hash：OCR 丢了名字/名字被读错
        content_ch = self._content_hash(content, "")
        self._seen_content_hashes[content_ch] = None

        logger.debug(f"📝 已记录AI发送的消息: {content[:50]}")

    # ---- 分隔符匹配 ----
    # OCR 可能把 "-" 识别为各种 Unicode 字符
    # ⚠️ 注意：不能用 "一" 做分隔符！"一" 在中文消息里极其常见（"一起""等一下"），
    #   用 "一" 做分隔会导致整条消息被错误截断。
    _SEPARATORS = [
        # 带空格的分隔符优先（最可靠）
        " - ", " — ", " – ", " · ", " : ", " 丨 ",
        # 左/右单边带空格（容错）
        " -", "- ", " —", "— ", " –", "– ",
        " ·", "· ", " :", ": ", " 丨", "丨 ",
        # 无空格版本（OCR 可能漏掉空格）：
        # 注意：名字长度验证会过滤掉误拆
        "-", "—", "–", "·", ":", "丨",
    ]

    @staticmethod
    def _is_garbage_text(text: str) -> bool:
        """
        检测 OCR 产生的垃圾文本（GPU 崩溃/渲染伪影导致的随机字符）。
        规则:
        1. 连续 10+ 个 ASCII 字母且无空格 → 渲染伪影
        2. 纯符号/数字且无中文 → 无意义
        3. 长度 < 2 个有意义字符
        """
        stripped = text.strip()
        if len(stripped) < 2:
            return True

        # 规则1: 检查是否有超长连续 ASCII 字母串（如 "Outputinitializationaboveina"）
        max_ascii_run = 0
        current_run = 0
        for ch in stripped:
            if ch.isascii() and ch.isalpha():
                current_run += 1
                max_ascii_run = max(max_ascii_run, current_run)
            else:
                current_run = 0
        if max_ascii_run >= 10:
            return True

        # 规则2: 纯英文但完全不像可读单词（全是辅音/元音都不合理）
        has_chinese = any('\u4e00' <= ch <= '\u9fff' for ch in stripped)
        has_space = ' ' in stripped or '\t' in stripped
        if not has_chinese and not has_space:
            # 纯英文无空格 → 超过20字符无空格 = 垃圾
            if len(stripped) > 20:
                return True

        return False

    def _extract_msg_by_separator(self, text: str) -> Optional[Tuple[str, str]]:
        """
        尝试从文本末尾提取 "内容 分隔符 名字"。
        优先从末尾找分隔符，避免消息内容中间出现类似符号被误拆。

        Returns:
            (消息内容, 玩家名)，没分隔符或名字不像真名返回 None。
        """
        # 策略1：优先从末尾找分隔符 + 短名字（最可靠）
        end_sep_result = self._extract_msg_end_separator(text)
        if end_sep_result is not None:
            return end_sep_result

        # 策略2：回退到原始方式（分隔符在中间的情况）
        for sep in self._SEPARATORS:
            if sep in text:
                parts = text.split(sep, 1)
                msg = parts[0].strip()
                name = parts[1].strip() if len(parts) > 1 else ""
                if not msg:
                    continue
                # 验证玩家名合理性
                if not self._is_plausible_name(name):
                    continue
                return msg, name
        return None

    def _extract_msg_end_separator(self, text: str) -> Optional[Tuple[str, str]]:
        """
        从文本末尾附近提取名字：格式 "内容 - 名字" 或 "内容-名字"。
        要求名字在文本最末尾，且长度在 2-6 字之间（典型玩家名长度）。

        额外兜底：OCR 常把 " - " 误读成中文 "一"（如 "好像人机 - 大号"
        → "好像人机一大号"）。对末尾模式 "内容一名字" 做特殊处理，
        只在名字合理时生效，避免误拆消息内容里的 "一起"/"等一下"。
        """
        name_len_min, name_len_max = 2, 6

        for sep in self._SEPARATORS:
            # 找最后一个分隔符
            idx = text.rfind(sep)
            if idx == -1:
                continue
            msg = text[:idx].strip()
            name = text[idx + len(sep):].strip()
            if not msg or len(msg) < 2:
                continue
            if len(name) < name_len_min or len(name) > name_len_max:
                continue
            if not self._is_plausible_name(name):
                continue
            return msg, name

        # ---- 兜底：OCR 把 " - " 错读成 "一" 的情况 ----
        # 游戏消息格式："内容 - 名字"，OCR 可能把 "-" 误读成 "一"。
        # 例如："好像人机一大号" → ("好像人机", "大号")
        #
        # 关键防护："一" 是中文极高频字（"一起""一下""第一种"），
        # 不能无脑 rfind() 拆。必须满足：
        #   1. "一" 之后刚好是 2-6 字候选名
        #   2. "一" 在文本末尾附近（距末尾 ≤ name_len_max+1 字）
        #   3. 消息内容长度 >= 名字长度的 2 倍（确保拆分合理）
        idx = text.rfind("一")
        if idx > 0:
            msg = text[:idx].strip()
            name = text[idx + 1:].strip()
            # 位置约束："一" 必须在文本末尾附近
            # len(text) - idx - 1 就是 "一" 后面的字符数 = len(name)
            chars_after_one = len(text) - idx - 1
            if chars_after_one > name_len_max:
                pass  # "一" 后面太长了，不像是分隔符误读
            elif len(msg) >= 2 and name_len_min <= len(name) <= name_len_max:
                # 内容必须至少是名字的 2 倍长，防止 "好一天"→("好","天") 这种误拆
                if len(msg) >= len(name) * 2 and self._is_plausible_name(name):
                    return msg, name
        return None

    @staticmethod
    def _is_plausible_name(name: str) -> bool:
        """验证候选玩家名是否合理（非仅空白、非单字、非纯标点）"""
        if not name:
            return False
        name = name.strip()
        if len(name) < 2:          # 单字不是名字
            return False
        if len(name) > 12:         # 太长不像玩家名
            return False
        # 不能全是标点/特殊字符
        has_alpha = any(c.isalpha() for c in name)
        has_cjk = any('\u4e00' <= c <= '\u9fff' for c in name)
        has_digit = any(c.isdigit() for c in name)
        if not (has_alpha or has_cjk or has_digit):
            return False
        return True

    # ---- 启发式无分隔符匹配 ----
    # 当 "-" 太小 OCR 没识别时，用末尾短名的模式猜测
    # 规则: "长消息内容 + 短玩家名" → 玩家名通常是 2-4 个字符
    @staticmethod
    def _extract_msg_heuristic(text: str) -> Optional[Tuple[str, str]]:
        """
        启发式: 如果文本较长（>=5字），且末尾像是短名（2-4字），
        则分离内容+名字。

        例如: "你好小明" → ("你好", "小明")
              "今天天气真好旅人007" → ("今天天气真好", "旅人007")
        """
        text_clean = text.strip()
        n = len(text_clean)

        # 太短的文本不拆（<5字可能是纯名字或纯消息）
        if n < 5:
            return None

        # 尝试从末尾截取 2、3、4 字作为候选名
        for name_len in (4, 3, 2):
            if n <= name_len:
                continue
            candidate_name = text_clean[-name_len:]
            # 名字中不能有太长的连续英文字母（>4 个字母 → 可能是英文消息）
            alpha_count = sum(1 for c in candidate_name if c.isascii() and c.isalpha())
            if alpha_count >= name_len:  # 全是英文字母 → 不截
                continue

            msg_content = text_clean[:-name_len].strip()
            # 确保剩余部分足够长（至少3字）
            if msg_content and len(msg_content) >= 3:
                return msg_content, candidate_name

        return None

    @staticmethod
    def _img_hash(img: np.ndarray) -> str:
        """快速图像哈希：64×64 分辨率，足以捕捉文字变化"""
        small = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST)
        return hashlib.md5(small.tobytes()).hexdigest()

    def reset(self):
        """重置状态（切换场景时调用）"""
        self._last_text = ""
        self._last_sender_name = ""
        self._last_timestamp = 0.0
        self._last_chat_hash = ""
        self._known_messages.clear()
        self._conversation_context.clear()
        self._seen_content_hashes.clear()
        logger.info("ChatOCR 状态已重置（含对话历史和内容哈希环）")

    @property
    def last_sender_name(self) -> str:
        """返回最近一次识别到的对方玩家名（可能为空）"""
        return self._last_sender_name

    def update_chat_region(self, region: Tuple[int, int, int, int]):
        """更新聊天区域坐标"""
        self.chat_region = region
        self._region_debug_saved = False  # 允许重新生成区域调试图
        logger.info(f"聊天区域已更新: {region}")

    def read_all(self, screen: np.ndarray) -> str:
        """
        读取聊天区域所有可见文字（不去重，只保留左侧别人的发言）。

        Returns:
            所有可见聊天文字
        """
        if not self._init_ocr():
            return ""

        left, top, right, bottom = self.chat_region
        try:
            chat_img = screen[top:bottom, left:right]
        except Exception:
            return ""

        if chat_img.size == 0:
            return ""

        # 预处理：轻量对比度增强
        if self.preprocess:
            chat_img = self._preprocess_for_ocr(chat_img)

        # 保持高分辨率以识别小字（名字等）
        h, w = chat_img.shape[:2]
        if w > 1200:
            scale = 1200.0 / w
            new_w, new_h = 1200, int(h * scale)
            chat_img = cv2.resize(chat_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            h, w = new_h, new_w

        self._save_preprocessed_debug(chat_img, "read_all")

        # PaddleOCR 3.x 需要 BGR 三通道图
        if len(chat_img.shape) == 2:
            chat_img = cv2.cvtColor(chat_img, cv2.COLOR_GRAY2BGR)

        results, _elapsed = self._ocr_infer(chat_img)

        if results:
            # ===== x 坐标过滤：只保留左侧气泡（别人的发言） =====
            left_texts = [
                t for t, y, xl, xr, xc in results
                if self._is_left_bubble(xl, xr, w)
            ]
            return "\n".join(left_texts)
        return ""

