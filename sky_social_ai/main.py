"""
光遇社交AI - 主控程序

光遇聊天流程：
    1. 发现椅子 → 点击坐下
    2. 坐下后 → AI主动打招呼（点击左下角按钮 → 输入文字 → 发送）
    3. 别人说话时，头顶会出现气泡
    4. 点击对方头顶气泡 → 左侧展开对话框 → OCR读取
    5. AI生成回复 → 点击聊天按钮 → 输入 → 发送 → 等待下一轮
"""

import time
import logging
import signal
import sys
import os
import difflib
from collections import deque
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np

from screen_capture import create_capture
from chair_detector import ChairDetector
from chat_button_detector import ChatButtonDetector
from speech_bubble_detector import SpeechBubbleDetector
from chat_ocr import ChatOCR
from llm_dialogue import DialogueEngine
from friend_manager import FriendManager
from input_simulator import create_input

try:
    import yaml
except ImportError:
    print("请安装 pyyaml: pip install pyyaml")
    sys.exit(1)


class State(Enum):
    """状态机状态"""
    IDLE = auto()              # 空闲中，扫描椅子
    APPROACHING = auto()       # 正在坐下（等待动画）
    SEATED = auto()            # 已坐下，等待消息
    OPENING_BUBBLE = auto()    # 点击对方头顶气泡
    READING_BUBBLE = auto()    # 读取左侧对话框内容
    THINKING = auto()          # AI 生成回复中
    OPENING_CHAT = auto()      # 点击左下角聊天按钮
    TYPING = auto()            # 输入文字中
    SENDING = auto()           # 发送
    CLOSING_CHAT = auto()      # 点击返回键关闭聊天
    ADDING_FRIEND = auto()     # 添加好友中
    ERROR = auto()             # 错误状态


class SkySocialAI:
    """光遇社交AI 主控制器"""

    def __init__(self, config_path: str = "config.yaml", setup_logging: bool = True):
        # 加载配置
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), config_path
        ) if not os.path.isabs(config_path) else config_path

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        platform = self.config.get("platform", "pc")
        self.pc_native = self.config.get("pc_native", False)
        self._key_bindings = self.config.get("key_bindings", {})
        self._scan_chat_only = self.config.get("scan_only_chat_region", False) and self.pc_native
        act_cfg = self.config.get("actions", {})

        # ===== 模块初始化 =====
        self.capture = create_capture(platform)
        window_offset = tuple(self.config.get("window_offset", [0, 0]))
        self.input = create_input(platform, offset=window_offset)

        # 椅子检测
        self.chair_detector = ChairDetector(
            template_dir=os.path.join(os.path.dirname(__file__),
                self.config.get("chair_detection", {}).get("template_dir", "templates")),
            template_names=self.config.get("chair_detection", {}).get("templates", []),
            threshold=self.config.get("chair_detection", {}).get("match_threshold", 0.75),
        )

        # 聊天按钮检测（左下角气泡图标）
        self.chat_button_detector = ChatButtonDetector(
            template_path=os.path.join(os.path.dirname(__file__),
                self.config.get("chat_button", {}).get("template_path", "templates/chat_button.png")),
            threshold=self.config.get("chat_button", {}).get("match_threshold", 0.75),
            search_region=self._parse_region(
                self.config.get("chat_button", {}).get("search_region")
            ),
        )

        # 头顶气泡检测
        self.bubble_detector = SpeechBubbleDetector(
            template_path=os.path.join(os.path.dirname(__file__),
                self.config.get("speech_bubble", {}).get("template_path", "templates/speech_bubble.png")),
            threshold=self.config.get("speech_bubble", {}).get("match_threshold", 0.75),
            search_region=self._parse_region(
                self.config.get("speech_bubble", {}).get("search_region")
            ),
        )

        # OCR、对话引擎、好友管理
        self.chat_ocr = ChatOCR(self.config)
        self.dialogue = DialogueEngine(self.config)
        self.friend_manager = FriendManager(self.input, self.config)

        # ===== 状态机 =====
        skip_chair = self.config.get("chair_detection", {}).get("skip_chair", False)
        self.state = State.SEATED if skip_chair else State.IDLE
        self._running = False

        # 参数
        self.scan_interval = act_cfg.get("scan_interval", 2.0)
        self.scan_interval_fast = act_cfg.get("scan_interval_fast", 0.4)
        self.typing_speed = act_cfg.get("typing_speed", 0.05)
        self.idle_timeout = act_cfg.get("idle_timeout_rounds", 30)
        # 批量聚合窗口（秒）：在此期间积累多条消息，一次性发给 LLM
        self.batch_window = act_cfg.get("batch_window", 1.5)
        # ===== 对话持久化配置 =====
        conv_cfg = self.config.get("conversation", {})
        self._max_conversation_rounds = conv_cfg.get("max_rounds", 300)  # 0=无上限
        self._max_context_lines = conv_cfg.get("max_context_lines", 30)    # 传给 LLM 的最大上下文行数
        self._max_context_messages = conv_cfg.get("max_context_messages", 20)  # OCR 上下文最多保留条数
        # ===== 内存监控 =====
        self._mem_log_interval = conv_cfg.get("mem_log_interval_rounds", 30)  # 每 N 轮打印内存状态
        # 响应链状态集：这些状态之间无需等待 scan_interval，直接连续执行
        self._chain_states = {
            State.THINKING, State.OPENING_CHAT, State.TYPING,
            State.SENDING, State.CLOSING_CHAT,
        }

        # ===== 对话状态 =====
        self._has_greeted = False            # 是否已打招招呼
        self._idle_rounds = 0                # 空闲轮次计数
        self._conversation_rounds = 0         # 对话轮次计数
        self._pending_message = ""            # 待处理消息（最新一条的纯文本）
        self._pending_context: list = []      # 待传给 LLM 的对话上下文（结构化）
        self._pending_reply = ""              # 待发送回复
        self._bubble_click_time = 0.0         # 点击气泡的时间
        self._chat_open_time = 0.0            # 打开聊天的时间
        self._approach_start_time = 0.0       # 开始坐下的时间
        self._last_send_time = 0.0            # 上次发送消息时间（冷却用）
        self._handled_messages = deque(maxlen=80)  # 已处理的消息文本（兜底去重，自动淘汰旧消息）
        self._chat_ui_open = False           # 聊天对话框是否已打开（C键是切换开关）
        # ===== 批量消息聚合 =====
        self._batch_messages: list = []       # 积累中的多条消息(display文本)
        self._batch_contexts: list = []       # 积累中的对话上下文
        self._batch_senders: list = []        # 积累中的发送者名
        self._batch_deadline = 0.0            # 收集窗口截止时间

        # 如果跳过椅子检测，立刻重置 OCR 历史
        if skip_chair:
            self.chat_ocr.reset()

        # 开场白
        self.greetings = self.config.get("persona", {}).get(
            "greetings", ["晚上好呀~", "嗨！", "你好呀！"]
        )

        # ===== 调试 =====
        debug_cfg = self.config.get("debug", {})
        self.save_screenshots = debug_cfg.get("save_screenshots", False)
        self.screenshot_dir = os.path.join(os.path.dirname(__file__),
            debug_cfg.get("screenshot_dir", "debug_screenshots"))
        self.screenshot_max_keep = debug_cfg.get("max_screenshots", 100)
        if self.save_screenshots:
            os.makedirs(self.screenshot_dir, exist_ok=True)

        # ===== 日志 =====
        self.logger = logging.getLogger("SkyAI")
        if setup_logging:
            log_level = debug_cfg.get("log_level", "INFO")
            logging.basicConfig(
                level=getattr(logging, log_level, logging.INFO),
                format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        self.logger.info("=" * 50)
        self.logger.info("光遇社交AI 初始化完成")
        self.logger.info(f"平台: {platform}  分辨率: {self.capture.get_resolution()}")
        self.logger.info("=" * 50)

        # PC 原生模式：启动时双击焦点位锁定游戏窗口
        # 焦点坐标为屏幕绝对坐标，不叠加窗口偏移
        if self.pc_native:
            fp = self.config.get("focus_point")
            if fp:
                self.logger.info(f"🖱️ 双击焦点位({fp[0]},{fp[1]})锁定游戏窗口...")
                self.input.click_raw(fp[0], fp[1])
                time.sleep(0.1)
                self.input.click_raw(fp[0], fp[1])
                time.sleep(0.3)

        # GUI 状态回调
        self._gui_status = None

    # ==================== 主循环 ====================

    def run(self):
        self._running = True
        signal.signal(signal.SIGINT, self._signal_handler)

        if self.state == State.SEATED:
            self.logger.info("🟢 跳过椅子检测，直接从\"已坐下\"开始")

        self.logger.info("🟢 开始运行 (Ctrl+C 停止)")
        try:
            while self._running:
                self._tick()

                # 变速扫描：按当前状态选择 sleep 时长
                if self.state == State.SEATED:
                    time.sleep(self.scan_interval_fast)  # 0.4s 快扫，不错过连发
                elif self.state not in self._chain_states:
                    time.sleep(self.scan_interval)        # IDLE等用配置值
                # 响应链状态跑完后回到 SEATED，上面已经睡了
                # 不会出现仍停在链状态的情况（_tick 已连续推进完毕）
        except KeyboardInterrupt:
            self.logger.info("收到停止信号")
        finally:
            self._shutdown()

    def _tick(self):
        try:
            # PC 原生 + 聊天区域直读模式：只截聊天区域（快 20~30 倍）
            if (self._scan_chat_only
                    and self.state == State.SEATED
                    and self.config.get("use_chat_area_directly", False)):
                chat_region = self.config.get("chat_area", {}).get("region")
                if chat_region:
                    screen = self.capture.capture_region(tuple(chat_region))
                else:
                    screen = self.capture.capture()
            else:
                screen = self.capture.capture()

            if self.save_screenshots:
                self._save_debug_screenshot(screen)
            self._notify_gui()

            # 状态机分发
            handlers = {
                State.IDLE: self._handle_idle,
                State.APPROACHING: self._handle_approaching,
                State.SEATED: self._handle_seated,
                State.OPENING_BUBBLE: self._handle_opening_bubble,
                State.READING_BUBBLE: self._handle_reading_bubble,
                State.THINKING: self._handle_thinking,
                State.OPENING_CHAT: self._handle_opening_chat,
                State.TYPING: self._handle_typing,
                State.SENDING: self._handle_sending,
                State.CLOSING_CHAT: self._handle_closing_chat,
                State.ADDING_FRIEND: self._handle_adding_friend,
                State.ERROR: self._handle_error,
            }

            # 连续推进响应链：THINKING→OPENING_CHAT→TYPING→SENDING→CLOSING_CHAT
            # 一次截图里全部跑完，不再等 scan_interval
            max_chain = 10  # 防死循环
            for _ in range(max_chain):
                handler = handlers.get(self.state, self._handle_idle)
                handler(screen)
                # 如果跳出了响应链，停下来等下次 tick
                if self.state not in self._chain_states:
                    break
        except Exception as e:
            self.logger.error(f"_tick 异常: {e}")
            import traceback
            traceback.print_exc()
            # 如果跳过椅子检测，保持在 SEATED 状态，不要切回 IDLE 扫椅子
            skip_chair = self.config.get("chair_detection", {}).get("skip_chair", False)
            if not skip_chair:
                self.state = State.IDLE

    # ==================== 状态处理函数 ====================

    def _handle_idle(self, screen: np.ndarray):
        """空闲状态：扫描椅子（如已配置跳过，直接回到已坐下）"""
        if self.config.get("chair_detection", {}).get("skip_chair", False):
            self.logger.debug("跳过椅子检测，直接回到已坐下状态")
            self.state = State.SEATED
            return

        result = self.chair_detector.detect(screen)

        if result:
            self.logger.info(f"🪑 发现椅子！置信度={result.confidence:.2f}")
            self.input.click(*result.center)
            self._approach_start_time = time.time()
            self.state = State.APPROACHING
            self._has_greeted = False
            self._idle_rounds = 0
            self._conversation_rounds = 0
            self._chat_ui_open = False
        else:
            self._idle_rounds += 1
            if self._idle_rounds % 5 == 0:
                self.logger.debug(f"扫描椅子中... (第{self._idle_rounds}轮)")
            if self._idle_rounds % 10 == 0:
                self._human_like_action()

    def _handle_approaching(self, screen: np.ndarray):
        """等待坐下动画完成"""
        elapsed = time.time() - self._approach_start_time
        if elapsed >= 2.5:  # 坐下动画约2秒
            self.logger.info("🧘 已坐下")
            self.chat_ocr.reset()
            self.state = State.SEATED
        else:
            self.logger.debug(f"坐下动画中... ({elapsed:.1f}s)")

    def _is_message_handled(self, message: str, threshold: float = 0.85) -> bool:
        """
        判断消息是否已处理过（模糊匹配）。
        OCR 同一句话可能多次识别出不同变体，不能仅靠精确字符串去重。

        注意：只用于 OCR 同一条消息的重复扫描去重，不能误杀真正的新消息。
        - 提高阈值到 0.85，避免 "你好" 和 "你好呀" 被错判为相同
        - 子串包含仅在长度相近（差 ≤ 2 字）时触发，避免短消息永久屏蔽长消息
        """
        msg_clean = message.strip()
        if not msg_clean:
            return True
        msg_len = len(msg_clean)
        for handled in self._handled_messages:
            h_len = len(handled)
            # 子串包含：仅当两条消息长度接近时（差 ≤ 2 字）才判定为重复
            # 避免 "你好" 永久屏蔽 "你好呀今天真开心" 这种正常的新消息
            if abs(msg_len - h_len) <= 2:
                if msg_clean in handled or handled in msg_clean:
                    return True
            similarity = difflib.SequenceMatcher(None, msg_clean, handled).ratio()
            if similarity >= threshold:
                return True
        return False

    def _check_batch_expiry(self):
        """检查批量收集窗口是否到期，到期则触发 THINKING"""
        if self._batch_messages and self._batch_deadline > 0:
            if time.time() >= self._batch_deadline:
                self.logger.info(
                    f"⏰ 批量收集窗口到期，共 {len(self._batch_messages)} 条消息"
                )
                self._flush_batch()

    def _flush_batch(self):
        """将积累的批量消息打包，进入 THINKING 状态"""
        if not self._batch_messages:
            self._batch_deadline = 0
            return
        self._pending_message = self._batch_messages[-1]  # 保持兼容
        self._pending_context = list(self._batch_contexts)  # 复制上下文
        self._batch_deadline = 0
        self.state = State.THINKING

    def _trim_context(self, contexts: list) -> list:
        """裁剪 OCR 上下文，保留最近 N 条消息，控制内存"""
        if len(contexts) <= self._max_context_messages:
            return contexts
        # 保留最新的消息（尾部），丢弃最早的消息（头部）
        trimmed = contexts[-self._max_context_messages:]
        self.logger.debug(
            f"📏 上下文裁剪: {len(contexts)} → {len(trimmed)} 条"
        )
        return trimmed

    def _log_memory_stats(self):
        """定期打印内存状态，便于监控是否泄漏"""
        stats_parts = [
            f"known_msgs={len(self.chat_ocr._known_messages)}",
            f"seen_hashes={len(self.chat_ocr._seen_content_hashes)}",
            f"handled={len(self._handled_messages)}",
            f"dialogue_hist={len(self.dialogue.history)}",
            f"batch_msgs={len(self._batch_messages)}",
            f"round={self._conversation_rounds}",
        ]
        self.logger.info(f"📊 内存状态: {', '.join(stats_parts)}")

    def _handle_seated(self, screen: np.ndarray):
        """已坐下状态：主动打招呼 或 监听对方发言（批量聚合模式）"""
        self._conversation_rounds += 1

        # 对话轮次上限检查（0 = 无上限，永不自动结束）
        if self._max_conversation_rounds > 0 and \
                self._conversation_rounds > self._max_conversation_rounds:
            self.logger.info(
                f"对话轮次达到上限 ({self._max_conversation_rounds})，自动结束。"
                "将 config.yaml 中 conversation.max_rounds 设为 0 可取消上限。"
            )
            self.state = State.IDLE
            return

        # ===== 内存监控：每 N 轮打印一次 =====
        if self._conversation_rounds % self._mem_log_interval == 0:
            self._log_memory_stats()

        # ===== 批量窗口到期 → 打包所有积累消息，进入 THINKING =====
        now = time.time()
        if self._batch_messages and self._batch_deadline > 0 and now >= self._batch_deadline:
            self._flush_batch()
            return

        # 还没打招呼：主动发开场白
        if not self._has_greeted:
            import random
            self._handled_messages.clear()     # 新会话，清掉旧消息记录
            self._pending_reply = random.choice(self.greetings)
            self._has_greeted = True
            self.logger.info(f"👋 主动打招呼: {self._pending_reply}")
            if self._gui_status:
                self._gui_status.chat("ai", self._pending_reply)
            self.state = State.OPENING_CHAT
            return

        # 发送后冷却，避免读到自己刚发的消息
        # OCR 有哈希去重 + 模糊匹配，1.5 秒冷却足够避免误读
        if time.time() - self._last_send_time < 1.5:
            return

        # OCR 不可用时，主动尝试初始化（而非被动等待）
        if not self.chat_ocr.is_available:
            if self._conversation_rounds % 5 == 0:
                self.logger.debug(
                    f"等待 OCR 恢复... (第{self._conversation_rounds}轮)"
                )
            # 主动触发初始化
            inited = self.chat_ocr._init_ocr()
            # 超过 60 轮 OCR 仍不可用 → 放弃，避免死循环
            if not inited and self._conversation_rounds > 60:
                self.logger.error(
                    "❌ OCR 长时间不可用（60轮+），请先修复 OCR 再启动。"
                    "安装命令: pip install paddlepaddle paddleocr"
                )
                self._running = False
            return

        use_chat_area = self.config.get("use_chat_area_directly", False)

        if use_chat_area:
            # 直接读取左侧对话框 → 返回结构化对话快照
            snapshot = self.chat_ocr.read_message(screen)
            if snapshot and snapshot.has_new:
                message = snapshot.latest_message
                if not message:
                    self._check_batch_expiry()
                    return
                # 兜底去重
                if self._is_message_handled(message):
                    self.logger.debug(f"⏭ 跳过已处理消息: {message}")
                    self._check_batch_expiry()
                    return
                self._handled_messages.append(message.strip())

                sender_name = snapshot.sender_name
                display_name = sender_name or "对方"

                # ===== 批量聚合：进入收集窗口 =====
                if not self._batch_messages:
                    # 第一条消息，开启收集窗口
                    self._batch_deadline = now + self.batch_window
                    self.logger.info(
                        f"📥 批量收集开始 (窗口{self.batch_window}s)，"
                        f"第一条 [{display_name}]: {message}"
                    )

                self._batch_messages.append(message)
                self._batch_senders.append(sender_name)
                # 用最新的快照覆盖上下文（保留最完整视图），裁剪过长的上下文
                self._batch_contexts = self._trim_context([
                    {"role": "context", "content": m.display}
                    for m in snapshot.messages
                ])
                # 日志打印
                ctx_lines = "\n  ".join(snapshot.conversation_text.split("\n"))
                self.logger.info(f"📩 收到消息 [{display_name}]: {message}")
                self.logger.info(f"📋 对话上下文:\n  {ctx_lines}")
                if self._gui_status:
                    self._gui_status.chat("other", message, sender_name)

                self._check_batch_expiry()
            else:
                self._check_batch_expiry()
                if self._conversation_rounds % 5 == 0 and not self._batch_messages:
                    self.logger.debug(f"等待对方发言... (第{self._conversation_rounds}轮)")
            return

        # 扫描对方头顶气泡（旧方式）
        bubble = self.bubble_detector.detect(screen)
        if bubble:
            self.logger.info(f"💬 发现头顶气泡！点击查看")
            self.input.click(*bubble.center)
            self._bubble_click_time = time.time()
            self.state = State.OPENING_BUBBLE
            return

        # 没发现气泡，继续等待
        if self._conversation_rounds % 5 == 0:
            self.logger.debug(f"等待对方发言... (第{self._conversation_rounds}轮)")

    def _handle_opening_bubble(self, screen: np.ndarray):
        """等待点击气泡后对话框展开"""
        elapsed = time.time() - self._bubble_click_time
        if elapsed >= 1.0:  # 等待对话框展开动画
            self.state = State.READING_BUBBLE
        else:
            self.logger.debug(f"等待对话框展开... ({elapsed:.1f}s)")

    def _handle_reading_bubble(self, screen: np.ndarray):
        """读取左侧对话框内容"""
        snapshot = self.chat_ocr.read_message(screen)
        if snapshot and snapshot.has_new:
            message = snapshot.latest_message
            if not message:
                self.state = State.SEATED
                return
            if self._is_message_handled(message):
                self.logger.debug(f"⏭ 跳过已处理消息: {message}")
                self.state = State.SEATED
                return
            self._handled_messages.append(message.strip())
            sender_name = snapshot.sender_name
            display_name = sender_name or "对方"
            self._pending_message = message
            self._pending_context = [
                {"role": "context", "content": m.display}
                for m in snapshot.messages
            ]
            self.logger.info(f"📩 收到消息 [{display_name}]: {message}")
            if self._gui_status:
                self._gui_status.chat("other", message, sender_name)
            self.state = State.THINKING
        else:
            elapsed = time.time() - self._bubble_click_time
            if elapsed > 5.0:
                self.logger.info("超时未读到消息，返回等待")
                self.state = State.SEATED
            else:
                self.logger.debug(f"等待OCR识别... ({elapsed:.1f}s)")

    def _handle_thinking(self, screen: np.ndarray):
        """AI 生成回复（支持批量消息聚合）"""
        # 批量模式：多条消息合并为一条 prompt
        if len(self._batch_messages) > 1:
            combined = "；".join(
                f"[{s or '某人'}] {m}"
                for m, s in zip(self._batch_messages, self._batch_senders)
            )
            message = combined
            self._pending_message = combined
            self.logger.info(f"📦 批量消息 ({len(self._batch_messages)}条): {combined}")
        elif self._batch_messages:
            message = self._batch_messages[0]
        else:
            message = self._pending_message

        # 清空批量缓存
        self._batch_messages.clear()
        self._batch_senders.clear()

        if self._is_friend_request(message):
            self.state = State.ADDING_FRIEND
            return

        self.logger.info("🤔 AI 思考中...")
        context = self._batch_contexts if self._batch_contexts else self._pending_context
        reply = self.dialogue.chat(message, context=context)
        self._pending_reply = reply
        self._pending_context = []
        self._batch_contexts = []
        self.logger.info(f"💡 AI回复: {reply}")
        if self._gui_status:
            self._gui_status.chat("ai", reply)
        self.state = State.OPENING_CHAT

    def _handle_opening_chat(self, screen: np.ndarray):
        """打开聊天输入框"""
        self.logger.info("📝 打开聊天输入框...")

        if self.pc_native:
            # PC 原生：C键呼出对话框（开关式，只按一次）→ Enter进入输入框
            open_key = self._key_bindings.get("open_chat", "c")
            enter_key = self._key_bindings.get("enter_input", "enter")
            if not self._chat_ui_open:
                self.logger.info(f"按键 '{open_key}' 呼出聊天")
                self.input.press_key(open_key)
                time.sleep(0.4)
                self._chat_ui_open = True
            else:
                self.logger.info("聊天对话框已打开，跳过C键（避免切换关闭）")
            self.logger.info(f"按键 '{enter_key}' 进入输入框")
            self.input.press_key(enter_key)
            time.sleep(0.3)
            self._chat_open_time = time.time()
            self.state = State.TYPING
            return

        # 旧方式：模板匹配聊天按钮
        btn = self.chat_button_detector.detect(screen)
        if btn:
            self.logger.info(f"检测到聊天按钮 ({btn.confidence:.2f})，点击")
            self.input.click(*btn.center)
            time.sleep(0.5)
            self._chat_open_time = time.time()
            self.state = State.TYPING
            return

        # 方法2：回退 — 点击配置的输入框区域
        chat_input = self.config.get("chat_input", {}).get("region")
        if chat_input:
            cx = (chat_input[0] + chat_input[2]) // 2
            cy = (chat_input[1] + chat_input[3]) // 2
            self.logger.info(f"使用回退坐标点击输入区 ({cx},{cy})")
            self.input.click(cx, cy)
            time.sleep(0.5)
            self._chat_open_time = time.time()
            self.state = State.TYPING
            return

        # 方法3：点击屏幕左下角
        h, w = self.capture.get_resolution()
        self.logger.info(f"使用默认左下角坐标点击 ({w//6},{h-80})")
        self.input.click(w // 6, h - 80)
        time.sleep(0.5)
        self._chat_open_time = time.time()
        self.state = State.TYPING

    def _handle_typing(self, screen: np.ndarray):
        """输入回复文字"""
        reply = self._pending_reply
        if reply:
            self.logger.info(f"⌨️ 输入: {reply}")
            self.input.type_text(reply, interval=self.typing_speed)
            time.sleep(0.5)
        self.state = State.SENDING

    def _handle_sending(self, screen: np.ndarray):
        """按回车发送消息"""
        self.logger.info("✉️ 发送消息")
        if self.pc_native:
            # PC 原生：SendInput 驱动级发送 Enter
            time.sleep(0.2)
            self.input.press_key("enter")
            self.logger.info("→ Enter 发送（PC原生）")
        else:
            self.input.press_key("enter")
        self._last_send_time = time.time()

        # ===== 防止 AI 回复自己 =====
        # AI 发消息后，聊天区域会显示这条消息。
        # 必须在 OCR 扫描到之前注入 known_messages，否则 OCR 会当成"新消息"触发新一轮回复。
        if self._pending_reply:
            self.chat_ocr.mark_sent_message(self._pending_reply)
            self._handled_messages.append(self._pending_reply.strip())

        time.sleep(0.8)
        self.state = State.CLOSING_CHAT

    def _handle_closing_chat(self, screen: np.ndarray):
        """关闭聊天面板"""
        if self.pc_native:
            close_key = self._key_bindings.get("close_chat", "esc")
            self.logger.info(f"↩️ 按 '{close_key}' 收起输入框（对话框保持打开）")
            self.input.press_key(close_key)
            time.sleep(0.3)
            # 注意：不重置 _chat_ui_open！只有收起输入框，对话框仍开着
            # 下次 OPENING_CHAT 会跳过 C 键，直接按 Enter 进入输入
            self.state = State.SEATED
            return

        back_btn = self.config.get("back_button", {}).get("region")
        if back_btn:
            cx = (back_btn[0] + back_btn[2]) // 2
            cy = (back_btn[1] + back_btn[3]) // 2
            self.logger.info(f"↩️ 点击返回按钮 ({cx},{cy})")
            self.input.click(cx, cy)
            time.sleep(0.5)
        else:
            self.logger.info("↩️ 按 ESC 关闭聊天")
            self.input.press_key("esc")
            time.sleep(0.3)
        self.state = State.SEATED

    def _handle_adding_friend(self, screen: np.ndarray):
        """添加好友"""
        self.logger.info("👋 执行加好友流程...")
        self.friend_manager.send_friend_request()
        self.state = State.SEATED

    def _handle_error(self, screen: np.ndarray):
        """错误恢复"""
        self.logger.warning("错误状态，恢复到 IDLE")
        self.state = State.IDLE

    # ==================== GUI 通知 ====================

    def _notify_gui(self):
        if self._gui_status:
            state_names = {
                State.IDLE: "idle",
                State.APPROACHING: "seated",
                State.SEATED: "seated",
                State.OPENING_BUBBLE: "thinking",
                State.READING_BUBBLE: "thinking",
                State.THINKING: "thinking",
                State.OPENING_CHAT: "thinking",
                State.TYPING: "responding",
                State.SENDING: "responding",
                State.CLOSING_CHAT: "responding",
                State.ADDING_FRIEND: "idle",
                State.ERROR: "error",
            }
            detail_map = {
                State.IDLE: f"扫描椅子中... (第{self._idle_rounds}轮)",
                State.APPROACHING: "正在坐下...",
                State.SEATED: f"等待发言... (第{self._conversation_rounds}轮)",
                State.OPENING_BUBBLE: "点击气泡查看消息...",
                State.READING_BUBBLE: "读取消息中...",
                State.THINKING: "AI 思考中...",
                State.OPENING_CHAT: "打开聊天中...",
                State.TYPING: "正在输入回复...",
                State.SENDING: "发送消息中...",
                State.CLOSING_CHAT: "关闭聊天面板...",
                State.ADDING_FRIEND: "添加好友中...",
            }
            state_name = state_names.get(self.state, "idle")
            detail = detail_map.get(self.state, "")
            self._gui_status.update(state_name, detail)

    # ==================== 辅助方法 ====================

    def _is_friend_request(self, message: str) -> bool:
        if not message:
            return False
        keywords = ["加好友", "加个好友", "交个朋友", "加我", "add friend", "be friend"]
        return any(kw in message.lower() for kw in keywords)

    def _human_like_action(self):
        import random
        action = random.choice(["idle", "look_around"])
        if action == "look_around":
            w, h = self.capture.get_resolution()
            offset_x = random.randint(-80, 80)
            offset_y = random.randint(-30, 30)
            self.input.move_to(w // 2 + offset_x, h // 2 + offset_y, duration=0.3)
            self.logger.debug("👀 模拟转动视角")

    def _save_debug_screenshot(self, screen: np.ndarray):
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{ts}_{self.state.name}.png"
        filepath = os.path.join(self.screenshot_dir, filename)
        cv2.imwrite(filepath, screen)
        self._cleanup_old_screenshots()

    def _cleanup_old_screenshots(self):
        """滚动清理：只保留最近 N 张截图，删除更早的"""
        try:
            files = sorted(
                [f for f in os.listdir(self.screenshot_dir) if f.endswith(".png")],
                key=lambda f: os.path.getmtime(os.path.join(self.screenshot_dir, f)),
                reverse=True,  # 新的在前
            )
            for old in files[self.screenshot_max_keep:]:
                os.remove(os.path.join(self.screenshot_dir, old))
                self.logger.debug(f"🧹 清理旧截图: {old}")
        except Exception:
            pass

    @staticmethod
    def _parse_region(cfg_value) -> Optional[tuple]:
        """解析坐标配置，列表转元组"""
        if cfg_value and isinstance(cfg_value, (list, tuple)) and len(cfg_value) == 4:
            return tuple(cfg_value)
        return None

    def _signal_handler(self, signum, frame):
        self.logger.info("\n收到中断信号...")
        self._running = False

    def _shutdown(self):
        self.logger.info("📤 发送关闭消息...")
        try:
            if self.pc_native:
                # 快速发送 "系统已关闭"
                open_key = self._key_bindings.get("open_chat", "c")
                enter_key = self._key_bindings.get("enter_input", "enter")
                close_key = self._key_bindings.get("close_chat", "esc")
                self.input.press_key(open_key)
                time.sleep(0.3)
                self.input.press_key(enter_key)
                time.sleep(0.2)
                self.input.type_text("系统已关闭", interval=0.02)
                time.sleep(0.2)
                self.input.press_key("enter")
                time.sleep(0.3)
                self.input.press_key(close_key)
                self.logger.info("✅ 已发送：系统已关闭")
        except Exception as e:
            self.logger.warning(f"发送关闭消息失败: {e}")

        self.logger.info("光遇社交AI 已停止")
        self.logger.info(f"本次对话引擎历史 {len(self.dialogue.history)//2} 轮")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="光遇社交AI")
    parser.add_argument(
        "-c", "--config", default="config.yaml",
        help="配置文件路径 (默认: config.yaml)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"配置文件不存在: {args.config}")
        sys.exit(1)

    ai = SkySocialAI(args.config)
    ai.run()


if __name__ == "__main__":
    main()
