"""
LLM 对话引擎 - 接入 DeepSeek API 生成回复

使用 OpenAI 兼容接口调用 DeepSeek
"""

import requests
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class DialogueEngine:
    """对话引擎 - DeepSeek API"""

    def __init__(self, config: dict):
        ds_cfg = config.get("deepseek", {})
        persona_cfg = config.get("persona", {})
        conv_cfg = config.get("conversation", {})

        self.api_key = ds_cfg.get("api_key", "")
        self.model = ds_cfg.get("model", "deepseek-chat")
        self.base_url = ds_cfg.get("base_url", "https://api.deepseek.com/v1")
        self.max_tokens = ds_cfg.get("max_tokens", 200)
        self.temperature = ds_cfg.get("temperature", 0.8)

        # 构建系统提示词
        self.system_prompt = persona_cfg.get("system_prompt", "").format(
            name=persona_cfg.get("name", "小光"),
            style=persona_cfg.get("style", "热情友善"),
        )

        # 对话历史
        self.history: List[Dict[str, str]] = []
        # 最大历史条数（节省token）
        self.max_history = 20
        # OCR 上下文最大行数（截断过长的屏幕对话记录）
        self.max_context_lines = conv_cfg.get("max_context_lines", 30)

        logger.info(f"对话引擎就绪: model={self.model}, url={self.base_url}")
        if self.api_key == "sk-your-api-key-here":
            logger.warning("⚠ 请先在 config.yaml 中设置 DeepSeek API Key！")

    def chat(self, message: str, context: Optional[List[Dict]] = None) -> str:
        """
        发送消息并获取 AI 回复

        Args:
            message: 对方的最新一条消息
            context: 完整对话上下文，格式:
                [{"role": "context", "content": "[小明] 你好"},
                 {"role": "context", "content": "[小红] 嗨"}]

        Returns:
            AI 生成的回复文本
        """
        if not self.api_key or self.api_key == "sk-your-api-key-here":
            return "（请先配置API Key）你好呀！"

        # 构建消息列表
        messages = [{"role": "system", "content": self.system_prompt}]

        # 加入对话历史（旧的 user/assistant 对）
        messages.extend(self.history[-self.max_history:])

        # 如果有结构化对话上下文，注入到系统提示中
        # 让 LLM 能看到完整的 "谁说了什么" 上下文
        # 裁剪过长的上下文以节省 token
        if context:
            ctx_lines = [item["content"] for item in context if item.get("content")]
            if len(ctx_lines) > self.max_context_lines:
                ctx_lines = ctx_lines[-self.max_context_lines:]
                logger.debug(
                    f"📏 上下文裁剪 (LLM): {len(context)} → {len(ctx_lines)} 行"
                )
            if ctx_lines:
                context_block = (
                    "【当前屏幕上的对话记录（从上到下是最旧到最新）】\n"
                    + "\n".join(ctx_lines)
                    + "\n\n请按照以上对话记录，自然回复下面这个人发来的最新消息。"
                    + "要记住上面每个人说了什么，不要问重复的问题，也不要回复自己之前说过的话。"
                    + "你只需要针对最新的那条消息做回应，不要说多余的话。"
                )
                messages.append({"role": "system", "content": context_block})

        # 加入当前消息
        messages.append({"role": "user", "content": message})

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                reply = data["choices"][0]["message"]["content"].strip()

                # 保存对话历史
                self.history.append({"role": "user", "content": message})
                self.history.append({"role": "assistant", "content": reply})

                # 限制历史长度
                if len(self.history) > self.max_history * 2:
                    self.history = self.history[-self.max_history * 2:]

                logger.info(f"AI回复: {reply[:80]}...")
                return reply
            else:
                logger.error(f"API错误 ({response.status_code}): {response.text}")
                return "（API调用失败）"

        except requests.exceptions.Timeout:
            logger.error("API请求超时")
            return "（回复超时，稍等一下~）"
        except Exception as e:
            logger.error(f"API异常: {e}")
            return "（出错了，待会再聊吧~）"

    def chat_stream(self, message: str) -> str:
        """
        流式对话（简化版，实际返回完整回复）

        如需真正的流式输出请使用 SSE 解析
        """
        return self.chat(message)

    def reset_history(self):
        """重置对话历史"""
        self.history.clear()
        logger.info("对话历史已重置")

    def get_history_summary(self) -> str:
        """获取对话摘要（调试用）"""
        if not self.history:
            return "（无对话历史）"
        lines = []
        for msg in self.history:
            role = "对方" if msg["role"] == "user" else "AI"
            lines.append(f"[{role}] {msg['content'][:50]}")
        return "\n".join(lines)
