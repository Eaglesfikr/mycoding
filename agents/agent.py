"""
agent.py — Agent 主循环

核心功能：
  - Agent 类，封装模型调用（OpenAI / Anthropic 双协议）
  - chat(user_message) 主方法
  - 指数退避重试（429 / 503 / 529）
  - 上下文窗口常量表
  - clear_history() / show_cost()
"""

import os
import time
from typing import Optional

# ── 模型上下文窗口常量表 ──────────────────────────────────────
# (model_prefix, context_window, input_price_per_1M, output_price_per_1M)
# 价格单位：美元 / 百万 token
# Qwen 系列原始价格单位为 RMB（~7 RMB/USD），此处已折算为 USD
MODEL_REGISTRY: list[tuple[str, int, float, float]] = [
    # ── 阿里云百炼 DashScope（通义千问 Qwen）──
    # Qwen3.8-Max
    ("qwen3.8-max", 131072, 1.71, 5.14),
    ("qwen3.8-flash", 131072, 0.11, 0.39),
    # Qwen3.7-Max
    ("qwen3.7-max", 131072, 1.71, 5.14),
    ("qwen3.7-plus", 131072, 0.30, 0.70),
    ("qwen3.7-flash", 131072, 0.02, 0.05),
    # Qwen-Max（旧版）
    ("qwen-max", 32768, 2.00, 6.00),
    # Qwen-Plus
    ("qwen-plus", 131072, 0.50, 2.00),
    # Qwen-Turbo
    ("qwen-turbo", 131072, 0.30, 0.60),
    # QwQ-Plus（推理专用）
    ("qwq-plus", 131072, 0.23, 0.57),
    # Qwen-Coder-Plus
    ("qwen-coder-plus", 131072, 0.50, 1.00),
    # Qwen-Long（长文本）
    ("qwen-long", 10000000, 0.07, 0.29),
    # 通义千问旧版兜底
    ("qwen", 131072, 0.50, 2.00),
    # ── Anthropic Claude ──
    ("claude-opus-5", 200000, 15.00, 75.00),
    ("claude-sonnet-5", 200000, 3.00, 15.00),
    ("claude-haiku-4", 200000, 0.80, 4.00),
    ("claude-opus-4", 200000, 15.00, 75.00),
    ("claude-sonnet-4", 200000, 3.00, 15.00),
    ("claude-haiku-3", 200000, 0.25, 1.25),
    ("claude-3-opus", 200000, 15.00, 75.00),
    ("claude-3-sonnet", 200000, 3.00, 15.00),
    ("claude-3-haiku", 200000, 0.25, 1.25),
    ("claude-2", 100000, 8.00, 24.00),
    ("claude-instant", 100000, 0.80, 2.40),
    # ── OpenAI GPT ──
    ("gpt-4o", 128000, 2.50, 10.00),
    ("gpt-4o-mini", 128000, 0.15, 0.60),
    ("gpt-4-turbo", 128000, 10.00, 30.00),
    ("gpt-4", 8192, 30.00, 60.00),
    ("gpt-3.5-turbo", 16384, 0.50, 1.50),
    ("o1", 200000, 15.00, 60.00),
    ("o3-mini", 200000, 1.10, 4.40),
    # ── DeepSeek ──
    ("deepseek", 65536, 0.14, 0.28),
    ("deepseek-chat", 65536, 0.14, 0.28),
    ("deepseek-reasoner", 65536, 0.55, 2.19),
    # 通用兜底
    ("default", 128000, 2.00, 8.00),
]

# ── 消息角色 ──────────────────────────────────────────────
SYSTEM_ROLE = "system"
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"
TOOL_ROLE = "tool"


def _lookup_model(model_name: str) -> tuple[int, float, float]:
    """查找模型的上下文窗口和价格，未匹配则返回 default

    按前缀长度降序匹配，确保 "gpt-4o-mini" 优先于 "gpt-4o"。
    """
    lower = model_name.lower()
    # 按前缀长度降序排序，避免 "gpt-4o" 吃掉 "gpt-4o-mini"
    sorted_registry = sorted(
        [r for r in MODEL_REGISTRY if r[0] != "default"],
        key=lambda r: len(r[0]),
        reverse=True,
    )
    for prefix, ctx, inp, out in sorted_registry:
        if lower.startswith(prefix):
            return ctx, inp, out
    # fallback to default
    for prefix, ctx, inp, out in MODEL_REGISTRY:
        if prefix == "default":
            return ctx, inp, out
    return 128000, 2.0, 8.0


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数（~4 chars = 1 token）"""
    return len(text) // 4 + 1


def _exponential_backoff(attempt: int, base: float = 2.0, max_delay: float = 60.0) -> float:
    """指数退避 + jitter"""
    delay = min(base * (2 ** (attempt - 1)), max_delay)
    jitter = delay * 0.1 * (hash(str(time.time_ns())) % 20 / 20.0)
    return delay + jitter


# ── Agent 类 ──────────────────────────────────────────────────
class Agent:
    """Agent 类，封装模型调用与对话管理。"""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str = "",
        api_base: str = "",
        max_turns: int = 50,
        max_cost: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("API_KEY", "")
        self.api_base = (api_base or os.environ.get("API_BASE", "")).rstrip("/")
        self.max_turns = max_turns
        self.max_cost = max_cost
        self.system_prompt = system_prompt or "You are Mini Code, a helpful AI assistant."

        # 对话历史
        self.messages: list[dict] = []

        # Token 统计
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # 模型信息
        self.context_window, self.input_price, self.output_price = _lookup_model(model)

        # 判断 API 协议
        self.is_anthropic = "/anthropic" in self.api_base.lower()

        # 初始化客户端
        self._init_client()

    def _init_client(self) -> None:
        """根据协议初始化对应的 SDK 客户端"""
        if self.is_anthropic:
            self._init_anthropic_client()
        else:
            self._init_openai_client()

    def _init_openai_client(self) -> None:
        """初始化 OpenAI-compatible 客户端"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "需要安装 openai SDK: pip install openai>=1.0.0"
            )
        self._openai_client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        self._openai_client_kwargs: dict = {}

    def _init_anthropic_client(self) -> None:
        """初始化 Anthropic 客户端"""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "需要安装 anthropic SDK: pip install anthropic>=0.25.0"
            )
        self._anthropic_client = anthropic.Anthropic(
            api_key=self.api_key,
            base_url=self.api_base,
        )
        self._anthropic_client_kwargs: dict = {}

    # ── 公共方法 ──────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """主方法：发送用户消息并返回助手回复"""
        # 添加用户消息
        self.messages.append({"role": USER_ROLE, "content": user_message})

        # 调用模型（含自动重试）
        reply_content = self._call_with_retry()

        # 添加助手回复
        if reply_content is not None:
            self.messages.append({"role": ASSISTANT_ROLE, "content": reply_content})

            # 成本检查
            if self.max_cost is not None:
                cost = self.show_cost(raw=True)
                if cost > self.max_cost:
                    raise RuntimeError(
                        f"费用已达上限 ${cost:.4f} (max_cost=${self.max_cost:.4f})"
                    )

        return reply_content or ""

    def clear_history(self) -> None:
        """清空对话历史"""
        self.messages.clear()
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def show_cost(self, raw: bool = False) -> float:
        """显示或返回累计费用"""
        input_cost = (self.total_input_tokens / 1_000_000) * self.input_price
        output_cost = (self.total_output_tokens / 1_000_000) * self.output_price
        total = input_cost + output_cost

        if raw:
            return total

        return total

    def get_cost_string(self) -> str:
        """返回格式化费用字符串"""
        input_cost = (self.total_input_tokens / 1_000_000) * self.input_price
        output_cost = (self.total_output_tokens / 1_000_000) * self.output_price
        total = input_cost + output_cost
        return (
            f"模型: {self.model} | "
            f"输入: {self.total_input_tokens:,} tokens (${input_cost:.4f}) | "
            f"输出: {self.total_output_tokens:,} tokens (${output_cost:.4f}) | "
            f"总计: ${total:.4f}"
        )

    # ── 内部方法 ──────────────────────────────────────────

    def _call_with_retry(self, max_retries: int = 5) -> Optional[str]:
        """调用模型，带指数退避重试"""
        last_error: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                if self.is_anthropic:
                    return self._call_anthropic()
                else:
                    return self._call_openai()
            except Exception as e:
                last_error = e
                status_code = self._extract_status_code(e)

                # 只在 429 / 503 / 529 且还有重试次数时重试
                if status_code in (429, 503, 529) and attempt < max_retries:
                    delay = _exponential_backoff(attempt)
                    time.sleep(delay)
                    continue

                # 其他错误 或 重试全部耗尽 → 直接抛出
                if status_code in (429, 503, 529):
                    # 重试耗尽，抛出 RuntimeError
                    raise RuntimeError(
                        f"模型调用失败（{max_retries} 次重试后）: {last_error}"
                    ) from last_error
                raise

    def _extract_status_code(self, error: Exception) -> int:
        """从异常中提取 HTTP 状态码"""
        err_str = str(error)
        for code_str in ["429", "503", "529", "401", "403", "400", "500"]:
            if code_str in err_str:
                return int(code_str)
        return 0

    def _call_openai(self) -> str:
        """使用 OpenAI-compatible API 调用"""
        messages = self._build_openai_messages()

        response = self._openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self._openai_client_kwargs,
        )

        # Token 统计
        if response.usage:
            self.total_input_tokens += response.usage.prompt_tokens or 0
            self.total_output_tokens += response.usage.completion_tokens or 0

        # 取回复内容
        choice = response.choices[0]
        if choice.message.content:
            return choice.message.content
        return ""

    def _call_anthropic(self) -> str:
        """使用 Anthropic API 调用"""
        import anthropic

        system_msg, messages = self._build_anthropic_messages()

        kwargs: dict = dict(
            model=self.model,
            max_tokens=4096,
            messages=messages,
        )
        if system_msg:
            kwargs["system"] = system_msg

        response = self._anthropic_client.messages.create(**kwargs)

        # Token 统计
        if response.usage:
            self.total_input_tokens += response.usage.input_tokens or 0
            self.total_output_tokens += response.usage.output_tokens or 0

        # 取回复内容
        content_parts = []
        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
        return "\n".join(content_parts)

    def _build_openai_messages(self) -> list[dict]:
        """构建 OpenAI 格式的 messages 列表"""
        msgs: list[dict] = []
        # System prompt
        msgs.append({"role": SYSTEM_ROLE, "content": self.system_prompt})
        # 对话历史
        msgs.extend(self.messages)
        return msgs

    def _build_anthropic_messages(self) -> tuple[Optional[str], list[dict]]:
        """构建 Anthropic 格式的 messages，返回 (system, messages)"""
        system_msg: Optional[str] = None
        msgs: list[dict] = []

        for msg in self.messages:
            role = msg["role"]
            if role == SYSTEM_ROLE:
                system_msg = msg["content"]
            elif role == USER_ROLE:
                msgs.append({"role": "user", "content": msg["content"]})
            elif role == ASSISTANT_ROLE:
                msgs.append({"role": "assistant", "content": msg["content"]})

        # 如果没有用户消息，加一个占位
        if not msgs:
            msgs.append({"role": "user", "content": "Hello!"})

        return system_msg or self.system_prompt, msgs