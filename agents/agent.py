"""
agent.py — Agent 主循环

核心功能：
  - Agent 类，封装模型调用（OpenAI / Anthropic 双协议）
  - chat(user_message) 主方法，支持多轮工具调用循环
  - 工具调用解析、权限检查、执行、结果回写
  - 指数退避重试（429 / 503 / 529）
  - 上下文窗口常量表
  - clear_history() / show_cost()
"""

import os
import json
import time
from typing import Optional

from agents.tools import (
    execute_tool,
    check_permission,
    get_active_tool_definitions,
    PermissionMode,
)
from agents.ui import (
    print_tool_call,
    print_tool_result,
    print_info,
    stop_spinner,
)

# ── 模型上下文窗口常量表 ──────────────────────────────────────
# (model_prefix, context_window, input_price_per_1M, output_price_per_1M)
# 价格单位：美元 / 百万 token
MODEL_REGISTRY: list[tuple[str, int, float, float]] = [
    # ── 阿里云百炼 DashScope（通义千问 Qwen）──
    ("qwen3.8-max", 131072, 1.71, 5.14),
    ("qwen3.8-flash", 131072, 0.11, 0.39),
    ("qwen3.7-max", 131072, 1.71, 5.14),
    ("qwen3.7-plus", 131072, 0.30, 0.70),
    ("qwen3.7-flash", 131072, 0.02, 0.05),
    ("qwen-max", 32768, 2.00, 6.00),
    ("qwen-plus", 131072, 0.50, 2.00),
    ("qwen-turbo", 131072, 0.30, 0.60),
    ("qwq-plus", 131072, 0.23, 0.57),
    ("qwen-coder-plus", 131072, 0.50, 1.00),
    ("qwen-long", 10000000, 0.07, 0.29),
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
    """查找模型的上下文窗口和价格，未匹配则返回 default"""
    lower = model_name.lower()
    sorted_registry = sorted(
        [r for r in MODEL_REGISTRY if r[0] != "default"],
        key=lambda r: len(r[0]),
        reverse=True,
    )
    for prefix, ctx, inp, out in sorted_registry:
        if lower.startswith(prefix):
            return ctx, inp, out
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
        permission_mode: PermissionMode = PermissionMode.DEFAULT,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("API_KEY", "")
        self.api_base = (api_base or os.environ.get("API_BASE", "")).rstrip("/")
        self.max_turns = max_turns
        self.max_cost = max_cost
        self.system_prompt = system_prompt or "You are Mini Code, a helpful AI assistant."
        self.permission_mode = permission_mode

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
            raise ImportError("需要安装 openai SDK: pip install openai>=1.0.0")
        self._openai_client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        self._openai_client_kwargs: dict = {}

    def _init_anthropic_client(self) -> None:
        """初始化 Anthropic 客户端"""
        try:
            import anthropic
        except ImportError:
            raise ImportError("需要安装 anthropic SDK: pip install anthropic>=0.25.0")
        self._anthropic_client = anthropic.Anthropic(
            api_key=self.api_key,
            base_url=self.api_base,
        )
        self._anthropic_client_kwargs: dict = {}

    # ── 公共方法 ──────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """主方法：发送用户消息，支持多轮工具调用循环"""
        self.messages.append({"role": USER_ROLE, "content": user_message})

        MAX_TOOL_TURNS = 10
        for turn in range(MAX_TOOL_TURNS):
            # 调用模型获取回复
            response = self._call_with_retry()

            # 解析工具调用
            if self.is_anthropic:
                tool_calls, text = self._parse_anthropic_tool_calls(response)
                assistant_msg = self._build_anthropic_assistant_msg(tool_calls, text)
            else:
                tool_calls, text = self._parse_openai_tool_calls(response)
                assistant_msg = self._build_openai_assistant_msg(tool_calls, text)

            # 添加助手消息到历史
            self.messages.append(assistant_msg)

            # 纯文本回复 → 结束
            if not tool_calls:
                return text

            # ── 处理工具调用 ──
            stop_spinner()

            for tc in tool_calls:
                tc_name = tc["name"]
                tc_args = tc["args"]

                # 显示工具调用
                print_tool_call(tc_name, tc_args)

                # 权限检查
                perm = check_permission(tc_name, self.permission_mode)

                if perm == "deny":
                    result = (
                        f"<tool-error>Permission denied: {tc_name} "
                        f"(mode={self.permission_mode.value})</tool-error>"
                    )
                    print_tool_result(result)
                elif perm == "ask":
                    approved = self._prompt_user_approval(tc_name)
                    if not approved:
                        result = f"<tool-error>Permission denied by user: {tc_name}</tool-error>"
                    else:
                        result = execute_tool(tc_name, tc_args)
                    print_tool_result(result)
                else:  # allow
                    result = execute_tool(tc_name, tc_args)
                    print_tool_result(result)

                # 工具结果回写消息历史
                self._add_tool_result(tc, result)

            # 继续下一轮，让模型处理工具结果

        return f"Reached maximum tool call turns ({MAX_TOOL_TURNS})."

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
        return total if raw else total

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

    def _call_with_retry(self, max_retries: int = 5):
        """调用模型，带指数退避重试，返回完整 response 对象"""
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

                if status_code in (429, 503, 529) and attempt < max_retries:
                    delay = _exponential_backoff(attempt)
                    time.sleep(delay)
                    continue

                if status_code in (429, 503, 529):
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

    # ── API 调用方法 ───────────────────────────────────────

    def _call_openai(self):
        """使用 OpenAI-compatible API 调用，返回完整 response"""
        messages = self._build_openai_messages()

        # 构建 tools 参数
        tools = []
        for t in get_active_tool_definitions():
            tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            })

        kwargs = dict(
            model=self.model,
            messages=messages,
            **self._openai_client_kwargs,
        )
        if tools:
            kwargs["tools"] = tools

        response = self._openai_client.chat.completions.create(**kwargs)

        # Token 统计
        if response.usage:
            self.total_input_tokens += response.usage.prompt_tokens or 0
            self.total_output_tokens += response.usage.completion_tokens or 0

        return response

    def _call_anthropic(self):
        """使用 Anthropic API 调用，返回完整 response"""
        system_msg, messages = self._build_anthropic_messages()

        tools = get_active_tool_definitions()

        kwargs: dict = dict(
            model=self.model,
            max_tokens=4096,
            messages=messages,
        )
        if system_msg:
            kwargs["system"] = system_msg
        if tools:
            kwargs["tools"] = tools

        response = self._anthropic_client.messages.create(**kwargs)

        # Token 统计
        if response.usage:
            self.total_input_tokens += response.usage.input_tokens or 0
            self.total_output_tokens += response.usage.output_tokens or 0

        return response

    # ── 工具调用解析 ───────────────────────────────────────

    def _parse_openai_tool_calls(self, response) -> tuple[list[dict], str]:
        """从 OpenAI response 解析 tool_calls，返回 (tool_calls, text)"""
        message = response.choices[0].message
        tool_calls = []
        text = message.content or ""

        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"_raw_args": tc.function.arguments}
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": args,
                })

        return tool_calls, text

    def _parse_anthropic_tool_calls(self, response) -> tuple[list[dict], str]:
        """从 Anthropic response 解析 tool_use，返回 (tool_calls, text)"""
        tool_calls = []
        text_parts = []

        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "args": dict(block.input),
                })
            elif block.type == "text":
                text_parts.append(block.text)

        return tool_calls, "\n".join(text_parts)

    # ── 消息构建 ──────────────────────────────────────────

    def _build_openai_assistant_msg(self, tool_calls: list[dict], text: str) -> dict:
        """构建 OpenAI 格式的助手消息（含 tool_calls）"""
        msg: dict = {"role": "assistant", "content": text}
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"]),
                    },
                }
                for tc in tool_calls
            ]
        return msg

    def _build_anthropic_assistant_msg(self, tool_calls: list[dict], text: str) -> dict:
        """构建 Anthropic 格式的助手消息（含 tool_use blocks）"""
        content: list[dict] = []
        if text:
            content.append({"type": "text", "text": text})
        for tc in tool_calls:
            content.append({
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["name"],
                "input": tc["args"],
            })
        return {"role": "assistant", "content": content}

    # ── 工具结果回写 ───────────────────────────────────────

    def _add_tool_result(self, tc: dict, result: str) -> None:
        """将工具执行结果添加到消息历史（双协议适配）"""
        if self.is_anthropic:
            # Anthropic 要求 tool_result 放在 user role 的 content block 中
            self.messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": result,
                    }
                ],
            })
        else:
            # OpenAI 使用 tool role
            self.messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

    # ── 用户权限确认 ───────────────────────────────────────

    def _prompt_user_approval(self, tool_name: str) -> bool:
        """向用户询问是否允许工具调用"""
        try:
            stop_spinner()
            response = input(f"  Allow {tool_name}? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return response in ("y", "yes")

    # ── 消息格式化 ────────────────────────────────────────

    def _build_openai_messages(self) -> list[dict]:
        """构建 OpenAI 格式的 messages 列表"""
        msgs: list[dict] = []
        msgs.append({"role": SYSTEM_ROLE, "content": self.system_prompt})
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
                # 保留 tool_calls 字段（Anthropic 将其编码为 content blocks）
                assistant_entry: dict = {"role": "assistant", "content": msg["content"]}
                msgs.append(assistant_entry)

        # 如果没有用户消息，加一个占位
        if not msgs:
            msgs.append({"role": "user", "content": "Hello!"})

        return system_msg or self.system_prompt, msgs