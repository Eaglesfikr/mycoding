"""
main.py — CLI 入口

功能：
  - 参数解析：--model, --plan, --resume, --yolo, --accept-edits, --dont-ask,
    --api-base, --max-cost, --max-turns
  - 加载 .env（python-dotenv），读取 API_KEY / API_BASE / MODEL
  - API 协议自动判断（路径含 /anthropic → Anthropic 协议，否则 OpenAI 协议）
  - REPL 交互循环：输入 → 调用 Agent → 输出
  - 一次性执行模式：python -m agents.main "prompt"
  - 退出命令：exit / quit
  - Ctrl+C 中断处理
"""

import io
import os
import sys
import argparse
import signal
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from agents.agent import Agent
from agents.ui import (
    print_welcome,
    print_goodbye,
    print_user_prompt,
    print_assistant_text,
    print_error,
    print_warning,
    print_info,
    print_cost,
    print_divider,
    start_spinner,
    stop_spinner,
)


# ── 常量 ────────────────────────────────────────────────────────
DOT_ENV_PATHS = [
    Path(".env"),
    Path.home() / ".env",
]


# ── 平台编码修复 ────────────────────────────────────────────
def _fix_platform_encoding() -> None:
    """修复终端 Unicode 编码问题

    不同平台有不同的编码问题：
    - Windows (原生)：控制台默认为 cp936/cp437，无法编码 emoji/中文
    - WSL/Linux：通常 UTF-8 正常，但如果 locale 非 UTF-8 则做检查
    """

    # 检查当前 stdout 编码，如果已经是 UTF-8 就跳过
    current_encoding = getattr(sys.stdout, "encoding", "").lower()
    if "utf" in current_encoding or "utf8" in current_encoding:
        return  # 系统已经是 UTF-8，无需修复

    # 尝试修复
    if sys.platform == "win32":
        _fix_windows_encoding()
    else:
        _fix_posix_encoding()


def _fix_windows_encoding() -> None:
    """Windows 原生终端的 UTF-8 编码修复"""
    # 方法 1: reconfigure（Python 3.7+）
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="surrogateescape")
        sys.stderr.reconfigure(encoding="utf-8", errors="surrogateescape")
        return
    except (ValueError, AttributeError):
        pass

    # 方法 2: TextIOWrapper 包装兜底
    try:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="surrogateescape"
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="surrogateescape"
        )
    except (ValueError, AttributeError):
        pass

    # 方法 3: 环境变量（影响 subprocess）
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def _fix_posix_encoding() -> None:
    """POSIX (Linux/macOS/WSL) 的 locale 检查"""
    import locale

    try:
        current = locale.getlocale(locale.LC_CTYPE)
        if current and "UTF-8" not in str(current).upper():
            # 尝试设置为 UTF-8
            locale.setlocale(locale.LC_CTYPE, "C.UTF-8")
    except (locale.Error, ValueError):
        pass


# ── 信号处理 ────────────────────────────────────────────────────
_interrupted = False


def _handle_sigint(signum, frame) -> None:
    """Ctrl+C 处理：第一次中断当前操作，第二次退出"""
    global _interrupted
    if _interrupted:
        print("\n")
        print_goodbye()
        sys.exit(0)
    _interrupted = True
    print("\n")
    print_warning("按 Ctrl+C 再次确认退出…")
    stop_spinner()


# ── 参数解析 ────────────────────────────────────────────────────


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        prog="Mini-code",
        description=" mini Code — 自进化 Harness Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python -m agents.main                          # 进入 REPL\n"
            "  python -m agents.main \"你好\"                   # 一次性问答\n"
            "  python -m agents.main --model claude-sonnet-4  # 指定模型\n"
            "  python -m agents.main --api-base https://...   # 自定义 API 地址\n"
        ),
    )

    # 模型选择
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型名称 (默认从 .env 读取 MODEL，或 gpt-4o)",
    )

    # API 配置
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="API 地址 (默认从 .env 读取 API_BASE)",
    )

    # 执行模式
    parser.add_argument(
        "--plan",
        action="store_true",
        help="规划模式（只读，拒绝编辑和 Shell）",
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="跳过所有权限确认",
    )
    parser.add_argument(
        "--accept-edits",
        action="store_true",
        help="自动允许编辑操作",
    )
    parser.add_argument(
        "--dont-ask",
        action="store_true",
        help="自动拒绝所有操作（CI 模式）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="恢复最近会话（迭代四实现）",
    )

    # 限制
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="最大对话轮数 (默认 50)",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help="最大费用上限（美元）",
    )

    # 用户消息（一次性模式）
    parser.add_argument(
        "message",
        type=str,
        nargs="?",
        default=None,
        help="一次性执行的消息（不提供则进入 REPL）",
    )

    return parser.parse_args(argv)


# ── .env 加载 ───────────────────────────────────────────────────


def load_env_files() -> None:
    """从多个位置加载 .env 文件"""
    loaded = False
    for env_path in DOT_ENV_PATHS:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
            loaded = True

    if not loaded:
        # 尝试加载当前目录的 .env
        cwd_env = Path.cwd() / ".env"
        if cwd_env.exists():
            load_dotenv(dotenv_path=cwd_env, override=False)
            loaded = True

    if not loaded:
        print_warning("未找到 .env 文件，请创建 .env 并设置 API_KEY")


# ── Agent 创建 ─────────────────────────────────────────────────
def create_agent(args: argparse.Namespace) -> Agent:
    """根据命令行参数和 .env 配置创建 Agent"""
    api_key = os.environ.get("API_KEY", "")
    api_base = os.environ.get("API_BASE", "")
    model = os.environ.get("MODEL", "gpt-4o")

    # 命令行覆盖
    if args.api_base:
        api_base = args.api_base
    if args.model:
        model = args.model

    if not api_key:
        print_warning("API_KEY 未设置，某些模型可能无法调用")
    if not api_base:
        print_warning("API_BASE 未设置，将使用 SDK 默认地址")

    max_turns = args.max_turns or 50
    max_cost = args.max_cost

    # 构建 system prompt（后续迭代会替换为 prompt.py 的 build_system_prompt）
    system_parts = ["You are Mini Code, a self-evolving AI coding assistant."]

    if args.plan:
        system_parts.append("You are in plan mode. Read-only: you can read files and search, "
                            "but cannot edit files or run shell commands.")
    if args.yolo:
        system_parts.append("YOLO mode: all tools are automatically approved.")
    if args.accept_edits:
        system_parts.append("Accept-edits mode: file edits are automatically approved.")
    if args.dont_ask:
        system_parts.append("Dont-ask mode: all tool calls are automatically rejected.")

    return Agent(
        model=model,
        api_key=api_key,
        api_base=api_base,
        max_turns=max_turns,
        max_cost=max_cost,
        system_prompt="\n".join(system_parts),
    )


# ── REPL 循环 ───────────────────────────────────────────────────


def run_repl(agent: Agent) -> None:
    """交互式 REPL 循环"""
    print_welcome()
    turn = 0

    while turn < agent.max_turns:
        try:
            # 读取用户输入
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+D / Ctrl+C
            print()
            break

        # 空输入跳过
        if not user_input:
            continue

        # 退出命令
        if user_input.lower() in ("exit", "quit"):
            break

        turn += 1

        # 显示用户输入
        print_user_prompt(user_input)

        # 调用 Agent
        try:
            start_spinner("Thinking…")
            reply = agent.chat(user_input)
            stop_spinner()
            print_assistant_text(reply)
            print_cost(agent.get_cost_string())
        except Exception as e:
            stop_spinner()
            print_error(f"调用失败: {e}")
            continue

        print_divider()

    print_goodbye()


def run_once(agent: Agent, message: str) -> None:
    """一次性执行模式"""
    print_info(f"模型: {agent.model} | API: {agent.api_base or '(default)'}")

    try:
        start_spinner("Thinking…")
        reply = agent.chat(message)
        stop_spinner()
        print_assistant_text(reply)
        print_cost(agent.get_cost_string())
    except Exception as e:
        stop_spinner()
        print_error(f"调用失败: {e}")
        sys.exit(1)


# ── 入口 ────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> None:
    """主入口"""
    # 平台编码修复
    _fix_platform_encoding()

    # 注册 Ctrl+C 处理器
    signal.signal(signal.SIGINT, _handle_sigint)

    # 加载 .env
    load_env_files()

    # 解析参数
    args = parse_args(argv)

    # 创建 Agent
    agent = create_agent(args)

    # 进入模式
    if args.message:
        run_once(agent, args.message)
    else:
        run_repl(agent)


if __name__ == "__main__":
    main()