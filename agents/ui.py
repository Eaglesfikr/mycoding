"""
ui.py — 终端输出 (Terminal UI)

提供美观的终端输出函数，包括欢迎信息、用户/助手对话气泡、
错误/警告/信息提示、工具调用展示、分隔线、Spinner 等。
"""

import sys
from typing import Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.theme import Theme

# ── 自定义颜色主题 ──────────────────────────────────────────────
custom_theme = Theme(
    {
        "info": "bold cyan",
        "warning": "bold yellow",
        "error": "bold red",
        "success": "bold green",
        "muted": "dim white",
        "cost": "bold magenta",
        "user": "bold blue",
        "assistant": "bold green",
        "tool_call": "bold yellow",
        "tool_result": "dim yellow",
        "divider": "dim white",
    }
)

_console = Console(theme=custom_theme, highlight=False)

# ── Spinner 实例 ────────────────────────────────────────────────
_spinner_progress: Optional[Progress] = None
_spinner_task_id: Optional[int] = None


def print_welcome() -> None:
    """打印欢迎界面"""
    welcome_text = Text()
    welcome_text.append("🧠 Mini Code", style="bold green")
    welcome_text.append("\n自进化 Harness Agent", style="muted")
    welcome_text.append("\n输入 ", style="")
    welcome_text.append("exit", style="bold yellow")
    welcome_text.append(" 或 ", style="")
    welcome_text.append("quit", style="bold yellow")
    welcome_text.append(" 退出\n", style="")
    _console.print()
    _console.print(
        Panel(welcome_text, border_style="green", padding=(1, 2))
    )
    _console.print()


def print_goodbye() -> None:
    """打印告别信息"""
    _console.print()
    _console.print(
        Panel(
            Text("👋 再见！", style="bold green"),
            border_style="green",
            padding=(1, 2),
        )
    )


def print_user_prompt(content: str) -> None:
    """打印用户输入"""
    text = Text()
    text.append("You", style="user")
    text.append(f"\n{content}")
    _console.print()
    _console.print(Panel(text, border_style="blue", padding=(1, 2)))


def print_assistant_text(content: str) -> None:
    """打印助手文本回复（支持 Markdown）"""
    _console.print()
    _console.print(
        Panel(
            Markdown(content) if content.strip() else Text("(空回复)", style="muted"),
            title="🤖 Mini Code",
            title_align="left",
            border_style="green",
            padding=(1, 2),
        )
    )


def print_error(message: str) -> None:
    """打印错误信息"""
    _console.print(f"  [error]✖ {message}[/error]")


def print_warning(message: str) -> None:
    """打印警告信息"""
    _console.print(f"  [warning]⚠ {message}[/warning]")


def print_info(message: str) -> None:
    """打印普通信息"""
    _console.print(f"  [info]ℹ {message}[/info]")


def print_tool_call(name: str, args: dict) -> None:
    """打印工具调用信息"""
    _console.print(
        Panel(
            Syntax(
                f"Tool: {name}\nArgs: {args}",
                "python",
                theme="monokai",
                word_wrap=True,
            ),
            title="🔧 Tool Call",
            title_align="left",
            border_style="yellow",
            padding=(1, 2),
        )
    )


def print_tool_result(result: str) -> None:
    """打印工具执行结果"""
    # 截断过长的输出
    max_len = 2000
    display = result if len(result) <= max_len else result[:max_len] + "\n... (truncated)"
    _console.print(
        Panel(
            Syntax(display, "text", theme="monokai", word_wrap=True),
            title="📦 Tool Result",
            title_align="left",
            border_style="dim yellow",
            padding=(1, 2),
        )
    )


def print_cost(cost_info: str) -> None:
    """打印费用信息"""
    _console.print(f"  [cost]💰 {cost_info}[/cost]")


def print_divider(char: str = "━") -> None:
    """打印分隔线"""
    _console.print()
    _console.print(f"  [divider]{char * 60}[/divider]")


def start_spinner(text: str = "Thinking…") -> None:
    """启动加载动画

    使用 rich 的 Progress + SpinnerColumn 实现非阻塞 Spinner。
    """
    global _spinner_progress, _spinner_task_id
    stop_spinner()  # 确保先停止旧的
    _spinner_progress = Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=_console,
    )
    _spinner_progress.__enter__()
    _spinner_task_id = _spinner_progress.add_task(text, total=None)


def stop_spinner() -> None:
    """停止加载动画"""
    global _spinner_progress, _spinner_task_id
    if _spinner_progress is not None:
        try:
            _spinner_progress.__exit__(None, None, None)
        except Exception:
            pass
        _spinner_progress = None
        _spinner_task_id = None