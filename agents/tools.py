"""
tools.py — 工具定义与执行 (Tool System)

功能：
  - ToolDef 类型定义（Anthropic tool schema 格式）
  - 10 个内置工具：read_file / write_file / edit_file / list_files /
    grep_search / run_shell / compact_context / agent / skill / tool_search
  - execute_tool() 工具执行路由
  - PermissionMode 权限模式 + check_permission() 权限判断
  - CONCURRENCY_SAFE_TOOLS / READ_ONLY_TOOLS / DEFERRED_TOOLS
  - get_active_tool_definitions() 过滤延迟工具
"""

import os
import re
import sys
import json
import glob as glob_module
import subprocess
from pathlib import Path
from enum import Enum
from typing import Any, Optional, Callable


# ── 类型定义 ──────────────────────────────────────────────────────

class PermissionMode(Enum):
    """权限模式"""
    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    BYPASS = "bypassPermissions"
    DONT_ASK = "dontAsk"
    PLAN = "plan"


# ── 工具分类常量 ──────────────────────────────────────────────

READ_ONLY_TOOLS = frozenset({
    "read_file",
    "list_files",
    "grep_search",
    "compact_context",
    "tool_search",
})

CONCURRENCY_SAFE_TOOLS = frozenset({
    "read_file",
    "list_files",
    "grep_search",
    "compact_context",
})

DEFERRED_TOOLS = frozenset({"skill", "tool_search"})


# ── 权限判断 ──────────────────────────────────────────────────────

def check_permission(tool_name: str, mode: PermissionMode) -> str:
    """检查工具调用权限，返回 'allow' / 'deny' / 'ask'"""
    if mode == PermissionMode.BYPASS:
        return "allow"
    if mode == PermissionMode.DONT_ASK:
        return "deny"
    if mode == PermissionMode.PLAN:
        return "allow" if tool_name in READ_ONLY_TOOLS else "deny"
    if mode == PermissionMode.ACCEPT_EDITS:
        if tool_name in ("write_file", "edit_file"):
            return "allow"
        return "ask"
    # DEFAULT: always ask
    return "ask"


# ── 工具实现 ──────────────────────────────────────────────────────

def read_file_impl(path: str, offset: Optional[int] = None,
                   limit: Optional[int] = None) -> str:
    """读取文件内容"""
    file_path = Path(path).expanduser().resolve()

    if not file_path.exists():
        return f"<tool-error>File not found: {path}</tool-error>"
    if not file_path.is_file():
        return f"<tool-error>Not a file: {path}</tool-error>"

    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"<tool-error>Failed to read {path}: {e}</tool-error>"

    lines = text.splitlines(keepends=True)
    total_lines = len(lines)

    # Apply offset (1-based)
    start = (offset - 1) if offset and offset > 0 else 0
    if start >= total_lines:
        # show the last line if offset beyond range
        start = max(0, total_lines - 1)

    # Apply limit
    if limit and limit > 0:
        end = start + limit
    else:
        end = total_lines

    selected = lines[start:end]
    # Truncate if too many lines
    MAX_LINES = 4000
    truncated = False
    if len(selected) > MAX_LINES:
        selected = selected[:MAX_LINES]
        truncated = True

    # Build output with line numbers
    line_start = start + 1
    numbered = []
    for i, line in enumerate(selected):
        numbered.append(f"{line_start + i:>6} {line}")

    output = "".join(numbered)
    if truncated:
        output += f"\n... (truncated, showing {MAX_LINES} of {total_lines} lines)"

    return (
        f"<file-content path=\"{file_path}\" lines=\"{line_start}-{line_start + len(selected) - 1}\">\n"
        f"{output}\n"
        f"</file-content>"
    )


def write_file_impl(path: str, content: str) -> str:
    """写入/覆盖文件"""
    file_path = Path(path).expanduser().resolve()

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        byte_count = len(content.encode("utf-8"))
        return (
            f"<file-write path=\"{file_path}\" bytes=\"{byte_count}\">\n"
            f"Successfully wrote {byte_count} bytes to {file_path}\n"
            f"</file-write>"
        )
    except Exception as e:
        return f"<tool-error>Failed to write {path}: {e}</tool-error>"


def edit_file_impl(path: str, old_string: str, new_string: str,
                   replace_all: bool = False) -> str:
    """精确字符串替换编辑"""
    file_path = Path(path).expanduser().resolve()

    if not file_path.exists():
        return f"<tool-error>File not found: {path}</tool-error>"
    if not file_path.is_file():
        return f"<tool-error>Not a file: {path}</tool-error>"

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"<tool-error>Failed to read {path}: {e}</tool-error>"

    if old_string not in content:
        return (
            f"<tool-error>old_string not found in {path}.\n"
            f"Searched for: {old_string!r}\n"
            f"Make sure the text matches exactly (including whitespace).</tool-error>"
        )

    count = content.count(old_string)

    if replace_all:
        new_content = content.replace(old_string, new_string)
        replacement_count = count
    else:
        new_content = content.replace(old_string, new_string, 1)
        replacement_count = 1

    try:
        file_path.write_text(new_content, encoding="utf-8")
    except Exception as e:
        return f"<tool-error>Failed to write {path}: {e}</tool-error>"

    return (
        f"<file-edit path=\"{file_path}\" replacements=\"{replacement_count}\">\n"
        f"Replaced {replacement_count} occurrence(s) in {file_path}\n"
        f"</file-edit>"
    )


def list_files_impl(pattern: str, path: Optional[str] = None) -> str:
    """Glob 文件查找"""
    search_root = Path(path).expanduser().resolve() if path else Path.cwd()

    if not search_root.exists():
        return f"<tool-error>Directory not found: {path or '.'}</tool-error>"

    glob_pattern = str(search_root / pattern)
    matches = sorted(glob_module.glob(glob_pattern, recursive=True))

    if not matches:
        return f"No files matched `{pattern}` in {search_root}"

    # Format output
    lines = [f"Found {len(matches)} file(s) matching `{pattern}` in {search_root}:\n"]
    for m in matches:
        rel = Path(m).relative_to(search_root) if path else Path(m).name
        size = Path(m).stat().st_size if Path(m).is_file() else 0
        lines.append(f"  {rel}  ({size:,} bytes)" if Path(m).is_file() else f"  {rel}/")

    return "\n".join(lines)


def grep_search_impl(pattern: str, path: Optional[str] = None,
                     include: Optional[str] = None,
                     output_mode: str = "content",
                     context: int = 0) -> str:
    """内容正则搜索（类似 ripgrep）"""
    search_root = Path(path).expanduser().resolve() if path else Path.cwd()

    if not search_root.exists():
        return f"<tool-error>Directory not found: {path or '.'}</tool-error>"

    try:
        regex = re.compile(pattern, re.MULTILINE)
    except re.error as e:
        return f"<tool-error>Invalid regex pattern: {e}</tool-error>"

    # Collect files
    all_files = []
    if search_root.is_file():
        all_files = [search_root]
    else:
        for root, dirs, files in os.walk(search_root):
            # Skip common binary/vcs directories
            dirs[:] = [d for d in dirs if not d.startswith((".", "__pycache__", "node_modules"))]
            for fname in files:
                fpath = Path(root) / fname
                # Apply include glob filter
                if include:
                    if not glob_module.fnmatch.fnmatch(fname, include):
                        continue
                all_files.append(fpath)

    # Search each file
    results: list[dict] = []
    for fpath in all_files:
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        lines = text.splitlines(keepends=True)
        for i, line in enumerate(lines):
            if regex.search(line):
                results.append({
                    "file": str(fpath),
                    "line_num": i + 1,
                    "text": line.rstrip("\n\r"),
                    "context_before": [],
                    "context_after": [],
                })
                # Add context lines
                if context > 0:
                    for ci in range(1, context + 1):
                        if i - ci >= 0:
                            results[-1]["context_before"].append(f"{i - ci + 1:>6} {lines[i - ci].rstrip()}")
                        if i + ci < len(lines):
                            results[-1]["context_after"].append(f"{i + ci + 1:>6} {lines[i + ci].rstrip()}")

    if not results:
        return f"No matches found for `{pattern}` in {search_root}"

    # Output formatting
    if output_mode == "files_with_matches":
        files_matched = sorted(set(r["file"] for r in results))
        return f"Found {len(files_matched)} file(s) matching `{pattern}`:\n" + "\n".join(files_matched)

    if output_mode == "count":
        from collections import Counter
        counter = Counter(r["file"] for r in results)
        counts = sorted(counter.items())
        return f"Matches for `{pattern}`:\n" + "\n".join(f"  {f}: {c}" for f, c in counts)

    # Default: content mode
    lines_out = [f"Found {len(results)} match(es) for `{pattern}`:\n"]
    current_file = None
    for r in results:
        if r["file"] != current_file:
            current_file = r["file"]
            lines_out.append(f"\n── {current_file} ──\n")
        if r["context_before"]:
            for ctx in r["context_before"]:
                lines_out.append(f"  {ctx}")
        lines_out.append(f"  >{r['line_num']:>5} | {r['text']}")
        if r["context_after"]:
            for ctx in r["context_after"]:
                lines_out.append(f"  {ctx}")

    return "\n".join(lines_out)


def run_shell_impl(command: str, timeout: int = 120,
                   description: Optional[str] = None) -> str:
    """执行 Shell 命令（WSL 自适应）"""
    # Platform detection
    if sys.platform == "win32":
        cmd_list = ["wsl", "bash", "-c", command]
    else:
        cmd_list = ["bash", "-c", command]

    try:
        result = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd(),
        )
    except subprocess.TimeoutExpired:
        return (
            f"<tool-error>Command timed out after {timeout}s:\n"
            f"```bash\n{command}\n```</tool-error>"
        )
    except FileNotFoundError as e:
        return f"<tool-error>Shell not found: {e}</tool-error>"
    except Exception as e:
        return f"<tool-error>Execution error: {e}</tool-error>"

    # Format output
    output_parts = []
    if result.stdout:
        output_parts.append(result.stdout)
    if result.stderr:
        output_parts.append(f"<stderr>\n{result.stderr}")

    full_output = "".join(output_parts)

    # Truncate if too long
    MAX_OUTPUT_LEN = 10000
    truncated = False
    if len(full_output) > MAX_OUTPUT_LEN:
        full_output = full_output[:MAX_OUTPUT_LEN]
        truncated = True

    if truncated:
        full_output += f"\n... (truncated, exceeded {MAX_OUTPUT_LEN} chars)"

    prefix = f"Exit code: {result.returncode}"
    if description:
        prefix = f"$ {description}\n{prefix}"

    return f"{prefix}\n{full_output}"


def compact_context_impl(reason: Optional[str] = None) -> str:
    """手动触发上下文压缩（存根）"""
    return "(stub) Context compaction not yet implemented (iteration 4)."


def agent_impl(name: str, prompt: str,
               isolation: bool = False) -> str:
    """子 Agent 调用（存根）"""
    return "(stub) Sub-agent execution not yet implemented (iteration 8)."


def skill_impl(name: str, args: Optional[str] = None) -> str:
    """Skill 执行（延迟工具存根）"""
    return "(deferred) Skill execution not yet implemented (iteration 6)."


def tool_search_impl(query: str) -> str:
    """工具搜索（延迟工具存根）"""
    return "(deferred) Tool search not yet implemented (iteration 6)."


# ── 执行路由 ──────────────────────────────────────────────────────

TOOL_IMPL_MAP: dict[str, Callable[..., str]] = {
    "read_file": read_file_impl,
    "write_file": write_file_impl,
    "edit_file": edit_file_impl,
    "list_files": list_files_impl,
    "grep_search": grep_search_impl,
    "run_shell": run_shell_impl,
    "compact_context": compact_context_impl,
    "agent": agent_impl,
    "skill": skill_impl,
    "tool_search": tool_search_impl,
}


def execute_tool(name: str, args: dict) -> str:
    """工具执行路由

    根据工具名查找实现，处理异常并返回字符串结果。
    所有异常都被捕获并返回错误字符串（不会中断 Agent 循环）。
    """
    impl = TOOL_IMPL_MAP.get(name)
    if impl is None:
        return f"<tool-error>Unknown tool: {name}</tool-error>"
    try:
        return impl(**args)
    except TypeError as e:
        return f"<tool-error>Invalid arguments for {name}: {e}\nArgs: {json.dumps(args, ensure_ascii=False)}</tool-error>"
    except Exception as e:
        return (
            f"<tool-error>{type(e).__name__} executing {name}: {e}\n"
        )


# ── 工具定义列表 ──────────────────────────────────────────────────

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file at the given path. "
            "Use offset (1-based) and limit to read specific line ranges. "
            "The output includes line numbers. For large files, the output is truncated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file",
                },
                "offset": {
                    "type": "integer",
                    "description": "1-based line number to start reading from",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write content to a file, creating it if it doesn't exist. "
            "Parent directories are created automatically. "
            "This will OVERWRITE the file if it already exists."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Edit a file by performing exact string replacements. "
            "By default (replace_all=false), only the first occurrence is replaced. "
            "Set replace_all=true to replace all occurrences. "
            "Returns an error if old_string is not found."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file",
                },
                "old_string": {
                    "type": "string",
                    "description": "Exact text to search for (including whitespace)",
                },
                "new_string": {
                    "type": "string",
                    "description": "Text to replace with",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences instead of just the first",
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "list_files",
        "description": (
            "List files matching a glob pattern. "
            "Supports recursive patterns with ** (e.g., '**/*.py'). "
            "Returns file paths sorted alphabetically with file sizes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match (e.g. '**/*.py', 'src/**/*.ts')",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (defaults to current working directory)",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "grep_search",
        "description": (
            "Search file contents using regular expressions. "
            "Supports context lines, various output modes (content/files_with_matches/count), "
            "and file type filtering via include glob."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression to search for",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in (defaults to current directory)",
                },
                "include": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g. '*.py', '*.{ts,js}')",
                },
                "output_mode": {
                    "type": "string",
                    "description": "'content' (default, shows matching lines), 'files_with_matches' (filenames only), or 'count' (count per file)",
                    "enum": ["content", "files_with_matches", "count"],
                },
                "context": {
                    "type": "integer",
                    "description": "Number of context lines before and after each match",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "run_shell",
        "description": (
            "Execute a shell command and return its output. "
            "The command runs in bash via WSL. "
            "Use for running scripts, build commands, git operations, etc. "
            "Output is limited to 10000 characters."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 120)",
                },
                "description": {
                    "type": "string",
                    "description": "Short description of what the command does",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "compact_context",
        "description": (
            "Manually trigger context compression when the conversation is getting long. "
            "Folds old messages into structured session memory to free up context window. "
            "Only use this when the conversation has exceeded ~60% of the context window."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why context compression is needed",
                },
            },
            "required": [],
        },
    },
    {
        "name": "agent",
        "description": (
            "Launch a sub-agent to handle a specific task independently. "
            "The sub-agent works in the background and reports back when done. "
            "Use for tasks that can run in parallel or that need focused attention."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "A short name for the sub-agent (e.g. 'explore', 'plan')",
                },
                "prompt": {
                    "type": "string",
                    "description": "The task description for the sub-agent",
                },
                "isolation": {
                    "type": "string",
                    "description": "Isolation mode: 'worktree' for isolated git worktree",
                    "enum": ["none", "worktree"],
                },
            },
            "required": ["name", "prompt"],
        },
    },
    {
        "name": "skill",
        "description": (
            "Execute a reusable skill (a packaged task template with instructions). "
            "Skills are discovered from project or user skill directories."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the skill to execute",
                },
                "args": {
                    "type": "string",
                    "description": "Arguments to pass to the skill",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "tool_search",
        "description": (
            "Search for available MCP tools or built-in tools by query. "
            "Returns tool definitions matching the search criteria."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find tools",
                },
            },
            "required": ["query"],
        },
    },
]


def get_active_tool_definitions() -> list[dict]:
    """返回当前可用的工具定义列表（排除延迟工具）"""
    return [t for t in TOOL_DEFINITIONS if t["name"] not in DEFERRED_TOOLS]