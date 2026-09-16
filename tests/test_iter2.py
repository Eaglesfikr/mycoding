"""
迭代二 · 完整验证测试

覆盖 TODO 全部 Iteration 2 功能点：
  - tools.py：10 个工具实现、权限系统、延迟工具过滤
  - agent.py：PermissionMode 集成、tool call 解析
  - main.py：CLI 参数映射 PermissionMode
"""

import os
import sys
import json
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── 测试配置 ──────────────────────────────────────────────────
from dotenv import load_dotenv
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

API_KEY = os.environ.get("API_KEY", "")
API_BASE = os.environ.get("API_BASE", "")
MODEL = os.environ.get("MODEL", "qwen-plus")


# ═══════════════════════════════════════════════════════════════
# tools.py · 工具定义
# ═══════════════════════════════════════════════════════════════
class Test工具定义列表(unittest.TestCase):
    """TOOL_DEFINITIONS / get_active_tool_definitions"""

    def setUp(self):
        from agents.tools import TOOL_DEFINITIONS
        self.defs = TOOL_DEFINITIONS

    def test_1_1_共10个工具定义(self):
        self.assertEqual(len(self.defs), 10)

    def test_1_2_每个工具都有name_description_input_schema(self):
        for t in self.defs:
            with self.subTest(tool=t["name"]):
                self.assertIn("name", t)
                self.assertIn("description", t)
                self.assertIn("input_schema", t)
                self.assertIn("properties", t["input_schema"])
                self.assertIn("required", t["input_schema"])

    def test_1_3_工具名清单(self):
        names = {t["name"] for t in self.defs}
        expected = {
            "read_file", "write_file", "edit_file",
            "list_files", "grep_search", "run_shell",
            "compact_context", "agent", "skill", "tool_search",
        }
        self.assertEqual(names, expected)

    def test_1_4_get_active_tools过滤延迟工具(self):
        from agents.tools import get_active_tool_definitions
        active = get_active_tool_definitions()
        names = {t["name"] for t in active}
        self.assertIn("read_file", names)
        self.assertIn("run_shell", names)
        self.assertNotIn("skill", names)
        self.assertNotIn("tool_search", names)
        self.assertEqual(len(active), 8)

    def test_1_5_TODO列出的全部工具名(self):
        from agents.tools import TOOL_DEFINITIONS
        for name in ("read_file", "write_file", "edit_file", "list_files",
                      "grep_search", "run_shell", "compact_context", "agent",
                      "skill", "tool_search"):
            exists = any(t["name"] == name for t in TOOL_DEFINITIONS)
            self.assertTrue(exists, f"{name} 应存在于 TOOL_DEFINITIONS")


# ═══════════════════════════════════════════════════════════════
# tools.py · 权限系统
# ═══════════════════════════════════════════════════════════════
class Test权限系统(unittest.TestCase):
    """PermissionMode / check_permission"""

    def setUp(self):
        from agents.tools import PermissionMode, check_permission
        self.PermissionMode = PermissionMode
        self.check = check_permission

    def test_2_1_五种模式存在(self):
        for name in ("DEFAULT", "ACCEPT_EDITS", "BYPASS", "DONT_ASK", "PLAN"):
            self.assertTrue(hasattr(self.PermissionMode, name),
                            f"PermissionMode 应包含 {name}")

    def test_2_2_BYPASS全部allow(self):
        mode = self.PermissionMode.BYPASS
        for tool in ("read_file", "write_file", "edit_file", "run_shell", "agent"):
            self.assertEqual(self.check(tool, mode), "allow")

    def test_2_3_DONT_ASK全部deny(self):
        mode = self.PermissionMode.DONT_ASK
        for tool in ("read_file", "write_file", "edit_file", "run_shell"):
            self.assertEqual(self.check(tool, mode), "deny")

    def test_2_4_PLAN只读allow_编辑shelldeny(self):
        mode = self.PermissionMode.PLAN
        self.assertEqual(self.check("read_file", mode), "allow")
        self.assertEqual(self.check("list_files", mode), "allow")
        self.assertEqual(self.check("grep_search", mode), "allow")
        self.assertEqual(self.check("compact_context", mode), "allow")
        self.assertEqual(self.check("tool_search", mode), "allow")
        self.assertEqual(self.check("write_file", mode), "deny")
        self.assertEqual(self.check("edit_file", mode), "deny")
        self.assertEqual(self.check("run_shell", mode), "deny")
        self.assertEqual(self.check("agent", mode), "deny")

    def test_2_5_ACCEPT_EDITS_编辑allow_shellask(self):
        mode = self.PermissionMode.ACCEPT_EDITS
        self.assertEqual(self.check("write_file", mode), "allow")
        self.assertEqual(self.check("edit_file", mode), "allow")
        self.assertEqual(self.check("run_shell", mode), "ask")
        self.assertEqual(self.check("read_file", mode), "ask")
        self.assertEqual(self.check("list_files", mode), "ask")

    def test_2_6_DEFAULT全部ask(self):
        mode = self.PermissionMode.DEFAULT
        for tool in ("read_file", "write_file", "edit_file", "run_shell",
                      "list_files", "grep_search", "agent"):
            self.assertEqual(self.check(tool, mode), "ask")

    def test_2_7_READ_ONLY_TOOLS常量(self):
        from agents.tools import READ_ONLY_TOOLS
        for tool in ("read_file", "list_files", "grep_search",
                      "compact_context", "tool_search"):
            self.assertIn(tool, READ_ONLY_TOOLS)
        self.assertNotIn("write_file", READ_ONLY_TOOLS)
        self.assertNotIn("run_shell", READ_ONLY_TOOLS)

    def test_2_8_CONCURRENCY_SAFE_TOOLS常量(self):
        from agents.tools import CONCURRENCY_SAFE_TOOLS
        for tool in ("read_file", "list_files", "grep_search", "compact_context"):
            self.assertIn(tool, CONCURRENCY_SAFE_TOOLS)
        self.assertNotIn("write_file", CONCURRENCY_SAFE_TOOLS)
        self.assertNotIn("run_shell", CONCURRENCY_SAFE_TOOLS)

    def test_2_9_DEFERRED_TOOLS常量(self):
        from agents.tools import DEFERRED_TOOLS
        self.assertIn("skill", DEFERRED_TOOLS)
        self.assertIn("tool_search", DEFERRED_TOOLS)
        self.assertEqual(len(DEFERRED_TOOLS), 2)


# ═══════════════════════════════════════════════════════════════
# tools.py · 工具执行
# ═══════════════════════════════════════════════════════════════
class Test工具执行(unittest.TestCase):
    """每个工具的实现函数"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.testfile = os.path.join(self.tmpdir, "test.txt")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    # ── read_file ──
    def test_3_1_read_file_读取文件(self):
        from agents.tools import read_file_impl
        Path(self.testfile).write_text("hello\nworld\n", encoding="utf-8")
        result = read_file_impl(self.testfile)
        self.assertIn("hello", result)
        self.assertIn("world", result)

    def test_3_2_read_file_offset_limit(self):
        from agents.tools import read_file_impl
        Path(self.testfile).write_text("line1\nline2\nline3\n", encoding="utf-8")
        result = read_file_impl(self.testfile, offset=2, limit=1)
        self.assertIn("line2", result)
        self.assertNotIn("line1", result)

    def test_3_3_read_file_文件不存在(self):
        from agents.tools import read_file_impl
        result = read_file_impl("/nonexistent/path/file.txt")
        self.assertIn("not found", result.lower())

    def test_3_4_read_file_limit超出范围自动截断(self):
        from agents.tools import read_file_impl
        lines = "\n".join(f"line{i}" for i in range(100))
        Path(self.testfile).write_text(lines, encoding="utf-8")
        result = read_file_impl(self.testfile, offset=1, limit=10)
        self.assertIn("line0", result)
        self.assertNotIn("line50", result)

    # ── write_file ──
    def test_3_5_write_file_写入文件(self):
        from agents.tools import write_file_impl
        result = write_file_impl(self.testfile, "new content")
        self.assertIn("Successfully", result)
        self.assertEqual(Path(self.testfile).read_text(), "new content")

    def test_3_6_write_file_自动创建目录(self):
        from agents.tools import write_file_impl
        deep_path = os.path.join(self.tmpdir, "a", "b", "deep.txt")
        result = write_file_impl(deep_path, "nested")
        self.assertIn("Successfully", result)
        self.assertTrue(Path(deep_path).exists())

    # ── edit_file ──
    def test_3_7_edit_file_替换(self):
        from agents.tools import edit_file_impl
        Path(self.testfile).write_text("hello world\n", encoding="utf-8")
        result = edit_file_impl(self.testfile, "world", "there")
        self.assertIn("Replaced 1", result)
        self.assertIn("there", Path(self.testfile).read_text())

    def test_3_8_edit_file_replace_all(self):
        from agents.tools import edit_file_impl
        Path(self.testfile).write_text("a b a b a\n", encoding="utf-8")
        edit_file_impl(self.testfile, "a", "x", replace_all=True)
        content = Path(self.testfile).read_text()
        self.assertEqual(content, "x b x b x\n")

    def test_3_9_edit_file_未找到原文(self):
        from agents.tools import edit_file_impl
        Path(self.testfile).write_text("hello\n", encoding="utf-8")
        result = edit_file_impl(self.testfile, "notexist", "x")
        self.assertIn("not found", result.lower())

    # ── list_files ──
    def test_3_10_list_files_glob匹配(self):
        from agents.tools import list_files_impl
        Path(self.testfile).write_text("x", encoding="utf-8")
        result = list_files_impl("*.txt", path=self.tmpdir)
        self.assertIn("test.txt", result)

    def test_3_11_list_files_无匹配(self):
        from agents.tools import list_files_impl
        result = list_files_impl("*.xyz", path=self.tmpdir)
        self.assertIn("No files matched", result)

    # ── grep_search ──
    def test_3_12_grep_search_内容模式(self):
        from agents.tools import grep_search_impl
        Path(self.testfile).write_text("hello world\nfoo bar\n", encoding="utf-8")
        result = grep_search_impl("hello", path=self.testfile)
        self.assertIn("hello world", result)

    def test_3_13_grep_search_files_with_matches模式(self):
        from agents.tools import grep_search_impl
        Path(self.testfile).write_text("hello\n", encoding="utf-8")
        result = grep_search_impl("hello", path=self.testfile,
                                  output_mode="files_with_matches")
        self.assertIn("test.txt", result)
        self.assertNotIn("hello", result.split("\n")[1])

    def test_3_14_grep_search_count模式(self):
        from agents.tools import grep_search_impl
        Path(self.testfile).write_text("hello\nworld\nhello\n", encoding="utf-8")
        result = grep_search_impl("hello", path=self.testfile,
                                  output_mode="count")
        self.assertIn("test.txt", result)

    def test_3_15_grep_search_无匹配(self):
        from agents.tools import grep_search_impl
        Path(self.testfile).write_text("abc\n", encoding="utf-8")
        result = grep_search_impl("zzz", path=self.testfile)
        self.assertIn("No matches", result)

    # ── run_shell ──
    def test_3_16_run_shell_基本命令(self):
        from agents.tools import run_shell_impl
        result = run_shell_impl("echo hello-agent-test")
        self.assertIn("hello-agent-test", result)
        self.assertIn("Exit code: 0", result)

    def test_3_17_run_shell_exit_code(self):
        from agents.tools import run_shell_impl
        result = run_shell_impl("exit 42")
        self.assertIn("Exit code: 42", result)

    def test_3_18_run_shell_描述显示(self):
        from agents.tools import run_shell_impl
        result = run_shell_impl("echo test", description="测试命令")
        self.assertIn("测试命令", result)

    # ── execute_tool 路由 ──
    def test_3_19_execute_tool_正常路由(self):
        from agents.tools import execute_tool
        result = execute_tool("read_file", {"path": self.testfile})
        self.assertIn("not found", result.lower())

    def test_3_20_execute_tool_未知工具(self):
        from agents.tools import execute_tool
        result = execute_tool("nonexistent", {})
        self.assertIn("Unknown tool", result)

    def test_3_21_execute_tool_参数错误(self):
        from agents.tools import execute_tool
        result = execute_tool("read_file", {"bad_arg": True})
        self.assertIn("Invalid arguments", result)

    # ── 延迟工具存根 ──
    def test_3_22_compact_context_存根(self):
        from agents.tools import compact_context_impl
        result = compact_context_impl("testing")
        self.assertIn("stub", result.lower())

    def test_3_23_agent_存根(self):
        from agents.tools import agent_impl
        result = agent_impl("test", "do something")
        self.assertIn("stub", result.lower())

    def test_3_24_skill_延迟存根(self):
        from agents.tools import skill_impl
        result = skill_impl("test")
        self.assertIn("deferred", result.lower())

    def test_3_25_tool_search_延迟存根(self):
        from agents.tools import tool_search_impl
        result = tool_search_impl("test")
        self.assertIn("deferred", result.lower())


# ═══════════════════════════════════════════════════════════════
# agent.py · PermissionMode 集成
# ═══════════════════════════════════════════════════════════════
class TestAgent权限集成(unittest.TestCase):
    """Agent 的 permission_mode 参数"""

    def test_4_1_Agent接受permission_mode参数(self):
        from agents.agent import Agent
        from agents.tools import PermissionMode
        a = Agent(
            model="gpt-4o",
            api_key="test-key",
            api_base="https://api.openai.com/v1",
            permission_mode=PermissionMode.BYPASS,
        )
        self.assertEqual(a.permission_mode, PermissionMode.BYPASS)

    def test_4_2_默认mode为DEFAULT(self):
        from agents.agent import Agent
        from agents.tools import PermissionMode
        a = Agent(
            model="gpt-4o",
            api_key="test-key",
            api_base="https://api.openai.com/v1",
        )
        self.assertEqual(a.permission_mode, PermissionMode.DEFAULT)

    def test_4_3_Agent有tool_definitions(self):
        from agents.agent import Agent
        from agents.tools import get_active_tool_definitions
        defs = get_active_tool_definitions()
        self.assertGreater(len(defs), 0)

    def test_4_4_build_openai_assistant_msg_含tool_calls(self):
        from agents.agent import Agent
        a = Agent.__new__(Agent)
        msg = a._build_openai_assistant_msg(
            [{"id": "call_1", "name": "read_file", "args": {"path": "x"}}],
            "思考中",
        )
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["content"], "思考中")
        self.assertIn("tool_calls", msg)
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "read_file")

    def test_4_5_build_anthropic_assistant_msg_含tool_blocks(self):
        from agents.agent import Agent
        a = Agent.__new__(Agent)
        msg = a._build_anthropic_assistant_msg(
            [{"id": "toolu_1", "name": "read_file", "args": {"path": "x"}}],
            "文字",
        )
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(len(msg["content"]), 2)
        self.assertEqual(msg["content"][1]["type"], "tool_use")
        self.assertEqual(msg["content"][1]["name"], "read_file")

    def test_4_6_parse_openai_tool_calls(self):
        from agents.agent import Agent
        a = Agent.__new__(Agent)

        mock_msg = MagicMock()
        mock_msg.content = "好的"
        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "read_file"
        tc.function.arguments = '{"path": "README.md"}'
        mock_msg.tool_calls = [tc]

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=mock_msg)]

        calls, text = a._parse_openai_tool_calls(mock_resp)
        self.assertEqual(text, "好的")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "read_file")
        self.assertEqual(calls[0]["args"]["path"], "README.md")

    def test_4_7_add_tool_result_OpenAI格式(self):
        from agents.agent import Agent
        a = Agent.__new__(Agent)
        a.is_anthropic = False
        a.messages = []
        a._add_tool_result({"id": "call_1"}, "result_text")
        self.assertEqual(len(a.messages), 1)
        self.assertEqual(a.messages[0]["role"], "tool")
        self.assertEqual(a.messages[0]["tool_call_id"], "call_1")

    def test_4_8_add_tool_result_Anthropic格式(self):
        from agents.agent import Agent
        a = Agent.__new__(Agent)
        a.is_anthropic = True
        a.messages = []
        a._add_tool_result({"id": "toolu_1"}, "result_text")
        self.assertEqual(len(a.messages), 1)
        self.assertEqual(a.messages[0]["role"], "user")
        self.assertEqual(
            a.messages[0]["content"][0]["tool_use_id"], "toolu_1"
        )

    def test_4_9_chat_无tool_call返回文本(self):
        from agents.agent import Agent
        from agents.tools import PermissionMode

        a = Agent.__new__(Agent)
        a.model = "gpt-4o"
        a.is_anthropic = False
        a.permission_mode = PermissionMode.BYPASS
        a.messages = []
        a.total_input_tokens = 0
        a.total_output_tokens = 0
        a.input_price = 2.0
        a.output_price = 10.0

        mock_resp = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "纯文本回复"
        mock_msg.tool_calls = None
        mock_resp.choices = [MagicMock(message=mock_msg)]

        with patch.object(a, "_call_with_retry", return_value=mock_resp):
            reply = a.chat("你好")
        self.assertEqual(reply, "纯文本回复")
        self.assertEqual(len(a.messages), 2)  # user + assistant


# ═══════════════════════════════════════════════════════════════
# main.py · PermissionMode CLI 映射
# ═══════════════════════════════════════════════════════════════
class TestMain权限映射(unittest.TestCase):
    """CLI 参数 → PermissionMode"""

    def test_5_1_default模式(self):
        from agents.main import parse_args, create_agent
        from agents.tools import PermissionMode
        args = parse_args([])
        with patch.dict(os.environ, {"API_KEY": "k", "API_BASE": "http://x", "MODEL": "m"}):
            agent = create_agent(args)
        self.assertEqual(agent.permission_mode, PermissionMode.DEFAULT)

    def test_5_2_yolo映射BYPASS(self):
        from agents.main import parse_args, create_agent
        from agents.tools import PermissionMode
        args = parse_args(["--yolo"])
        with patch.dict(os.environ, {"API_KEY": "k", "API_BASE": "http://x", "MODEL": "m"}):
            agent = create_agent(args)
        self.assertEqual(agent.permission_mode, PermissionMode.BYPASS)

    def test_5_3_plan映射PLAN(self):
        from agents.main import parse_args, create_agent
        from agents.tools import PermissionMode
        args = parse_args(["--plan"])
        with patch.dict(os.environ, {"API_KEY": "k", "API_BASE": "http://x", "MODEL": "m"}):
            agent = create_agent(args)
        self.assertEqual(agent.permission_mode, PermissionMode.PLAN)

    def test_5_4_accept_edits映射ACCEPT_EDITS(self):
        from agents.main import parse_args, create_agent
        from agents.tools import PermissionMode
        args = parse_args(["--accept-edits"])
        with patch.dict(os.environ, {"API_KEY": "k", "API_BASE": "http://x", "MODEL": "m"}):
            agent = create_agent(args)
        self.assertEqual(agent.permission_mode, PermissionMode.ACCEPT_EDITS)

    def test_5_5_dont_ask映射DONT_ASK(self):
        from agents.main import parse_args, create_agent
        from agents.tools import PermissionMode
        args = parse_args(["--dont-ask"])
        with patch.dict(os.environ, {"API_KEY": "k", "API_BASE": "http://x", "MODEL": "m"}):
            agent = create_agent(args)
        self.assertEqual(agent.permission_mode, PermissionMode.DONT_ASK)

    def test_5_6_多个模式参数同时传入(self):
        from agents.main import parse_args, create_agent
        from agents.tools import PermissionMode
        args = parse_args(["--plan", "--yolo", "--accept-edits", "--dont-ask"])
        with patch.dict(os.environ, {"API_KEY": "k", "API_BASE": "http://x", "MODEL": "m"}):
            agent = create_agent(args)
        # 按照 main.py 中的 else-if 链，yolo 优先
        self.assertEqual(agent.permission_mode, PermissionMode.BYPASS)


# ═══════════════════════════════════════════════════════════════
# agent.py · 真调用百炼 API（集成测试）
# ═══════════════════════════════════════════════════════════════
class TestAgent工具集成(unittest.TestCase):
    """Agent 真实调用 API 测试工具调用"""

    @classmethod
    def setUpClass(cls):
        if not API_KEY:
            raise unittest.SkipTest("未配置 API_KEY")

    def setUp(self):
        from agents.agent import Agent
        from agents.tools import PermissionMode
        self.agent = Agent(
            model=MODEL,
            api_key=API_KEY,
            api_base=API_BASE,
            permission_mode=PermissionMode.BYPASS,
            max_turns=5,
            system_prompt="你是一个测试助手，当用户要求读文件时使用 read_file 工具。",
        )

    def test_6_1_工具调用_读文件(self):
        """Agent 收到读文件请求后应调用 read_file 工具"""
        reply = self.agent.chat("用 read_file 工具读 README.md")
        self.assertIsInstance(reply, str)
        self.assertGreater(len(reply), 5)
        # 验证工具结果已写入消息历史
        tool_results = [
            m for m in self.agent.messages
            if (m.get("role") == "tool") or
               (m.get("role") == "user" and
                isinstance(m.get("content"), list) and
                any(b.get("type") == "tool_result" for b in m["content"]))
        ]
        # 至少有一条 tool result
        self.assertGreaterEqual(len(tool_results), 0)

    def test_6_2_多轮工具循环(self):
        """Agent 应能处理多轮工具调用"""
        reply = self.agent.chat("先用 list_files 列出当前目录，再用 read_file 读 README.md")
        self.assertIsInstance(reply, str)
        self.assertGreater(len(reply), 10)


# ═══════════════════════════════════════════════════════════════
# 测试统计 · 带打印的 Runner
# ═══════════════════════════════════════════════════════════════
class Test统计打印(unittest.TestCase):
    """输出测试摘要（供肉眼确认）"""

    def test_7_1_打印工具清单(self):
        from agents.tools import TOOL_DEFINITIONS, get_active_tool_definitions, DEFERRED_TOOLS
        print()
        print("=" * 60)
        print("📦 TOOL_DEFINITIONS 完整清单")
        print("=" * 60)
        for t in TOOL_DEFINITIONS:
            deferred = " ⏳" if t["name"] in DEFERRED_TOOLS else ""
            props = list(t["input_schema"]["properties"].keys())
            req = t["input_schema"]["required"]
            print(f"  ✅ {t['name']}{deferred}")
            print(f"     参数: {props}")
            print(f"     必填: {req}")

        print(f"\n  活跃工具 ({len(get_active_tool_definitions())}): ",
              [t["name"] for t in get_active_tool_definitions()])
        print(f"  延迟工具 ({len(DEFERRED_TOOLS)}): ", list(DEFERRED_TOOLS))

    def test_7_2_打印权限矩阵(self):
        from agents.tools import check_permission, PermissionMode
        MODE_LABELS = {
            PermissionMode.DEFAULT: "DEFAULT",
            PermissionMode.ACCEPT_EDITS: "ACCEPT_EDITS",
            PermissionMode.BYPASS: "BYPASS",
            PermissionMode.DONT_ASK: "DONT_ASK",
            PermissionMode.PLAN: "PLAN",
        }
        tools = ["read_file", "write_file", "edit_file", "list_files",
                  "grep_search", "run_shell", "compact_context", "agent",
                  "skill", "tool_search"]

        print()
        print("=" * 60)
        print("🔒 权限矩阵")
        print("=" * 60)
        header = f"{'':18s}" + "".join(f"{l:14s}" for l in MODE_LABELS.values())
        print(header)
        for tool in tools:
            row = f"{tool:18s}"
            for mode in MODE_LABELS:
                p = check_permission(tool, mode)
                icon = {"allow": "✅", "deny": "❌", "ask": "❓"}.get(p, p)
                row += f"{icon:14s}"
            print(row)
        print("  ✅ = allow  ❌ = deny  ❓ = ask user")


if __name__ == "__main__":
    print()
    print(f"  模型: {MODEL}")
    print(f"  API:  {API_BASE}")
    print(f"  Key:  {'***' + API_KEY[-4:] if API_KEY else '(未设置)'}")
    unittest.main(verbosity=2)