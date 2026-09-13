"""
迭代一 · 完整验证测试

覆盖 TODO 全部 24 项功能点。
Agent 相关用例真调用百炼 API，其余纯逻辑无需网络。
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

API_KEY = os.environ.get("API_KEY", "")
API_BASE = os.environ.get("API_BASE", "")
MODEL = os.environ.get("MODEL", "qwen-plus")


# ═══════════════════════════════════════════════════════════════
# main.py · CLI 入口
# ═══════════════════════════════════════════════════════════════
class TestCLI入口(unittest.TestCase):
    """main.py — 参数解析"""

    def setUp(self):
        from agents.main import parse_args
        self.parse = parse_args

    def test_1_1_无参数时全部默认值(self):
        args = self.parse([])
        for k in ("model", "api_base", "max_turns", "max_cost", "message"):
            self.assertIsNone(getattr(args, k))
        for k in ("plan", "yolo", "accept_edits", "dont_ask", "resume"):
            self.assertFalse(getattr(args, k))

    def test_1_2_全部参数同时传入(self):
        argv = [
            "--model", "qwen-plus",
            "--api-base", "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "--plan", "--yolo", "--accept-edits", "--dont-ask", "--resume",
            "--max-turns", "10", "--max-cost", "0.5",
            "你好",
        ]
        args = self.parse(argv)
        self.assertEqual(args.model, "qwen-plus")
        self.assertEqual(args.api_base, "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.assertTrue(args.plan)
        self.assertTrue(args.yolo)
        self.assertTrue(args.accept_edits)
        self.assertTrue(args.dont_ask)
        self.assertTrue(args.resume)
        self.assertEqual(args.max_turns, 10)
        self.assertEqual(args.max_cost, 0.5)
        self.assertEqual(args.message, "你好")

    def test_1_3_message为可选位置参数(self):
        self.assertIsNone(self.parse([]).message)
        self.assertEqual(self.parse(["hello"]).message, "hello")

    def test_1_4_TODO列出的全部参数名(self):
        for name, argv in [
            ("--model", ["--model", "x"]),
            ("--plan", ["--plan"]),
            ("--yolo", ["--yolo"]),
            ("--accept-edits", ["--accept-edits"]),
            ("--dont-ask", ["--dont-ask"]),
            ("--resume", ["--resume"]),
            ("--api-base", ["--api-base", "http://x"]),
            ("--max-turns", ["--max-turns", "5"]),
            ("--max-cost", ["--max-cost", "1.0"]),
        ]:
            with self.subTest(arg=name):
                self.parse(argv)


class TestCLI环境加载(unittest.TestCase):
    """main.py — .env 加载与 Agent 创建"""

    def test_2_1_create_agent读取环境变量(self):
        from agents.main import parse_args, create_agent
        args = parse_args([])
        with patch.dict(os.environ, {
            "API_KEY": "sk-test",
            "API_BASE": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "MODEL": "qwen-turbo",
        }):
            agent = create_agent(args)
        self.assertEqual(agent.api_key, "sk-test")
        self.assertEqual(agent.api_base, "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(agent.model, "qwen-turbo")

    def test_2_2_CLI参数覆盖环境变量(self):
        from agents.main import parse_args, create_agent
        args = parse_args(["--model", "qwen-plus"])
        with patch.dict(os.environ, {"MODEL": "qwen-turbo"}):
            agent = create_agent(args)
        self.assertEqual(agent.model, "qwen-plus")

    def test_2_3_plan模式注入system_prompt(self):
        from agents.main import parse_args, create_agent
        args = parse_args(["--plan"])
        with patch.dict(os.environ, {"API_KEY": "k", "API_BASE": "http://x", "MODEL": "m"}):
            agent = create_agent(args)
        self.assertIn("plan mode", agent.system_prompt)

    def test_2_4_yolo模式注入system_prompt(self):
        from agents.main import parse_args, create_agent
        args = parse_args(["--yolo"])
        with patch.dict(os.environ, {"API_KEY": "k", "API_BASE": "http://x", "MODEL": "m"}):
            agent = create_agent(args)
        self.assertIn("YOLO mode", agent.system_prompt)


# ═══════════════════════════════════════════════════════════════
# agent.py · 模型注册表
# ═══════════════════════════════════════════════════════════════
class Test模型注册表(unittest.TestCase):
    """agent.py — context window 常量表"""

    def setUp(self):
        from agents.agent import _lookup_model
        self.lookup = _lookup_model

    def test_3_1_百炼模型价格准确(self):
        for name, ctx, inp, out in [
            ("qwen-plus", 131072, 0.50, 2.00),
            ("qwen-turbo", 131072, 0.30, 0.60),
            ("qwen-max", 32768, 2.00, 6.00),
            ("qwen3.8-flash", 131072, 0.11, 0.39),
            ("qwen3.8-max", 131072, 1.71, 5.14),
            ("qwq-plus", 131072, 0.23, 0.57),
            ("qwen-long", 10000000, 0.07, 0.29),
            ("qwen-coder-plus", 131072, 0.50, 1.00),
        ]:
            with self.subTest(model=name):
                c, i, o = self.lookup(name)
                self.assertEqual(c, ctx)
                self.assertEqual(i, inp)
                self.assertEqual(o, out)

    def test_3_2_其他平台模型(self):
        for name, ctx, inp, out in [
            ("gpt-4o", 128000, 2.50, 10.00),
            ("gpt-4o-mini", 128000, 0.15, 0.60),
            ("claude-sonnet-4", 200000, 3.00, 15.00),
            ("claude-haiku-4", 200000, 0.80, 4.00),
            ("deepseek-chat", 65536, 0.14, 0.28),
            ("o1", 200000, 15.00, 60.00),
        ]:
            with self.subTest(model=name):
                c, i, o = self.lookup(name)
                self.assertEqual(c, ctx)
                self.assertEqual(i, inp)
                self.assertEqual(o, out)

    def test_3_3_未注册模型用兜底(self):
        c, i, o = self.lookup("nonexistent-model-v99")
        self.assertEqual(c, 128000)
        self.assertEqual(i, 2.00)
        self.assertEqual(o, 8.00)

    def test_3_4_长前缀优先于短前缀(self):
        _, i, _ = self.lookup("qwen3.8-flash")
        self.assertEqual(i, 0.11)
        _, i, _ = self.lookup("qwen")
        self.assertEqual(i, 0.50)

    def test_3_5_型号后缀不影响匹配(self):
        c, _, _ = self.lookup("qwen3.8-flash-20250901")
        self.assertEqual(c, 131072)


class Test指数退避(unittest.TestCase):
    """agent.py — 指数退避重试"""

    def setUp(self):
        from agents.agent import _exponential_backoff
        self.b = _exponential_backoff

    def test_4_1_延迟单调递增(self):
        ds = [self.b(i) for i in range(1, 6)]
        for i in range(4):
            self.assertGreater(ds[i + 1], ds[i])

    def test_4_2_首次延迟约2秒(self):
        self.assertAlmostEqual(self.b(1), 2.1, delta=0.5)

    def test_4_3_第五次延迟约32至66秒(self):
        d = self.b(5)
        self.assertGreater(d, 32)
        self.assertLess(d, 66)

    def test_4_4_jitter产生变化(self):
        self.assertNotAlmostEqual(self.b(2), self.b(2), places=2)


class Test状态码提取(unittest.TestCase):
    """agent.py — 异常状态码提取"""

    def setUp(self):
        from agents.agent import _extract_status_code
        self.extract = _extract_status_code

    def test_5_1_可重试码_429_503_529(self):
        for code in (429, 503, 529):
            self.assertEqual(self.extract(Exception(str(code))), code)

    def test_5_2_其他码(self):
        self.assertEqual(self.extract(Exception("400 Bad Request")), 400)
        self.assertEqual(self.extract(Exception("unknown")), 0)


# ═══════════════════════════════════════════════════════════════
# Agent · 真调用百炼 API
# ═══════════════════════════════════════════════════════════════
class TestAgent基础功能(unittest.TestCase):
    """Agent 类核心方法"""

    @classmethod
    def setUpClass(cls):
        if not API_KEY:
            raise unittest.SkipTest("未配置 API_KEY")

    def setUp(self):
        from agents.agent import Agent
        self.agent = Agent(
            model=MODEL,
            api_key=API_KEY,
            api_base=API_BASE,
            system_prompt="You are a helpful assistant.",
        )

    def test_6_1_chat返回字符串(self):
        reply = self.agent.chat("用一句话介绍你自己")
        self.assertIsInstance(reply, str)
        self.assertGreater(len(reply), 5)

    def test_6_2_chat追加消息历史(self):
        self.agent.chat("第一轮")
        self.agent.chat("第二轮")
        self.assertEqual(len(self.agent.messages), 4)
        self.assertEqual(self.agent.messages[0]["role"], "user")
        self.assertEqual(self.agent.messages[1]["role"], "assistant")

    def test_6_3_clear_history清空(self):
        self.agent.chat("随便说点什么")
        self.assertGreater(len(self.agent.messages), 0)
        self.agent.clear_history()
        self.assertEqual(self.agent.messages, [])
        self.assertEqual(self.agent.total_input_tokens, 0)
        self.assertEqual(self.agent.total_output_tokens, 0)

    def test_6_4_show_cost输出格式(self):
        self.agent.chat("hi")
        s = self.agent.get_cost_string()
        self.assertIn(MODEL, s)
        self.assertIn("$", s)
        self.assertIn("tokens", s)

    def test_6_5_max_turns限制(self):
        self.agent.max_turns = 2
        self.agent.chat("1")
        self.agent.chat("2")
        with self.assertRaises(RuntimeError):
            self.agent.chat("3")

    def test_6_6_system_prompt生效(self):
        from agents.agent import Agent
        a = Agent(
            model=MODEL, api_key=API_KEY, api_base=API_BASE,
            system_prompt="你只说英文，每次回答以 'EN:' 开头",
        )
        reply = a.chat("你好")
        self.assertTrue(reply.startswith("EN:"), f"实际开头: {reply[:50]}")

    def test_6_7_多轮对话保持上下文(self):
        self.agent.chat("记住一个秘密数字：42")
        self.agent.chat("我刚才让你记住的数字是多少？")
        self.assertIn("42", self.agent.messages[-1]["content"])


class TestAgent协议(unittest.TestCase):
    """协议自动判断"""

    def test_7_1_百炼为OpenAI协议(self):
        from agents.agent import Agent
        a = Agent(model=MODEL, api_key=API_KEY, api_base=API_BASE)
        self.assertFalse(a.is_anthropic)
        self.assertIsNotNone(a._openai_client)

    def test_7_2_Anthropic协议(self):
        from agents.agent import Agent
        a = Agent(
            model="claude-sonnet-4",
            api_key="test-key",
            api_base="https://api.anthropic.com/v1/anthropic",
        )
        self.assertTrue(a.is_anthropic)


class TestAgent重试(unittest.TestCase):
    """指数退避重试逻辑（mock 避免真等）"""

    def setUp(self):
        from agents.agent import Agent, _lookup_model
        a = object.__new__(Agent)
        a.model = MODEL
        a.api_key = API_KEY
        a.api_base = API_BASE
        a.max_turns = 50
        a.system_prompt = "t"
        a.messages = []
        a.total_input_tokens = 0
        a.total_output_tokens = 0
        a.context_window, a.input_price, a.output_price = _lookup_model(MODEL)
        a.is_anthropic = False
        a._openai_client = None
        a._openai_client_kwargs = {}
        self.agent = a

    def test_8_1_429重试后成功(self):
        with patch.object(self.agent, "_call_openai",
                          side_effect=[Exception("429"), "ok"]):
            self.assertEqual(self.agent._call_with_retry(max_retries=3), "ok")

    def test_8_2_400不重试直接抛(self):
        with patch.object(self.agent, "_call_openai",
                          side_effect=Exception("400")):
            with self.assertRaises(Exception):
                self.agent._call_with_retry()

    def test_8_3_重试耗尽抛RuntimeError(self):
        with patch.object(self.agent, "_call_openai",
                          side_effect=Exception("429")):
            with self.assertRaises(RuntimeError):
                self.agent._call_with_retry(max_retries=2)


# ═══════════════════════════════════════════════════════════════
# ui.py · 终端输出
# ═══════════════════════════════════════════════════════════════
class TestUI(unittest.TestCase):
    """全部 13 个 UI 函数"""

    def test_9_1_函数存在(self):
        import agents.ui as ui
        for name in ("print_welcome", "print_goodbye", "print_user_prompt",
                      "print_assistant_text", "print_error", "print_warning",
                      "print_info", "print_tool_call", "print_tool_result",
                      "print_cost", "print_divider", "start_spinner", "stop_spinner"):
            self.assertTrue(callable(getattr(ui, name)), f"{name} 应可调用")

    def test_9_2_执行不抛异常(self):
        import agents.ui as ui
        ui.print_welcome()
        ui.print_goodbye()
        ui.print_user_prompt("你好")
        ui.print_assistant_text("**Markdown**")
        ui.print_assistant_text("")
        ui.print_error("错误"); ui.print_warning("警告"); ui.print_info("信息")
        ui.print_tool_call("read_file", {"path": "x"})
        ui.print_tool_result("内容" * 2000)
        ui.print_cost("$0.01")
        ui.print_divider()
        ui.start_spinner("加载")
        ui.stop_spinner()
        ui.stop_spinner()
        self.assertTrue(True)


# ═══════════════════════════════════════════════════════════════
# 项目文件
# ═══════════════════════════════════════════════════════════════
class Test项目文件(unittest.TestCase):
    """requirements.txt / .env 模板"""

    def setUp(self):
        self.root = Path(__file__).resolve().parent.parent

    def test_10_1_requirements含全部依赖(self):
        content = (self.root / "requirements.txt").read_text()
        for pkg in ("anthropic", "openai", "python-dotenv", "rich", "tqdm"):
            self.assertIn(pkg, content)

    def test_10_2_env模板存在(self):
        content = (self.root / ".env.template").read_text()
        self.assertIn("API_KEY", content)
        self.assertIn("API_BASE", content)
        self.assertIn("MODEL", content)

    def test_10_3_agents_main模块完整(self):
        import agents.main as m
        for attr in ("main", "parse_args", "create_agent", "run_repl", "run_once"):
            self.assertTrue(hasattr(m, attr))


if __name__ == "__main__":
    print(f"  模型: {MODEL}")
    print(f"  API:  {API_BASE}")
    print(f"  Key:  {'***' + API_KEY[-4:] if API_KEY else '(未设置)'}")
    unittest.main(verbosity=2)