"""pytest 共享夹具。

这里提供一个**完全离线的假 LLM**，让测试不依赖任何 API Key 和网络，
评审者只需 `pytest` 就能验证 Agent 循环、工具调用与记忆管理是否正确。

假 LLM 的构造器放在 ``tests/helpers.py``，测试中直接 ``from tests.helpers import ...``。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from code_review_agent.config import Settings
from code_review_agent.llm import LLMClient
from tests.helpers import FakeOpenAI

# --------------------------------------------------------------------------- #
# 临时工作区：内置若干「有味道」的示例代码，供各测试断言
# --------------------------------------------------------------------------- #
SAMPLES: dict[str, str] = {
    "app/__init__.py": "",
    "app/service.py": '''"""示例服务模块（刻意内置若干坏味道，供测试断言）。"""

import os

API_KEY = "sk-abcdefghijklmnopqrstuvwxyz012345"


def fetch(user, cache={}):
    """带可变默认参数与裸 except 的示例。"""
    try:
        if user:
            return cache.get(user)
        else:
            return None
    except:
        pass


def classify(value):
    if value > 100:
        return "big"
    elif value > 50:
        return "medium"
    elif value > 20:
        return "small"
    elif value > 10:
        return "tiny"
    elif value > 5:
        return "mini"
    elif value > 1:
        return "micro"
    elif value > 0:
        return "zero"
    else:
        return "negative"


def run(cmd):
    eval(cmd)
    os.system(cmd)
''',
    "app/utils.js": """function f(a) {
  var x = 1;
  if (a == 1) { console.log('debug'); }
  try { g(); } catch (e) {}
  return x;
}
""",
    "README.md": "# 示例项目\n\n用于测试。\n",
}


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """创建一个包含多语言示例代码的临时工作区。"""
    for rel, content in SAMPLES.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return tmp_path


@pytest.fixture
def settings(workspace: Path) -> Settings:
    return Settings(
        provider="deepseek",
        model="fake-model",
        api_key="fake-key",
        base_url="http://fake.local/v1",
        workspace=str(workspace),
        max_steps=6,
        memory_window=6,
    )


@pytest.fixture
def make_llm_client():
    """返回 (client, fake_openai)；client 是真实的 LLMClient，只是注入了假 SDK。"""

    def _factory(script):
        fake = FakeOpenAI(script)
        client = LLMClient(
            model="fake-model",
            api_key="fake-key",
            base_url="http://fake.local/v1",
            max_retries=0,
            client=fake,
        )
        return client, fake

    return _factory
