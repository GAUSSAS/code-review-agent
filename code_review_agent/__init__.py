"""代码审查 Agent —— 一个用于教学演示的轻量 Agent 实现。

包结构::

    code_review_agent/
    ├── agent.py       Agent 主循环（输入→推理→工具调用→输出）
    ├── llm.py         OpenAI 兼容 LLM 客户端（重试/退避/异常翻译）
    ├── memory.py      上下文记忆（滑窗 + LLM 摘要压缩）
    ├── prompts.py     Prompt 模板（系统提示 / Few-shot / 任务模板）
    ├── config.py      配置加载（环境变量 + .env）
    ├── cli.py         命令行入口
    └── tools/         工具层
        ├── base.py    工具协议、注册表、路径安全守卫
        ├── analyzer.py 静态分析器（AST + 启发式）
        └── builtin.py 7 个内置工具
"""

from .agent import AgentError, AgentEvent, ReviewAgent, StepLimitExceeded
from .config import Settings
from .llm import ChatMessage, LLMClient, LLMConfigError, LLMRetryExhausted

__version__ = "1.1.0"

__all__ = [
    "AgentError",
    "AgentEvent",
    "ReviewAgent",
    "StepLimitExceeded",
    "Settings",
    "ChatMessage",
    "LLMClient",
    "LLMConfigError",
    "LLMRetryExhausted",
    "__version__",
]
