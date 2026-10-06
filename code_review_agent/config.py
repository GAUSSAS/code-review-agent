"""配置加载：环境变量优先，.env 兜底。

刻意保持简单（不引入 pydantic-settings 等额外依赖），
让评审者一眼能看懂配置从哪里来、优先级如何。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# 常见服务商的默认地址，用户只填 base_url 即可切换
PROVIDER_PRESETS: dict[str, str] = {
    "deepseek": "https://api.deepseek.com/v1",
    "openai": "https://api.openai.com/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
}

DEFAULT_MODEL_BY_PROVIDER: dict[str, str] = {
    "deepseek": "deepseek-chat",
    "openai": "gpt-4o-mini",
    "dashscope": "qwen-plus",
    "moonshot": "moonshot-v1-8k",
    "zhipu": "glm-4-flash",
    "ollama": "qwen2.5-coder:7b",
}

_TRUTHY = {"1", "true", "yes", "on", "y"}


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in _TRUTHY


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _as_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def load_dotenv(path: Path | str = ".env", *, override: bool = False) -> dict[str, str]:
    """极简 .env 解析器：支持 KEY=VALUE、# 注释、可选的引号包裹。

    返回解析到的键值对；同时按需写入 ``os.environ``。
    """
    env_path = Path(path)
    parsed: dict[str, str] = {}
    if not env_path.is_file():
        return parsed

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        parsed[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return parsed


@dataclass
class Settings:
    """运行期配置。"""

    provider: str = "deepseek"
    model: str = "deepseek-chat"
    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"

    temperature: float = 0.2
    max_tokens: int = 4096
    timeout: float = 60.0
    max_retries: int = 4

    max_steps: int = 12
    memory_window: int = 24
    max_file_bytes: int = 200_000
    workspace: str = "."
    verbose: bool = False

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    @classmethod
    def from_env(
        cls,
        env_file: Path | str | None = ".env",
        *,
        workspace: str | None = None,
        overrides: dict[str, object] | None = None,
    ) -> "Settings":
        if env_file:
            load_dotenv(env_file)

        provider = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
        model = os.getenv("LLM_MODEL", "").strip() or DEFAULT_MODEL_BY_PROVIDER.get(
            provider, "deepseek-chat"
        )
        base_url = os.getenv("LLM_BASE_URL", "").strip() or PROVIDER_PRESETS.get(
            provider, PROVIDER_PRESETS["deepseek"]
        )

        settings = cls(
            provider=provider,
            model=model,
            api_key=os.getenv("LLM_API_KEY", "").strip(),
            base_url=base_url,
            temperature=_as_float(os.getenv("LLM_TEMPERATURE"), 0.2),
            max_tokens=_as_int(os.getenv("LLM_MAX_TOKENS"), 4096),
            timeout=_as_float(os.getenv("LLM_TIMEOUT"), 60.0),
            max_retries=_as_int(os.getenv("LLM_MAX_RETRIES"), 4),
            max_steps=_as_int(os.getenv("AGENT_MAX_STEPS"), 12),
            memory_window=_as_int(os.getenv("AGENT_MEMORY_WINDOW"), 24),
            max_file_bytes=_as_int(os.getenv("AGENT_MAX_FILE_BYTES"), 200_000),
            workspace=os.getenv("AGENT_WORKSPACE", "."),
            verbose=_as_bool(os.getenv("AGENT_VERBOSE"), False),
        )

        if workspace:
            settings.workspace = workspace
        for key, value in (overrides or {}).items():
            if value is not None and hasattr(settings, key):
                setattr(settings, key, value)
        return settings
