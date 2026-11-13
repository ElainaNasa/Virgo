"""运行配置与默认值。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4
from pathlib import Path


OFFICIAL_PROVIDER_URLS = {'zhipu': 'https://open.bigmodel.cn/api/paas/v4', 'minimax': 'https://api.minimax.cn/v1', 'qwen': 'https://maas.qianwenaiapi.com/compatible-mode/v1', 'mimo': 'https://api.xiaomimimo.com/v1', 'moonshot': 'https://api.moonshot.cn/v1'}


@dataclass(frozen=True)
class Settings:
    """系统级配置；密钥只从环境变量读取。"""

    workspace: Path
    state_dir: Path
    local_base_url: str
    local_model: str
    local_api_key: str | None
    offline: bool
    request_timeout: float
    log_dir: Path | None = None
    log_level: str = "INFO"
    log_console: bool = False
    router_provider: str = "auto"
    router_base_url: str = "https://api.openai.com/v1"
    router_model: str = "gpt-5.4"
    router_api_key: str | None = None
    router_timeout: float = 60.0
    execution_provider: str = "auto"
    execution_base_url: str = ""
    execution_model: str = ""
    execution_api_key: str | None = None
    execution_timeout: float = 60.0
    jev_base_url: str = "https://api.typesafe.ai/v1/systemone"
    jev_model: str = "jev-latest"
    jev_api_key: str | None = None
    jev_timeout: float = 30.0
    max_tool_calls: int = 50
    collaboration_tool_call_limit: int = 30
    context_compaction_threshold_tokens: int = 200_000
    deepseek_v4_flash_pricing: str = "nonpeak"
    inferai_base_url: str = "https://inferaiapi.com/v1"
    inferai_api_key: str | None = None
    inferai_max_retries: int = 2
    inferai_wire_api: str = "chat_completions"
    storage_group: str | None = None
    storage_run_id: str | None = None

    initial_context_provider: str = "auto"
    initial_context_base_url: str = "https://api.deepseek.com"
    initial_context_model: str | None = None
    initial_context_api_key: str | None = None
    initial_context_timeout: float = 60.0

    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    zhipu_api_key: str | None = None
    minimax_base_url: str = "https://api.minimax.cn/v1"
    minimax_api_key: str | None = None
    qwen_base_url: str = "https://maas.qianwenaiapi.com/compatible-mode/v1"
    qwen_api_key: str | None = None
    mimo_base_url: str = "https://api.xiaomimimo.com/v1"
    mimo_api_key: str | None = None
    moonshot_base_url: str = "https://api.moonshot.cn/v1"
    moonshot_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_api_key: str | None = None

    execution_moonshot_api_key: str | None = None
    execution_mimo_api_key: str | None = None
    execution_zhipu_api_key: str | None = None
    execution_deepseek_api_key: str | None = None
    execution_minimax_api_key: str | None = None
    execution_qwen_api_key: str | None = None

    @classmethod
    def from_env(cls, workspace: str | Path | None = None) -> "Settings":
        root = Path(workspace or os.getenv("VIRGO_WORKSPACE", Path.cwd())).resolve()
        _load_dotenv(root / ".env")
        state_dir = Path(os.getenv("VIRGO_STATE_DIR", str(root / "var"))).resolve()
        mode = os.getenv("VIRGO_OFFLINE", "1").strip().lower()
        router_provider = os.getenv("VIRGO_ROUTER_PROVIDER", "auto").strip().lower()
        execution_api_key = os.getenv("VIRGO_EXECUTION_API_KEY") or os.getenv("VIRGO_LOCAL_API_KEY")
        inferai_api_key = os.getenv("VIRGO_INFERAI_API_KEY") or execution_api_key or os.getenv("OPENAI_API_KEY")
        router_api_key = os.getenv("VIRGO_ROUTER_API_KEY")
        if not router_api_key and router_provider == "deepseek":
            router_api_key = execution_api_key
        router_api_key = router_api_key or os.getenv("OPENAI_API_KEY") or execution_api_key
        return cls(
            zhipu_base_url=os.getenv("VIRGO_ZHIPU_BASE_URL", OFFICIAL_PROVIDER_URLS["zhipu"]),
            zhipu_api_key=os.getenv("VIRGO_ZHIPU_API_KEY") or os.getenv("ZHIPU_API_KEY"),
            minimax_base_url=os.getenv("VIRGO_MINIMAX_BASE_URL", OFFICIAL_PROVIDER_URLS["minimax"]),
            minimax_api_key=os.getenv("VIRGO_MINIMAX_API_KEY") or os.getenv("MINIMAX_API_KEY"),
            qwen_base_url=os.getenv("VIRGO_QWEN_BASE_URL", OFFICIAL_PROVIDER_URLS["qwen"]),
            qwen_api_key=os.getenv("VIRGO_QWEN_API_KEY") or os.getenv("QWEN_API_KEY"),
            mimo_base_url=os.getenv("VIRGO_MIMO_BASE_URL", OFFICIAL_PROVIDER_URLS["mimo"]),
            mimo_api_key=os.getenv("VIRGO_MIMO_API_KEY") or os.getenv("MIMO_API_KEY"),
            moonshot_base_url=os.getenv("VIRGO_MOONSHOT_BASE_URL", OFFICIAL_PROVIDER_URLS["moonshot"]),
            moonshot_api_key=os.getenv("VIRGO_MOONSHOT_API_KEY") or os.getenv("MOONSHOT_API_KEY"),
            deepseek_base_url=os.getenv("VIRGO_DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            deepseek_api_key=os.getenv("VIRGO_DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY"),
            execution_moonshot_api_key=os.getenv("VIRGO_EXECUTION_MOONSHOT_API_KEY"),
            execution_mimo_api_key=os.getenv("VIRGO_EXECUTION_MIMO_API_KEY"),
            execution_zhipu_api_key=os.getenv("VIRGO_EXECUTION_ZHIPU_API_KEY"),
            execution_deepseek_api_key=os.getenv("VIRGO_EXECUTION_DEEPSEEK_API_KEY"),
            execution_minimax_api_key=os.getenv("VIRGO_EXECUTION_MINIMAX_API_KEY"),
            execution_qwen_api_key=os.getenv("VIRGO_EXECUTION_QWEN_API_KEY"),
            workspace=root,
            state_dir=state_dir,
            local_base_url=os.getenv("VIRGO_LOCAL_BASE_URL", "http://127.0.0.1:8000/v1"),
            local_model=os.getenv("VIRGO_LOCAL_MODEL", "deepseek-flash"),
            local_api_key=os.getenv("VIRGO_LOCAL_API_KEY"),
            offline=mode not in {"0", "false", "no"},
            request_timeout=float(os.getenv("VIRGO_REQUEST_TIMEOUT", "60")),
            log_dir=Path(os.getenv("VIRGO_LOG_DIR", str(state_dir / "logs"))).resolve(),
            log_level=os.getenv("VIRGO_LOG_LEVEL", "INFO"),
            log_console=os.getenv("VIRGO_LOG_CONSOLE", "0").strip().lower() in {"1", "true", "yes"},
            router_provider=router_provider,
            router_base_url=os.getenv("VIRGO_ROUTER_BASE_URL", "https://api.openai.com/v1"),
            router_model=os.getenv("VIRGO_ROUTER_MODEL", "gpt-5.4"),
            router_api_key=os.getenv("VIRGO_ROUTER_API_KEY") if router_provider in OFFICIAL_PROVIDER_URLS else router_api_key,
            router_timeout=float(os.getenv("VIRGO_ROUTER_TIMEOUT", os.getenv("VIRGO_REQUEST_TIMEOUT", "60"))),
            execution_provider=os.getenv("VIRGO_EXECUTION_PROVIDER", "auto").strip().lower(),
            execution_base_url=os.getenv("VIRGO_EXECUTION_BASE_URL", os.getenv("VIRGO_LOCAL_BASE_URL", "http://127.0.0.1:8000/v1")),
            execution_model=os.getenv("VIRGO_EXECUTION_MODEL", os.getenv("VIRGO_LOCAL_MODEL", "deepseek-flash")),
            execution_api_key=os.getenv("VIRGO_EXECUTION_API_KEY") if os.getenv("VIRGO_EXECUTION_PROVIDER", "auto").strip().lower() in OFFICIAL_PROVIDER_URLS else execution_api_key,
            execution_timeout=float(os.getenv("VIRGO_EXECUTION_TIMEOUT", os.getenv("VIRGO_REQUEST_TIMEOUT", "60"))),
            jev_base_url=os.getenv("VIRGO_JEV_BASE_URL", "https://api.typesafe.ai/v1/systemone"),
            jev_model=os.getenv("VIRGO_JEV_MODEL", "jev-latest"),
            jev_api_key=os.getenv("VIRGO_JEV_API_KEY"),
            jev_timeout=float(os.getenv("VIRGO_JEV_TIMEOUT", "30")),
            max_tool_calls=max(1, int(os.getenv("VIRGO_MAX_TOOL_CALLS", "50"))),
            collaboration_tool_call_limit=max(1, int(os.getenv("VIRGO_COLLABORATION_TOOL_CALL_LIMIT", "30"))),
            context_compaction_threshold_tokens=max(1, int(os.getenv("VIRGO_CONTEXT_COMPACTION_TOKENS", "200000"))),
            deepseek_v4_flash_pricing=os.getenv("VIRGO_DEEPSEEK_V4_FLASH_PRICING", "nonpeak").strip().lower(),
            inferai_base_url=os.getenv("VIRGO_INFERAI_BASE_URL", "https://inferaiapi.com/v1"),
            inferai_api_key=inferai_api_key,
            inferai_max_retries=max(0, int(os.getenv("VIRGO_INFERAI_MAX_RETRIES", "2"))),
            inferai_wire_api=os.getenv("VIRGO_INFERAI_WIRE_API", "chat_completions").strip().lower(),
            initial_context_provider=os.getenv("VIRGO_INITIAL_CONTEXT_PROVIDER", "auto").strip().lower(),
            initial_context_base_url=os.getenv("VIRGO_INITIAL_CONTEXT_BASE_URL", "https://api.deepseek.com"),
            initial_context_model=os.getenv("VIRGO_INITIAL_CONTEXT_MODEL") or None,
            initial_context_api_key=(os.getenv("VIRGO_INITIAL_CONTEXT_API_KEY") if os.getenv("VIRGO_INITIAL_CONTEXT_PROVIDER", "auto").strip().lower() in OFFICIAL_PROVIDER_URLS else os.getenv("VIRGO_INITIAL_CONTEXT_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or execution_api_key),
            initial_context_timeout=float(os.getenv("VIRGO_INITIAL_CONTEXT_TIMEOUT", "60")),
        )


def _load_dotenv(path: Path) -> None:
    """读取本地配置文件；已有环境变量优先。"""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def new_storage_run_id() -> str:
    """为一次评测脚本运行生成可读且不易碰撞的目录名。"""
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
