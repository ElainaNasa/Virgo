"""根据配置创建执行模型和路由模型服务。"""

from __future__ import annotations

import re

from virgo.config import Settings, OFFICIAL_PROVIDER_URLS
from virgo.providers.base import ModelProvider
from virgo.providers.offline import OfflineProvider
from virgo.providers.openai_compatible import OpenAICompatibleProvider
from virgo.providers.openai_responses import OpenAIResponsesProvider
from virgo.providers.jev import JevProvider


def _official_provider(settings: Settings, provider: str, stage: str) -> ModelProvider:
    """官方渠道固定使用 Chat Completions，阶段配置优先于渠道配置。"""
    base_url = getattr(settings, f"{provider}_base_url")
    stage_url = getattr(settings, f"{stage}_base_url")
    default_urls = {"execution": {"", settings.local_base_url},
                    "router": {"https://api.openai.com/v1"},
                    "initial_context": {"https://api.deepseek.com"}}
    if stage_url and stage_url not in default_urls[stage]:
        base_url = stage_url
    dedicated_key = getattr(settings, f"execution_{provider}_api_key", None) if stage == "execution" else None
    api_key = dedicated_key or getattr(settings, f"{stage}_api_key") or getattr(settings, f"{provider}_api_key")
    if not api_key:
        raise ValueError(f"官方 {provider} {stage} 服务需要 VIRGO_{stage.upper()}_API_KEY 或 VIRGO_{provider.upper()}_API_KEY")
    return OpenAICompatibleProvider(base_url, api_key, getattr(settings, f"{stage}_timeout"))


def build_execution_provider(settings: Settings) -> ModelProvider:
    provider = settings.execution_provider
    if provider == "auto":
        if not settings.offline:
            return OfficialModelProvider(settings)
        provider = "offline"
    if provider in OFFICIAL_PROVIDER_URLS:
        return _official_provider(settings, provider, "execution")
    base_url = settings.execution_base_url or settings.local_base_url
    api_key = settings.inferai_api_key if provider == "inferai" else (settings.execution_api_key or settings.local_api_key)
    timeout = settings.execution_timeout or settings.request_timeout
    if provider == "offline":
        return OfflineProvider()
    if provider == "local":
        return OpenAICompatibleProvider(base_url, api_key, timeout)
    if provider == "inferai":
        if not api_key:
            raise ValueError("已选择 InferAI 执行模型，但未设置 VIRGO_INFERAI_API_KEY、VIRGO_EXECUTION_API_KEY 或 OPENAI_API_KEY")
        if settings.inferai_wire_api == "responses":
            return OpenAIResponsesProvider(settings.inferai_base_url, api_key, timeout, settings.inferai_max_retries)
        if settings.inferai_wire_api in {"chat_completions", "chat", "chat/completions"}:
            return OpenAICompatibleProvider(settings.inferai_base_url, api_key, timeout)
        raise ValueError(f"未知 InferAI wire API：{settings.inferai_wire_api}")
    if provider in {"cloud", "openai", "openai_compatible", "openrouter", "deepseek"}:
        if not api_key:
            raise ValueError("已选择云端执行模型，但未设置 VIRGO_EXECUTION_API_KEY 或 VIRGO_LOCAL_API_KEY")
        return OpenAICompatibleProvider(base_url, api_key, timeout)
    raise ValueError(f"未知执行模型服务：{settings.execution_provider}")


def build_router_provider(settings: Settings) -> ModelProvider:
    provider = settings.router_provider
    if provider == "auto":
        provider = "offline" if settings.offline else "local"
    if provider == "offline":
        return OfflineProvider()
    if provider in OFFICIAL_PROVIDER_URLS:
        return _official_provider(settings, provider, "router")
    if provider == "jev":
        if not settings.jev_api_key:
            raise ValueError("已选择 Jev 路由模型，但未设置 VIRGO_JEV_API_KEY")
        return JevProvider(settings.jev_base_url, settings.jev_api_key, settings.jev_model, settings.jev_timeout)
    if provider == "local":
        return OpenAICompatibleProvider(settings.local_base_url, settings.local_api_key, settings.request_timeout)
    if provider in {"cloud", "openai", "openai_compatible", "openrouter", "deepseek"}:
        if not settings.router_api_key:
            raise ValueError("已选择云端路由模型，但未设置 VIRGO_ROUTER_API_KEY 或 OPENAI_API_KEY")
        return OpenAICompatibleProvider(settings.router_base_url, settings.router_api_key, settings.router_timeout)
    raise ValueError(f"未知路由模型服务：{settings.router_provider}")


def build_initial_context_provider(settings: Settings) -> ModelProvider:
    """初始上下文可独立配置；auto 保留原来的服务选择规则。"""
    if settings.initial_context_provider in OFFICIAL_PROVIDER_URLS:
        return _official_provider(settings, settings.initial_context_provider, "initial_context")
    if settings.initial_context_provider == "deepseek":
        if not settings.initial_context_api_key:
            raise ValueError("DeepSeek 初始上下文需要 VIRGO_INITIAL_CONTEXT_API_KEY、DEEPSEEK_API_KEY 或 VIRGO_EXECUTION_API_KEY")
        return OpenAICompatibleProvider(settings.initial_context_base_url,
            settings.initial_context_api_key, settings.initial_context_timeout)
    if settings.initial_context_provider != "auto":
        raise ValueError(f"未知初始上下文服务：{settings.initial_context_provider}")
    if settings.router_provider == "jev":
        return build_execution_provider(settings)
    return build_router_provider(settings)


class OfficialModelProvider:
    """每次按所选模型的官方渠道分发，避免所有模型发往同一个服务。"""
    work_throttle_at_transport = True

    def __init__(self, settings):
        self.settings = settings

    def complete(self, **kwargs):
        from virgo.models import ModelPool
        channel = ModelPool().get(kwargs["model"]).provider
        if kwargs["model"] == "glm-5.3-flash":
            # 模型归属仍用于最终失败熔断；传输优先走 Qwen，渠道不可用时先尝试智谱，再交由外层熔断。
            try:
                return self._complete_channel("qwen", kwargs)
            except Exception as exc:
                message = str(exc).lower()
                http = re.search(r"http\s+(\d{3})", message)
                code = int(http.group(1)) if http else None
                unavailable = (code in {401, 403, 404, 408, 429} or (code is not None and code >= 500)) if code is not None else (
                    isinstance(exc, (TimeoutError, ConnectionError)) or any(term in message for term in (
                        "timed out", "timeout", "urlerror", "unexpected_eof_while_reading",
                        "eof occurred in violation of protocol", "connection reset", "connection refused",
                        "connection aborted", "remote end closed", "name resolution", "nodename nor servname",
                    ))
                )
                if not unavailable:
                    raise
                response = self._complete_channel("zhipu", kwargs)
                # 保留实际渠道，便于日志定位，避免只看到模型归属渠道。
                if hasattr(response, "raw") and isinstance(response.raw, dict):
                    response.raw["_virgo_provider_fallback"] = {
                        "from": "qwen", "to": "zhipu", "reason": str(exc)[:1000],
                    }
                return response
        return self._complete_channel(channel, kwargs)

    def _complete_channel(self, channel, kwargs):
        """按渠道独立读取地址和密钥，降级时保留原始模型和请求参数。"""
        base_url = getattr(self.settings, f"{channel}_base_url")
        api_key = getattr(self.settings, f"execution_{channel}_api_key") or getattr(self.settings, f"{channel}_api_key")
        if not api_key:
            raise ValueError(f"模型 {kwargs['model']} 的官方渠道需要 VIRGO_EXECUTION_{channel.upper()}_API_KEY 或 VIRGO_{channel.upper()}_API_KEY")
        return OpenAICompatibleProvider(base_url, api_key, self.settings.execution_timeout).complete(**kwargs)
