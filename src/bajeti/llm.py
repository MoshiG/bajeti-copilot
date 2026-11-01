"""One small interface for every LLM backend.

The rest of the codebase only ever calls `LLMClient.chat(...)`. Swapping a vLLM server on a
rented GPU for a hosted API is a config change, and tests run against `FakeLLM` with no network.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from bajeti.config import LLMConfig

Message = dict[str, Any]          # {"role": "user"|"assistant"|"system"|"tool", "content": ...}
ToolSpec = dict[str, Any]         # OpenAI-style function schema


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResult:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMClient(Protocol):
    def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None) -> ChatResult: ...


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    """Small open models sometimes emit slightly broken JSON; fail soft rather than crash a run."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        return {"_raw": raw, "_error": "invalid_json"}


class OpenAICompatClient:
    """Talks to anything that speaks the OpenAI chat-completions API (vLLM, hosted providers)."""

    def __init__(self, cfg: LLMConfig, client: Any | None = None) -> None:
        self.cfg = cfg
        if client is None:
            from openai import OpenAI  # imported lazily so tests never need network setup

            client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout_s)
        self._client = client

    def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None) -> ChatResult:
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=_parse_arguments(tc.function.arguments),
            )
            for tc in (msg.tool_calls or [])
        ]
        usage = getattr(resp, "usage", None)
        return ChatResult(
            content=msg.content,
            tool_calls=calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


class FakeLLM:
    """Scripted LLM for tests: returns queued responses in order and records every call."""

    def __init__(self, responses: Iterable[ChatResult | str]) -> None:
        self._queue = [r if isinstance(r, ChatResult) else ChatResult(content=r) for r in responses]
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None) -> ChatResult:
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self._queue:
            raise RuntimeError("FakeLLM ran out of scripted responses")
        return self._queue.pop(0)


def build_llm(cfg: LLMConfig) -> LLMClient:
    return OpenAICompatClient(cfg)
