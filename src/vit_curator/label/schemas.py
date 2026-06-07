"""Schema types for vLLM client responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatUsage:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True)
class ChatResult:
    content: str
    finish_reason: str | None
    latency_ms: float
    ttft_ms: float | None
    usage: ChatUsage | None


# Alias for backward compatibility
Usage = ChatUsage


def parse_usage(resp_json: dict[str, Any]) -> ChatUsage | None:
    usage = resp_json.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    return ChatUsage(
        prompt_tokens=int(prompt_tokens) if isinstance(prompt_tokens, int) else None,
        completion_tokens=int(completion_tokens) if isinstance(completion_tokens, int) else None,
        total_tokens=int(total_tokens) if isinstance(total_tokens, int) else None,
    )


def extract_content(resp_json: dict[str, Any]) -> tuple[str, str | None, ChatUsage | None]:
    """Extract content, finish_reason, and usage from an OpenAI-style response."""
    choices = resp_json.get("choices") or []
    if not choices:
        raise ValueError("missing choices in response")
    c0 = choices[0]
    finish_reason = c0.get("finish_reason")
    msg = c0.get("message") or {}
    content = msg.get("content")
    if content is None:
        raise ValueError("missing message.content in response")
    if not isinstance(content, str):
        raise ValueError(f"unexpected content type: {type(content)}")
    return content, finish_reason, parse_usage(resp_json)
