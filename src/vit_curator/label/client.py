"""VLM OpenAI-compatible HTTP client for structured label inference."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import orjson

from vit_curator.label.schemas import ChatResult, extract_content, parse_usage


class VllmStructuredMode:
    STRUCTURED_OUTPUTS = "structured_outputs"
    GUIDED_JSON = "guided_json"


@dataclass
class VllmClient:
    base_url: str
    timeout_s: float = 120.0
    _mode: str = VllmStructuredMode.STRUCTURED_OUTPUTS

    def _endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/v1/chat/completions"

    async def _post(self, client: httpx.AsyncClient, payload: dict[str, Any]) -> dict[str, Any]:
        r = await client.post(self._endpoint(), json=payload)
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        return r.json()

    def _payload(
        self,
        *,
        model: str,
        prompt: str,
        image_path: Path,
        max_tokens: int,
        temperature: float,
        schema: dict[str, Any] | None,
        structured_mode: str,
        stream: bool,
        stream_include_usage: bool,
    ) -> dict[str, Any]:
        image_url = image_path.as_posix()
        if not image_url.startswith("file://"):
            image_url = "file://" + image_url

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if stream:
            payload["stream"] = True
            if stream_include_usage:
                payload["stream_options"] = {"include_usage": True}

        if schema is not None:
            if structured_mode == VllmStructuredMode.STRUCTURED_OUTPUTS:
                payload["extra_body"] = {"structured_outputs": {"json": schema}}
            else:
                payload["extra_body"] = {"guided_json": schema}

        return payload

    async def _stream_response(
        self,
        *,
        client: httpx.AsyncClient,
        payload: dict[str, Any],
        start: float,
    ) -> ChatResult:
        content_parts: list[str] = []
        finish_reason: str | None = None
        usage = None
        ttft_ms: float | None = None

        async with client.stream("POST", self._endpoint(), json=payload) as r:
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
            async for line in r.aiter_lines():
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = orjson.loads(data)
                except Exception as e:
                    raise ValueError(f"invalid stream JSON chunk: {data[:200]!r}") from e

                choices = chunk.get("choices") or []
                if choices:
                    c0 = choices[0]
                    delta = c0.get("delta") or {}
                    piece = delta.get("content")
                    if isinstance(piece, str):
                        if ttft_ms is None and piece:
                            ttft_ms = (time.perf_counter() - start) * 1000.0
                        content_parts.append(piece)
                    fr = c0.get("finish_reason")
                    if fr is not None:
                        finish_reason = fr

                if usage is None:
                    usage = parse_usage(chunk)

        content = "".join(content_parts)
        if not content:
            raise ValueError("empty content in stream response")
        latency_ms = (time.perf_counter() - start) * 1000.0
        return ChatResult(
            content=content,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            usage=usage,
        )

    async def classify_one(
        self,
        *,
        http: httpx.AsyncClient,
        model: str,
        prompt: str,
        image_path: Path,
        max_tokens: int,
        temperature: float,
        schema: dict[str, Any] | None,
        stream: bool,
        stream_include_usage: bool,
    ) -> ChatResult:
        start = time.perf_counter()
        last_err: Exception | None = None
        for mode in (self._mode, VllmStructuredMode.GUIDED_JSON):
            try:
                payload = self._payload(
                    model=model,
                    prompt=prompt,
                    image_path=image_path,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    schema=schema,
                    structured_mode=mode,
                    stream=stream,
                    stream_include_usage=stream_include_usage,
                )
                if stream:
                    result = await self._stream_response(client=http, payload=payload, start=start)
                else:
                    resp_json = await self._post(http, payload)
                    content, finish_reason, usage = extract_content(resp_json)
                    latency_ms = (time.perf_counter() - start) * 1000.0
                    result = ChatResult(
                        content=content,
                        finish_reason=finish_reason,
                        latency_ms=latency_ms,
                        ttft_ms=None,
                        usage=usage,
                    )
                self._mode = mode  # persist last working mode
                return result
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"classify failed in all structured modes: {last_err}")

    async def probe(self) -> str:
        """Best-effort probe to detect structured mode support."""
        async with httpx.AsyncClient(timeout=self.timeout_s) as http:
            payload = {
                "model": "dummy",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 4,
                "temperature": 0.0,
                "extra_body": {"structured_outputs": {"choice": ["pong"]}},
            }
            try:
                await self._post(http, payload)
                self._mode = VllmStructuredMode.STRUCTURED_OUTPUTS
                return self._mode
            except Exception:
                payload["extra_body"] = {
                    "guided_json": {
                        "type": "object",
                        "properties": {"pong": {"type": "string"}},
                        "required": ["pong"],
                        "additionalProperties": False,
                    }
                }
                await self._post(http, payload)
                self._mode = VllmStructuredMode.GUIDED_JSON
                return self._mode
